"""Run the frozen registration study with sampled POSIX resource controls.

This is an integrity/resource guard for reviewed code, not an OS sandbox or a
hard-real-time guarantee. Raw output is retained on failure; retries need a new
output root. The numerical study and its scientific acceptance rules are unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

GIB = 1024**3
MIB = 1024**2
SCHEMA = "noether-twin-registration-development-v1"
VERSIONS = {"numpy": "2.4.6", "scipy": "1.17.1", "hcipy": "0.7.0", "dLux": "0.15.1",
            "jax": "0.10.2", "jaxlib": "0.10.2"}
FLAGS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
         "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "JAX_PLATFORMS": "cpu",
         "JAX_ENABLE_X64": "true", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
         "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INDEX": "1", "UV_OFFLINE": "1"}
UNIT_IDENTITIES = [
    {"id": f"shift-{state}-mean", "state_index": state, "kind": "expected_mean",
     "calibration_seed": None, "prediction_seed": None} for state in range(3)
] + [
    {"id": f"shift-{state}-noise-{repeat:02d}", "state_index": state, "kind": "poisson",
     "calibration_seed": 911110000 + state * 100 + repeat,
     "prediction_seed": 921110000 + state * 100 + repeat}
    for state in range(3) for repeat in range(16)
]
EXPECTED_UNITS = [row["id"] for row in UNIT_IDENTITIES]
EXPECTED_PACKET_FILES = (
    {"protocol.json", "events.jsonl", "analysis.json", "manifest.json"}
    | {f"{unit}-{kind}.json" for unit in EXPECTED_UNITS for kind in ("fit", "outcome")}
    | {f"arrays/{unit}-{kind}.npz" for unit in EXPECTED_UNITS for kind in ("calibration", "predictions", "scoring")}
    | {f"arrays/{kind}-{index:06d}.npz" for kind, count in (("reference", 72), ("prediction", 459))
       for index in range(count)}
)


class GuardError(RuntimeError):
    """A stable failure code without machine-specific paths."""


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 1800
    rss_bytes: int = 4 * GIB
    packet_logical_bytes: int = 512 * MIB
    packet_allocated_bytes: int = GIB
    runtime_bytes: int = 256 * MIB


def require(condition, reason):
    if not condition:
        raise GuardError(reason)


def digest(path, *, maximum=512 * MIB):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "hash_input_bound")
        total, result = 0, hashlib.sha256()
        while block := os.read(descriptor, 65536):
            total += len(block)
            require(total <= maximum, "hash_input_growth")
            result.update(block)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "hash_input_changed")
        return result.hexdigest()
    finally:
        os.close(descriptor)


def read_json(path, maximum=4 * MIB):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, "json_input_bound")
        data = bytearray()
        while chunk := os.read(descriptor, min(65536, maximum + 1 - len(data))):
            data.extend(chunk)
            require(len(data) <= maximum, "json_input_growth")
        return json.loads(data)
    finally:
        os.close(descriptor)


def safe_relative(value):
    require(isinstance(value, str), "relative_path_type")
    path = PurePosixPath(value)
    require(value == path.as_posix() and not path.is_absolute() and ".." not in path.parts
            and value not in ("", "."), "unsafe_relative_path")
    return path


def entry_type(info, *, live):
    regular, directory = stat.S_ISREG(info.st_mode), stat.S_ISDIR(info.st_mode)
    require(regular or directory, "nonregular_output")
    require(not regular or info.st_nlink <= (2 if live else 1), "output_hard_link")
    return regular, directory


def inventory(root, *, live=False):
    """No-follow inode accounting; final closure rejects transient hard links."""
    seen, rows = set(), {}
    logical, allocated, entries = 0, 0, 0

    def walk(descriptor, prefix):
        nonlocal logical, allocated, entries
        with os.scandir(descriptor) as iterator:
            for entry in iterator:
                entries += 1
                require(entries <= 50000, "output_entry_bound")
                relative = prefix + entry.name
                temporary = entry.name.startswith(".pending-") and len(entry.name) == 41 and all(
                    char in "0123456789abcdef" for char in entry.name[9:])
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    if live and temporary:
                        continue
                    raise GuardError("output_entry_disappeared") from None
                # The producer briefly links .pending-UUID to its final name.
                regular, directory = entry_type(info, live=live)
                key = info.st_dev, info.st_ino
                rows[relative] = (info.st_dev, info.st_ino, info.st_size, info.st_blocks, info.st_mtime_ns)
                if key not in seen:
                    seen.add(key)
                    logical += info.st_size if regular else 0
                    allocated += info.st_blocks * 512
                if directory:
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    try:
                        current = os.fstat(child)
                        require((current.st_dev, current.st_ino) == key, "output_directory_replaced")
                        walk(child, relative + "/")
                    finally:
                        os.close(child)

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        allocated += os.fstat(descriptor).st_blocks * 512
        walk(descriptor, "")
    finally:
        os.close(descriptor)
    return {"logical": logical, "allocated": allocated, "entries": rows}


def cpu_identity():
    model = platform.processor()
    if sys.platform == "darwin":
        model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3).decode().strip()
    elif Path("/proc/cpuinfo").is_file():
        with Path("/proc/cpuinfo").open() as stream:
            data = stream.read(65536)
        model = next((line.split(":", 1)[1].strip() for line in data.splitlines()
                      if line.startswith(("model name", "Hardware")) and ":" in line), model)
    require(len(model) <= 256, "cpu_identity_bound")
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "cpu": model}


def native_identity():
    """Bounded hashes of native NumPy/SciPy/JAX libraries; no package imports."""
    result, total, entries = {}, 0, 0
    for package in ("numpy", "scipy", "jax", "jaxlib"):
        spec = importlib.util.find_spec(package)
        require(spec is not None and spec.origin is not None, "native_package_missing")
        package_root = Path(spec.origin).parent
        for root in (package_root, package_root.parent / (package + ".libs")):
            if not root.exists():
                continue
            pending = [root]
            while pending:
                directory = pending.pop()
                with os.scandir(directory) as iterator:
                    for entry in iterator:
                        entries += 1
                        require(entries <= 50000, "native_inventory_entry_bound")
                        info = entry.stat(follow_symlinks=False)
                        require(not stat.S_ISLNK(info.st_mode), "native_inventory_symlink")
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode) and (entry.name.endswith((".so", ".dylib")) or ".so." in entry.name):
                            total += info.st_size
                            require(total <= GIB and info.st_size <= 512 * MIB, "native_inventory_byte_bound")
                            relative = str(Path(entry.path).relative_to(package_root.parent))
                            require(relative not in result, "native_inventory_duplicate")
                            result[relative] = {"bytes": info.st_size, "sha256": digest(entry.path)}
    require(result, "native_inventory_empty")
    return result


def repository_binding(repository, *, require_git=False):
    """Never discover a parent repository for an unpacked source snapshot.

    Source archives remain runnable. Release tooling can require an explicit
    checkout rooted here before issuing any Git command.
    """
    repository = repository.resolve(strict=True)
    marker = repository / ".git"
    require(not marker.is_symlink(), "repository_git_marker_symlink")
    if not marker.exists():
        require(not require_git, "explicit_repository_required")
        return {"kind": "source_snapshot", "git_discovery": "disabled"}
    require(marker.is_dir() or marker.is_file(), "repository_git_marker_type")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CEILING_DIRECTORIES=str(repository.parent), GIT_OPTIONAL_LOCKS="0")
    root = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--show-toplevel"], env=env, timeout=5,
    ).decode().strip()
    require(Path(root).resolve(strict=True) == repository, "repository_root_mismatch")
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"], env=env, timeout=5,
    ).decode().strip()
    require(len(head) in (40, 64) and all(char in "0123456789abcdef" for char in head), "repository_head_format")
    return {"kind": "explicit_checkout", "head": head}


def source_identity(repository):
    """Check the actual release tree without importing scientific packages."""
    provenance = read_json(repository / "results/registration_development_v1/provenance.json")
    for name, expected in provenance["source_files"].items():
        require(digest(repository / safe_relative(name)) == expected, "acquisition_source_changed")
    source_map = read_json(repository / "source-map.json")
    require(len(source_map["files"]) <= 256, "source_inventory_entry_bound")
    names = set(source_map["files"]) | {"source-map.json"}
    require("scripts/run_registration.py" in names and "tests/test_registration_launcher.py" in names,
            "launcher_source_inventory_missing")
    python_paths = {str(path.relative_to(repository)) for folder in ("src", "scripts", "tests")
                    for path in (repository / folder).rglob("*.py")}
    require(len(python_paths) <= 256, "python_source_inventory_bound")
    require(python_paths == {name for name in names if name.endswith(".py")}, "python_source_inventory_changed")
    total = sum((repository / safe_relative(name)).stat().st_size for name in names)
    require(total <= 32 * MIB, "source_inventory_byte_bound")
    source = {name: digest(repository / safe_relative(name), maximum=8 * MIB) for name in sorted(names)}
    for name, row in source_map["files"].items():
        require(source[name] == row["snapshot_sha256"], "snapshot_source_changed")
    return {"source_sha256": source, "acquisition_source_sha256": provenance["source_files"],
            "repository": repository_binding(repository)}


def release_identity(repository):
    require(sys.version_info[:3] == (3, 11, 15), "python_version_mismatch")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "scientific_version_mismatch")
    return {**source_identity(repository),
            "python_version": platform.python_version(), "interpreter_sha256": digest(Path(sys.executable).resolve()),
            "platform": cpu_identity(), "versions": versions, "native_libraries": native_identity(),
            "numerical_environment": FLAGS, "replay_claim": "provenance_only;not_a_bitwise_replay_guarantee"}


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_group(child):
    for sig, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()
            if not group_exists(child.pid):
                child.wait(timeout=1)
                return
            time.sleep(0.05)
    raise GuardError("owned_group_survived_cleanup")


def rss_bytes(pgid):
    rows = subprocess.check_output(["ps", "-axo", "pgid=,rss="], timeout=3, env={"LC_ALL": "C"}).decode()
    require(len(rows) <= 4 * MIB, "process_inventory_bound")
    return sum(int(parts[1]) * 1024 for row in rows.splitlines()
               if len(parts := row.split()) == 2 and int(parts[0]) == pgid)


def verify_packet(packet, acquisition_source):
    before = inventory(packet)
    manifest = read_json(packet / "manifest.json")
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete_development", "manifest_schema")
    declared = manifest["files"]
    require(isinstance(declared, dict) and len(declared) < 2000, "manifest_entry_bound")
    actual = {name for name in before["entries"] if (packet / name).is_file()}
    require(actual == set(declared) | {"manifest.json"}, "manifest_inventory")
    require(actual == EXPECTED_PACKET_FILES, "packet_artifact_coverage")
    for name, expected in declared.items():
        require(digest(packet / safe_relative(name)) == expected, "packet_hash_mismatch")
    header = read_json(packet / "protocol.json")
    require(header["source"] == acquisition_source, "packet_source_binding")
    protocol = header["protocol"]
    require(protocol["schema"] == SCHEMA and protocol["units"] == UNIT_IDENTITIES,
            "protocol_unit_identities")
    actual_outcomes = {name for name in actual if name.endswith("-outcome.json")}
    require(actual_outcomes == {name + "-outcome.json" for name in EXPECTED_UNITS}, "outcome_unit_identities")
    for expected in protocol["units"]:
        outcome = read_json(packet / (expected["id"] + "-outcome.json"))
        require(all(outcome[key] == value for key, value in expected.items()), "outcome_identity_mismatch")
    analysis = read_json(packet / "analysis.json")
    require(analysis["schema"] == SCHEMA and analysis["status"] == "complete_development"
            and analysis["units"] == 51 and analysis["noisy_units"] == 48 and analysis["mean_diagnostics"] == 3
            and analysis["physical_states"] == 3 and protocol["selection_or_evaluation_cohort"] is False,
            "analysis_coverage")
    require(analysis["counts"]["reference"] == 72 and analysis["counts"]["prediction"] == 459
            and 2295 <= analysis["counts"]["objective"] <= 48960, "analysis_callback_coverage")
    require(inventory(packet) == before, "packet_changed_during_verification")
    return {"units": 51, "manifest_sha256": digest(packet / "manifest.json"),
            "scope": "execution_and_file_integrity;not_independent_optical_recomputation"}


def _write_json(path, data):
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return len(payload)


def supervise(output, command, *, environment, identity, minimum_free, reserve_paths, limits=Limits(),
              packet_check=verify_packet, command_identity=None):
    """Internal supervision entry point; CLI fixes the acquisition command."""
    started = time.monotonic()
    command_identity = {"argv_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                        **(command_identity or {"purpose": "synthetic_guard_check"})}
    output.mkdir(exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    for directory in ("tmp", "cache", "jax-cache", "mpl-cache"):
        (runtime / directory).mkdir()
    packet = output / "packet"
    paths = [output, output.parent, *reserve_paths]
    bindings = [(path.stat().st_dev, path.stat().st_ino) for path in paths]
    runtime_binding = (runtime.stat().st_dev, runtime.stat().st_ino)
    child, reason, requested, evidence = None, None, None, None
    peak, free_minimum = 0, None
    initial = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupt(signum, frame):
        nonlocal requested
        requested = signum

    def guard():
        nonlocal peak, free_minimum
        require(requested is None, "interrupted")
        require(time.monotonic() - started <= limits.wall_seconds, "wall_limit")
        require((runtime.stat().st_dev, runtime.stat().st_ino) == runtime_binding, "runtime_identity_changed")
        require({path.name for path in output.iterdir()} <= {"packet", "runtime"}, "unexpected_output_root_entry")
        for path, bound in zip(paths, bindings, strict=True):
            require((path.stat().st_dev, path.stat().st_ino) == bound, "reserve_path_identity_changed")
            volume = os.statvfs(path)
            available = volume.f_bavail * volume.f_frsize
            free_minimum = available if free_minimum is None else min(free_minimum, available)
            require(available >= minimum_free, "free_space_reserve")
        if packet.exists():
            size = inventory(packet, live=child is not None and child.poll() is None)
            require(size["logical"] <= limits.packet_logical_bytes, "packet_logical_limit")
            require(size["allocated"] <= limits.packet_allocated_bytes, "packet_allocated_limit")
        size = inventory(runtime, live=child is not None and child.poll() is None)
        require(max(size["logical"], size["allocated"]) <= limits.runtime_bytes, "runtime_output_limit")
        if child is not None:
            peak = max(peak, rss_bytes(child.pid))
            require(peak <= limits.rss_bytes, "group_rss_limit")

    for sig in handlers:
        signal.signal(sig, interrupt)
    try:
        guard()
        initial = identity()
        guard()
        _write_json(runtime / "admission.json", {"identity": initial, "limits": asdict(limits),
                    "minimum_free_bytes": minimum_free, "reserve_path_count": len(paths),
                    "command": command_identity})
        guard()
        env = dict(environment)
        env.update(TMPDIR=str(runtime / "tmp"), XDG_CACHE_HOME=str(runtime / "cache"),
                   JAX_COMPILATION_CACHE_DIR=str(runtime / "jax-cache"), MPLCONFIGDIR=str(runtime / "mpl-cache"))
        with (runtime / "run.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            _write_json(runtime / "owned-process.json", {"pid": child.pid, "pgid": child.pid})
            while child.poll() is None:
                guard()
                time.sleep(0.2)
            require(child.returncode == 0, "child_nonzero_exit")
            require(not group_exists(child.pid), "descendants_survived_leader")
        guard()
        require(identity() == initial, "source_or_runtime_identity_changed")
        evidence = packet_check(packet, initial["acquisition_source_sha256"])
        guard()
    except BaseException as error:
        reason = str(error) if isinstance(error, GuardError) else type(error).__name__
    finally:
        if child is not None and group_exists(child.pid):
            try:
                terminate_group(child)
            except BaseException:
                reason = (reason or "cleanup") + ";owned_group_survived_cleanup"
        absent = child is None or not group_exists(child.pid)
        highwater = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if highwater > limits.rss_bytes:
            reason = (reason or "final") + ";child_highwater_rss_limit"
        try:
            final = inventory(runtime)
            require(inventory(runtime) == final, "unstable_runtime_inventory")
            if packet.exists():
                final_packet = inventory(packet)
                require(inventory(packet) == final_packet, "unstable_packet_inventory")
                require(final_packet["logical"] <= limits.packet_logical_bytes, "final_packet_logical_limit")
                require(final_packet["allocated"] <= limits.packet_allocated_bytes, "final_packet_allocated_limit")
            else:
                final_packet = None
        except BaseException as error:
            reason = (reason or "final") + ";" + (str(error) if isinstance(error, GuardError) else type(error).__name__)
            final = {"logical": None, "allocated": None}
            final_packet = None
        elapsed = time.monotonic() - started
        if requested is not None or elapsed > limits.wall_seconds:
            reason = reason or "interruption_or_wall_limit_at_closure"
        receipt = {"schema": "noether-registration-supervision-v1", "status": "completed" if reason is None else "stopped",
                   "reason": reason, "returncode": child.returncode if child else None,
                   "owned_pgid": child.pid if child else None, "group_absent": absent,
                   "elapsed_seconds": elapsed, "peak_group_rss_bytes": peak, "child_highwater_rss_bytes": highwater,
                   "minimum_free_bytes": free_minimum, "evidence": evidence, "scientific_promotion": False,
                   "command": command_identity,
                   "runtime_before_receipt": {key: final[key] for key in ("logical", "allocated")},
                   "packet_bytes": None if final_packet is None else {key: final_packet[key] for key in ("logical", "allocated")}}
        try:
            require((output.stat().st_dev, output.stat().st_ino) == bindings[0], "output_identity_changed_at_receipt")
            require((runtime.stat().st_dev, runtime.stat().st_ino) == runtime_binding, "runtime_identity_changed_at_receipt")
            if {path.name for path in output.iterdir()} - {"packet", "runtime"}:
                receipt.update(status="stopped", reason=(receipt["reason"] or "final") + ";unexpected_output_root_entry")
            temporary = runtime / (".pending-" + uuid.uuid4().hex)
            _write_json(temporary, receipt)
            try:
                charged = inventory(runtime)
                if max(charged["logical"], charged["allocated"]) > limits.runtime_bytes:
                    receipt.update(status="stopped", reason=(receipt["reason"] or "final") + ";terminal_receipt_runtime_limit")
            except BaseException:
                receipt.update(status="stopped", reason=(receipt["reason"] or "final") + ";terminal_inventory_unavailable")
            if receipt["status"] == "stopped":
                payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                with temporary.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            # Publication is exclusive. Temporary preparation may change; the
            # final receipt never replaces an earlier terminal record.
            os.link(temporary, output / "receipt.json")
            temporary.unlink()
            descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-free-gib", required=True, type=int)
    parser.add_argument("--reserve-path", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    require(sys.platform in ("darwin", "linux"), "unsupported_platform")
    require(args.min_free_gib >= 1, "minimum_free_must_be_positive")
    repository = Path(__file__).resolve().parents[1]
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "output_already_exists")
    require(output.parent.is_dir(), "output_parent_missing")
    reserve_paths = [path.resolve(strict=True) for path in args.reserve_path]
    env = {key: os.environ[key] for key in ("HOME", "USER", "PATH", "LANG", "LC_ALL") if key in os.environ}
    env.update(FLAGS)
    bootstrap = """import sys,runpy,json
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2])
sys.path.insert(0,str(root/'src'))
import noether_optics
assert Path(noether_optics.__file__).resolve()==root/'src/noether_optics/__init__.py'
sys.argv=['noether_optics.bench.twin_registration',str(output)]
try: runpy.run_module('noether_optics.bench.twin_registration',run_name='__main__')
finally:
 origins={n:str(Path(m.__file__).resolve().relative_to(root)) for n,m in sys.modules.items()
          if n.startswith('noether_optics') and getattr(m,'__file__',None)}
 print(json.dumps({'source_origins':origins}),flush=True)
"""
    command = [sys.executable, "-I", "-B", "-c", bootstrap, str(repository), str(output / "packet")]
    with (repository / ".noether-twin.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = supervise(output, command, environment=env, identity=lambda: release_identity(repository),
                            minimum_free=args.min_free_gib * GIB, reserve_paths=reserve_paths,
                            command_identity={"module": "noether_optics.bench.twin_registration",
                                              "output": "packet", "isolated_python": True})
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
