"""Resource-guard checks with synthetic children; no optical acquisition."""

import fcntl
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_registration.py"
SPEC = importlib.util.spec_from_file_location("registration_launcher_checks", SOURCE)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def run_child(tmp_path, code, *, limits=None, identity=None, packet_check=None):
    output = tmp_path / "attempt"
    result = runner.supervise(
        output, [sys.executable, "-I", "-B", "-c", code],
        environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        identity=identity or (lambda: {"acquisition_source_sha256": {}}), minimum_free=0,
        reserve_paths=[tmp_path], limits=limits or runner.Limits(wall_seconds=5, rss_bytes=256 * runner.MIB),
        packet_check=packet_check or (lambda packet, source: {"synthetic_packet": True}),
    )
    assert result == json.loads((output / "receipt.json").read_text())
    assert result["group_absent"]
    if result["owned_pgid"] is not None:
        assert not runner.group_exists(result["owned_pgid"])
    assert result["scientific_promotion"] is False
    return output, result


TINY = "from pathlib import Path; Path('packet').mkdir(); Path('packet/retained.txt').write_text('evidence')"


def test_existing_output_is_unchanged(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run_child(tmp_path, TINY)
    assert list(output.iterdir()) == [sentinel] and sentinel.read_text() == "preserve"


def test_competing_launcher_lock_refuses_before_output(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    (repository / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "__file__", str(repository / "scripts/run_registration.py"))
    output = tmp_path / "fresh"
    with (repository / ".noether-twin.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main([str(output), "--min-free-gib", "4"])
    assert not output.exists()


def test_success_retains_packet_identity_and_terminal_receipt(tmp_path):
    output, result = run_child(tmp_path, TINY)
    assert result["status"] == "completed" and result["returncode"] == 0
    assert (output / "packet/retained.txt").read_text() == "evidence"
    assert result["evidence"] == {"synthetic_packet": True}
    assert result["command"]["purpose"] == "synthetic_guard_check"
    assert len(result["command"]["argv_sha256"]) == 64
    assert (output / "runtime/admission.json").is_file()


def test_timeout_terminates_descendant_and_retains_partial_output(tmp_path):
    code = '''
import subprocess,sys,signal,time,json
from pathlib import Path
Path('packet').mkdir(); Path('packet/partial').write_text('retained')
child=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'])
def stop(*args):
 child.wait(timeout=2)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path('runtime/descendant.json').write_text(json.dumps({'pid':child.pid}))
time.sleep(30)
'''
    output, result = run_child(tmp_path, code, limits=runner.Limits(wall_seconds=0.6, rss_bytes=256 * runner.MIB))
    assert result["status"] == "stopped" and "wall_limit" in result["reason"]
    assert (output / "packet/partial").read_text() == "retained"
    descendant = json.loads((output / "runtime/descendant.json").read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)


@pytest.mark.parametrize("target,reason", [("packet", "packet_logical_limit"), ("runtime", "runtime_output_limit")])
def test_output_limits_preserve_excess_artifact(tmp_path, target, reason):
    code = TINY + f"; Path('{target}/large').write_bytes(b'x'*131072); import time; time.sleep(30)"
    limits = runner.Limits(wall_seconds=3, rss_bytes=256 * runner.MIB,
                           packet_logical_bytes=65536, runtime_bytes=65536)
    output, result = run_child(tmp_path, code, limits=limits)
    assert result["status"] == "stopped" and reason in result["reason"]
    assert (output / target / "large").stat().st_size == 131072


def test_allocated_bytes_guard_uses_injected_metadata(tmp_path, monkeypatch):
    original = runner.inventory

    def allocated(root, **kwargs):
        result = original(root, **kwargs)
        if root.name == "packet":
            result["allocated"] = 2 * runner.GIB
        return result

    monkeypatch.setattr(runner, "inventory", allocated)
    output, result = run_child(tmp_path, TINY + "; import time; time.sleep(30)")
    assert result["status"] == "stopped" and "packet_allocated_limit" in result["reason"]
    assert (output / "packet/retained.txt").exists()


def test_sampled_rss_limit_stops_owned_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "rss_bytes", lambda group: 512 * runner.MIB)
    _, result = run_child(tmp_path, "import time;time.sleep(30)")
    assert result["status"] == "stopped" and "group_rss_limit" in result["reason"]


def test_free_space_admission_failure_has_no_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
    output = tmp_path / "attempt"
    result = runner.supervise(output, [sys.executable, "-c", "raise AssertionError('must not run')"],
                               environment={}, identity=lambda: {}, minimum_free=1, reserve_paths=[])
    assert result["status"] == "stopped" and result["reason"] == "free_space_reserve"
    assert result["owned_pgid"] is None and result["returncode"] is None
    assert (output / "receipt.json").is_file()


def test_changed_identity_stops_before_success(tmp_path):
    identities = iter([{"acquisition_source_sha256": {}}, {"acquisition_source_sha256": {}, "changed": True}])
    output, result = run_child(tmp_path, TINY, identity=lambda: next(identities))
    assert result["status"] == "stopped" and result["reason"] == "source_or_runtime_identity_changed"
    assert (output / "packet/retained.txt").exists()


def test_nonzero_exit_retains_log_and_returncode(tmp_path):
    output, result = run_child(tmp_path, "print('retained error',flush=True);raise SystemExit(7)")
    assert result["status"] == "stopped" and result["returncode"] == 7
    assert result["reason"] == "child_nonzero_exit"
    assert "retained error" in (output / "runtime/run.log").read_text()


def test_unexpected_root_output_cannot_escape_accounting(tmp_path):
    output, result = run_child(tmp_path, TINY + "; Path('unexpected').write_text('retain');import time;time.sleep(30)")
    assert result["status"] == "stopped" and "unexpected_output_root_entry" in result["reason"]
    assert (output / "unexpected").read_text() == "retain"


@pytest.mark.parametrize("info,reason", [
    (SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_nlink=1), "nonregular_output"),
    (SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_nlink=2), "output_hard_link"),
])
def test_final_forbidden_links_use_metadata_injection(info, reason):
    with pytest.raises(runner.GuardError, match=reason):
        runner.entry_type(info, live=False)


def packet_fixture(tmp_path):
    root = tmp_path / "packet"
    root.mkdir()
    source = {"source.py": "f" * 64}
    protocol = {"schema": runner.SCHEMA, "units": runner.UNIT_IDENTITIES, "selection_or_evaluation_cohort": False}
    runner._write_json(root / "protocol.json", {"protocol": protocol, "source": source})
    for unit in runner.UNIT_IDENTITIES:
        runner._write_json(root / (unit["id"] + "-outcome.json"), unit)
    runner._write_json(root / "analysis.json", {"schema": runner.SCHEMA, "status": "complete_development",
                       "units": 51, "noisy_units": 48, "mean_diagnostics": 3, "physical_states": 3,
                       "counts": {"reference": 72, "prediction": 459, "objective": 2295}})
    # This guard checks custody, not array contents or optical computations.
    for name in runner.EXPECTED_PACKET_FILES - {"manifest.json"}:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic retained evidence")
    files = {str(path.relative_to(root)): runner.digest(path) for path in root.rglob("*") if path.is_file()}
    runner._write_json(root / "manifest.json", {"schema": runner.SCHEMA, "status": "complete_development", "files": files})
    return root, source


def test_packet_verification_checks_exact_declared_unit_identities(tmp_path):
    packet, source = packet_fixture(tmp_path)
    assert runner.verify_packet(packet, source)["units"] == 51
    header = json.loads((packet / "protocol.json").read_text())
    header["protocol"]["units"][0]["id"] = "undeclared-unit"
    (packet / "protocol.json").write_text(json.dumps(header))
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["files"]["protocol.json"] = runner.digest(packet / "protocol.json")
    (packet / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(runner.GuardError, match="protocol_unit_identities"):
        runner.verify_packet(packet, source)


def test_packet_verification_detects_late_extra_file(tmp_path, monkeypatch):
    packet, source = packet_fixture(tmp_path)
    original = runner.digest
    once = []

    def late(path, **kwargs):
        result = original(path, **kwargs)
        if not once:
            once.append(True)
            (packet / "late-extra.txt").write_text("retained")
        return result

    monkeypatch.setattr(runner, "digest", late)
    with pytest.raises(runner.GuardError, match="packet_changed_during_verification"):
        runner.verify_packet(packet, source)
    assert (packet / "late-extra.txt").exists()


@pytest.mark.parametrize("missing", ["arrays/reference-000000.npz", "shift-0-mean-fit.json", "events.jsonl"])
def test_packet_cannot_omit_artifact_even_with_consistent_manifest(tmp_path, missing):
    packet, source = packet_fixture(tmp_path)
    (packet / missing).unlink()
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["files"].pop(missing)
    (packet / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(runner.GuardError, match="packet_artifact_coverage"):
        runner.verify_packet(packet, source)


def test_missing_manifest_stops_successful_child(tmp_path):
    output, result = run_child(tmp_path, TINY, packet_check=runner.verify_packet)
    assert result["status"] == "stopped" and result["reason"] == "FileNotFoundError"
    assert (output / "packet/retained.txt").exists()


def test_native_library_inventory_is_bounded_relative_and_content_sensitive(tmp_path, monkeypatch):
    for name in ("numpy", "scipy", "jax", "jaxlib"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "library.so").write_bytes(name.encode())
    monkeypatch.setattr(runner.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(tmp_path / name / "__init__.py")))
    before = runner.native_identity()
    assert set(before) == {name + "/library.so" for name in ("numpy", "scipy", "jax", "jaxlib")}
    assert before["numpy/library.so"]["sha256"] == hashlib.sha256(b"numpy").hexdigest()
    (tmp_path / "numpy/library.so").write_bytes(b"changed")
    assert runner.native_identity()["numpy/library.so"] != before["numpy/library.so"]


def test_curated_release_source_admits_itself_without_scientific_imports():
    repository = SOURCE.parents[1]
    before = set(sys.modules)
    result = runner.source_identity(repository)
    expected = json.loads((repository / "source-map.json").read_text())
    assert set(result["source_sha256"]) == set(expected["files"]) | {"source-map.json"}
    assert "tests/test_registration_launcher.py" in result["source_sha256"]
    assert "scripts/run_registration.py" in result["source_sha256"]
    assert len(result["acquisition_source_sha256"]) == 8
    assert not ({"numpy", "scipy", "jax", "hcipy", "dLux"} & (set(sys.modules) - before))


def test_source_map_cannot_omit_launcher_test(tmp_path, monkeypatch):
    repository = SOURCE.parents[1]
    original = runner.read_json

    def missing(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path.name == "source-map.json":
            value["files"].pop("tests/test_registration_launcher.py")
        return value

    monkeypatch.setattr(runner, "read_json", missing)
    with pytest.raises(runner.GuardError, match="launcher_source_inventory_missing"):
        runner.source_identity(repository)


def test_source_map_cannot_accept_changed_launcher_hash(monkeypatch):
    repository = SOURCE.parents[1]
    original = runner.read_json

    def changed(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path.name == "source-map.json":
            value["files"]["scripts/run_registration.py"]["snapshot_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(runner, "read_json", changed)
    with pytest.raises(runner.GuardError, match="snapshot_source_changed"):
        runner.source_identity(repository)


@pytest.mark.parametrize("require_git", [False, True])
def test_source_archive_never_discovers_parent_repository(tmp_path, monkeypatch, require_git):
    (tmp_path / ".git").mkdir()
    repository = tmp_path / "archive"
    repository.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("A parent Git repository must never be discovered")

    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    if require_git:
        with pytest.raises(runner.GuardError, match="explicit_repository_required"):
            runner.repository_binding(repository, require_git=True)
    else:
        assert runner.repository_binding(repository) == {
            "kind": "source_snapshot", "git_discovery": "disabled",
        }


def test_explicit_git_checkout_must_have_exact_repository_root(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    (repository / ".git").mkdir(parents=True)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: str(tmp_path).encode())
    with pytest.raises(runner.GuardError, match="repository_root_mismatch"):
        runner.repository_binding(repository, require_git=True)


def test_repository_binding_ignores_inherited_git_overrides(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    (repository / ".git").mkdir(parents=True)
    monkeypatch.setenv("GIT_DIR", "/inherited/unrelated/repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/inherited/unrelated/worktree")
    calls = []

    def git(argv, *, env, timeout):
        calls.append(argv)
        assert "GIT_DIR" not in env and "GIT_WORK_TREE" not in env
        assert env["GIT_CEILING_DIRECTORIES"] == str(tmp_path)
        assert timeout == 5
        return str(repository).encode() if "--show-toplevel" in argv else b"a" * 40

    monkeypatch.setattr(runner.subprocess, "check_output", git)
    assert runner.repository_binding(repository, require_git=True) == {"kind": "explicit_checkout", "head": "a" * 40}
    assert len(calls) == 2


def test_git_marker_symlink_is_refused_without_following_it(tmp_path, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == tmp_path / ".git" or original(path))
    with pytest.raises(runner.GuardError, match="repository_git_marker_symlink"):
        runner.repository_binding(tmp_path)


def test_admission_metadata_limit_prevents_child_start(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The child must not start after an admission limit violation")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    _, result = run_child(
        tmp_path, TINY, identity=lambda: {"acquisition_source_sha256": {}, "oversize": "x" * 131072},
        limits=runner.Limits(wall_seconds=5, rss_bytes=256 * runner.MIB, runtime_bytes=65536),
    )
    assert result["status"] == "stopped" and "runtime_output_limit" in result["reason"]
    assert result["owned_pgid"] is None and result["returncode"] is None


def test_admission_checks_output_parent_reserve(tmp_path, monkeypatch):
    original = runner.os.statvfs
    monkeypatch.setattr(
        runner.os, "statvfs",
        lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096) if path == tmp_path else original(path),
    )
    result = runner.supervise(
        tmp_path / "attempt", [sys.executable, "-c", "raise AssertionError('must not run')"],
        environment={}, identity=lambda: {}, minimum_free=1, reserve_paths=[],
    )
    assert result["status"] == "stopped" and result["reason"] == "free_space_reserve"
    assert result["owned_pgid"] is None
