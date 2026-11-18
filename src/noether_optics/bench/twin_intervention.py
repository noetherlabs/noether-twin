"""Static optical intervention qualification with independent HCIPy evidence.

The candidate receives calibration images, nominal physics and known commands.
True command-response gain stays in the reference adapter. This is a trusted-code
boundary, not a security sandbox. The deterministic fixtures are development
cases; neither pixels nor commands are independent statistical replicates.

Execution order:
  matched-gain controls -> calibration -> frozen fits -> frozen predictions
  -> held-out reference -> data-only reductions -> immutable complete manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import resource
import sys
import time
import tomllib
import uuid
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "noether-static-intervention-v1"
ACTIVE = (6, 7, 8, 11, 12, 13, 16, 17, 18)
RESOLUTIONS = (64, 128, 256)
CASES = (("case-a", 1.0), ("case-b", 1.1))
STEPS_NM = (0.01, 0.1, 1.0)
LIMITS = {
    "reference": 384,
    "forward": 16000,
    "jvp": 148,
    "wall_seconds": 1800,
    "rss_bytes": 1024**3,
    "packet_bytes": 512 * 1024**2,
    "temporary_bytes": 128 * 1024**2,
    "free_bytes": 2 * 1024**3,
}
SOURCE_FILES = (
    "src/noether_optics/bench/refsim/config.py",
    "src/noether_optics/bench/refsim/system.py",
    "src/noether_optics/twin/static_mirror.py",
    "src/noether_optics/bench/twin_intervention.py",
    "pyproject.toml",
    "uv.lock",
)
VERSIONS = {
    "hcipy": "0.7.0",
    "dLux": "0.15.1",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "equinox": "0.13.8",
    "optax": "0.2.8",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
}
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,180}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def physical_contract():
    # No JAX or HCIPy import is needed to read or verify stored evidence.
    return {
        "diameter_m": 8.0,
        "obscuration_ratio": 0.14,
        "wavelength_m": 551e-9,
        "focal_length_m": 64.0,
        "actuators_across": 5,
        "coupling": 0.45,
        "cutoff_spacings": 3.0,
        "output_samples": 65,
        "pixels_per_lambda_d": 4.0,
        "defocus_opd_m": 30e-9,
    }


def focal_axis():
    spec = physical_contract()
    scale = spec["wavelength_m"] * spec["focal_length_m"] / (4 * spec["diameter_m"])
    return np.arange(-32, 33, dtype=np.float64) * scale


def core_mask():
    p, q = np.meshgrid(np.arange(-32, 33), np.arange(-32, 33), indexing="xy")
    return p * p + q * q <= 16


def check_focal_grid(x_m, y_m):
    axis = focal_axis()
    for name, values in (("x", x_m), ("y", y_m)):
        values = np.asarray(values)
        if values.dtype != np.float64 or not np.array_equal(values, axis):
            raise ValueError(f"Mismatched physical focal {name} coordinates")


def core_power(power, x_m, y_m):
    check_focal_grid(x_m, y_m)
    values = np.asarray(power)
    if values.shape != (65, 65) or values.dtype != np.float64:
        raise ValueError("Pixel power must be a 65 by 65 float64 array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Pixel power must be finite and nonnegative")
    return float(np.sum(values[core_mask()], dtype=np.float64))


def choose_action(scores):
    expected = {name for name, _ in evaluation_commands()}
    if set(scores) != expected or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("Action scores must cover exactly the 72 finite held-out identities")
    maximum = max(scores.values())
    tied = sorted(name for name, value in scores.items() if maximum - value <= 1e-10)
    return {"chosen_id": tied[0], "predicted_maximum": maximum, "predicted_ties": tied}


def action_regret(choice, reference_scores):
    # The predicted near-tie rule deliberately does not modify actual regret.
    choose_action(reference_scores)  # validate coverage and finiteness
    maximum = max(reference_scores.values())
    chosen = choice["chosen_id"]
    if chosen not in reference_scores:
        raise ValueError("Chosen action is not in the reference set")
    return {
        "chosen_id": chosen,
        "reference_maximum": maximum,
        "reference_chosen": reference_scores[chosen],
        "regret": maximum - reference_scores[chosen],
        "reference_best_ids": sorted(k for k, v in reference_scores.items() if v == maximum),
        "reference_near_ties": sorted(k for k, v in reference_scores.items() if maximum - v <= 1e-10),
    }


def error_metrics(predicted, reference, *, scale=1.0):
    predicted, reference = np.asarray(predicted) * scale, np.asarray(reference) * scale
    if predicted.shape != reference.shape or not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Error metrics require matching finite arrays")
    delta = predicted - reference
    norm = float(np.linalg.norm(reference.ravel()))
    absolute = float(np.linalg.norm(delta.ravel()))
    return {
        "absolute_l1": float(np.sum(np.abs(delta))),
        "absolute_l2": absolute,
        "reference_l2": norm,
        "relative_l2": absolute / norm if norm > 1e-12 else None,
        "relative_status": "defined" if norm > 1e-12 else "reference_norm_at_or_below_1e-12",
    }


def calibration_commands():
    commands = []
    for i, actuator in enumerate(ACTIVE):
        for sign_name, sign in (("neg", -1), ("pos", 1)):
            value = np.zeros(25, dtype=np.float64)
            value[actuator] = sign * 20e-9
            commands.append((f"cal-{i:02d}-{sign_name}", value))
    return commands


def evaluation_commands():
    commands = []
    for i, actuator in enumerate(ACTIVE):
        for amplitude in (10, 30):
            for s_name, s in (("sneg", -1), ("spos", 1)):
                for t_name, t in (("tneg", -1), ("tpos", 1)):
                    value = np.zeros(25, dtype=np.float64)
                    value[actuator] = t * amplitude * 1e-9
                    value[ACTIVE[(i + 1) % 9]] = t * s * amplitude * 1e-9
                    commands.append((f"eval-{i:02d}-{amplitude}-{s_name}-{t_name}", value))
    return commands


def derivative_anchors():
    values = dict(evaluation_commands())
    result = []
    for i in (0, 3, 6):
        for amplitude in (10, 30):
            for sign in ("sneg", "spos"):
                name = f"eval-{i:02d}-{amplitude}-{sign}-tpos"
                direction = np.zeros(25, dtype=np.float64)
                direction[ACTIVE[i]], direction[ACTIVE[(i + 1) % 9]] = 1, -1
                result.append((name, values[name], direction))
    return result


def component_commands():
    axis, mix, direction_a, direction_b = (np.zeros(25, dtype=np.float64) for _ in range(4))
    axis[ACTIVE[4]], direction_a[ACTIVE[0]] = 15e-9, 1
    mix[ACTIVE[0]], mix[ACTIVE[8]], direction_b[ACTIVE[4]] = -7e-9, 12e-9, 1
    return (("component-axis", axis, direction_a), ("component-mix", mix, direction_b))


def reference_ledger():
    records = []
    for case, gain in CASES:
        for purpose, values in (
            ("primary", [("zero", np.zeros(25))] + calibration_commands() + evaluation_commands()),
            ("component", [(name, command) for name, command, _ in component_commands()]),
        ):
            for name, command in values:
                records.append(
                    {
                        "id": f"ref-{case}-{name}",
                        "purpose": purpose,
                        "case": case,
                        "gain": gain,
                        "grid": 256,
                        "command_id": name,
                        "command_m": command.tolist(),
                    }
                )
        for purpose, anchors in (("derivative", derivative_anchors()), ("component-derivative", component_commands())):
            for name, command, direction in anchors:
                for step_index, step_nm in enumerate(STEPS_NM):
                    for sign_name, sign in (("minus", -1), ("plus", 1)):
                        records.append(
                            {
                                "id": f"ref-{case}-{name}-h{step_index}-{sign_name}",
                                "purpose": purpose,
                                "case": case,
                                "gain": gain,
                                "grid": 256,
                                "command_id": name,
                                "step_nm": step_nm,
                                "sign": sign,
                                "command_m": (command + sign * step_nm * 1e-9 * direction).tolist(),
                            }
                        )
        for name, command, _ in derivative_anchors():
            records.append(
                {
                    "id": f"ref-{case}-{name}-grid512",
                    "purpose": "convergence",
                    "case": case,
                    "gain": gain,
                    "grid": 512,
                    "command_id": name,
                    "command_m": command.tolist(),
                }
            )
    if len(records) != 378 or len({r["id"] for r in records}) != 378:
        raise ValueError("Reference ledger does not contain exactly 378 unique calls")
    return records


def source_identity():
    root = Path(__file__).resolve().parents[3]
    return {name: file_hash(root / name) for name in SOURCE_FILES}


def protocol(source=None):
    return {
        "schema": SCHEMA,
        "physics": physical_contract(),
        "source": source or source_identity(),
        "reference_ledger": reference_ledger(),
        "resolutions": list(RESOLUTIONS),
        "limits": LIMITS,
        "calibration": {
            "grid_points": 81,
            "bounds": [0.8, 1.2],
            "max_refinement_evaluations": 40,
            "bracket_width": 1e-6,
            "exact_multiple_grid_minima": "unavailable",
            "exhaustion": "unavailable",
            "nonfinite": "retain_and_stop",
            "claim": "local_search_only",
        },
        "decision": {
            "integer_radius_squared": 16,
            "centres": 49,
            "predicted_tolerance": 1e-10,
            "tie_break": "lexicographic_id",
            "regret": "actual_reference_max_minus_selected",
        },
        "gates": {
            "coordinate_epsilons": 64,
            "surface_absolute_m": 5e-18,
            "field_relative_l2": 1e-10,
            "power_absolute_l1": 1e-10,
            "jvp_absolute_l1_per_nm": 1e-10,
            "jvp_relative_l2": 1e-5,
        },
        "planned_counts": {"reference": 378, "jvp": 148, "forward_upper_bound": 14056},
    }


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular evidence file: {path.name}")
    with path.open() as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _publish(path, payload):
    """Publish without replacing an existing final file, even with a racing writer."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.parent / (".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json(path, value):
    _publish(path, canonical(value) + b"\n")


def _array_bytes(arrays):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, member.getvalue())
    return stream.getvalue()


def _safe_id(name):
    if not isinstance(name, str) or _ID.fullmatch(name) is None:
        raise ValueError("Invalid evidence identity")
    return name


def _tree_bytes(root):
    total = 0
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        for name in subdirectories + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted in evidence packets")
        total += sum((Path(directory) / name).stat().st_size for name in filenames)
    return total


class EvidencePacket:
    """One writer; durable dispatch precedes every bounded numerical callback."""

    def __init__(self, root, contract, runtime, admission, *, resume=False, storage_roots=()):
        self.root = Path(root)
        self.contract = contract
        self.contract_hash = digest(contract)
        if self.root.is_symlink():
            raise ValueError("Evidence root must not be a symlink")
        if resume:
            if not self.root.is_dir():
                raise ValueError("Cannot resume an absent packet")
        else:
            self.root.mkdir(exist_ok=False)
        _tree_bytes(self.root)
        self.storage_roots = storage_roots or ((self.root, self.root.stat().st_dev),)
        if (self.root / ".writer.lock").is_symlink():
            raise ValueError("Writer lock must not be a symlink")
        self.lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                self.header = _read_json(self.root / "protocol.json")
                if self.header["contract"] != contract or self.header["runtime"] != runtime:
                    raise ValueError("Resume source, protocol or runtime identity changed")
                if self.header["admission"] != admission:
                    raise ValueError("Resume admission identity changed")
                self._check_layout()
            else:
                for name in ("attempts", "prepared", "records", "arrays", "models", "choices"):
                    (self.root / name).mkdir()
                self.header = {
                    "contract": contract,
                    "contract_sha256": self.contract_hash,
                    "runtime": runtime,
                    "admission": admission,
                    "started_unix": time.time(),
                }
                _publish_json(self.root / "protocol.json", self.header)
            self._audit_journal()
            self.complete = (self.root / "manifest.json").exists()
            self.packet_bytes = _tree_bytes(self.root)
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def _check_layout(self):
        allowed = {
            ".writer.lock",
            "protocol.json",
            "attempts",
            "prepared",
            "records",
            "arrays",
            "models",
            "choices",
            "analysis.json",
            "report.md",
            "manifest.json",
            "stop.json",
        }
        for path in self.root.iterdir():
            if path.name not in allowed and not path.name.startswith(".pending-"):
                raise ValueError(f"Unexpected packet entry: {path.name}")
        for directory, suffix in (
            ("attempts", ".json"),
            ("prepared", ".json"),
            ("records", ".json"),
            ("arrays", ".npz"),
            ("models", ".json"),
            ("choices", ".json"),
        ):
            for path in (self.root / directory).iterdir():
                if not path.is_file() or (path.suffix != suffix and not path.name.startswith(".pending-")):
                    raise ValueError("Unexpected nested packet entry")
        _tree_bytes(self.root)

    def checkpoint(self, additional_bytes=0, *, terminal_publication=False):
        if self.complete:
            raise RuntimeError("Completed evidence is read-only")
        if self.root.stat().st_dev != self.header["admission"]["device"]:
            raise RuntimeError("Research filesystem identity changed")
        for path, expected_device in self.storage_roots:
            if Path(path).stat().st_dev != expected_device:
                raise RuntimeError("Research or backing filesystem identity changed")
            disk = os.statvfs(path)
            if disk.f_bavail * disk.f_frsize < LIMITS["free_bytes"] + additional_bytes:
                raise RuntimeError("Research or backing free-space floor reached")
        if not terminal_publication and time.time() - self.header["started_unix"] >= LIMITS["wall_seconds"]:
            raise RuntimeError("Continuous acquisition wall-time limit reached")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == "darwin" else 1024
        if not terminal_publication and rss > LIMITS["rss_bytes"]:
            raise RuntimeError("Peak process RSS limit reached")
        if self.packet_bytes + additional_bytes > LIMITS["packet_bytes"]:
            raise RuntimeError("Evidence packet size limit reached")

    def write_json(self, path, value, *, terminal_publication=False):
        payload = canonical(value) + b"\n"
        self.checkpoint(len(payload), terminal_publication=terminal_publication)
        _publish(path, payload)
        self.packet_bytes += len(payload)

    def _read_output(self, record, *, allow_unpublished=False):
        identity = _safe_id(record["id"])
        if record["protocol_sha256"] != self.contract_hash:
            raise ValueError("Record identity or protocol mismatch")
        sequence = record["attempt"]
        if type(sequence) is not int or not 0 <= sequence < len(self.attempts):
            raise ValueError("Prepared output has no matching dispatch")
        attempt = _read_json(self.attempts[sequence])
        if any(record[k] != attempt[k] for k in ("id", "kind", "protocol_sha256", "inputs", "started_unix")):
            raise ValueError("Prepared output differs from its dispatch")
        if record["status"] not in ("complete", "nonfinite", "exception"):
            raise ValueError("Unknown terminal output status")
        if (
            not isinstance(record["seconds"], int | float)
            or record["seconds"] < 0
            or not math.isfinite(record["seconds"])
        ):
            raise ValueError("Invalid retained callback duration")
        for name, value in record.get("scalars", {}).items():
            _safe_id(name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("Invalid retained float64 scalar")
        result = {name: np.asarray(value, dtype=np.float64) for name, value in record.get("scalars", {}).items()}
        if record.get("array_file"):
            expected_path = f"arrays/{identity}-a{sequence:06d}.npz"
            if record["array_file"] != expected_path:
                raise ValueError("Unexpected array path")
            if result:
                raise ValueError("Raw output cannot also contain scalar-only output")
            if re.fullmatch(r"[0-9a-f]{64}", record["array_sha256"]) is None or not record["array_schema"]:
                raise ValueError("Invalid prepared array hash or empty schema")
            for name, schema in record["array_schema"].items():
                _safe_id(name)
                if set(schema) != {"shape", "dtype"} or schema["dtype"] not in ("<f8", "<c16", "|b1"):
                    raise ValueError("Invalid prepared raw array schema")
                if not isinstance(schema["shape"], list) or any(type(n) is not int or n < 0 for n in schema["shape"]):
                    raise ValueError("Invalid prepared array shape")
            path = self.root / expected_path
            if not path.exists() and allow_unpublished:
                return None
            if path.is_symlink() or file_hash(path) != record["array_sha256"]:
                raise ValueError("Evidence array hash mismatch")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(record["array_schema"]):
                    raise ValueError("Array schema differs from archive")
                for name in archive.files:
                    _safe_id(name)
                    value = archive[name]
                    schema = record["array_schema"][name]
                    if list(value.shape) != schema["shape"] or value.dtype.str != schema["dtype"]:
                        raise ValueError("Array shape or dtype changed")
                    if value.dtype not in (np.dtype("float64"), np.dtype("complex128"), np.dtype("bool")):
                        raise ValueError("Unexpected raw array dtype")
                    result[name] = value
        elif record["array_schema"]:
            raise ValueError("Array schema has no corresponding raw artifact")
        finite = all(np.isfinite(value).all() for value in result.values())
        if (record["status"] == "complete" and not finite) or (record["status"] == "nonfinite" and finite):
            raise ValueError("Terminal status differs from retained numerical output")
        return result

    def _audit_journal(self):
        """Validate every published output before resuming, including later stages."""
        if self.header["contract_sha256"] != self.contract_hash:
            raise ValueError("Protocol hash differs")
        self.attempts = sorted((self.root / "attempts").glob("*.json"))
        self.counts = dict.fromkeys(("reference", "forward", "jvp"), 0)
        inputs_by_id = {}
        for sequence, path in enumerate(self.attempts):
            attempt = _read_json(path)
            _safe_id(attempt["id"])
            if path.name != f"{sequence:06d}.json" or attempt["protocol_sha256"] != self.contract_hash:
                raise ValueError("Invalid dispatch ledger sequence or protocol")
            if attempt["kind"] not in self.counts:
                raise ValueError("Unknown dispatch category")
            binding = (attempt["kind"], attempt["inputs"])
            if inputs_by_id.setdefault(attempt["id"], binding) != binding:
                raise ValueError("Retried numerical identity changed inputs")
            self.counts[attempt["kind"]] += 1
        if any(self.counts[k] > LIMITS[k] for k in self.counts):
            raise ValueError("Retained dispatch ledger exceeds resource limits")
        self.dispatch_inputs = inputs_by_id
        self.prepared_by_id = {}
        bound_arrays = set()
        complete_prepared = {}
        for path in sorted((self.root / "prepared").glob("*.json")):
            record = _read_json(path)
            if path.name != f"{record['attempt']:06d}.json":
                raise ValueError("Prepared receipt filename differs from dispatch")
            output = self._read_output(record, allow_unpublished=True)
            self.prepared_by_id.setdefault(record["id"], []).append(record)
            if record["array_file"] and (self.root / record["array_file"]).exists():
                bound_arrays.add(record["array_file"])
            if output is not None:
                if record["id"] in complete_prepared:
                    raise ValueError("Multiple fully published prepared outputs for one identity")
                complete_prepared[record["id"]] = record
        actual_arrays = {"arrays/" + path.name for path in (self.root / "arrays").glob("*.npz")}
        if actual_arrays != bound_arrays:
            raise ValueError("Unbound raw array artifact")
        for path in (self.root / "records").glob("*.json"):
            record = _read_json(path)
            if path.stem != record["id"] or complete_prepared.get(path.stem) != record:
                raise ValueError("Terminal record lacks its fully published prepared output")
        for folder in ("models", "choices"):
            for path in (self.root / folder).glob("*.json"):
                value = _read_json(path)
                position = value["frozen_after_attempt"]
                if (
                    value["protocol_sha256"] != self.contract_hash
                    or type(position) is not int
                    or not -1 <= position < len(self.attempts)
                ):
                    raise ValueError("Frozen receipt protocol or dispatch position differs")

    def load_record(self, identity):
        identity = _safe_id(identity)
        record = _read_json(self.root / "records" / (identity + ".json"))
        if record["id"] != identity:
            raise ValueError("Record identity mismatch")
        prepared = _read_json(self.root / "prepared" / f"{record['attempt']:06d}.json")
        if record != prepared:
            raise ValueError("Terminal record differs from prepared output")
        result = self._read_output(record)
        return record, result

    def perform(self, kind, identity, callback, *, inputs, scalar_only=False):
        identity = _safe_id(identity)
        if self.dispatch_inputs.get(identity, (kind, inputs)) != (kind, inputs):
            raise ValueError("Retried numerical identity changed inputs")
        if any(record["status"] != "complete" for record in self.prepared_by_id.get(identity, [])):
            raise RuntimeError(
                "Retained prepared numerical failure; do not retry or replace it with a later terminal outcome"
            )
        destination = self.root / "records" / (identity + ".json")
        if not destination.exists():
            recoverable = []
            for prepared in self.prepared_by_id.get(identity, []):
                if prepared["kind"] != kind or prepared["inputs"] != inputs:
                    raise ValueError("Existing prepared output input identity differs")
                output = self._read_output(prepared, allow_unpublished=True)
                if output is not None:
                    recoverable.append(prepared)
            if len(recoverable) > 1:
                raise ValueError("Ambiguous prepared outputs; cannot select a retry")
            if recoverable:
                self.write_json(destination, recoverable[0], terminal_publication=True)
        if destination.exists():
            record, arrays = self.load_record(identity)
            if record["kind"] != kind or record["inputs"] != inputs:
                raise ValueError("Existing record input identity differs")
            if record["status"] != "complete":
                raise RuntimeError("Retained terminal numerical failure; do not retry")
            return arrays
        self.checkpoint()
        if kind not in self.counts or self.counts[kind] >= LIMITS[kind]:
            raise RuntimeError("Numerical call budget exhausted")
        sequence = len(self.attempts)
        attempt = {
            "id": identity,
            "kind": kind,
            "protocol_sha256": self.contract_hash,
            "inputs": inputs,
            "started_unix": time.time(),
        }
        attempt_path = self.root / "attempts" / f"{sequence:06d}.json"
        self.write_json(attempt_path, attempt)
        self.attempts.append(attempt_path)
        self.counts[kind] += 1
        self.dispatch_inputs[identity] = (kind, inputs)
        start = time.monotonic()
        arrays = {}
        error = None
        try:
            arrays = {name: np.asarray(value) for name, value in callback().items()}
            for name, value in arrays.items():
                _safe_id(name)
                if value.dtype not in (np.dtype("float64"), np.dtype("complex128"), np.dtype("bool")):
                    raise ValueError("Numerical outputs must be float64, complex128 or boolean")
            finite = all(np.isfinite(value).all() for value in arrays.values())
            status = "complete" if finite else "nonfinite"
        except Exception as exc:
            status, error = "exception", {"type": type(exc).__name__, "message": str(exc)}
        record = {
            **attempt,
            "attempt": sequence,
            "status": status,
            "seconds": time.monotonic() - start,
            "error": error,
            "scalars": {},
            "array_file": None,
            "array_schema": {},
        }
        payload = None
        if arrays and scalar_only and status == "complete":
            if any(value.shape != () or value.dtype != np.float64 for value in arrays.values()):
                raise ValueError("Scalar-only record returned non-scalar data")
            record["scalars"] = {name: float(value) for name, value in arrays.items()}
        elif arrays:
            payload = _array_bytes(arrays)
            if len(payload) > LIMITS["temporary_bytes"]:
                raise RuntimeError("One array publication exceeds temporary allowance")
            path = self.root / "arrays" / f"{identity}-a{sequence:06d}.npz"
            expected_hash = hashlib.sha256(payload).hexdigest()
            record.update(
                array_file="arrays/" + path.name,
                array_sha256=expected_hash,
                array_schema={k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()},
            )
        # The descriptor binds a completed callback to one already charged
        # attempt. Only a descriptor plus its fully published archive permits
        # recovery without another callback (and without new compile timing).
        self.write_json(self.root / "prepared" / f"{sequence:06d}.json", record, terminal_publication=True)
        self.prepared_by_id.setdefault(identity, []).append(record)
        if payload is not None:
            self.checkpoint(len(payload), terminal_publication=True)
            _publish(path, payload)
            self.packet_bytes += len(payload)
        self.write_json(destination, record, terminal_publication=True)
        if status != "complete":
            raise RuntimeError(f"Retained numerical failure: {identity} ({status})")
        self.checkpoint()
        return arrays

    def freeze(self, directory, identity, payload):
        destination = self.root / directory / (_safe_id(identity) + ".json")
        value = {"protocol_sha256": self.contract_hash, **payload}
        if destination.exists():
            old = _read_json(destination)
            if {k: v for k, v in old.items() if k != "frozen_after_attempt"} != value:
                raise ValueError("Frozen evidence would change")
        else:
            value["frozen_after_attempt"] = len(self.attempts) - 1
            self.checkpoint(len(canonical(value)))
            self.write_json(destination, value)
        return file_hash(destination)

    @classmethod
    def reader(cls, root):
        """Data-only reader: creates no lock or directories and imports no optical backend."""
        reader = cls.__new__(cls)
        reader.root = Path(root)
        if reader.root.is_symlink():
            raise ValueError("Evidence root must not be a symlink")
        reader.header = _read_json(reader.root / "protocol.json")
        reader.contract = reader.header["contract"]
        reader.contract_hash = digest(reader.contract)
        if reader.header["contract_sha256"] != reader.contract_hash:
            raise ValueError("Protocol hash differs")
        if reader.contract != protocol(source=reader.contract["source"]):
            raise ValueError("Unsupported or changed scientific protocol")
        reader._check_layout()
        reader._audit_journal()
        return reader


def fit_scalar_gain(objective):
    """Bounded local refinement; exact multiple grid minima make the arm unavailable."""
    trace = []

    def evaluate(gain, phase, bracket):
        value = float(objective(float(gain), len(trace)))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite calibration objective")
        trace.append(
            {"sequence": len(trace), "gain": float(gain), "objective": value, "phase": phase, "bracket": list(bracket)}
        )
        return value

    grid = np.linspace(0.8, 1.2, 81, dtype=np.float64)
    values = [evaluate(g, "grid", (0.8, 1.2)) for g in grid]
    winners = [i for i, value in enumerate(values) if value == min(values)]
    receipt = {
        "gain": None,
        "trace": trace,
        "grid_minima": winners,
        "claim": "local_search_only",
        "boundary": any(i in (0, 80) for i in winners),
    }
    if len(winners) != 1:
        return {**receipt, "status": "ambiguous_grid_minimum", "final_bracket": None, "bracket_width": None}
    index = winners[0]
    low, high = float(grid[max(0, index - 1)]), float(grid[min(80, index + 1)])
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left = evaluate(left, "refinement", (low, high))
    f_right = evaluate(right, "refinement", (low, high))
    calls = 2
    while high - low > 1e-6 and calls < 40:
        if f_left <= f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = evaluate(left, "refinement", (low, high))
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = evaluate(right, "refinement", (low, high))
        calls += 1
    receipt.update(final_bracket=[low, high], bracket_width=high - low, refinement_evaluations=calls)
    if high - low > 1e-6:
        return {**receipt, "status": "refinement_exhausted"}
    best = min(trace, key=lambda row: (row["objective"], row["gain"]))
    return {
        **receipt,
        "status": "converged_local",
        "gain": best["gain"],
        "objective": best["objective"],
        "boundary": best["gain"] in (0.8, 1.2),
    }


def reference_backend(*, defocus_opd_m=30e-9, registration_x_pitch=0.0):
    """Own HCIPy grids, aperture and DM; do not consume candidate operators.

    Each factory owns its diversity and cache. The default preserves the v1
    optical contract; a phase-diversity pair uses two separate factories. A
    nonzero registration translates the independently constructed actuator
    centres in pitch units. Registration is immutable within each factory, so
    different registrations cannot share cached geometry.
    """
    if isinstance(defocus_opd_m, bool) or not isinstance(defocus_opd_m, int | float):
        raise ValueError("defocus_opd_m must be a finite real number")
    if not math.isfinite(defocus_opd_m):
        raise ValueError("defocus_opd_m must be a finite real number")
    if (
        isinstance(registration_x_pitch, bool)
        or not isinstance(registration_x_pitch, int | float)
        or not math.isfinite(registration_x_pitch)
    ):
        raise ValueError("registration_x_pitch must be a finite real number")
    import hcipy as hp

    spec = physical_contract()
    spec["defocus_opd_m"] = defocus_opd_m
    cache = {}

    def evaluate(grid_size, command_m, true_gain, *, component=False):
        if cache.get("grid_size") != grid_size:
            cache.clear()
            grid = hp.make_pupil_grid(grid_size, spec["diameter_m"])
            aperture = hp.make_obstructed_circular_aperture(spec["diameter_m"], spec["obscuration_ratio"])(grid)
            area = math.pi * (spec["diameter_m"] / 2) ** 2 * (1 - spec["obscuration_ratio"] ** 2)
            input_field = aperture / math.sqrt(area)
            diversity = spec["defocus_opd_m"] * (2 * (grid.x**2 + grid.y**2) / (spec["diameter_m"] / 2) ** 2 - 1)
            pitch = spec["diameter_m"] / 5
            if registration_x_pitch == 0:
                # Preserve the original zero-registration reference path.
                influences = hp.make_gaussian_influence_functions(
                    grid, 5, pitch, crosstalk=spec["coupling"], cutoff=spec["cutoff_spacings"]
                )
            else:
                # HCIPy's own continuous poke construction, never candidate arrays.
                centers = hp.make_actuator_positions(5, pitch).shifted([registration_x_pitch * pitch, 0.0])
                sigma = pitch / math.sqrt(-2 * math.log(spec["coupling"]))
                pokes = hp.make_gaussian_pokes(None, centers, sigma, pitch / sigma * spec["cutoff_spacings"])
                influences = hp.evaluate_supersampled(pokes, grid, 1, make_sparse=True)
            axis = focal_axis()
            scale = spec["wavelength_m"] * spec["focal_length_m"] / (4 * spec["diameter_m"])
            focal_grid = hp.CartesianGrid(hp.SeparatedCoords([axis, axis]), weights=scale**2)
            cache.update(
                grid_size=grid_size,
                grid=grid,
                aperture=aperture,
                field=input_field,
                diversity=diversity,
                mirror=hp.DeformableMirror(influences),
                propagator=hp.FraunhoferPropagator(grid, focal_grid, spec["focal_length_m"]),
            )
        grid, mirror = cache["grid"], cache["mirror"]
        mirror.actuators = np.asarray(command_m, dtype=np.float64) * true_gain
        wavefront = hp.Wavefront(cache["field"].copy(), spec["wavelength_m"])
        wavefront.electric_field *= np.exp(2j * np.pi * cache["diversity"] / spec["wavelength_m"])
        output = cache["propagator"](mirror(wavefront))
        values = {
            "field": np.asarray(output.electric_field.shaped, dtype=np.complex128),
            "power": np.asarray(output.power.shaped, dtype=np.float64),
            "incident": np.asarray(np.sum(np.abs(cache["field"]) ** 2 * grid.weights), dtype=np.float64),
        }
        if component:
            values.update(
                surface=np.asarray(mirror.surface.shaped, dtype=np.float64),
                pupil_x=np.asarray(grid.x.shaped, dtype=np.float64),
                pupil_y=np.asarray(grid.y.shaped, dtype=np.float64),
                aperture=np.asarray(cache["aperture"].shaped > 0, dtype=bool),
            )
        return values

    return evaluate


def candidate_backend():
    """The backend interface contains no oracle, true gain or held-out observations."""
    import jax

    from noether_optics.twin.static_mirror import (
        StaticMirrorSpec,
        directional_derivative,
        pixel_amplitude,
        pupil_inputs,
    )

    spec = StaticMirrorSpec(**physical_contract())
    compiled = {}
    incident = {}

    def evaluate(kind, n, command_m, model_gain, *, direction=None, component=False):
        key = kind, n
        compilation_seconds = 0.0
        if key not in compiled:
            start = time.monotonic()
            command_shape = jax.ShapeDtypeStruct((25,), np.float64)
            gain_shape = jax.ShapeDtypeStruct((), np.float64)
            if kind == "forward":
                operation = jax.jit(lambda u, g: pixel_amplitude(u, g, spec=spec, pupil_samples=n))
                compiled[key] = operation.lower(command_shape, gain_shape).compile()
            else:
                operation = jax.jit(lambda u, v, g: directional_derivative(u, v, g, spec=spec, pupil_samples=n))
                compiled[key] = operation.lower(command_shape, command_shape, gain_shape).compile()
            compilation_seconds = time.monotonic() - start
        if n not in incident:
            field = np.asarray(pupil_inputs(spec, n)["input_field"])
            incident[n] = float(np.sum(np.abs(field) ** 2) * (spec.diameter_m / n) ** 2)
        args = (command_m, model_gain) if kind == "forward" else (command_m, direction, model_gain)
        result = np.asarray(compiled[key](*args))  # synchronizes before timing/publication
        if kind == "jvp":
            return {"jvp": result, "compile_seconds": np.asarray(compilation_seconds)}
        values = {
            "amplitude": result,
            "power": np.asarray(np.abs(result) ** 2, dtype=np.float64),
            "compile_seconds": np.asarray(compilation_seconds),
            "incident": np.asarray(incident[n]),
        }
        if component:
            inputs = {name: np.asarray(value) for name, value in pupil_inputs(spec, n).items()}
            values.update(
                surface=model_gain * np.tensordot(command_m, inputs["influence_functions"], axes=1),
                pupil_x=inputs["x_m"],
                pupil_y=inputs["y_m"],
                aperture=inputs["aperture"],
                incident=np.asarray(np.sum(np.abs(inputs["input_field"]) ** 2) * (spec.diameter_m / n) ** 2),
            )
        return values

    return evaluate


def _reference_call(packet, item, backend):
    return packet.perform(
        "reference",
        item["id"],
        lambda: backend(
            item["grid"], np.asarray(item["command_m"]), item["gain"], component=item["purpose"] == "component"
        ),
        inputs=item,
    )


def _candidate_call(packet, backend, identity, n, command, gain, *, receipt=None, direction=None, component=False):
    kind = "forward" if direction is None else "jvp"
    inputs = {
        "resolution": n,
        "command_m": np.asarray(command).tolist(),
        "model_gain": float(gain),
        "model_receipt_sha256": receipt,
        "component_control": component,
        "direction": None if direction is None else np.asarray(direction).tolist(),
    }
    return packet.perform(
        kind,
        identity,
        lambda: backend(kind, n, np.asarray(command), gain, direction=direction, component=component),
        inputs=inputs,
    )


def _central_difference(packet, case, name, step_index):
    plus = packet.load_record(f"ref-{case}-{name}-h{step_index}-plus")[1]["power"]
    minus = packet.load_record(f"ref-{case}-{name}-h{step_index}-minus")[1]["power"]
    return (plus - minus) / (2 * STEPS_NM[step_index] * 1e-9)


def component_gate_report(packet):
    rows = []
    spec = physical_contract()
    scale = spec["wavelength_m"] * spec["focal_length_m"] / (4 * spec["diameter_m"])
    for case, _ in CASES:
        for name, _, _ in component_commands():
            reference = packet.load_record(f"ref-{case}-{name}")[1]
            candidate = packet.load_record(f"control-{case}-{name}")[1]
            derivative = packet.load_record(f"control-jvp-{case}-{name}")[1]["jvp"]
            coordinate_error = max(float(np.max(np.abs(candidate[k] - reference[k]))) for k in ("pupil_x", "pupil_y"))
            surface_error = float(np.max(np.abs(candidate["surface"] - reference["surface"])))
            field = error_metrics(candidate["amplitude"], 1j * scale * reference["field"])
            power = error_metrics(candidate["power"], reference["power"])
            derivatives = [
                error_metrics(derivative, _central_difference(packet, case, name, step), scale=1e-9)
                for step in range(3)
            ]
            checks = {
                "coordinates": coordinate_error <= 64 * np.finfo(np.float64).eps * spec["diameter_m"],
                "aperture": np.array_equal(candidate["aperture"], reference["aperture"]),
                "surface": surface_error <= 5e-18,
                "field": field["relative_l2"] is not None and field["relative_l2"] <= 1e-10,
                "power": power["absolute_l1"] <= 1e-10,
            }
            for step in (0, 1):
                values = derivatives[step]
                checks[f"jvp-h{step}"] = values["absolute_l1"] <= 1e-10 or (
                    values["relative_l2"] is not None and values["relative_l2"] <= 1e-5
                )
            checks = {name: bool(passed) for name, passed in checks.items()}
            rows.append(
                {
                    "case": case,
                    "command": name,
                    "coordinate_error_m": coordinate_error,
                    "surface_error_m": surface_error,
                    "field": field,
                    "power": power,
                    "derivatives_per_nm": derivatives,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return {"rows": rows, "passed": all(row["passed"] for row in rows)}


def _fit_arm(packet, candidate, case, n):
    calibration = []
    calibration_hashes = {}
    for name, command in calibration_commands():
        identity = f"ref-{case}-{name}"
        calibration.append((name, command, packet.load_record(identity)[1]["power"]))
        calibration_hashes[identity] = file_hash(packet.root / "records" / (identity + ".json"))
    calibration_digest = digest(calibration_hashes)

    def objective(gain, proposal):
        total = 0.0
        for index, (_, command, observed) in enumerate(calibration):
            identity = f"fit-{case}-n{n}-p{proposal:03d}-c{index:02d}"

            def predict_loss(command=command, observed=observed):
                values = candidate("forward", n, command, gain)
                prediction = values["power"]
                with np.errstate(over="ignore", invalid="ignore"):
                    loss = np.sum((prediction - observed) ** 2, dtype=np.float64)
                if not np.isfinite(prediction).all() or not np.isfinite(loss):
                    return {"power": prediction, "loss": np.asarray(loss)}
                return {"loss": np.asarray(loss), "compile_seconds": values["compile_seconds"]}

            values = packet.perform(
                "forward",
                identity,
                predict_loss,
                scalar_only=True,
                inputs={
                    "calibration_sha256": calibration_digest,
                    "resolution": n,
                    "gain": gain,
                    "command_m": command.tolist(),
                    "proposal": proposal,
                    "calibration_index": index,
                },
            )
            total += float(values["loss"])
        return total

    fit = fit_scalar_gain(objective)
    fit.update(resolution=n, calibration_records=calibration_hashes, calibration_sha256=calibration_digest)
    if fit["status"] == "converged_local":
        final_hashes = {}
        for index, (_, command, _) in enumerate(calibration):
            identity = f"fit-final-{case}-n{n}-c{index:02d}"
            _candidate_call(packet, candidate, identity, n, command, fit["gain"])
            final_hashes[identity] = file_hash(packet.root / "records" / (identity + ".json"))
        fit["final_calibration_records"] = final_hashes
    else:
        fit["final_calibration_records"] = {}
    receipt_hash = packet.freeze("models", f"{case}-n{n}-recalibrated", fit)
    return fit, receipt_hash


def _validate_scientific_output(kind, inputs, arrays):
    """Check physical array schemas even when a later stage is incomplete."""
    if "calibration_index" in inputs:
        if set(arrays) != {"loss", "compile_seconds"} or any(v.shape != () for v in arrays.values()):
            raise ValueError("Calibration objective has invalid scalar schema")
        if float(arrays["loss"]) < 0 or float(arrays["compile_seconds"]) < 0:
            raise ValueError("Calibration loss or compilation duration is negative")
        return
    if kind == "jvp":
        required = {"jvp", "compile_seconds"}
    elif kind == "reference":
        required = {"power", "field", "incident"}
    else:
        required = {"power", "amplitude", "incident", "compile_seconds"}
    component = inputs.get("component_control", False) or inputs.get("purpose") == "component"
    if component and kind != "jvp":
        required |= {"pupil_x", "pupil_y", "surface", "aperture"}
    if set(arrays) != required:
        raise ValueError("Numerical output schema differs from its declared call")
    for name, value in arrays.items():
        shape = (
            ()
            if name in ("incident", "compile_seconds")
            else ((256, 256) if name in ("pupil_x", "pupil_y", "surface", "aperture") else (65, 65))
        )
        dtype = np.dtype(
            "complex128" if name in ("field", "amplitude") else "bool" if name == "aperture" else "float64"
        )
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("Numerical output shape or dtype differs from its declared call")
    if "power" in arrays:
        core_power(arrays["power"], focal_axis(), focal_axis())
    if "incident" in arrays and float(arrays["incident"]) <= 0:
        raise ValueError("Incident power must be positive")
    if "compile_seconds" in arrays and float(arrays["compile_seconds"]) < 0:
        raise ValueError("Compilation duration is negative")


def scientific_preflight(packet):
    """Replay the acquired prefix without dispatching or publishing anything.

    Every existing dispatch, including an interrupted one, must belong to this
    exact prefix. An early held-out dispatch cannot hide behind a later retry.
    The same acquisition control flow supplies call declarations and freezes.
    """
    packet._check_layout()
    packet._audit_journal()
    attempts_by_id = {}
    for sequence, path in enumerate(packet.attempts):
        attempt = _read_json(path)
        attempts_by_id.setdefault(attempt["id"], []).append((sequence, attempt))

    class EndOfPrefix(Exception):
        pass

    class Prefix:
        root, contract = packet.root, packet.contract

        def __init__(self):
            self.seen = set()
            self.freezes = {"models": set(), "choices": set()}
            self.last_attempt = -1

        def perform(self, kind, identity, callback, *, inputs, scalar_only=False):
            attempts = attempts_by_id.get(identity, [])
            if any(record["status"] != "complete" for record in packet.prepared_by_id.get(identity, [])):
                raise RuntimeError("Retained prepared numerical failure; a later terminal outcome cannot erase it")
            for sequence, attempt in attempts:
                if (
                    (identity not in self.seen and sequence <= self.last_attempt)
                    or attempt["kind"] != kind
                    or attempt["inputs"] != inputs
                ):
                    raise ValueError("Existing dispatch violates the declared acquisition order or inputs")
                if identity not in self.seen:
                    self.last_attempt = sequence
            self.seen.add(identity)
            if not (self.root / "records" / (identity + ".json")).exists():
                # Prepared-but-unpublished records are checked by the generic
                # journal audit. Recovery belongs to the real writer, after all
                # existing scientific evidence has passed this read-only audit.
                for prepared in packet.prepared_by_id.get(identity, []):
                    arrays = packet._read_output(prepared, allow_unpublished=True)
                    if arrays is not None and prepared["status"] == "complete":
                        _validate_scientific_output(kind, inputs, arrays)
                raise EndOfPrefix()
            record, arrays = packet.load_record(identity)
            if record["status"] != "complete":
                raise RuntimeError("Retained terminal numerical failure; do not retry")
            if not attempts or record["attempt"] != attempts[-1][0]:
                raise ValueError("A numerical retry postdates its terminal output")
            _validate_scientific_output(kind, inputs, arrays)
            return arrays

        def load_record(self, identity):
            return packet.load_record(identity)

        def freeze(self, directory, identity, payload):
            path = self.root / directory / (identity + ".json")
            if not path.exists():
                raise EndOfPrefix()
            expected = {"protocol_sha256": packet.contract_hash, **payload, "frozen_after_attempt": self.last_attempt}
            if _read_json(path) != expected:
                raise ValueError("Frozen evidence does not reproduce from the acquired prefix")
            self.freezes[directory].add(identity)
            return file_hash(path)

    def never_dispatch(*args, **kwargs):
        raise AssertionError("Scientific preflight must never dispatch a backend")

    prefix = Prefix()
    with contextlib.suppress(EndOfPrefix):
        _acquire(prefix, never_dispatch, never_dispatch, publish=False)
    if set(attempts_by_id) - prefix.seen:
        raise ValueError("Dispatch outside the declared acquisition prefix, including premature held-out evidence")
    for directory, seen in prefix.freezes.items():
        if {p.stem for p in (packet.root / directory).glob("*.json")} != seen:
            raise ValueError("Frozen receipt outside the declared acquisition prefix")


def acquire(packet, *, reference=None, candidate=None):
    """Run the complete fixed study only after auditing every existing artifact."""
    scientific_preflight(packet)
    return _acquire(packet, reference or reference_backend(), candidate or candidate_backend())


def _acquire(packet, reference, candidate, *, publish=True):
    ledger = packet.contract["reference_ledger"]
    for item in ledger:
        if item["purpose"] in ("component", "component-derivative"):
            _reference_call(packet, item, reference)
    for case, gain in CASES:
        for name, command, direction in component_commands():
            _candidate_call(packet, candidate, f"control-{case}-{name}", 256, command, gain, component=True)
            _candidate_call(
                packet, candidate, f"control-jvp-{case}-{name}", 256, command, gain, direction=direction, component=True
            )
    gates = component_gate_report(packet)
    packet.freeze("models", "component-gates", gates)
    if not gates["passed"]:
        raise RuntimeError("Matched-grid component qualification failed; retain protocol and diagnose")
    for item in ledger:
        if item["purpose"] == "primary" and not item["command_id"].startswith("eval-"):
            _reference_call(packet, item, reference)
    # Complete every choice before acquiring any held-out reference outcome.
    for case, _ in CASES:
        for n in RESOLUTIONS:
            fit, fitted_receipt = _fit_arm(packet, candidate, case, n)
            nominal_receipt = packet.freeze(
                "models",
                f"{case}-n{n}-nominal",
                {"status": "nominal", "gain": 1.0, "resolution": n, "calibration_records": {}},
            )
            for arm, gain, receipt in (
                ("nominal", 1.0, nominal_receipt),
                ("recalibrated", fit["gain"], fitted_receipt),
            ):
                identity = f"{case}-n{n}-{arm}"
                if gain is None:
                    packet.freeze(
                        "choices",
                        identity,
                        {
                            "status": "unavailable",
                            "reason": fit["status"],
                            "model_receipt_sha256": receipt,
                            "predictions": {},
                        },
                    )
                    continue
                predictions, scores = {}, {}
                for name, command in [("zero", np.zeros(25))] + evaluation_commands():
                    record_id = f"pred-{identity}-{name}"
                    result = _candidate_call(packet, candidate, record_id, n, command, gain, receipt=receipt)
                    predictions[name] = file_hash(packet.root / "records" / (record_id + ".json"))
                    if name != "zero":
                        scores[name] = core_power(result["power"], focal_axis(), focal_axis())
                derivatives = {}
                for name, command, direction in derivative_anchors():
                    record_id = f"jvp-{identity}-{name}"
                    _candidate_call(
                        packet, candidate, record_id, n, command, gain, receipt=receipt, direction=direction
                    )
                    derivatives[name] = file_hash(packet.root / "records" / (record_id + ".json"))
                packet.freeze(
                    "choices",
                    identity,
                    {
                        "status": "available",
                        "model_receipt_sha256": receipt,
                        "predictions": predictions,
                        "derivatives": derivatives,
                        "scores": scores,
                        **choose_action(scores),
                    },
                )
    for item in ledger:
        if item["purpose"] not in ("component", "component-derivative"):
            _reference_call(packet, item, reference)
    return finalize(packet) if publish else None


def analyze(packet):
    """Reconstruct decisions and all errors solely from verified retained arrays."""
    scientific_preflight(packet)
    expected = set()
    attempts = sorted((packet.root / "attempts").glob("*.json"))
    counts = dict.fromkeys(("reference", "forward", "jvp"), 0)
    dispatch = []
    for sequence, path in enumerate(attempts):
        value = _read_json(path)
        if path.name != f"{sequence:06d}.json" or value["protocol_sha256"] != packet.contract_hash:
            raise ValueError("Dispatch sequence or protocol mismatch")
        if value["kind"] not in counts:
            raise ValueError("Unknown numerical call category")
        counts[value["kind"]] += 1
        dispatch.append(value)
    if any(counts[k] > LIMITS[k] for k in counts):
        raise ValueError("Retained dispatch ledger exceeds resource limits")

    def require(identity, kind, inputs=None):
        record, arrays = packet.load_record(identity)
        expected.add(identity)
        if record["status"] != "complete" or record["kind"] != kind:
            raise ValueError("Incomplete/failed numerical record in complete analysis")
        if inputs is not None and record["inputs"] != inputs:
            raise ValueError("Recorded numerical inputs differ from frozen declaration")
        sequence = record["attempt"]
        if not isinstance(sequence, int) or not 0 <= sequence < len(dispatch):
            raise ValueError("Record has no matching dispatch")
        attempt = dispatch[sequence]
        if any(record[k] != attempt[k] for k in ("id", "kind", "protocol_sha256", "inputs", "started_unix")):
            raise ValueError("Terminal record does not match its durable dispatch")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite data cannot be a successful terminal record")
        return record, arrays

    def frozen(directory, identity):
        value = _read_json(packet.root / directory / (identity + ".json"))
        if value["protocol_sha256"] != packet.contract_hash:
            raise ValueError("Frozen receipt belongs to another protocol")
        sequence = value["frozen_after_attempt"]
        if not isinstance(sequence, int) or not -1 <= sequence < len(dispatch):
            raise ValueError("Frozen receipt has invalid dispatch position")
        return value

    reference = {}
    heldout_ids = {item["id"] for item in packet.contract["reference_ledger"] if item["command_id"].startswith("eval-")}
    heldout_attempts = [index for index, attempt in enumerate(dispatch) if attempt["id"] in heldout_ids]
    for item in packet.contract["reference_ledger"]:
        record, arrays = require(item["id"], "reference", item)
        core_power(arrays["power"], focal_axis(), focal_axis())
        if arrays["field"].shape != (65, 65) or arrays["field"].dtype != np.complex128:
            raise ValueError("Reference complex field schema differs")
        if arrays["incident"].shape != () or float(arrays["incident"]) <= 0:
            raise ValueError("Reference incident power must be positive")
        reference[item["id"]] = arrays
    for case, gain in CASES:
        for name, command, direction in component_commands():
            for kind, prefix, tangent in (("forward", "control", None), ("jvp", "control-jvp", direction)):
                require(
                    f"{prefix}-{case}-{name}",
                    kind,
                    {
                        "resolution": 256,
                        "command_m": command.tolist(),
                        "model_gain": gain,
                        "model_receipt_sha256": None,
                        "component_control": True,
                        "direction": None if tangent is None else tangent.tolist(),
                    },
                )
    gates = component_gate_report(packet)
    saved_gates = frozen("models", "component-gates")
    if not gates["passed"] or any(saved_gates[k] != gates[k] for k in gates):
        raise ValueError("Retained component gate does not reproduce")
    rows, convergence, freeze_positions = [], [], []
    model_ids, choice_ids = {"component-gates"}, set()
    for case, _ in CASES:
        reference_zero = reference[f"ref-{case}-zero"]["power"]
        reference_scores = {
            name: core_power(reference[f"ref-{case}-{name}"]["power"], focal_axis(), focal_axis())
            for name, _ in evaluation_commands()
        }
        calibration_hashes = {
            f"ref-{case}-{name}": file_hash(packet.root / "records" / f"ref-{case}-{name}.json")
            for name, _ in calibration_commands()
        }
        calibration_digest = digest(calibration_hashes)
        for n in RESOLUTIONS:
            fitted_id = f"{case}-n{n}-recalibrated"
            fitted = frozen("models", fitted_id)

            def replay_objective(gain, proposal, *, case=case, n=n, calibration_digest=calibration_digest):
                total = 0.0
                for index, (_, command) in enumerate(calibration_commands()):
                    _, values = require(
                        f"fit-{case}-n{n}-p{proposal:03d}-c{index:02d}",
                        "forward",
                        {
                            "calibration_sha256": calibration_digest,
                            "resolution": n,
                            "gain": gain,
                            "command_m": command.tolist(),
                            "proposal": proposal,
                            "calibration_index": index,
                        },
                    )
                    if set(values) != {"loss", "compile_seconds"} or values["loss"].shape != ():
                        raise ValueError("Calibration objective record has invalid scalar schema")
                    total += float(values["loss"])
                return total

            rebuilt_fit = fit_scalar_gain(replay_objective)
            for key, value in rebuilt_fit.items():
                if fitted[key] != value:
                    raise ValueError("Frozen scalar fit does not reproduce from its full objective trace")
            if (
                fitted["calibration_records"] != calibration_hashes
                or fitted["calibration_sha256"] != calibration_digest
            ):
                raise ValueError("Frozen calibration input chain differs")
            final_hashes, final_objective = {}, 0.0
            if fitted["gain"] is not None:
                for index, (name, command) in enumerate(calibration_commands()):
                    identity = f"fit-final-{case}-n{n}-c{index:02d}"
                    record, arrays = require(
                        identity,
                        "forward",
                        {
                            "resolution": n,
                            "command_m": command.tolist(),
                            "model_gain": fitted["gain"],
                            "model_receipt_sha256": None,
                            "component_control": False,
                            "direction": None,
                        },
                    )
                    if record["attempt"] > fitted["frozen_after_attempt"]:
                        raise ValueError("Final calibration predictions postdate the model freeze")
                    observed = reference[f"ref-{case}-{name}"]["power"]
                    final_objective += float(np.sum((arrays["power"] - observed) ** 2, dtype=np.float64))
                    final_hashes[identity] = file_hash(packet.root / "records" / (identity + ".json"))
                if final_objective != fitted["objective"]:
                    raise ValueError("Selected gain objective does not reproduce from its retained final images")
            if fitted["final_calibration_records"] != final_hashes:
                raise ValueError("Final calibration image hash chain differs")
            for arm in ("nominal", "recalibrated"):
                identity = f"{case}-n{n}-{arm}"
                model_ids.add(identity)
                choice_ids.add(identity)
                model = frozen("models", identity)
                receipt_hash = file_hash(packet.root / "models" / (identity + ".json"))
                choice = frozen("choices", identity)
                freeze_positions.append(choice["frozen_after_attempt"])
                if choice["model_receipt_sha256"] != receipt_hash:
                    raise ValueError("Choice binds a different model receipt")
                if arm == "nominal" and (model["status"] != "nominal" or model["gain"] != 1.0):
                    raise ValueError("Nominal arm must use unfitted gain one")
                row = {
                    "case": case,
                    "resolution": n,
                    "arm": arm,
                    "model_gain": model["gain"],
                    "calibration_status": model["status"],
                }
                if model["gain"] is None:
                    if (
                        choice["status"] != "unavailable"
                        or choice["reason"] != model["status"]
                        or choice["predictions"]
                    ):
                        raise ValueError("Unsuccessful fit did not retain an unavailable arm")
                    rows.append({**row, "status": "unavailable", "reason": model["status"]})
                    continue
                if choice["status"] != "available":
                    raise ValueError("Eligible model arm was silently withheld")
                predicted, prediction_hashes, scores = {}, {}, {}
                incident = None
                for name, command in [("zero", np.zeros(25))] + evaluation_commands():
                    record_id = f"pred-{identity}-{name}"
                    record, arrays = require(
                        record_id,
                        "forward",
                        {
                            "resolution": n,
                            "command_m": command.tolist(),
                            "model_gain": model["gain"],
                            "model_receipt_sha256": receipt_hash,
                            "component_control": False,
                            "direction": None,
                        },
                    )
                    if not model["frozen_after_attempt"] < record["attempt"] <= choice["frozen_after_attempt"]:
                        raise ValueError("Prediction is not between model and choice freezes")
                    predicted[name] = arrays["power"]
                    current_incident = float(arrays["incident"])
                    if current_incident <= 0 or (incident is not None and current_incident != incident):
                        raise ValueError("Candidate input flux changed across commands")
                    incident = current_incident
                    prediction_hashes[name] = file_hash(packet.root / "records" / (record_id + ".json"))
                    if name != "zero":
                        scores[name] = core_power(arrays["power"], focal_axis(), focal_axis())
                rebuilt_choice = choose_action(scores)
                if choice["predictions"] != prediction_hashes or choice["scores"] != scores:
                    raise ValueError("Choice predictions or scores do not reproduce")
                if any(choice[k] != v for k, v in rebuilt_choice.items()):
                    raise ValueError("Frozen selected action or near-tie set differs")
                metrics = []
                for name, _ in evaluation_commands():
                    truth = reference[f"ref-{case}-{name}"]["power"]
                    metrics.append(
                        {
                            "command_id": name,
                            "forward": error_metrics(predicted[name], truth),
                            "action": error_metrics(predicted[name] - predicted["zero"], truth - reference_zero),
                            "candidate_captured_power": float(np.sum(predicted[name])),
                            "reference_captured_power": float(np.sum(truth)),
                            "captured_power_error": float(np.sum(predicted[name]) - np.sum(truth)),
                        }
                    )
                derivatives, derivative_hashes = [], {}
                for name, command, direction in derivative_anchors():
                    record_id = f"jvp-{identity}-{name}"
                    record, arrays = require(
                        record_id,
                        "jvp",
                        {
                            "resolution": n,
                            "command_m": command.tolist(),
                            "model_gain": model["gain"],
                            "model_receipt_sha256": receipt_hash,
                            "component_control": False,
                            "direction": direction.tolist(),
                        },
                    )
                    if not model["frozen_after_attempt"] < record["attempt"] <= choice["frozen_after_attempt"]:
                        raise ValueError("JVP is not between model and choice freezes")
                    derivative_hashes[name] = file_hash(packet.root / "records" / (record_id + ".json"))
                    derivatives.append(
                        {
                            "command_id": name,
                            "steps": [
                                {
                                    "step_nm": step_nm,
                                    **error_metrics(
                                        arrays["jvp"], _central_difference(packet, case, name, step), scale=1e-9
                                    ),
                                }
                                for step, step_nm in enumerate(STEPS_NM)
                            ],
                        }
                    )
                if choice["derivatives"] != derivative_hashes:
                    raise ValueError("JVP hash chain differs")
                rows.append(
                    {
                        **row,
                        "status": "available",
                        "candidate_incident_power": incident,
                        "reference_incident_power": float(reference[f"ref-{case}-zero"]["incident"]),
                        "actions": metrics,
                        "derivatives_per_nm": derivatives,
                        "decision": {**rebuilt_choice, **action_regret(rebuilt_choice, reference_scores)},
                    }
                )
        for name, _, _ in derivative_anchors():
            primary, fine = reference[f"ref-{case}-{name}"], reference[f"ref-{case}-{name}-grid512"]
            convergence.append(
                {
                    "case": case,
                    "command_id": name,
                    "power_512_vs_256": error_metrics(fine["power"], primary["power"]),
                    "incident_power_256": float(primary["incident"]),
                    "incident_power_512": float(fine["incident"]),
                    "captured_power_256": float(np.sum(primary["power"])),
                    "captured_power_512": float(np.sum(fine["power"])),
                }
            )
    if min(heldout_attempts) <= max(freeze_positions):
        raise ValueError("Held-out reference was acquired before all choices were frozen")
    actual = {p.stem for p in (packet.root / "records").glob("*.json")}
    if actual != expected:
        raise ValueError("Terminal record coverage differs from the full declared study")
    bound_arrays = {
        record["array_file"]
        for records in packet.prepared_by_id.values()
        for record in records
        if record["array_file"] and (packet.root / record["array_file"]).exists()
    }
    actual_arrays = {"arrays/" + path.name for path in (packet.root / "arrays").glob("*.npz")}
    if actual_arrays != bound_arrays:
        raise ValueError("Unbound or missing raw array artifacts")
    for folder, identities in (("models", model_ids), ("choices", choice_ids)):
        if {p.stem for p in (packet.root / folder).glob("*.json")} != identities:
            raise ValueError("Frozen model/choice coverage differs")
    for attempt in dispatch:
        if attempt["id"] not in expected:
            raise ValueError("Dispatch contains an undeclared numerical identity")
        terminal = packet.load_record(attempt["id"])[0]
        if attempt["inputs"] != terminal["inputs"] or attempt["kind"] != terminal["kind"]:
            raise ValueError("Retried numerical identity changed inputs")
    unavailable = sum(row["status"] == "unavailable" for row in rows)
    return {
        "schema": SCHEMA,
        "protocol_sha256": packet.contract_hash,
        "status": "complete_with_unavailable_arms" if unavailable else "complete_success",
        "counts": counts,
        "terminal_records": len(expected),
        "unavailable_arms": unavailable,
        "component_gates": gates,
        "arms": rows,
        "convergence": convergence,
        "limits": "Static noiseless deterministic fixtures; no independent replicates, "
        "global fit, control or hardware claim.",
    }


def render_report(analysis):
    lines = [
        "# Static mirror-intervention qualification",
        "",
        f"Status: **{analysis['status']}**.",
        "",
        "Independent HCIPy reference and dLux candidates on fixed physical focal samples. "
        "The reference pupil has 256 samples per side; the 512-grid check is reported separately.",
        "",
        "| Case | Pupil samples | Arm | Gain | Maximum forward L1 error | Core-power regret |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in analysis["arms"]:
        if row["status"] == "unavailable":
            cells = [row["case"], str(row["resolution"]), row["arm"], "unavailable", row["reason"], "unavailable"]
        else:
            cells = [
                row["case"],
                str(row["resolution"]),
                row["arm"],
                f"{row['model_gain']:.9g}",
                f"{max(v['forward']['absolute_l1'] for v in row['actions']):.9g}",
                f"{row['decision']['regret']:.9g}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "All forward, differential-action and derivative errors, fluxes, chosen IDs and ties are retained in "
        "`analysis.json`. Derivatives use all three declared finite-difference steps; "
        "no step was selected after observing errors.",
        "",
        "Core power sums 49 preweighted pixel powers at integer coordinates p²+q² ≤ 16. "
        "Candidate scores within 1e-10 of the predicted maximum are tied; "
        "the lexicographically first ID is selected. "
        "Regret uses the actual reference maximum without this predicted tie rule.",
        "",
        "Scalar recalibration uses only eighteen signed axial probes, an 81-point gain grid and at most forty "
        "bounded local-refinement evaluations. Multiple exact grid minima "
        "or exhausted refinement leave that arm unavailable.",
        "",
        analysis["limits"],
        "",
        "The fixed mask is a discrete aperture metric. Finite-grid parity is not continuum convergence; "
        "known diversity and illumination constrain the calibration. "
        "These fixtures do not establish identifiability "
        "under unrestricted nuisance, controller superiority, dynamic robustness or hardware reliability.",
        "",
    ]
    return "\n".join(lines)


def _file_manifest(root):
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.name not in (".writer.lock", "manifest.json")
    }


def finalize(packet):
    analysis = analyze(packet)
    for name, payload in (
        ("analysis.json", canonical(analysis) + b"\n"),
        ("report.md", render_report(analysis).encode()),
    ):
        path = packet.root / name
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("Previously published analysis differs from raw evidence")
        else:
            packet.checkpoint(len(payload))
            _publish(path, payload)
            packet.packet_bytes += len(payload)
    manifest = {
        "schema": SCHEMA,
        "protocol_sha256": packet.contract_hash,
        "status": analysis["status"],
        "counts": analysis["counts"],
        "files": _file_manifest(packet.root),
    }
    packet.write_json(packet.root / "manifest.json", manifest)
    packet.complete = True
    return verify_packet(packet.root)


def verify_packet(root):
    packet = EvidencePacket.reader(root)
    manifest = _read_json(packet.root / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["protocol_sha256"] != packet.contract_hash:
        raise ValueError("Completion manifest belongs to another protocol")
    if manifest["files"] != _file_manifest(packet.root):
        raise ValueError("Completion manifest file closure or hash differs")
    analysis = analyze(packet)
    if _read_json(packet.root / "analysis.json") != analysis:
        raise ValueError("Analysis does not reproduce from retained evidence")
    if (packet.root / "report.md").read_text() != render_report(analysis):
        raise ValueError("Report differs from regenerated analysis")
    if manifest["status"] != analysis["status"] or manifest["counts"] != analysis["counts"]:
        raise ValueError("Completion status/counts differ")
    return analysis


def runtime_identity():
    """Acquisition-only runtime check; report verification does not call this."""
    required_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
    }
    for key, expected in required_environment.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"Acquisition requires {key}={expected} before importing numerical backends")
    if os.environ.get("JAX_ENABLE_X64", "").lower() not in ("1", "true"):
        raise RuntimeError("Acquisition requires JAX_ENABLE_X64=true before importing JAX")
    if "--xla_cpu_multi_thread_eigen=false" not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError("Acquisition requires single-worker Eigen CPU configuration")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS or platform.python_version() != "3.11.15":
        raise RuntimeError("Acquisition runtime differs from the declared package/Python versions")
    root = Path(__file__).resolve().parents[3]
    with (root / "uv.lock").open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    if not any(p["name"].lower() == "dlux" and p["version"] == "0.15.1" for p in packages):
        raise RuntimeError("Resolve and review the declared dLux dependency lock before acquisition")
    import dLux
    import hcipy
    import jax

    if not jax.config.x64_enabled or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Acquisition requires JAX float64 on CPU")
    dependency_source = {}
    for package, paths in (
        (
            hcipy,
            (
                "field/util.py",
                "aperture/generic.py",
                "optics/deformable_mirror.py",
                "mode_basis/gaussian_pokes.py",
                "propagation/fraunhofer.py",
            ),
        ),
        (dLux, ("wavefronts.py", "utils/propagation.py")),
    ):
        directory = Path(package.__file__).resolve().parent
        for relative in paths:
            dependency_source[package.__name__ + "/" + relative] = file_hash(directory / relative)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "backend": "cpu",
        "x64": True,
        "thread_environment": {**required_environment, "XLA_FLAGS": os.environ["XLA_FLAGS"]},
        "dependency_source": dependency_source,
        "interpreter_sha256": file_hash(Path(sys.executable).resolve()),
    }


def validate_admission(admission, output):
    """Check an externally approved internal-image receipt without publishing private paths."""
    if admission.get("status") != "verified_internal_copy_mounted" or admission.get("writable") is not True:
        raise ValueError("A verified writable internal-copy admission is required")
    mount, image, output = Path(admission["mount"]), Path(admission["image"]), Path(output)
    if mount.is_symlink() or image.is_symlink() or not mount.is_dir() or not image.is_dir():
        raise ValueError("Admission mount and backing image must be real directories")
    if mount.stat().st_dev != admission["st_dev"] or admission["mount"] == admission.get("source_mount"):
        raise ValueError("Admission does not identify the distinct writable internal volume")
    if not output.resolve().is_relative_to(mount.resolve()) or output.resolve() == mount.resolve():
        raise ValueError("New run output must be beneath the admitted internal research volume")
    for name in ("TMPDIR", "JAX_COMPILATION_CACHE_DIR", "XDG_CACHE_HOME"):
        value = os.environ.get(name)
        if value is None or not Path(value).is_dir() or not Path(value).resolve().is_relative_to(mount.resolve()):
            raise ValueError(f"{name} must point to an existing directory inside the admitted research volume")
    storage = ((mount, admission["st_dev"]), (image.parent, image.parent.stat().st_dev))
    for path, _ in storage:
        disk = os.statvfs(path)
        if disk.f_bavail * disk.f_frsize < LIMITS["free_bytes"]:
            raise RuntimeError("Admission does not meet the research/backing free-space reserve")
    receipt = {
        "run_id": output.name,
        "device": admission["st_dev"],
        "backing_device": image.parent.stat().st_dev,
        "volume_uuid": admission["volume_uuid"],
        "admission_sha256": digest(admission),
        "storage_kind": "internal_image_fresh_host",
        "free_space_floor_bytes": LIMITS["free_bytes"],
    }
    return receipt, storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("protocol", help="Print the exact declaration without running optics")
    run = commands.add_parser("run", help="Acquire the bounded static study after external admission")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify", help="Verify and regenerate a completed packet without optical execution")
    verify.add_argument("packet", type=Path)
    args = parser.parse_args(argv)
    if args.command == "protocol":
        print(json.dumps(protocol(), indent=2, allow_nan=False))
        return 0
    if args.command == "verify":
        result = verify_packet(args.packet)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}))
        return 0
    admission, storage = validate_admission(_read_json(args.admission), args.output)
    runtime = runtime_identity()
    with contextlib.closing(
        EvidencePacket(args.output, protocol(), runtime, admission, resume=args.resume, storage_roots=storage)
    ) as packet:
        if packet.complete:
            result = verify_packet(packet.root)
        else:
            try:
                result = acquire(packet)
            except Exception as exc:
                stop = {
                    "status": "stopped_incomplete",
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "counts": packet.counts,
                    "protocol_sha256": packet.contract_hash,
                }
                if not (packet.root / "stop.json").exists():
                    # A lost mount/floor failure must not trigger fallback writes.
                    with contextlib.suppress(OSError, RuntimeError):
                        packet.write_json(packet.root / "stop.json", stop, terminal_publication=True)
                print(json.dumps(stop, allow_nan=False), file=sys.stderr)
                return 1
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
