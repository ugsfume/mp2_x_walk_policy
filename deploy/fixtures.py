"""Fixture bundles: a compressed NPZ of arrays plus a JSON manifest that records
the dtype, shape and SHA-256 of every array and of the NPZ itself, so a bundle
can be verified before anything is compared against it.

    write_bundle("dir/fixtures", {"observations": obs, ...}, metadata={...})
        -> dir/fixtures.json + dir/fixtures.npz
    arrays, metadata = load_bundle("dir/fixtures.json")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "mp2_x_walk_fixtures/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_bundle(stem: str | Path, arrays: Mapping[str, np.ndarray], *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write ``<stem>.npz`` and ``<stem>.json``; return the manifest path.
    ``stem`` must not carry an extension."""
    stem = Path(stem).expanduser().resolve()
    if stem.suffix:
        raise ValueError(f"pass the bundle stem without an extension, got {stem.name}")
    if not arrays:
        raise ValueError("at least one array is required")
    normalized = {}
    for name, value in sorted(arrays.items()):
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError(f"array {name!r} must not contain objects")
        normalized[name] = np.array(value, copy=True, order="C")
    stem.parent.mkdir(parents=True, exist_ok=True)
    data_path = stem.with_suffix(".npz")
    np.savez_compressed(data_path, **normalized)
    manifest = {
        "schema": SCHEMA,
        "data": {"path": data_path.name, "sha256": sha256_file(data_path)},
        "arrays": [
            {"name": name, "dtype": v.dtype.name, "shape": list(v.shape), "sha256": sha256_array(v)}
            for name, v in normalized.items()
        ],
        "metadata": dict(metadata or {}),
    }
    manifest_path = stem.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_bundle(manifest_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a bundle, verifying the NPZ hash and every array's dtype, shape and hash."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{manifest_path}: unsupported schema {manifest.get('schema')!r}")
    data_path = manifest_path.parent / manifest["data"]["path"]
    observed = sha256_file(data_path)
    if observed != manifest["data"]["sha256"]:
        raise ValueError(f"{data_path}: sha256 {observed} != manifest {manifest['data']['sha256']}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(data_path, allow_pickle=False) as archive:
        expected = {spec["name"] for spec in manifest["arrays"]}
        if set(archive.files) != expected:
            raise ValueError(f"{data_path}: arrays {sorted(archive.files)} != manifest {sorted(expected)}")
        for spec in manifest["arrays"]:
            value = np.array(archive[spec["name"]], copy=True, order="C")
            if value.dtype.name != spec["dtype"] or list(value.shape) != spec["shape"]:
                raise ValueError(f"array {spec['name']!r}: {value.dtype.name} {value.shape} != manifest")
            if sha256_array(value) != spec["sha256"]:
                raise ValueError(f"array {spec['name']!r}: content hash mismatch")
            arrays[spec["name"]] = value
    return arrays, dict(manifest.get("metadata", {}))
