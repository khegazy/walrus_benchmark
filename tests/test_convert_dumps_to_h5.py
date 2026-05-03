"""Unit tests for ``scripts/convert_dumps_to_h5.py``.

The conversion script lives outside the ``walrus`` package, so it is loaded
via ``importlib`` from the repo's ``scripts/`` directory. Tests build their
own synthetic Well-format HDF5 files (simpler than ``the_well`` 's
``write_dummy_data`` and lets us control timestamps for overlap detection).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_dumps_to_h5.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "convert_dumps_to_h5", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv():
    """The loaded conversion module, exposed as ``cv`` to tests."""
    return _load_script_module()


# ----------------------------------------------------------------------------
# Synthetic Well-format builder
# ----------------------------------------------------------------------------


def _write_well_file(
    path: Path,
    *,
    n_trajectories: int,
    time: np.ndarray,
    scalar_fields: Sequence[str] = ("density",),
    vector_fields: Sequence[str] = ("velocity",),
    spatial: tuple = (4, 4),
    spatial_dims: Sequence[str] = ("x", "y"),
):
    """Write a minimal Well-format HDF5 file sufficient for the converter.

    Stores ``len(scalar_fields)`` time-varying scalars in ``t0_fields`` and
    ``len(vector_fields)`` time-varying vector fields in ``t1_fields``,
    each with shape ``(n_trajectories, len(time), *spatial[, len(spatial_dims)])``.
    """
    n_t = int(time.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = np.array(list(spatial_dims), dtype=object)
        dims.create_dataset("time", data=time.astype(np.float32))
        for i, name in enumerate(spatial_dims):
            dims.create_dataset(name, data=np.linspace(0, 1, spatial[i], dtype=np.float32))

        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = np.array(list(scalar_fields), dtype=object)
        for s in scalar_fields:
            data = np.random.default_rng(0).standard_normal(
                (n_trajectories, n_t, *spatial)
            ).astype(np.float32)
            ds = t0.create_dataset(s, data=data)
            ds.attrs["time_varying"] = True
            ds.attrs["sample_varying"] = True
            ds.attrs["dim_varying"] = [True, True]

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = np.array(list(vector_fields), dtype=object)
        for v in vector_fields:
            data = np.random.default_rng(1).standard_normal(
                (n_trajectories, n_t, *spatial, len(spatial_dims))
            ).astype(np.float32)
            ds = t1.create_dataset(v, data=data)
            ds.attrs["time_varying"] = True
            ds.attrs["sample_varying"] = True
            ds.attrs["dim_varying"] = [True, True]

        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = np.array([], dtype=h5py.string_dtype())


def _make_well_dataset(
    root: Path,
    splits_time: dict,
    *,
    n_trajectories_per_file: int = 1,
    scalar_fields: Sequence[str] = ("density",),
    vector_fields: Sequence[str] = ("velocity",),
):
    """Create a Well-format dataset under ``root/data/{train,valid,test}/file_00.h5``.

    ``splits_time`` maps split name -> 1D ``np.ndarray`` of timestamps.
    """
    for split, time in splits_time.items():
        _write_well_file(
            root / "data" / split / "file_00.h5",
            n_trajectories=n_trajectories_per_file,
            time=time,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
        )


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------


def test_group_channels_scalars_only(cv):
    groups = cv.group_channels_by_field(["density", "pressure"])
    assert groups == [("density", [0]), ("pressure", [1])]


def test_group_channels_vector_2d(cv):
    groups = cv.group_channels_by_field(
        ["density", "velocity_x", "velocity_y", "vorticity"]
    )
    assert groups == [
        ("density", [0]),
        ("velocity", [1, 2]),
        ("vorticity", [3]),
    ]


def test_group_channels_vector_3d(cv):
    groups = cv.group_channels_by_field(
        ["density", "velocity_x", "velocity_y", "velocity_z"]
    )
    assert groups == [("density", [0]), ("velocity", [1, 2, 3])]


def test_group_channels_tensor(cv):
    groups = cv.group_channels_by_field(
        ["stress_xx", "stress_xy", "stress_yx", "stress_yy"]
    )
    assert groups == [("stress", [0, 1, 2, 3])]


def test_apply_field_renames_scalar(cv):
    assert cv.apply_field_renames(
        ["density", "internal_energy", "pressure"]
    ) == ["density", "energy", "pressure"]


def test_apply_field_renames_preserves_vector_suffix(cv):
    # Vector component suffixes survive a rename of the base name.
    assert cv.apply_field_renames(["internal_energy_x", "internal_energy_y"]) == [
        "energy_x",
        "energy_y",
    ]


def test_apply_field_renames_passthrough(cv):
    assert cv.apply_field_renames(["velocity_x", "vorticity"]) == [
        "velocity_x",
        "vorticity",
    ]


def test_infer_experiment_name(cv, tmp_path):
    dump_dir = tmp_path / "experiments" / "eval_demo" / "viz" / "ds" / "full_trajectory_dumps"
    dump_dir.mkdir(parents=True)
    assert cv.infer_experiment_name(str(dump_dir)) == "eval_demo"


def test_infer_experiment_name_missing(cv, tmp_path):
    with pytest.raises(ValueError):
        cv.infer_experiment_name(str(tmp_path / "no" / "marker" / "here"))


# ----------------------------------------------------------------------------
# find_dump_files
# ----------------------------------------------------------------------------


def _touch_dump(dump_dir: Path, name: str, shape=(1, 4, 4, 4, 2)):
    arr = np.zeros(shape, dtype=np.float32)
    np.save(dump_dir / name, arr)


def test_find_dump_files_pairs_and_sorts(cv, tmp_path):
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    for j in (2, 0, 1):
        _touch_dump(dump_dir, f"ypred_ds_rollout_valid_epoch1_rank0_{j}.npy")
        _touch_dump(dump_dir, f"yref_ds_rollout_valid_epoch1_rank0_{j}.npy")
    # Add an unmatched ypred (no yref) — should be skipped with a warning.
    _touch_dump(dump_dir, "ypred_ds_rollout_valid_epoch1_rank0_99.npy")

    dumps = cv.find_dump_files(str(dump_dir))
    # Sorted by (dataset, split, epoch, rank, batch); batches 0,1,2.
    assert [d.batch for d in dumps] == [0, 1, 2]
    for d in dumps:
        assert d.dataset_name == "ds"
        assert d.split == "rollout_valid"
        assert d.rank == 0


def test_find_dump_files_split_filter(cv, tmp_path):
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    for split in ("rollout_valid", "rollout_test"):
        _touch_dump(dump_dir, f"ypred_ds_{split}_epoch1_rank0_0.npy")
        _touch_dump(dump_dir, f"yref_ds_{split}_epoch1_rank0_0.npy")

    valid_only = cv.find_dump_files(str(dump_dir), split="valid")
    test_only = cv.find_dump_files(str(dump_dir), split="test")
    assert len(valid_only) == 1 and valid_only[0].split == "rollout_valid"
    assert len(test_only) == 1 and test_only[0].split == "rollout_test"


# ----------------------------------------------------------------------------
# Field-name detection from Well files
# ----------------------------------------------------------------------------


def test_flat_field_names_from_well_file_2d(cv, tmp_path):
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={"valid": np.linspace(0, 1, 5)},
        scalar_fields=("density", "pressure"),
        vector_fields=("velocity",),
    )
    flat = cv.flat_field_names_from_well_file(
        str(root / "data" / "valid" / "file_00.h5"), n_spatial_dims=2
    )
    assert flat == ["density", "pressure", "velocity_x", "velocity_y"]


def test_flat_field_names_from_well_file_3d(cv, tmp_path):
    root = tmp_path / "ds"
    _write_well_file(
        root / "data" / "valid" / "file_00.h5",
        n_trajectories=1,
        time=np.linspace(0, 1, 4),
        scalar_fields=("density",),
        vector_fields=("velocity",),
        spatial=(4, 4, 4),
        spatial_dims=("x", "y", "z"),
    )
    flat = cv.flat_field_names_from_well_file(
        str(root / "data" / "valid" / "file_00.h5"), n_spatial_dims=3
    )
    assert flat == ["density", "velocity_x", "velocity_y", "velocity_z"]


# ----------------------------------------------------------------------------
# Trajectory info / overlap-aware offset table
# ----------------------------------------------------------------------------


def test_trajectory_info_returns_steps_and_timestamps(cv, tmp_path):
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={"train": np.linspace(0, 1, 5)},
        n_trajectories_per_file=2,
    )
    info = cv.trajectory_info_in_split(str(root), "train")
    assert info == [(5, 0.0, 1.0), (5, 0.0, 1.0)]


def _make_cfg(
    *,
    dataset_name: str,
    dataset_path: Path,
    n_steps_input: int,
    start_rollout_at: int = -1,
    min_dt_stride: int = 1,
    max_dt_stride: int = 1,
):
    """Build the minimal OmegaConf-shaped object the converter reads."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {
                "module_parameters": {
                    "n_steps_input": n_steps_input,
                    "min_dt_stride": min_dt_stride,
                    "max_dt_stride": max_dt_stride,
                    "start_rollout_valid_output_at_t": start_rollout_at,
                    "well_dataset_info": {
                        dataset_name: {"path": str(dataset_path)},
                    },
                }
            }
        }
    )


def _make_dump_record(cv, *, dataset_name: str, split: str, rank: int, batch: int):
    return cv.DumpFile(
        pred_path=f"<fake>_rank{rank}_{batch}.npy",
        ref_path=f"<fake>_rank{rank}_{batch}.npy",
        dataset_name=dataset_name,
        split=split,
        epoch=1,
        rank=rank,
        batch=batch,
    )


def test_auto_compute_t0s_no_overlap(cv, tmp_path):
    """Three independent splits with disjoint timestamps -> cumulative offsets
    should add full trajectory lengths."""
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={
            "train": np.linspace(0, 1, 5),    # last t = 1
            "valid": np.linspace(2, 3, 5),    # first t = 2 (no overlap)
            "test":  np.linspace(4, 5, 5),
        },
    )
    cfg = _make_cfg(dataset_name="ds", dataset_path=root, n_steps_input=2)
    dumps = [
        _make_dump_record(cv, dataset_name="ds", split="rollout_valid", rank=0, batch=0),
        _make_dump_record(cv, dataset_name="ds", split="rollout_test",  rank=0, batch=0),
    ]
    shapes = [(1, 3, 4, 4, 3), (1, 3, 4, 4, 3)]  # (B, T, x, y, C)
    t0s = cv.auto_compute_t0s(cfg, dumps, shapes, world_size=1, dt_stride=1)
    # Train traj 0 -> offset 0; valid traj 0 -> offset 5; test traj 0 -> offset 10.
    # Last input local index = max(2, -1) - 1 = 1.
    assert t0s == [5 + 1, 10 + 1]


def test_auto_compute_t0s_overlap_subtracted(cv, tmp_path):
    """Splits that share a boundary timestamp -> offset should subtract one."""
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={
            "train": np.linspace(0, 1, 5),    # last = 1
            "valid": np.linspace(1, 2, 5),    # first = 1 (shared with train)
            "test":  np.linspace(2, 3, 5),    # first = 2 (shared with valid)
        },
    )
    cfg = _make_cfg(dataset_name="ds", dataset_path=root, n_steps_input=2)
    dumps = [
        _make_dump_record(cv, dataset_name="ds", split="rollout_valid", rank=0, batch=0),
        _make_dump_record(cv, dataset_name="ds", split="rollout_test",  rank=0, batch=0),
    ]
    shapes = [(1, 3, 4, 4, 3), (1, 3, 4, 4, 3)]
    t0s = cv.auto_compute_t0s(cfg, dumps, shapes, world_size=1, dt_stride=1)
    # With overlap, valid offset = 5 - 1 = 4, test = 4 + 5 - 1 = 8.
    # last_input_local = 1.
    assert t0s == [4 + 1, 8 + 1]


def test_auto_compute_t0s_uses_start_rollout_when_larger(cv, tmp_path):
    """If start_rollout_valid_output_at_t > Ni*dt, last input index follows it."""
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={
            "train": np.linspace(0, 5, 20),    # 20 steps, last = 5
            "valid": np.linspace(6, 7, 20),    # 20 steps, first = 6 (no overlap)
            "test":  np.linspace(8, 9, 20),
        },
    )
    cfg = _make_cfg(
        dataset_name="ds",
        dataset_path=root,
        n_steps_input=3,
        start_rollout_at=17,  # exceeds Ni*dt=3
    )
    dumps = [_make_dump_record(cv, dataset_name="ds", split="rollout_valid", rank=0, batch=0)]
    shapes = [(1, 2, 4, 4, 3)]
    t0s = cv.auto_compute_t0s(cfg, dumps, shapes, world_size=1, dt_stride=1)
    # last_input_local = max(3, 17) - 1 = 16. Valid offset = 20.
    assert t0s == [20 + 16]


def test_auto_compute_t0s_world_size_stripe(cv, tmp_path):
    """Multi-rank stripe: rank 0 batches 0,1,2 with W=4 -> traj 0,4,8 in split."""
    root = tmp_path / "ds"
    n_traj_valid = 10
    _make_well_dataset(
        root,
        splits_time={
            "train": np.linspace(0, 1, 5),
            "valid": np.linspace(2, 3, 7),
            "test":  np.linspace(4, 5, 5),
        },
        n_trajectories_per_file=1,
    )
    # Re-create valid with multiple trajectories.
    _write_well_file(
        root / "data" / "valid" / "file_00.h5",
        n_trajectories=n_traj_valid,
        time=np.linspace(2, 3, 7),
        scalar_fields=("density",),
        vector_fields=("velocity",),
    )
    cfg = _make_cfg(dataset_name="ds", dataset_path=root, n_steps_input=2)
    dumps = [
        _make_dump_record(cv, dataset_name="ds", split="rollout_valid", rank=0, batch=j)
        for j in (0, 1, 2)
    ]
    shapes = [(1, 3, 4, 4, 3)] * 3
    t0s = cv.auto_compute_t0s(cfg, dumps, shapes, world_size=4, dt_stride=1)
    # Train has 5 steps (1 traj). Valid trajectory k has offset 5 + 7*k.
    # With W=4, rank 0 sees split-local trajectory indices 0, 4, 8.
    # last_input_local = 1.
    assert t0s == [
        5 + 0 * 7 + 1,
        5 + 4 * 7 + 1,
        5 + 8 * 7 + 1,
    ]


def test_auto_compute_t0s_out_of_range(cv, tmp_path):
    """Trajectory index past the available count should raise IndexError."""
    root = tmp_path / "ds"
    _make_well_dataset(
        root,
        splits_time={
            "train": np.linspace(0, 1, 5),
            "valid": np.linspace(2, 3, 5),
            "test":  np.linspace(4, 5, 5),
        },
    )
    cfg = _make_cfg(dataset_name="ds", dataset_path=root, n_steps_input=2)
    dumps = [_make_dump_record(cv, dataset_name="ds", split="rollout_valid", rank=0, batch=5)]
    shapes = [(1, 3, 4, 4, 3)]
    with pytest.raises(IndexError):
        cv.auto_compute_t0s(cfg, dumps, shapes, world_size=1, dt_stride=1)


# ----------------------------------------------------------------------------
# End-to-end via main()
# ----------------------------------------------------------------------------


def test_write_forecast_h5_forecast_steps_filter(cv, tmp_path):
    """Only the requested forecast steps should appear in the HDF5; others
    are dropped, and out-of-range entries are silently warned away."""
    T, X, Y, C = 5, 4, 4, 3
    pred = np.random.default_rng(0).standard_normal((T, X, Y, C)).astype(np.float32)
    ref = np.random.default_rng(1).standard_normal((T, X, Y, C)).astype(np.float32)
    out = tmp_path / "f.h5"
    cv.write_forecast_h5(
        str(out),
        pred_sample=pred,
        ref_sample=ref,
        field_names=["density", "velocity_x", "velocity_y"],
        field_groups=[("density", [0]), ("velocity", [1, 2])],
        t0=100,
        dt_stride=2,
        dataset_name="ds",
        split="rollout_valid",
        source_file="<test>",
        forecast_steps=[1, 3, 99],  # 99 is out of range -> skipped
    )
    with h5py.File(out, "r") as f:
        # n_steps suffix = forecast_step * dt_stride.
        density_keys = sorted(k for k in f["density"].keys())
        assert density_keys == ["pred_2", "pred_6", "target_2", "target_6"]
        # time_index pred_2 -> t0 + 2 = 102; pred_6 -> 106.
        assert int(f["time_index/pred_2"][0]) == 102
        assert int(f["time_index/pred_6"][0]) == 106
        # Round-trip: density at forecast_step=1 == pred[0, ..., 0].
        np.testing.assert_allclose(
            np.asarray(f["density/pred_2"][0]), pred[0, ..., 0], rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(f["density/pred_6"][0]), pred[2, ..., 0], rtol=1e-6
        )


def test_end_to_end(cv, tmp_path, monkeypatch):
    """Synthetic dataset + dump pair -> kinet-style HDF5, with auto t0 and
    correct group structure / data round-trip."""
    # 1. Build a Well dataset whose scalar list includes ``internal_energy``
    #    (which the converter renames to ``energy`` in the output).
    ds_root = tmp_path / "wellds"
    _make_well_dataset(
        ds_root,
        splits_time={
            "train": np.linspace(0, 1, 5),
            "valid": np.linspace(2, 3, 5),
            "test":  np.linspace(4, 5, 5),
        },
        scalar_fields=("density", "internal_energy"),
        vector_fields=("velocity",),
    )

    # 2. Build an extended_config.yaml that points at it.
    experiment_name = "eval_synth"
    exp_dir = tmp_path / "experiments" / experiment_name
    exp_dir.mkdir(parents=True)
    cfg_yaml = (
        "data:\n"
        "  module_parameters:\n"
        "    n_steps_input: 2\n"
        "    min_dt_stride: 1\n"
        "    max_dt_stride: 1\n"
        "    start_rollout_valid_output_at_t: -1\n"
        "    well_dataset_info:\n"
        f"      synth:\n"
        f"        path: {ds_root}\n"
    )
    (exp_dir / "extended_config.yaml").write_text(cfg_yaml)

    # 3. Build a synthetic ypred/yref pair under viz/synth/full_trajectory_dumps.
    #    4 channels: density, internal_energy, velocity_x, velocity_y.
    dump_dir = exp_dir / "viz" / "synth" / "full_trajectory_dumps"
    dump_dir.mkdir(parents=True)
    B, T, X, Y, C = 1, 5, 4, 4, 4
    rng = np.random.default_rng(42)
    ypred = rng.standard_normal((B, T, X, Y, C)).astype(np.float32)
    yref = rng.standard_normal((B, T, X, Y, C)).astype(np.float32)
    np.save(dump_dir / "ypred_synth_rollout_valid_epoch1_rank0_0.npy", ypred)
    np.save(dump_dir / "yref_synth_rollout_valid_epoch1_rank0_0.npy", yref)

    # 4. Run main() with --forecast-steps 1 3 -> only those steps persist;
    #    --split valid, --world-size 1.
    monkeypatch.setattr(
        sys, "argv",
        [
            "convert_dumps_to_h5.py",
            str(dump_dir),
            "--split", "valid",
            "--world-size", "1",
            "--forecast-steps", "1", "3",
        ],
    )
    cv.main()

    # 5. Validate output. Valid traj 0 offset = 5 (no overlap), Ni=2, dt=1
    #    -> last_input_local = 1, t0 = 5 + 1 = 6.
    out_path = dump_dir / "h5" / "forecast_t0-6.h5"
    assert out_path.exists()
    with h5py.File(out_path, "r") as f:
        # internal_energy was renamed to energy.
        assert set(f.keys()) == {"density", "energy", "velocity", "time_index"}
        assert f["density/pred_1"].shape == (1, X, Y)
        assert f["energy/pred_1"].shape == (1, X, Y)
        assert f["velocity/pred_1"].shape == (2, X, Y)
        # --forecast-steps 1 3 -> only pred_1 and pred_3 (and target_*).
        density_pred_keys = sorted(
            int(k.split("_")[1]) for k in f["density"].keys() if k.startswith("pred_")
        )
        assert density_pred_keys == [1, 3]
        # time_index pred_1 = t0 + 1 = 7; pred_3 = 9.
        assert int(f["time_index/pred_1"][0]) == 7
        assert int(f["time_index/pred_3"][0]) == 9
        # The root attr field_names reflects the post-rename channel names.
        assert list(f.attrs["field_names"]) == [
            "density",
            "energy",
            "velocity_x",
            "velocity_y",
        ]
        # Round-trip: energy at forecast_step=1 == ypred[0, 0, ..., 1].
        np.testing.assert_allclose(
            np.asarray(f["energy/pred_1"][0]), ypred[0, 0, ..., 1], rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(f["velocity/pred_1"]),
            np.moveaxis(ypred[0, 0, ..., 2:], -1, 0),
            rtol=1e-6,
        )
        # Reference round-trip.
        np.testing.assert_allclose(
            np.asarray(f["density/target_3"][0]), yref[0, 2, ..., 0], rtol=1e-6
        )
