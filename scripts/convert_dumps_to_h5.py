"""Convert Walrus full_trajectory_dumps (.npy) to kinet-style HDF5.

Walrus evaluation runs save rollout predictions and references as flat-channel
NumPy arrays under ``<experiment>/viz/<dataset>/full_trajectory_dumps/``:

    ypred_<dataset>_<split>_epoch<N>_rank<R>_<batch_idx>.npy
    yref_<dataset>_<split>_epoch<N>_rank<R>_<batch_idx>.npy

Each array has shape ``(B, T, H, W, C)`` for 2D data or ``(B, T, H, W, D, C)``
for 3D, where ``C`` is the flat field-channel axis.

This script reorganises those dumps into HDF5 files that match the kinet
"forecast" layout (e.g. ``forecast_t0-5000.h5`` in the kinet experiments tree):

    forecast_t0-{t0}.h5
        density/pred_{n_steps}            (1, X, Y[, Z])
        density/target_{n_steps}          (1, X, Y[, Z])
        velocity/pred_{n_steps}           (n_components, X, Y[, Z])
        velocity/target_{n_steps}         (n_components, X, Y[, Z])
        ...
        t_index/pred_{n_steps}            (1,)  -> [t0 + n_steps]
        t_index/target_{n_steps}          (1,)

Field names and their channel order are detected automatically from the
evaluation config (``walrus/configs/<experiment_name>.yaml``) and the Well
dataset it points at; the user is not expected to specify them.

Usage
-----
    python scripts/convert_dumps_to_h5.py <dump_dir> --t0s 5000 5500 6000 ...

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


_DUMP_PATTERN = re.compile(
    r"^ypred_(?P<dset>.+?)_(?P<split>rollout_(?:valid|test))"
    r"_epoch(?P<epoch>\d+)_rank(?P<rank>\d+)_(?P<batch>\d+)\.npy$"
)


@dataclass
class DumpFile:
    pred_path: str
    ref_path: str
    dataset_name: str
    split: str
    epoch: int
    rank: int
    batch: int


def find_dump_files(dump_dir: str, split: Optional[str] = None) -> List[DumpFile]:
    """Find matching ypred/yref pairs in ``dump_dir``.

    Sorted by (dataset, split, epoch, rank, batch) so the order is deterministic
    and matches the user-supplied ``--t0s`` list.
    """
    pred_files = sorted(glob.glob(os.path.join(dump_dir, "ypred_*.npy")))
    out: List[DumpFile] = []
    for p in pred_files:
        m = _DUMP_PATTERN.match(os.path.basename(p))
        if m is None:
            continue
        if split is not None and m.group("split") != f"rollout_{split}":
            continue
        ref = p.replace("/ypred_", "/yref_")
        if not os.path.exists(ref):
            print(f"Warning: no matching yref for {p}; skipping", file=sys.stderr)
            continue
        out.append(
            DumpFile(
                pred_path=p,
                ref_path=ref,
                dataset_name=m.group("dset"),
                split=m.group("split"),
                epoch=int(m.group("epoch")),
                rank=int(m.group("rank")),
                batch=int(m.group("batch")),
            )
        )
    out.sort(key=lambda d: (d.dataset_name, d.split, d.epoch, d.rank, d.batch))
    return out


def infer_experiment_name(dump_dir: str) -> str:
    """Recover the experiment name from a dump dir path of the form
    ``.../experiments/<name>/viz/<dataset>/full_trajectory_dumps``."""
    parts = os.path.abspath(dump_dir).split(os.sep)
    try:
        i = parts.index("experiments")
    except ValueError as e:
        raise ValueError(
            f"Cannot infer experiment name from {dump_dir!r}: no 'experiments' "
            "component in the path. Pass --experiment-name explicitly."
        ) from e
    if i + 1 >= len(parts):
        raise ValueError(f"Malformed dump_dir {dump_dir!r}.")
    return parts[i + 1]


def load_run_config(dump_dir: str, experiment_name: str) -> Any:
    """Return the configuration that was used by the run that produced the
    dumps.

    Prefers the snapshot at ``<experiment_dir>/extended_config.yaml`` because
    it is frozen at run time; falls back to composing
    ``walrus/configs/<experiment_name>.yaml`` via Hydra (which may have been
    edited after the run, so values like ``n_steps_input`` could drift).
    """
    from omegaconf import OmegaConf

    experiment_dir = os.path.dirname(os.path.dirname(os.path.abspath(dump_dir)))
    while os.path.basename(experiment_dir) != experiment_name and experiment_dir not in ("/", ""):
        parent = os.path.dirname(experiment_dir)
        if parent == experiment_dir:
            experiment_dir = ""
            break
        experiment_dir = parent
    snapshot = os.path.join(experiment_dir, "extended_config.yaml") if experiment_dir else ""
    if snapshot and os.path.isfile(snapshot):
        print(f"Loading run config from {snapshot}", file=sys.stderr)
        return OmegaConf.load(snapshot)

    # Fallback: compose from the live configs.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(repo_root, "walrus", "configs")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Walrus configs directory not found at {config_dir}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    print(
        f"No extended_config.yaml found for {experiment_name!r}; composing "
        f"live config — values may have drifted from the actual run.",
        file=sys.stderr,
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=experiment_name)


def well_dataset_info_for(cfg: Any, dataset_name: str) -> Dict[str, Any]:
    """Pick the entry from ``data.module_parameters.well_dataset_info`` whose
    key matches the dataset name embedded in the dump filename."""
    info = cfg.data.module_parameters.well_dataset_info
    if dataset_name in info:
        return info[dataset_name]
    # Some dataset configs use a different key than the runtime dataset name —
    # fall back to the only entry if there's just one.
    keys = list(info.keys())
    if len(keys) == 1:
        return info[keys[0]]
    raise KeyError(
        f"Dataset {dataset_name!r} not found in well_dataset_info keys "
        f"{keys}. Add it to the config or rename the dump."
    )


def find_well_file(dataset_path: str, split: str) -> str:
    """Locate any Well-format HDF5 inside ``<dataset_path>/data/<split>/``.

    Falls back to other splits if the requested one is empty.
    """
    candidate_splits = [split, "valid", "test", "train"]
    seen = set()
    for s in candidate_splits:
        if s in seen:
            continue
        seen.add(s)
        files = sorted(
            glob.glob(os.path.join(dataset_path, "data", s, "*.h5"))
            + glob.glob(os.path.join(dataset_path, "data", s, "*.hdf5"))
        )
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No Well-format HDF5 found under {dataset_path}/data/<split>/"
    )


def flat_field_names_from_well_file(
    well_file: str, n_spatial_dims: int
) -> List[str]:
    """Reproduce ``flatten_field_names(metadata, include_constants=False)``
    for a Well-format file: rank-0 fields first, then rank-1 expanded with
    spatial-dim suffixes (``_x, _y[, _z]``), then rank-2 with the cartesian
    product (``_xx, _xy, _yx, _yy, ...``).

    Only ``time_varying`` fields are included, matching the dump pipeline.
    The expansion order follows
    ``the_well.data.datasets.WellDataset`` (datasets.py:494-516).
    """
    with h5py.File(well_file, "r") as f:
        spatial_dims_attr = f["dimensions"].attrs.get("spatial_dims")
        if spatial_dims_attr is None:
            spatial_dims_attr = ["x", "y", "z"][:n_spatial_dims]
        spatial_dims = list(spatial_dims_attr)[:n_spatial_dims]

        flat: List[str] = []
        for i in range(3):
            ti = f"t{i}_fields"
            if ti not in f:
                continue
            ti_field_dims = [
                "".join(xyz)
                for xyz in itertools.product(spatial_dims, repeat=i)
            ]
            for field in f[ti].attrs.get("field_names", []):
                if field not in f[ti]:
                    continue
                if not bool(f[ti][field].attrs.get("time_varying", True)):
                    continue
                for dims in ti_field_dims:
                    flat.append(f"{field}_{dims}" if dims else str(field))
    return flat


def group_channels_by_field(field_names: List[str]) -> List[Tuple[str, List[int]]]:
    """Group flat channel names into ``(group_name, [channel_indices])``.

    Strips a trailing ``_x``/``_y``/``_z`` (rank-1 vector) or ``_xx``/``_xy``/...
    (rank-2 tensor) suffix to produce the group name, preserving the order
    in which channels first appear. Channels with no recognised suffix become
    single-component scalar groups.
    """
    suffix_re = re.compile(r"_([xyz]{1,2})$")
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, name in enumerate(field_names):
        m = suffix_re.search(name)
        group = name[: m.start()] if m else name
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(idx)
    return [(g, groups[g]) for g in order]


def trajectory_info_in_split(
    dataset_path: str, split: str
) -> List[Tuple[int, float, float]]:
    """Return per-trajectory ``(n_steps, t_first, t_last)`` for
    ``<dataset_path>/data/<split>/``.

    Walks files in alphabetical order; for each file, reads the leading two
    axes of the first available time-varying field (any of ``t0_fields``,
    ``t1_fields``, ``t2_fields``) — Well-format stores them as
    ``(n_trajectories, n_timesteps, ...spatial...[, components])`` — and
    pulls the trajectory's first/last simulation-time values from
    ``dimensions/time``. Within a single file, every trajectory shares the
    same time axis, so the ``(t_first, t_last)`` pair is reused. The
    timestamps are used downstream to detect (and not double-count)
    boundary overlap between splits, e.g. the 1-frame overlap that
    ``scripts/convert_lbm_to_well.py`` introduces between
    ``train``/``valid``/``test``.
    """
    d = os.path.join(dataset_path, "data", split)
    if not os.path.isdir(d):
        return []
    files = sorted(
        glob.glob(os.path.join(d, "*.h5"))
        + glob.glob(os.path.join(d, "*.hdf5"))
    )
    out: List[Tuple[int, float, float]] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            n_traj = n_steps = None
            for ti in ("t0_fields", "t1_fields", "t2_fields"):
                if ti not in f:
                    continue
                for fn in f[ti].attrs.get("field_names", []):
                    name = str(fn)
                    if name in f[ti]:
                        ds = f[ti][name]
                        if ds.ndim >= 2:
                            n_traj, n_steps = int(ds.shape[0]), int(ds.shape[1])
                            break
                if n_traj is not None:
                    break
            if n_traj is None:
                raise ValueError(
                    f"Cannot determine trajectory count from {fp}: no "
                    "non-empty t<i>_fields with a named field."
                )
            t = f["dimensions/time"][:] if "dimensions/time" in f else None
            if t is None or t.size == 0:
                t_first = float("nan")
                t_last = float("nan")
            else:
                t_first = float(t[0])
                t_last = float(t[-1])
        out.extend([(n_steps, t_first, t_last)] * n_traj)
    return out


def auto_compute_t0s(
    cfg: Any,
    dumps: List["DumpFile"],
    sample_shapes: List[Tuple[int, ...]],
    world_size: int,
    dt_stride: int,
) -> List[int]:
    """Compute one t0 per (file, sample) by treating the Well dataset as the
    concatenation ``train -> valid -> test`` and assigning each dump sample
    to the trajectory the distributed sampler actually fed it.

    Convention: ``t0`` is the global-timeline index of the **last input
    frame** of the rollout (matching the kinet ``forecast_t0-N.h5`` files,
    where ``t_index/pred_<n_steps> = t0 + n_steps``).

    Within a trajectory, the rollout's first prediction is at local index
    ``rollout_start = max(Ni * dt_stride, start_rollout_valid_output_at_t)``
    (per ``the_well.data.datasets`` and Walrus's
    ``MixedWellDataModule``), so the last input frame is at
    ``rollout_start - dt_stride`` and

        t0 = global_offset + rollout_start - dt_stride

    Walrus dumps come from a striped distributed sampler: rank ``R`` running
    with world size ``W`` and per-rank batch size ``B`` sees split-local
    trajectory indices ``(j * W + R) * B + b`` for batch ``j`` and in-batch
    sample ``b`` (verified empirically against rayleigh_taylor). With
    ``W=1`` this reduces to the simple sequential walk.

    Multi-dataset eval configs (one ``well_dataset_info`` per Walrus dataset
    name) are handled per-dataset.
    """
    n_steps_input = int(cfg.data.module_parameters.n_steps_input)
    start_rollout_at = int(
        cfg.data.module_parameters.get("start_rollout_valid_output_at_t", -1)
    )
    if start_rollout_at < 0:
        rollout_start_local = n_steps_input * dt_stride
    else:
        rollout_start_local = max(n_steps_input * dt_stride, start_rollout_at)
    last_input_local = rollout_start_local - dt_stride

    # Per-dataset cache of split-local trajectory offsets.
    cache: Dict[str, Dict[str, Any]] = {}

    def cache_for(dataset_name: str) -> Dict[str, Any]:
        if dataset_name in cache:
            return cache[dataset_name]
        ds_info = well_dataset_info_for(cfg, dataset_name)
        path = ds_info["path"]
        train_info = trajectory_info_in_split(path, "train")
        valid_info = trajectory_info_in_split(path, "valid")
        test_info = trajectory_info_in_split(path, "test")

        # Walk train -> valid -> test, recording each trajectory's offset in a
        # global timeline. If a trajectory's first timestamp matches the
        # previous trajectory's last timestamp, treat the boundary frame as
        # shared (Walrus's kinet conversion script overlaps splits by one
        # frame); only count (n_steps - 1) toward the cumulative.
        cumulative = 0
        prev_t_last: Optional[float] = None
        offsets = {"train": [], "valid": [], "test": []}
        for split, info in (("train", train_info), ("valid", valid_info),
                            ("test", test_info)):
            for n_steps, t_first, t_last in info:
                shares_boundary = (
                    prev_t_last is not None
                    and not (t_first != t_first)  # skip NaN
                    and not (prev_t_last != prev_t_last)
                    and t_first == prev_t_last
                )
                start = cumulative - 1 if shares_boundary else cumulative
                offsets[split].append(start)
                cumulative = start + n_steps
                prev_t_last = t_last
        cache[dataset_name] = {
            "offsets": offsets,
            "counts": {"train": len(train_info), "valid": len(valid_info),
                       "test": len(test_info)},
        }
        return cache[dataset_name]

    t0s: List[int] = []
    for d, shape in zip(dumps, sample_shapes):
        info = cache_for(d.dataset_name)
        split_key = d.split.replace("rollout_", "")
        offsets = info["offsets"][split_key]
        n_traj_in_split = info["counts"][split_key]
        B = int(shape[0])
        for b in range(B):
            traj_idx = (d.batch * world_size + d.rank) * B + b
            if traj_idx >= n_traj_in_split:
                raise IndexError(
                    f"Computed split-local trajectory index {traj_idx} for "
                    f"rank={d.rank} batch={d.batch} b={b} (world_size="
                    f"{world_size}, B={B}) exceeds the {n_traj_in_split} "
                    f"{split_key} trajectories of dataset "
                    f"{d.dataset_name!r}. Check --world-size or pass --t0s."
                )
            t0s.append(offsets[traj_idx] + last_input_local)
    return t0s


def resolve_dt_stride(cfg: Any, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    mp = cfg.data.module_parameters
    dt_min = int(mp.min_dt_stride)
    dt_max = int(mp.max_dt_stride)
    if dt_min != dt_max:
        print(
            f"Warning: data config has min_dt_stride={dt_min} != "
            f"max_dt_stride={dt_max}. Using min_dt_stride. Pass --dt-stride "
            "to override.",
            file=sys.stderr,
        )
    return dt_min


def write_forecast_h5(
    out_path: str,
    pred_sample: np.ndarray,  # (T, H, W[, D], C)
    ref_sample: np.ndarray,
    field_names: List[str],
    field_groups: List[Tuple[str, List[int]]],
    t0: int,
    dt_stride: int,
    dataset_name: str,
    split: str,
    source_file: str,
) -> None:
    T = pred_sample.shape[0]
    spatial_axes = pred_sample.ndim - 2  # T, ...spatial..., C
    # Move components-axis from last to first per kinet convention.
    # Resulting per-step array shape: (C_g, *spatial)
    with h5py.File(out_path, "w") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["split"] = split
        f.attrs["dt_stride"] = dt_stride
        f.attrs["t0"] = t0
        f.attrs["field_names"] = np.array(field_names, dtype=h5py.string_dtype())
        f.attrs["source_file"] = source_file

        for group_name, channel_idxs in field_groups:
            f.create_group(group_name)
        f.create_group("t_index")

        for t in range(T):
            n_steps = (t + 1) * dt_stride
            for group_name, channel_idxs in field_groups:
                # Pred / target slices: pick channels, move components-axis to front.
                pred = pred_sample[t][..., channel_idxs]
                ref = ref_sample[t][..., channel_idxs]
                pred = np.moveaxis(pred, -1, 0).astype(np.float64, copy=False)
                ref = np.moveaxis(ref, -1, 0).astype(np.float64, copy=False)
                f[group_name].create_dataset(f"pred_{n_steps}", data=pred)
                f[group_name].create_dataset(f"target_{n_steps}", data=ref)
            t_index_value = np.array([t0 + n_steps], dtype=np.int64)
            f["t_index"].create_dataset(f"pred_{n_steps}", data=t_index_value)
            f["t_index"].create_dataset(f"target_{n_steps}", data=t_index_value)
        _ = spatial_axes  # silence unused; kept for future shape assertions


def main():
    parser = argparse.ArgumentParser(
        description="Convert Walrus full_trajectory_dumps to kinet-style HDF5."
    )
    parser.add_argument("dump_dir", help="Path to a full_trajectory_dumps/ directory.")
    parser.add_argument(
        "--t0s",
        nargs="+",
        type=int,
        default=None,
        help="Optional manual t0 list, one per (file, sample) in sorted order. "
        "If omitted, t0 is auto-computed by treating the Well dataset as the "
        "concatenation train -> valid -> test and using "
        "t0 = global_offset + n_steps_input for each sample.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Eval experiment name (matches walrus/configs/<name>.yaml). "
        "Inferred from dump_dir if omitted.",
    )
    parser.add_argument(
        "--dt-stride",
        type=int,
        default=None,
        help="Override the dt_stride read from the data config.",
    )
    parser.add_argument(
        "--field-names",
        nargs="+",
        default=None,
        help="Manual override for the channel->field mapping. "
        "Defaults to auto-detection from the Well dataset.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default=None,
        help="Restrict to dumps from one split (valid or test).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of ranks the run that produced the dumps used. "
        "The distributed sampler stripes split-local trajectories as "
        "(batch_idx * world_size + rank) * batch_size + b. Default 1 "
        "(single-rank or local). Set to the number of GPUs the eval used "
        "(e.g. 4 for --gpus-per-node=4 with 1 node).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write forecast_t0-*.h5 files. Default: <dump_dir>/h5/.",
    )
    args = parser.parse_args()

    dumps = find_dump_files(args.dump_dir, split=args.split)
    if not dumps:
        sys.exit(f"No ypred/yref pairs found in {args.dump_dir}")

    # Total samples = sum of B over all files.
    sample_shapes = []
    total_samples = 0
    for d in dumps:
        shape = np.load(d.pred_path, mmap_mode="r").shape
        sample_shapes.append(shape)
        total_samples += shape[0]
    experiment_name = args.experiment_name or infer_experiment_name(args.dump_dir)
    cfg = load_run_config(args.dump_dir, experiment_name)
    dt_stride = resolve_dt_stride(cfg, args.dt_stride)

    if args.t0s is None:
        t0s = auto_compute_t0s(
            cfg, dumps, sample_shapes, args.world_size, dt_stride
        )
        print(
            f"Auto-computed t0s (world_size={args.world_size}) for "
            f"{len(t0s)} samples: {t0s}",
            file=sys.stderr,
        )
    else:
        if len(args.t0s) != total_samples:
            sys.exit(
                f"--t0s has {len(args.t0s)} entries but {total_samples} "
                f"samples were found across {len(dumps)} files."
            )
        t0s = list(args.t0s)

    output_dir = args.output_dir or os.path.join(args.dump_dir, "h5")
    os.makedirs(output_dir, exist_ok=True)

    t0_iter = iter(t0s)
    for d, shape in zip(dumps, sample_shapes):
        # Detect field names per dump (datasets may differ across files).
        n_spatial_dims = len(shape) - 3  # B, T, ...spatial..., C
        if args.field_names is not None:
            field_names = list(args.field_names)
            well_file = "<override>"
        else:
            ds_info = well_dataset_info_for(cfg, d.dataset_name)
            split_for_well = d.split.replace("rollout_", "")
            well_file = find_well_file(ds_info["path"], split_for_well)
            field_names = flat_field_names_from_well_file(well_file, n_spatial_dims)

        c_axis = shape[-1]
        if len(field_names) != c_axis:
            sys.exit(
                f"Channel count mismatch for {d.pred_path}: dump has C={c_axis} "
                f"but {len(field_names)} field names were detected "
                f"({field_names}). Source: {well_file}."
            )

        field_groups = group_channels_by_field(field_names)

        ypred = np.load(d.pred_path)  # (B, T, ...spatial..., C)
        yref = np.load(d.ref_path)

        for b in range(shape[0]):
            t0 = next(t0_iter)
            out_path = os.path.join(output_dir, f"forecast_t0-{t0}.h5")
            write_forecast_h5(
                out_path,
                pred_sample=ypred[b],
                ref_sample=yref[b],
                field_names=field_names,
                field_groups=field_groups,
                t0=t0,
                dt_stride=dt_stride,
                dataset_name=d.dataset_name,
                split=d.split,
                source_file=d.pred_path,
            )
            print(
                f"Wrote {out_path}  "
                f"(T={ypred.shape[1]}, groups={[g for g,_ in field_groups]})"
            )


if __name__ == "__main__":
    main()
