#!/usr/bin/env python3
"""Convert kinet LBM HDF5 data to Well format for Walrus training.

Auto-detects which macroscopic fields are present in the source HDF5 and
converts them. Skips known LBM/diagnostic fields. Exits gracefully if an
unknown dataset key is encountered.

Usage:
    python convert_lbm_to_well.py --input /path/to/source.h5 \
        --bc-x periodic --bc-y periodic

    python convert_lbm_to_well.py --input /path/to/source.h5 \
        --bc-x open --bc-y periodic --crop-to-even

    # Different BCs on each boundary (left,right):
    python convert_lbm_to_well.py --input /path/to/source.h5 \
        --bc-x wall,open --bc-y periodic

    # Output is saved alongside the input as <name>_well-format.h5

    # Verify an already-converted file
    python convert_lbm_to_well.py --verify /path/to/converted.h5
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# ---------------------------------------------------------------------------
# Field registries
# ---------------------------------------------------------------------------

# Known macroscopic fields to convert.
# Maps source HDF5 key → Well field name.
# Tensor order and broadcast are auto-detected from shape.
FIELD_MAP = {
    "density": "density",
    "pressure": "pressure",
    "velocity": "velocity",
    "vorticity": "vorticity",
    "temperature": "temperature",
    "energy": "internal_energy",
}

# Known fields to skip (LBM distributions, diagnostics, coordinates).
SKIP_FIELDS = {
    # LBM distribution functions
    "F", "Feq", "F_collided",
    "G", "Geq", "G_collided",
    # Diagnostics / solver internals
    "stability_scale", "stabilized_time_scale", "stability_eps",
    "time_scale", "tau_thermal",
    # Coordinate / index arrays (handled separately)
    "time", "time_index",
}

# BC type string → Well bc_type
BC_TYPE_MAP = {
    "periodic": "PERIODIC",
    "open": "OPEN",
    "wall": "WALL",
}

# Valid BC type names
VALID_BC_TYPES = set(BC_TYPE_MAP.keys())


def classify_field(key: str, shape: tuple) -> dict | None:
    """Classify an HDF5 dataset key. Returns a spec dict or None to skip.

    Raises SystemExit for unknown keys.
    """
    if key in SKIP_FIELDS:
        return None

    if key not in FIELD_MAP:
        print(f"\nERROR: Unknown dataset key '{key}' with shape {shape}.")
        print(f"  Add it to FIELD_MAP (to convert) or SKIP_FIELDS (to ignore)")
        print(f"  in {__file__}")
        sys.exit(1)

    well_name = FIELD_MAP[key]

    # All field arrays have shape (batch, components, time, x, y)
    if len(shape) != 5:
        print(f"\nERROR: Field '{key}' has unexpected rank {len(shape)} (expected 5).")
        sys.exit(1)

    n_components = shape[1]

    if n_components == 1:
        # Scalar field — check if it needs broadcasting (spatial dims are 1)
        needs_broadcast = shape[3] == 1 and shape[4] == 1
        return {
            "src_key": key,
            "well_name": well_name,
            "tensor_order": 0,
            "broadcast": needs_broadcast,
        }
    else:
        # Vector field (components > 1)
        return {
            "src_key": key,
            "well_name": well_name,
            "tensor_order": 1,
            "n_components": n_components,
        }


def extract_system_parameters(src: h5py.File) -> dict:
    """Extract simulation parameters from source HDF5 attributes."""
    params = {}
    if "system_parameters" in src.attrs:
        sp = json.loads(src.attrs["system_parameters"])
        params.update(sp)
    return params


def convert_scalar_field(src: h5py.File, spec: dict, spatial_shape: tuple,
                         temporal_stride: int = 1) -> np.ndarray:
    """Convert a scalar field from (1,1,T,X,Y) to (1,T,X,Y) as float32."""
    data = src[spec["src_key"]][:]  # (1, 1, T, X, Y) or (1, 1, T, 1, 1)
    # Squeeze the component dimension: (1,1,T,...) → (1,T,...)
    data = data[:, 0, ::temporal_stride, :, :]  # (1, T, X, Y) or (1, T, 1, 1)

    if spec.get("broadcast", False):
        n_traj, n_time = data.shape[0], data.shape[1]
        data = np.broadcast_to(data, (n_traj, n_time, *spatial_shape)).copy()

    return data.astype(np.float32)


def convert_vector_field(src: h5py.File, spec: dict,
                         temporal_stride: int = 1) -> np.ndarray:
    """Convert a vector field from (1,C,T,X,Y) to (1,T,X,Y,C) as float32."""
    data = src[spec["src_key"]][:]  # (1, C, T, X, Y)
    data = data[:, :, ::temporal_stride, :, :]  # apply temporal stride
    data = np.transpose(data, (0, 2, 3, 4, 1))  # (1, T, X, Y, C)
    return data.astype(np.float32)


def parse_bc_spec(spec: str) -> tuple[str, str]:
    """Parse a BC spec string into (left/low, right/high) BC types.

    Accepts:
        "periodic"    → ("periodic", "periodic")
        "open"        → ("open", "open")
        "wall,open"   → ("wall", "open")
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0])
    elif len(parts) == 2:
        return (parts[0], parts[1])
    else:
        print(f"ERROR: Invalid BC spec '{spec}'. Use 'type' or 'left_type,right_type'.")
        sys.exit(1)


def _write_bc_group(bcs_group: h5py.Group, name: str, well_bc_type: str,
                    dim_name: str, mask: np.ndarray):
    """Write a single boundary condition group to HDF5."""
    bc_grp = bcs_group.create_group(name)
    bc_grp.create_dataset("mask", data=mask)
    bc_grp.attrs["bc_type"] = well_bc_type
    bc_grp.attrs["associated_dims"] = [dim_name]
    bc_grp.attrs["associated_fields"] = []
    bc_grp.attrs["sample_varying"] = False
    bc_grp.attrs["time_varying"] = False


def write_bcs(bcs_group: h5py.Group, dim_name: str, bc_spec: str, dim_size: int):
    """Write boundary condition groups for one axis.

    If both boundaries have the same type, writes one group with both
    endpoints marked (or all-zeros for periodic). If they differ, writes
    separate groups for each boundary.
    """
    bc_left, bc_right = parse_bc_spec(bc_spec)

    if bc_left == bc_right:
        # Same BC on both boundaries
        well_type = BC_TYPE_MAP[bc_left]
        mask = np.zeros(dim_size, dtype=np.int8)
        if bc_left != "periodic":
            mask[0] = 1
            mask[-1] = 1
        _write_bc_group(bcs_group, f"{dim_name}_{bc_left}", well_type, dim_name, mask)
    else:
        # Different BCs on each boundary — one group per boundary
        for bc_type, idx in [(bc_left, 0), (bc_right, -1)]:
            well_type = BC_TYPE_MAP[bc_type]
            mask = np.zeros(dim_size, dtype=np.int8)
            mask[idx] = 1
            suffix = "left" if idx == 0 else "right"
            _write_bc_group(bcs_group, f"{dim_name}_{bc_type}_{suffix}",
                            well_type, dim_name, mask)



def write_well_hdf5(
    out_path: str,
    dataset_name: str,
    bc_x: str,
    bc_y: str,
    t0_data: dict[str, np.ndarray],
    t1_data: dict[str, np.ndarray],
    time_coords: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    sim_params: dict,
):
    """Write a Well-format HDF5 file."""
    n_spatial_dims = 2
    first_field = next(iter(t0_data.values())) if t0_data else next(iter(t1_data.values()))
    n_traj = first_field.shape[0]

    with h5py.File(out_path, "w") as f:
        # ── Root attributes ──
        f.attrs["dataset_name"] = dataset_name
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = n_spatial_dims
        f.attrs["n_trajectories"] = n_traj
        param_names = list(sim_params.keys())
        f.attrs["simulation_parameters"] = param_names
        for k, v in sim_params.items():
            f.attrs[k] = v

        # ── Dimensions ──
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        t_ds = dims.create_dataset("time", data=time_coords)
        t_ds.attrs["time_varying"] = True
        t_ds.attrs["sample_varying"] = False
        x_ds = dims.create_dataset("x", data=x_coords)
        x_ds.attrs["time_varying"] = False
        x_ds.attrs["sample_varying"] = False
        y_ds = dims.create_dataset("y", data=y_coords)
        y_ds.attrs["time_varying"] = False
        y_ds.attrs["sample_varying"] = False

        # ── Boundary conditions ──
        bcs = f.create_group("boundary_conditions")
        write_bcs(bcs, "x", bc_x, len(x_coords))
        write_bcs(bcs, "y", bc_y, len(y_coords))

        # ── Scalars ──
        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = param_names
        for k, v in sim_params.items():
            ds = scalars.create_dataset(k, data=v)
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"] = False

        # ── t0_fields (scalar fields) ──
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = list(t0_data.keys())
        for name, arr in t0_data.items():
            ds = t0.create_dataset(name, data=arr, dtype=np.float32)
            ds.attrs["dim_varying"] = np.array([True, True])
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True

        # ── t1_fields (vector fields) ──
        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = list(t1_data.keys())
        for name, arr in t1_data.items():
            ds = t1.create_dataset(name, data=arr, dtype=np.float32)
            ds.attrs["dim_varying"] = np.array([True, True])
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True

        # ── t2_fields (empty) ──
        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []

    print(f"  Written: {out_path}")


# Valid spatial dimensions for the Walrus model's deterministic kernel selection.
# Derived from patch_dict keys {0,1,4,8,12,16,24,32} * per_axis_tokens (32).
WALRUS_VALID_DIMS = sorted([32 * k for k in [1, 4, 8, 12, 16, 24, 32]])
# = [32, 128, 256, 384, 512, 768, 1024]


def _adjust_dim_to_valid(size: int, bc_type: str) -> tuple[int, str]:
    """Adjust a spatial dimension to a valid Walrus model dimension.

    Returns (target_size, method) where method is 'crop', 'tile', or 'none'.
    - If size is already valid: no change.
    - If size > max valid: crop to max valid (1024).
    - If size < min valid (32) and BC is periodic: tile up to 32.
    - If size < min valid (32) and BC is not periodic: pad up to 32.
    - Otherwise: crop down to nearest valid dim <= size.
    """
    if size in WALRUS_VALID_DIMS:
        return size, "none"

    # Find the largest valid dim <= size
    valid_below = [d for d in WALRUS_VALID_DIMS if d <= size]
    if valid_below:
        return max(valid_below), "crop"

    # size < min valid dim (32) — need to extend
    target = WALRUS_VALID_DIMS[0]  # 32
    bc_left, _ = parse_bc_spec(bc_type)
    if bc_left == "periodic":
        return target, "tile"
    return target, "pad"


def _apply_spatial_adjustment_t0(arr: np.ndarray, target_x: int, target_y: int,
                                  method_x: str, method_y: str) -> np.ndarray:
    """Adjust spatial dims of a t0 (scalar) array: (n_traj, T, X, Y)."""
    if method_x == "crop":
        offset_x = (arr.shape[2] - target_x) // 2
        arr = arr[:, :, offset_x:offset_x + target_x, :]
    elif method_x == "tile":
        reps = (target_x + arr.shape[2] - 1) // arr.shape[2]
        arr = np.tile(arr, (1, 1, reps, 1))[:, :, :target_x, :]
    elif method_x == "pad":
        pad_x = target_x - arr.shape[2]
        arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_x), (0, 0)), mode="constant")

    if method_y == "crop":
        offset_y = (arr.shape[3] - target_y) // 2
        arr = arr[:, :, :, offset_y:offset_y + target_y]
    elif method_y == "tile":
        reps = (target_y + arr.shape[3] - 1) // arr.shape[3]
        arr = np.tile(arr, (1, 1, 1, reps))[:, :, :, :target_y]
    elif method_y == "pad":
        pad_y = target_y - arr.shape[3]
        arr = np.pad(arr, ((0, 0), (0, 0), (0, 0), (0, pad_y)), mode="constant")
    return arr


def _apply_spatial_adjustment_t1(arr: np.ndarray, target_x: int, target_y: int,
                                  method_x: str, method_y: str) -> np.ndarray:
    """Adjust spatial dims of a t1 (vector) array: (n_traj, T, X, Y, C)."""
    if method_x == "crop":
        offset_x = (arr.shape[2] - target_x) // 2
        arr = arr[:, :, offset_x:offset_x + target_x, :, :]
    elif method_x == "tile":
        reps = (target_x + arr.shape[2] - 1) // arr.shape[2]
        arr = np.tile(arr, (1, 1, reps, 1, 1))[:, :, :target_x, :, :]
    elif method_x == "pad":
        pad_x = target_x - arr.shape[2]
        arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_x), (0, 0), (0, 0)), mode="constant")

    if method_y == "crop":
        offset_y = (arr.shape[3] - target_y) // 2
        arr = arr[:, :, :, offset_y:offset_y + target_y, :]
    elif method_y == "tile":
        reps = (target_y + arr.shape[3] - 1) // arr.shape[3]
        arr = np.tile(arr, (1, 1, 1, reps, 1))[:, :, :, :target_y, :]
    elif method_y == "pad":
        pad_y = target_y - arr.shape[3]
        arr = np.pad(arr, ((0, 0), (0, 0), (0, 0), (0, pad_y), (0, 0)), mode="constant")
    return arr


def convert_file(input_path: str, bc_x: str, bc_y: str,
                 dataset_name: str, crop_to_even: bool,
                 pad_to_multiple: int = 0, temporal_stride: int = 1):
    """Convert a single kinet LBM HDF5 file to Well format."""
    print(f"Converting: {input_path}")

    with h5py.File(input_path, "r") as src:
        # Parse discretization attribute for grid sizes and physical coordinates
        disc = json.loads(src.attrs["discretization"])
        spatial_grid = disc["spatial"]["grid"]   # [X, Y]
        spatial_res = disc["spatial"]["resolution"]

        spatial_x = spatial_grid[0]
        spatial_y = spatial_grid[1]

        # Read time coordinates from the dataset
        time_coords = src["time"][::temporal_stride]
        if temporal_stride > 1:
            print(f"  Temporal stride {temporal_stride}: {len(src['time'][:])} -> {len(time_coords)} timesteps")

        # Crop spatial dims to even numbers for patch divisibility
        crop_x = None
        crop_y = None
        if crop_to_even:
            if spatial_x % 2 != 0:
                crop_x = spatial_x - 1
                print(f"  Cropping x from {spatial_x} to {crop_x} for even divisibility")
                spatial_x = crop_x
            if spatial_y % 2 != 0:
                crop_y = spatial_y - 1
                print(f"  Cropping y from {spatial_y} to {crop_y} for even divisibility")
                spatial_y = crop_y

        # Determine spatial adjustments for model compatibility
        method_x, method_y = "none", "none"
        target_x, target_y = spatial_x, spatial_y
        if pad_to_multiple > 0:
            target_x, method_x = _adjust_dim_to_valid(spatial_x, bc_x)
            target_y, method_y = _adjust_dim_to_valid(spatial_y, bc_y)
            if method_x != "none":
                print(f"  Adjusting x from {spatial_x} to {target_x} ({method_x}) for model compatibility")
            if method_y != "none":
                print(f"  Adjusting y from {spatial_y} to {target_y} ({method_y}) for model compatibility")

        spatial_shape = (spatial_x, spatial_y)

        # Generate physical coordinate arrays from discretization
        x_coords = np.arange(target_x, dtype=np.float64) * spatial_res
        y_coords = np.arange(target_y, dtype=np.float64) * spatial_res

        # Classify all datasets in the HDF5
        field_specs = []
        for key in src.keys():
            if not isinstance(src[key], h5py.Dataset):
                continue
            spec = classify_field(key, src[key].shape)
            if spec is not None:
                field_specs.append(spec)

        print(f"  Converting fields: {[s['well_name'] for s in field_specs]}")

        # Convert fields
        t0_data = {}
        t1_data = {}
        for spec in field_specs:
            if spec["tensor_order"] == 0:
                arr = convert_scalar_field(src, spec, spatial_shape, temporal_stride)
                if crop_x is not None:
                    off = (arr.shape[2] - crop_x) // 2
                    arr = arr[:, :, off:off + crop_x, :]
                if crop_y is not None:
                    off = (arr.shape[3] - crop_y) // 2
                    arr = arr[:, :, :, off:off + crop_y]
                arr = _apply_spatial_adjustment_t0(arr, target_x, target_y, method_x, method_y)
                t0_data[spec["well_name"]] = arr
            else:
                arr = convert_vector_field(src, spec, temporal_stride)
                if crop_x is not None:
                    off = (arr.shape[2] - crop_x) // 2
                    arr = arr[:, :, off:off + crop_x, :, :]
                if crop_y is not None:
                    off = (arr.shape[3] - crop_y) // 2
                    arr = arr[:, :, :, off:off + crop_y, :]
                arr = _apply_spatial_adjustment_t1(arr, target_x, target_y, method_x, method_y)
                t1_data[spec["well_name"]] = arr

        # Extract simulation parameters
        sim_params = extract_system_parameters(src)

    # Output in same directory as input, with _well-format appended
    input_dir = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    stride_suffix = f"_t-stride-{temporal_stride}" if temporal_stride > 1 else ""
    out_path = os.path.join(input_dir, f"{base}{stride_suffix}_well-format.h5")

    write_well_hdf5(
        out_path=out_path,
        dataset_name=dataset_name,
        bc_x=bc_x,
        bc_y=bc_y,
        t0_data=t0_data,
        t1_data=t1_data,
        time_coords=time_coords,
        x_coords=x_coords,
        y_coords=y_coords,
        sim_params=sim_params,
    )

    return out_path


def split_to_well_dirs(
    flat_h5_path: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    """Split a flat Well-format HDF5 into train/valid/test directories.

    Creates the Well directory layout expected by WellDataset:
        {basename}_well-format/data/{train,valid,test}/file_00.h5

    Splits are temporal: contiguous blocks of timesteps. Adjacent splits
    overlap by 1 timestep so that input/output windows at boundaries are
    complete.

    Parameters
    ----------
    flat_h5_path : str
        Path to the flat Well-format HDF5 file (output of convert_file).
    ratios : tuple of 3 floats
        Train/valid/test fractions (must sum to 1).
    """
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, (
        f"Split ratios must sum to 1, got {ratios} (sum={sum(ratios)})"
    )

    # Build output directory next to the flat file
    base = os.path.splitext(flat_h5_path)[0]  # drop .h5
    splits = ["train", "valid", "test"]

    with h5py.File(flat_h5_path, "r") as src:
        time_coords = src["dimensions/time"][:]
        n_time = len(time_coords)

        # Compute split boundaries
        n_train = max(1, int(round(n_time * ratios[0])))
        n_valid = max(1, int(round(n_time * ratios[1])))
        # test gets the remainder
        n_test = n_time - n_train - n_valid + 2  # +2 for overlaps
        assert n_test >= 1, f"Not enough timesteps ({n_time}) for the given split ratios"

        # Ranges with 1-timestep overlap at boundaries
        ranges = {
            "train": (0, n_train),
            "valid": (n_train - 1, n_train - 1 + n_valid + 1),
            "test":  (n_train + n_valid - 1, n_time),
        }

        for split in splits:
            t_start, t_end = ranges[split]
            t_end = min(t_end, n_time)  # clamp
            split_dir = os.path.join(base, "data", split)
            os.makedirs(split_dir, exist_ok=True)
            out_path = os.path.join(split_dir, "file_00.h5")

            with h5py.File(out_path, "w") as dst:
                # Copy root attributes
                for attr_name, attr_val in src.attrs.items():
                    dst.attrs[attr_name] = attr_val

                # Dimensions — slice time, copy spatial as-is
                dims_dst = dst.create_group("dimensions")
                for attr_name, attr_val in src["dimensions"].attrs.items():
                    dims_dst.attrs[attr_name] = attr_val
                t_ds = dims_dst.create_dataset("time", data=time_coords[t_start:t_end])
                for attr_name, attr_val in src["dimensions/time"].attrs.items():
                    t_ds.attrs[attr_name] = attr_val
                for dim_name in src["dimensions"]:
                    if dim_name == "time":
                        continue
                    dim_ds = dims_dst.create_dataset(dim_name, data=src[f"dimensions/{dim_name}"][:])
                    for attr_name, attr_val in src[f"dimensions/{dim_name}"].attrs.items():
                        dim_ds.attrs[attr_name] = attr_val

                # Boundary conditions — copy as-is
                src.copy("boundary_conditions", dst)

                # Scalars — copy as-is
                src.copy("scalars", dst)

                # t0_fields — slice time dimension (axis 1)
                t0_dst = dst.create_group("t0_fields")
                for attr_name, attr_val in src["t0_fields"].attrs.items():
                    t0_dst.attrs[attr_name] = attr_val
                for field_name in src["t0_fields"].attrs.get("field_names", []):
                    if field_name in src["t0_fields"]:
                        data = src[f"t0_fields/{field_name}"][:, t_start:t_end]
                        ds = t0_dst.create_dataset(field_name, data=data)
                        for attr_name, attr_val in src[f"t0_fields/{field_name}"].attrs.items():
                            ds.attrs[attr_name] = attr_val

                # t1_fields — slice time dimension (axis 1)
                t1_dst = dst.create_group("t1_fields")
                for attr_name, attr_val in src["t1_fields"].attrs.items():
                    t1_dst.attrs[attr_name] = attr_val
                for field_name in src["t1_fields"].attrs.get("field_names", []):
                    if field_name in src["t1_fields"]:
                        data = src[f"t1_fields/{field_name}"][:, t_start:t_end]
                        ds = t1_dst.create_dataset(field_name, data=data)
                        for attr_name, attr_val in src[f"t1_fields/{field_name}"].attrs.items():
                            ds.attrs[attr_name] = attr_val

                # t2_fields — copy as-is (usually empty)
                src.copy("t2_fields", dst)

            print(f"  {split}: {out_path} (timesteps {t_start}–{t_end-1}, n={t_end - t_start})")

    print(f"\n  Well directory created: {base}/")
    return base


def verify_well_file(path: str):
    """Verify the structure of a Well-format HDF5 file."""
    print(f"Verifying: {path}")
    errors = []

    with h5py.File(path, "r") as f:
        # Check root attributes
        for attr in ["dataset_name", "grid_type", "n_spatial_dims", "n_trajectories"]:
            if attr not in f.attrs:
                errors.append(f"Missing root attribute: {attr}")

        # Check required groups
        for grp in ["dimensions", "boundary_conditions", "scalars", "t0_fields", "t1_fields", "t2_fields"]:
            if grp not in f:
                errors.append(f"Missing group: {grp}")

        # Check dimensions
        if "dimensions" in f:
            dims = f["dimensions"]
            for dim in ["time", "x", "y"]:
                if dim not in dims:
                    errors.append(f"Missing dimension: {dim}")

        # Check field shapes
        if "dimensions" in f and "t0_fields" in f:
            n_traj = f.attrs.get("n_trajectories", 0)
            n_time = len(f["dimensions/time"])
            n_x = len(f["dimensions/x"])
            n_y = len(f["dimensions/y"])
            expected_t0 = (n_traj, n_time, n_x, n_y)

            for name in f["t0_fields"].attrs.get("field_names", []):
                if name in f["t0_fields"]:
                    shape = f["t0_fields"][name].shape
                    if shape != expected_t0:
                        errors.append(f"t0_fields/{name}: expected {expected_t0}, got {shape}")

            if "t1_fields" in f:
                for name in f["t1_fields"].attrs.get("field_names", []):
                    if name in f["t1_fields"]:
                        shape = f["t1_fields"][name].shape
                        # Vector fields: last dim is n_components (variable)
                        expected_prefix = (n_traj, n_time, n_x, n_y)
                        if shape[:4] != expected_prefix:
                            errors.append(f"t1_fields/{name}: expected {expected_prefix}+(C,), got {shape}")

        # Check field attributes
        for grp_name in ["t0_fields", "t1_fields"]:
            if grp_name in f:
                for name in f[grp_name].attrs.get("field_names", []):
                    if name in f[grp_name]:
                        ds = f[grp_name][name]
                        for attr in ["dim_varying", "sample_varying", "time_varying"]:
                            if attr not in ds.attrs:
                                errors.append(f"{grp_name}/{name} missing attr: {attr}")

        # Print tree
        print(f"\n  Root attrs: {dict(f.attrs)}")
        print(f"  Groups: {list(f.keys())}")
        if "dimensions" in f:
            for k in f["dimensions"]:
                print(f"  dimensions/{k}: {f['dimensions'][k].shape}")
        if "boundary_conditions" in f:
            for k in f["boundary_conditions"]:
                bc = f["boundary_conditions"][k]
                print(f"  boundary_conditions/{k}: type={bc.attrs.get('bc_type')}, mask={bc['mask'].shape}")
        for grp_name in ["t0_fields", "t1_fields"]:
            if grp_name in f:
                names = f[grp_name].attrs.get("field_names", [])
                print(f"  {grp_name}: {list(names)}")
                for name in names:
                    if name in f[grp_name]:
                        print(f"    {name}: {f[grp_name][name].shape} {f[grp_name][name].dtype}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("\n  OK — all checks passed")
        return True


def main():
    parser = argparse.ArgumentParser(description="Convert kinet LBM data to Well format")
    parser.add_argument("--input", type=str, help="Path to source HDF5 file")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Well dataset_name attribute (default: parent directory name)")
    parser.add_argument("--bc-x", type=str, default="periodic",
                        help="Boundary condition on x-axis: 'periodic', 'open', 'wall', "
                             "or 'left,right' e.g. 'wall,open' (default: periodic)")
    parser.add_argument("--bc-y", type=str, default="periodic",
                        help="Boundary condition on y-axis: 'periodic', 'open', 'wall', "
                             "or 'low,high' e.g. 'wall,open' (default: periodic)")
    parser.add_argument("--crop-to-even", action="store_true",
                        help="Crop odd spatial dimensions to even for patch divisibility")
    parser.add_argument("--pad-to-multiple", type=int, default=0,
                        help="Adjust spatial dims to be multiples of N for model compatibility. "
                             "Dims >= N are cropped down; dims < N are tiled (periodic BC) "
                             "or zero-padded (non-periodic). Walrus requires N=32.")
    parser.add_argument("--split", action="store_true",
                        help="Split output into data/{train,valid,test}/ directories "
                             "with temporal splits (default ratios 0.8/0.1/0.1)")
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1",
                        help="Comma-separated train,valid,test ratios (default: 0.8,0.1,0.1)")
    parser.add_argument("--temporal-stride", type=int, default=1,
                        help="Take every Nth timestep from the source data (default: 1, no striding)")
    parser.add_argument("--verify", type=str, default=None,
                        help="Path to a Well-format HDF5 to verify (skips conversion)")
    args = parser.parse_args()

    if args.verify:
        ok = verify_well_file(args.verify)
        sys.exit(0 if ok else 1)

    if not args.input:
        parser.error("--input is required for conversion")

    # Validate BC specs
    for flag, spec in [("--bc-x", args.bc_x), ("--bc-y", args.bc_y)]:
        for bc in parse_bc_spec(spec):
            if bc not in VALID_BC_TYPES:
                parser.error(f"Invalid BC type '{bc}' in {flag}. "
                             f"Valid types: {', '.join(sorted(VALID_BC_TYPES))}")

    # Default dataset name from parent directory
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.dirname(os.path.abspath(args.input)))

    out_path = convert_file(args.input, args.bc_x, args.bc_y,
                            dataset_name, args.crop_to_even,
                            pad_to_multiple=args.pad_to_multiple,
                            temporal_stride=args.temporal_stride)

    # Auto-verify the flat file
    print()
    verify_well_file(out_path)

    # Optionally split into train/valid/test directories
    if args.split:
        ratios = tuple(float(r) for r in args.split_ratios.split(","))
        if len(ratios) != 3:
            parser.error("--split-ratios must have exactly 3 values (train,valid,test)")
        print(f"\nSplitting into train/valid/test with ratios {ratios}...")
        well_dir = split_to_well_dirs(out_path, ratios)

        # Verify each split file
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(well_dir, "data", split, "file_00.h5")
            print()
            verify_well_file(split_path)


if __name__ == "__main__":
    main()
