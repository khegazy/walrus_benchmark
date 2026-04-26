# LBM to Well Format Converter

`convert_lbm_to_well.py` converts HDF5 files produced by the kinet lattice Boltzmann solver into the [Well format](https://github.com/PolymathicAI/the_well) used by Walrus for training, finetuning, and evaluation.

## What it does

The script reads a kinet HDF5 file and:

1. **Auto-detects macroscopic fields** (density, pressure, velocity, temperature, etc.) by scanning all datasets in the file.
2. **Skips LBM-internal fields** (distribution functions F/G, diagnostics, solver internals) automatically.
3. **Reorders axes** to match Well conventions:
   - Scalar fields: `(batch, 1, T, X, Y)` &rarr; `(n_traj, T, X, Y)` stored in `t0_fields/`
   - Vector fields: `(batch, C, T, X, Y)` &rarr; `(n_traj, T, X, Y, C)` stored in `t1_fields/`
   - Spatially constant fields (e.g. temperature with shape `(1,1,T,1,1)`) are broadcast to the full spatial grid.
4. **Reads grid metadata** from the `discretization` HDF5 attribute to set spatial coordinates and dimensions.
5. **Copies simulation parameters** from the `system_parameters` HDF5 attribute into Well root attributes and the `scalars/` group.
6. **Writes boundary conditions** in the Well mask format based on user-specified BC types.
7. **Converts data to float32** for storage efficiency.
8. **Runs a structural verification** on the output file automatically.

The output file is saved alongside the input as `<original_name>_well-format.hdf5`.

### Handling unknown fields

If the script encounters an HDF5 dataset key it doesn't recognize, it exits with a message like:

```
ERROR: Unknown dataset key 'magnetic_field' with shape (1, 1, 100, 256, 256).
  Add it to FIELD_MAP (to convert) or SKIP_FIELDS (to ignore)
  in scripts/convert_lbm_to_well.py
```

To fix this, open the script and add the key to one of the two registries at the top:

- **`FIELD_MAP`** &mdash; if the field should be converted. Maps the source key name to the Well field name (which must match an entry in the Walrus field index).
- **`SKIP_FIELDS`** &mdash; if the field should be ignored (e.g. solver diagnostics, intermediate quantities).

## Requirements

Run from the walrus conda environment:

```bash
module load conda && conda activate walrus
```

## Usage

### Basic conversion

```bash
python scripts/convert_lbm_to_well.py --input /path/to/source.h5
```

This assumes periodic boundary conditions on both axes (the default).

### Specifying boundary conditions

Use `--bc-x` and `--bc-y` to set boundary conditions. Valid types are `periodic`, `open`, and `wall`.

```bash
# Periodic on both axes (default)
python scripts/convert_lbm_to_well.py --input /path/to/source.h5

# Open (Neumann) on x, periodic on y
python scripts/convert_lbm_to_well.py --input /path/to/source.h5 \
    --bc-x open --bc-y periodic
```

For different BCs on each side of an axis, use a comma-separated pair (`left,right`):

```bash
# Wall on left x-boundary, open on right x-boundary
python scripts/convert_lbm_to_well.py --input /path/to/source.h5 \
    --bc-x wall,open --bc-y periodic
```

### Cropping odd dimensions

Walrus uses patch-based processing which requires even spatial dimensions. If your grid has an odd dimension (e.g. 3001), use `--crop-to-even` to trim it by one:

```bash
python scripts/convert_lbm_to_well.py --input /path/to/sod.h5 \
    --bc-x open --bc-y periodic --crop-to-even
```

### Custom dataset name

By default, the Well `dataset_name` attribute is set to the parent directory name. Override it with:

```bash
python scripts/convert_lbm_to_well.py --input /path/to/source.h5 \
    --dataset-name my_experiment
```

### Verifying a converted file

To check the structure of an already-converted Well file without re-converting:

```bash
python scripts/convert_lbm_to_well.py --verify /path/to/converted_well-format.hdf5
```

## All options

| Flag | Description | Default |
|---|---|---|
| `--input` | Path to source kinet HDF5 file | (required) |
| `--bc-x` | Boundary condition on x-axis: `periodic`, `open`, `wall`, or `left,right` | `periodic` |
| `--bc-y` | Boundary condition on y-axis: `periodic`, `open`, `wall`, or `low,high` | `periodic` |
| `--crop-to-even` | Crop odd spatial dimensions to even | off |
| `--dataset-name` | Well `dataset_name` attribute | parent dir name |
| `--verify` | Verify a Well file (skips conversion) | &mdash; |

## Currently registered fields

### Converted (`FIELD_MAP`)

| Source key | Well name | Well group |
|---|---|---|
| `density` | `density` | `t0_fields` |
| `pressure` | `pressure` | `t0_fields` |
| `temperature` | `temperature` | `t0_fields` |
| `energy` | `internal_energy` | `t0_fields` |
| `vorticity` | `vorticity` | `t0_fields` |
| `velocity` | `velocity` | `t1_fields` |

### Skipped (`SKIP_FIELDS`)

LBM distributions: `F`, `Feq`, `F_collided`, `G`, `Geq`, `G_collided`

Diagnostics/internals: `stability_scale`, `stabilized_time_scale`, `stability_eps`, `time_scale`, `tau_thermal`

Coordinates: `time`, `time_index`

## Output structure

The converted file follows the Well HDF5 format:

```
root attrs: dataset_name, grid_type, n_spatial_dims, n_trajectories,
            simulation_parameters, [individual param values]

dimensions/
  time (T,)
  x    (X,)
  y    (Y,)

boundary_conditions/
  x_periodic/   or   x_wall_left/ + x_open_right/
    mask (X,) int8

scalars/
  [simulation parameters as scalar datasets]

t0_fields/    (scalar fields)
  density      (n_traj, T, X, Y) float32
  pressure     (n_traj, T, X, Y) float32
  ...

t1_fields/    (vector fields)
  velocity     (n_traj, T, X, Y, 2) float32

t2_fields/    (empty)
```
