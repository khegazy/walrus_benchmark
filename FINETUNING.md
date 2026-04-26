# Fine-Tuning Walrus on Custom Data (Step-by-Step)

This guide walks through the entire process of taking raw kinet LBM simulation data, converting it to the Well format, writing the Hydra configs, and running fine-tuning on NERSC. It documents every file you need to create/edit and every gotcha encountered along the way.

---

## Prerequisites

### Environment

```bash
module load conda
conda activate walrus
```

If you haven't set up the conda environment yet, install from the repo root:

```bash
pip install .                 # or: pip install ".[test]" for linting/testing too
```

### Pretrained checkpoint

The pretrained Walrus checkpoint must exist at:

```
walrus/checkpoints/walrus.pt
```

This file is loaded by the default checkpoint config. If you don't have it, ask the Polymathic team or check their distribution instructions.

### Source data

Your raw kinet HDF5 files live somewhere like:

```
/global/cfs/cdirs/m4790/Data/kinet/<physics>/<subdir>/<file>.h5
```

Each file has shape convention `(batch, components, time, x, y)` in float64, with fields like `density`, `velocity`, `temperature`, `energy`, `vorticity`, and LBM distribution functions (`F`, `Feq`, `G`, `Geq`) which are excluded from conversion.

---

## Step 0: Zero-Shot Evaluation (Before Finetuning)

Before finetuning, evaluate the pretrained Walrus model on your data to establish a baseline. This runs the model in `validation_mode` — no training occurs, just forward passes on validation and test data with full metrics and optional prediction dumps.

### What you need

1. **Well-format data** with properly split `data/{train,valid,test}/` files (see Step 2 for conversion, or "Splitting an existing file" below)
2. **A data config** for evaluation
3. **A top-level evaluation config** with `validation_mode: True`
4. **GPU compute** — the full model is too large for CPU. Use FSDP across 4 GPUs for reasonable speed.

### Important: use properly split data

Do NOT symlink a single unsplit file into all three split directories. In `validation_mode`, Walrus runs full validation on the entire valid and test splits (the `max_samples` parameter only limits training, not validation). With 10000 timesteps per split, validation takes ~33 minutes and exceeds the 30-minute debug queue limit.

Instead, create proper temporal splits so each split is a manageable size. The conversion script's `--split` flag does this automatically. If you already have an unsplit file, see "Splitting an existing Well-format file" below.

### Splitting an existing Well-format file

If your data was converted without `--split`, create temporal splits manually. This is a Python script that reads the source file and creates three split files with non-overlapping time ranges:

```python
import h5py
import numpy as np

src = "/path/to/<basename>_well-format.h5"
well_dir = "/path/to/<basename>_well-format"

# Define splits (adjust ratios as needed)
splits = {"train": (0, 8000), "valid": (8000, 9000), "test": (9000, None)}

with h5py.File(src, "r") as f:
    total_t = f["dimensions/time"].shape[0]
    full_time = f["dimensions/time"][:]

    for split_name, (t_start, t_end) in splits.items():
        if t_end is None:
            t_end = total_t
        out_path = f"{well_dir}/data/{split_name}/file_00.h5"

        with h5py.File(out_path, "w") as out:
            # Copy top-level attributes
            for attr_name, attr_val in f.attrs.items():
                out.attrs[attr_name] = attr_val

            # Copy and slice dimensions (CRITICAL: must slice time!)
            dim_grp = out.create_group("dimensions")
            for k in f["dimensions"]:
                if k == "time":
                    dim_grp.create_dataset("time", data=full_time[t_start:t_end])
                else:
                    f["dimensions"].copy(f["dimensions"][k], dim_grp, k)
            for attr_name, attr_val in f["dimensions"].attrs.items():
                dim_grp.attrs[attr_name] = attr_val

            # Copy non-time-varying groups verbatim
            for grp_name in ["boundary_conditions", "scalars"]:
                if grp_name in f:
                    f.copy(grp_name, out)

            # Copy and slice time-varying field groups
            for tN in ["t0_fields", "t1_fields", "t2_fields"]:
                if tN in f:
                    grp = out.create_group(tN)
                    # Copy group-level attrs (CRITICAL: includes field_names)
                    for attr_name, attr_val in f[tN].attrs.items():
                        grp.attrs[attr_name] = attr_val
                    # Slice each field along the time axis
                    for field_name in f[tN]:
                        ds = f[tN][field_name]
                        data = ds[:, t_start:t_end]  # shape: (samples, time, ...)
                        new_ds = grp.create_dataset(field_name, data=data)
                        for attr_name, attr_val in ds.attrs.items():
                            new_ds.attrs[attr_name] = attr_val
```

**Three critical gotchas when splitting:**
1. **`dimensions/time` must be sliced** to match the field data. The Well library reads `_f["dimensions"]["time"].shape[-1]` to compute the number of valid samples. If this doesn't match the actual field data length, it creates phantom samples that crash with `KeyError: 'output_fields'`.
2. **Group-level `field_names` attr** must be copied for `t0_fields`, `t1_fields`, `t2_fields`. Using `h5py.create_group()` doesn't copy attrs from the source. Missing this causes `KeyError: "can't locate attribute: 'field_names'"`.
3. **Don't forget `t1_fields`** which contains velocity (a vector field stored separately from scalar fields in `t0_fields`).

### Create an evaluation data config

Create `walrus/configs/data/<name>_eval.yaml`:
- `batch_size: 1` (safer for memory)
- `max_rollout_steps: 50` (autoregressive evaluation length; must be less than the number of timesteps in your smallest split minus `n_steps_input`)
- **`full_well_field_index`** (must match the 67-field pretrained checkpoint)
- **Absolute path** to the well-format directory

Example (`walrus/configs/data/lbm_doubly_periodic_eval.yaml`):

```yaml
defaults:
- field_index_map_override: full_well_field_index
well_base_path: null
wandb_data_name: lbm_doubly_periodic_eval
module_parameters:
  _target_: walrus.data.MixedWellDataModule
  batch_size: 1
  n_steps_input: 10
  n_steps_output: 1
  min_dt_stride: 1
  max_dt_stride: 1
  max_samples: 100
  max_rollout_steps: 50
  well_dataset_info:
    lbm_doubly_periodic:
      include_filters: []
      exclude_filters: []
      path: /absolute/path/to/<basename>_well-format
```

Note: `max_samples` only limits training data, not validation. Validation always iterates through the full split. Control validation time by controlling the size of your valid/test splits.

### Create a top-level evaluation config

Create `walrus/configs/eval_<name>.yaml`. Key differences from a finetuning config:
- `validation_mode: True` — runs only the validation/test loops, no training
- `finetune: False` — don't set up finetuning paths
- `experiment: defaults` — no finetuning experiment setup needed
- `distribution: fsdp` — use multi-GPU for speed (the full model is slow on 1 GPU)
- `finetuning_mods: defaults` — no learnable RoPE needed for eval
- `folder_override:` set to a specific output directory
- `dump_prediction_to_disk: True` — saves raw predictions as `.npy` files
- `image_validation: True` and `video_validation: True` — saves visual outputs

Example (`walrus/configs/eval_doubly_periodic.yaml`):

```yaml
defaults:
  - trainer: globalnorm
  - optimizer: adam
  - lr_scheduler: inv_sqrt_w_sqrt_ramps_longer
  - model: extended_isotropic
  - data: lbm_doubly_periodic_eval
  - experiment: defaults
  - server: local
  - distribution: fsdp
  - logger: none
  - checkpoint: defaults
  - finetuning_mods: defaults
  - _self_

data_workers: 1
name: eval_doubly_periodic
finetune: False
validation_mode: True
automatic_setup: True
experiment_dir: /absolute/path/to/experiments

folder_override: /absolute/path/to/experiments/eval_doubly_periodic
checkpoint_override: ""

model:
  drop_path: 0.0
  input_field_drop: 0.0

trainer:
  max_epoch: 1
  val_frequency: 1
  rollout_val_frequency: 1
  short_validation_length: 200
  max_rollout_steps: 20
  skip_spectral_metrics: True
  enable_amp: False
  image_validation: True
  video_validation: True
  dump_prediction_to_disk: True
  num_detailed_logs: 5
  revin:
    _target_: walrus.trainer.normalization_strat.SamplewiseRevNormalization
    _partial_: True
```

Notes on `max_epoch: 1`: In validation mode, `validate()` always forces `full=True` regardless of `max_epoch` or `short_validation_length`. These settings don't actually limit validation length. The only way to control validation time is through the data split size.

### Run the evaluation (SLURM batch script)

The full Walrus model needs multiple GPUs. Create a SLURM script at `walrus/run_scripts/eval_<name>.sh`:

```bash
#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH -C gpu&hbm80g
#SBATCH -A <your_account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=6
#SBATCH -J eval_<name>
#SBATCH --output=eval_<name>_%j.log
#SBATCH -q debug

export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

module load conda
conda activate walrus

cd /path/to/walrus/walrus

srun python -u $(which torchrun) \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29500 \
    train.py --config-name=eval_<name>
```

Submit: `sbatch walrus/run_scripts/eval_<name>.sh`

With ~1000 timesteps per split and 4 GPUs, evaluation takes ~10 minutes (well within the 30-minute debug queue).

### Where to find results

After evaluation completes, results are in:

```
walrus/experiments/eval_<name>/
├── extended_config.yaml                    # frozen resolved config
├── checkpoints/                            # (empty for eval-only)
├── viz/
│   ├── <dataset_name>/                     # named after the HDF5 dataset_name attr
│   │   ├── full_trajectory_dumps/          # raw .npy prediction/reference arrays
│   │   │   ├── yref_*_rollout_valid_*.npy  # ground truth
│   │   │   └── ypred_*_rollout_valid_*.npy # model predictions
│   │   ├── rollout_losses/                 # per-timestep loss curves
│   │   ├── rollout_video/                  # .mp4 prediction vs reference videos
│   │   ├── density Histogram/              # per-field histogram comparisons
│   │   ├── pressure Histogram/
│   │   ├── temperature Histogram/
│   │   ├── velocity_x Histogram/
│   │   ├── velocity_y Histogram/
│   │   └── vorticity Histogram/
│   └── loss_dicts/
│       ├── valid_loss_dict_*.pkl           # one-step validation metrics
│       ├── test_loss_dict_*.pkl            # one-step test metrics
│       ├── rollout_valid_loss_dict_*.pkl   # rollout validation metrics
│       ├── rollout_test_loss_dict_*.pkl    # rollout test metrics
│       └── *_time_logs_*.pkl              # per-timestep metric logs
```

Note: the subdirectory is named after the `dataset_name` attribute in the HDF5 file (e.g., `sys_Re-5e4_Ma-1en1`), not the config name.

Load the loss dicts in Python to inspect metrics:

```python
import pickle
with open("walrus/experiments/eval_<name>/viz/loss_dicts/valid_loss_dict_epoch2_rank0.pkl", "rb") as f:
    metrics = pickle.load(f)
for k, v in metrics.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v:.6f}")
```

Load raw predictions for custom analysis:

```python
import numpy as np
yref = np.load(".../full_trajectory_dumps/yref_<dataset>_rollout_valid_epoch2_rank0_0.npy")
ypred = np.load(".../full_trajectory_dumps/ypred_<dataset>_rollout_valid_epoch2_rank0_0.npy")
print(f"Reference shape: {yref.shape}, Prediction shape: {ypred.shape}")
```

### Example baseline results (doubly_periodic, Re=5e4, Ma=0.1)

| Metric | Valid | Test |
|--------|-------|------|
| One-step MAE | 5.56e-6 | 4.97e-6 |
| Rollout MAE (20 steps) | 7.92e-5 | 7.95e-5 |
| One-step VRMSE | ~0.0012 | ~0.0012 |
| Rollout VRMSE (20 steps) | ~0.018 | ~0.018 |

### Comparing with post-finetuning results

After finetuning, re-run evaluation pointing to the finetuned checkpoint:

```bash
python train.py --config-name=eval_<name> \
    checkpoint.coalesced_checkpoint_path=/path/to/finetuned/checkpoint.pt \
    folder_override=/path/to/eval_output_finetuned/
```

Compare the loss dicts from both runs to see if finetuning improved performance.

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'output_fields'` | `dimensions/time` in split file still has full (unsplit) timestep count | Re-slice `dimensions/time` to match the actual number of timesteps in the split |
| `KeyError: "can't locate attribute: 'field_names'"` | Missing group-level `field_names` attr on `t0_fields`/`t1_fields` | Copy group attrs from source file to split files |
| Shape mismatch loading checkpoint (68 vs 67 fields) | Stale `extended_config.yaml` from previous run with different field index | Either delete `extended_config.yaml` from the experiment directory, or use a new output directory via CLI: `folder_override=/path/to/new_eval_dir/` |
| `Key 'finetune' is not in struct` | Missing `finetune` key in top-level eval config | Add `finetune: False` to the eval config |
| Job timeout (>30 min) | Validation split has too many timesteps | Use properly split data with ~1000 timesteps per valid/test split |

---

## Step 1: Inspect Your Source Data

Before converting, understand what you're working with. Open the source HDF5 in Python:

```python
import h5py, json

with h5py.File("/path/to/source.h5", "r") as f:
    print("Keys:", list(f.keys()))
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

    # Check discretization for grid sizes
    disc = json.loads(f.attrs["discretization"])
    print("Spatial grid:", disc["spatial"]["grid"])    # e.g. [256, 256] or [3001, 5]
    print("Resolution:", disc["spatial"]["resolution"])

    # Check what simulation parameters exist
    if "system_parameters" in f.attrs:
        print("Params:", json.loads(f.attrs["system_parameters"]))

    # Check number of timesteps
    print("Timesteps:", f["time"].shape)
```

Things to note:
- **Spatial dimensions** — Walrus only supports specific sizes: `{32, 128, 256, 384, 512, 768, 1024}`. The conversion script handles this, but you need to know your starting dimensions.
- **Number of timesteps** — This determines how much train/valid/test data you get after splitting.
- **Boundary conditions** — You need to know these for each axis (periodic, wall, open). They aren't stored in the HDF5; you know them from your simulation setup.

---

## Step 2: Convert to Well Format

The conversion script is at `scripts/convert_lbm_to_well.py`.

### Basic usage

```bash
python scripts/convert_lbm_to_well.py \
    --input /path/to/source.h5 \
    --bc-x <wall|periodic|open> \
    --bc-y <wall|periodic|open> \
    --crop-to-even \
    --pad-to-multiple 32 \
    --split --split-ratios 0.8,0.1,0.1
```

### What each flag does

| Flag | Purpose |
|------|---------|
| `--input` | Path to the source kinet HDF5 file |
| `--bc-x`, `--bc-y` | Boundary conditions per axis. Use `periodic`, `wall`, or `open`. For mixed BCs use comma syntax: `--bc-x wall,open` |
| `--crop-to-even` | If spatial dims are odd, crops by 1 to make even. Needed before padding. |
| `--pad-to-multiple 32` | **Required.** Adjusts spatial dims to the nearest valid Walrus size. Crops large dims down, tiles periodic or zero-pads non-periodic dims up. |
| `--split` | Splits the single HDF5 into `train/valid/test/` directories (temporal split). |
| `--split-ratios` | Default `0.8,0.1,0.1`. Adjusts train/valid/test proportions. |
| `--dataset-name` | Optional. Sets the `dataset_name` attribute. Defaults to parent directory name. |
| `--verify` | Verify an already-converted file without re-converting. |

### Example: Doubly-periodic (256x256, periodic everywhere)

```bash
python scripts/convert_lbm_to_well.py \
    --input /global/cfs/cdirs/m4790/Data/kinet/doubly_periodic/weakly_compressible_isoT_fluids/your_file.h5 \
    --bc-x periodic --bc-y periodic \
    --pad-to-multiple 32 \
    --split --split-ratios 0.8,0.1,0.1
```

Since 256 is already a valid Walrus dimension, no spatial adjustment is needed.

### Example: Sod shock tube (3001x5, wall/open x, periodic y)

```bash
python scripts/convert_lbm_to_well.py \
    --input /global/cfs/cdirs/m4790/Data/kinet/sod/compressible_fluids/sys_.../D2Q9_shape-3001-5_T-1000_....h5 \
    --bc-x wall,open --bc-y periodic \
    --crop-to-even \
    --pad-to-multiple 32 \
    --split --split-ratios 0.8,0.1,0.1
```

This will:
- Crop x from 3001 → 3000 (even), then crop to 1024 (nearest valid dim ≤ 3000)
- Tile y from 5 → 32 (periodic tiling, since 5 < 32)

### Output structure

After conversion with `--split`, you get:

```
<basename>_well-format/
└── data/
    ├── train/
    │   └── file_00.h5
    ├── valid/
    │   └── file_00.h5
    └── test/
        └── file_00.h5
```

The script auto-verifies each output file. Check the printed output for any `ERRORS`.

### Verify an existing file

```bash
python scripts/convert_lbm_to_well.py --verify /path/to/converted_well-format/data/train/file_00.h5
```

### What fields get converted

The conversion script maps these kinet fields to Well fields:

| kinet key | Well name | Walrus field index |
|-----------|-----------|-------------------|
| `density` | `density` | 28 |
| `velocity` | `velocity` (→ `velocity_x`, `velocity_y`) | 4, 5 |
| `pressure` | `pressure` | 3 |
| `temperature` | `temperature` | 46 |
| `energy` | `internal_energy` | 45 |
| `vorticity` | `vorticity` | 67 (new, not in pretrained model) |

LBM distribution functions (`F`, `Feq`, `G`, `Geq`, etc.) and solver diagnostics are automatically skipped.

---

## Step 3: Choose a Field Index Map

This is where a critical gotcha lives. The **field index map** tells Walrus which column index each physical field occupies in its input/output tensors. You have two choices:

### Option A: `full_well_field_index` (67 fields) — **USE THIS FOR FINETUNING**

This matches the pretrained `walrus.pt` checkpoint exactly (67 fields, indices 0–66). When you load the checkpoint, the encoder/decoder weights align perfectly. **Vorticity is excluded** since the pretrained model doesn't know about it.

File: `walrus/configs/data/field_index_map_override/full_well_field_index.yaml`

### Option B: `lbm_field_index` (68 fields) — **DO NOT USE FOR FINETUNING**

This adds `vorticity: 67` as field index 67, making 68 total fields. This causes a **shape mismatch** when loading `walrus.pt` because the checkpoint's encoder/decoder weight matrices are sized for 67 fields, not 68.

File: `walrus/configs/data/field_index_map_override/lbm_field_index.yaml`

**Rule: Always use `full_well_field_index` in your data config when loading the pretrained checkpoint.**

---

## Step 4: Create a Data Config

Create a file at `walrus/configs/data/<your_dataset>.yaml`. This tells Walrus where to find your Well-format data and how to batch it.

### Template

```yaml
defaults:
- field_index_map_override: full_well_field_index    # MUST be 67-field version for finetuning
well_base_path: null                                  # null = paths below are absolute
wandb_data_name: <your_dataset_name>
module_parameters:
  _target_: walrus.data.MixedWellDataModule
  batch_size: 2                 # per-GPU batch size. Reduce to 1 if you run out of memory.
  n_steps_input: 10             # number of input timesteps the model sees
  n_steps_output: 1             # number of output timesteps to predict
  min_dt_stride: 1              # minimum stride between timesteps
  max_dt_stride: 1              # maximum stride (1 = consecutive timesteps)
  max_samples: 2000             # max trajectories to use (set high, actual is limited by data)
  max_rollout_steps: 200        # max steps for autoregressive rollout evaluation
  well_dataset_info:
    <your_dataset_name>:
      include_filters: []
      exclude_filters: []
      path: /absolute/path/to/<basename>_well-format     # the directory containing data/
```

### Important notes

- **`path` must be absolute.** Training runs from `walrus/`, so relative paths break. Use the full path, e.g. `/global/cfs/cdirs/m4790/Data/kinet/sod/.../D2Q9_..._well-format`.
- **`n_steps_input`** — Must be ≤ the number of timesteps in your smallest split minus `n_steps_output`. If your valid split only has 12 timesteps, `n_steps_input: 10` with `n_steps_output: 1` leaves only 1 possible window. Use a smaller value (e.g. 2) if your dataset is small.
- **`batch_size`** — Start with 1–2 for the full model. The full Walrus model (`extended_isotropic`) is large; 4 GPUs with FSDP can typically handle `batch_size: 2`.
- **`max_rollout_steps`** — Set lower (e.g. 5–20) for initial testing to speed up validation. Increase later for real evaluation.

### Working example (sod subsonic)

```yaml
defaults:
- field_index_map_override: full_well_field_index
well_base_path: null
wandb_data_name: lbm_sod_subsonic
module_parameters:
  _target_: walrus.data.MixedWellDataModule
  batch_size: 1
  n_steps_input: 2
  n_steps_output: 1
  min_dt_stride: 1
  max_dt_stride: 1
  max_samples: 2000
  max_rollout_steps: 5
  well_dataset_info:
    lbm_sod_subsonic:
      include_filters: []
      exclude_filters: []
      path: /global/cfs/cdirs/m4790/Data/kinet/sod/compressible_fluids/sys_Pr-71en2_vuy-2_visc-25en3/D2Q9_shape-3001-5_T-1000_H-8908d7_well-format
```

---

## Step 5: Create a Top-Level Config

Create a file at `walrus/configs/<your_experiment>.yaml`. This is the main config that composes all the config groups together.

### Template

```yaml
defaults:
  - trainer: globalnorm
  - optimizer: adam
  - lr_scheduler: inv_sqrt_w_sqrt_ramps_longer
  - model: extended_isotropic
  - data: <your_data_config_name>          # matches filename in configs/data/ (without .yaml)
  - experiment: finetune_example
  - server: local
  - distribution: fsdp                     # fsdp for multi-GPU, local for single GPU
  - logger: none                           # or wandb if you want logging
  - checkpoint: defaults                   # loads walrus/checkpoints/walrus.pt
  - finetuning_mods: all
  - _self_                                 # *** MUST BE LAST ***

data_workers: 1                            # increase if I/O bottlenecked (up to ~10)
name: <experiment_name>                    # used in checkpoint directory naming
finetune: True
automatic_setup: True
experiment_dir: /absolute/path/to/experiments   # where checkpoints and logs go

# Override the folder_override from finetune_example (which sets "experiments/test")
folder_override: ""
checkpoint_override: ""

# Optimizer — lower learning rate for finetuning
optimizer:
  lr: 1e-4

# Model — disable regularization during finetuning
model:
  drop_path: 0.0
  input_field_drop: 0.0

# Trainer overrides
trainer:
  max_epoch: 51
  grad_acc_steps: 1
  clip_gradient: 10
  val_frequency: 5                         # validate every N epochs
  rollout_val_frequency: 5                 # autoregressive rollout eval every N epochs
  skip_spectral_metrics: True              # speeds up validation
  enable_amp: False                        # mixed precision off (more stable for finetuning)
  revin:                                   # MUST override if using globalnorm trainer
    _target_: walrus.trainer.normalization_strat.SamplewiseRevNormalization
    _partial_: True
```

### Critical gotchas explained

1. **`_self_` must be the LAST entry in `defaults:`.**
   Hydra processes defaults in order. If `_self_` is first, later defaults (like `experiment: finetune_example`) override your values. With `_self_` last, your config wins.

2. **Override `trainer.revin` to `SamplewiseRevNormalization`.**
   The `globalnorm` trainer defaults to `GlobalRevNormalization`, which requires a `stats.yaml` file in your dataset directory (pre-computed per-field statistics). Your custom data won't have this. `SamplewiseRevNormalization` computes statistics per-sample at runtime — no extra files needed.

3. **Override `folder_override: ""`.**
   The `finetune_example` experiment config sets `folder_override: "experiments/test"`. If you don't override this to `""`, all your experiments dump into the same `experiments/test/` directory.

4. **`experiment_dir` must be an absolute path.**
   Example: `/global/u2/k/khegazy/projects/pde/walrus/walrus/experiments`

5. **`distribution: fsdp` for multi-GPU, `distribution: local` for single GPU.**
   If using `local`, you also don't need `torchrun` — just `python train.py`.

### Working example (finetune_sod_subsonic.yaml)

```yaml
defaults:
  - trainer: globalnorm
  - optimizer: adam
  - lr_scheduler: inv_sqrt_w_sqrt_ramps_longer
  - model: extended_isotropic
  - data: lbm_sod_subsonic
  - experiment: finetune_example
  - server: local
  - distribution: fsdp
  - logger: none
  - checkpoint: defaults
  - finetuning_mods: all
  - _self_
data_workers: 1
name: finetune_sod_subsonic
finetune: True
automatic_setup: True
experiment_dir: /global/u2/k/khegazy/projects/pde/walrus/walrus/experiments

folder_override: ""
checkpoint_override: ""

optimizer:
  lr: 1e-4

model:
  drop_path: 0.0
  input_field_drop: 0.0

trainer:
  max_epoch: 51
  grad_acc_steps: 1
  clip_gradient: 10
  val_frequency: 5
  rollout_val_frequency: 5
  skip_spectral_metrics: True
  enable_amp: False
  revin:
    _target_: walrus.trainer.normalization_strat.SamplewiseRevNormalization
    _partial_: True
```

---

## Step 6: Create a Run Script

You have two options depending on how you want to run.

### Option A: Interactive (e.g. on a NERSC compute node via `salloc`)

First get an interactive GPU session:

```bash
salloc --nodes=1 --gpus-per-node=4 --time=02:00:00 -q interactive -C gpu -A <your_account>
```

Then run:

```bash
module load conda
conda activate walrus
cd walrus

# Multi-GPU with FSDP (4 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 train.py --config-name=<your_experiment>

# Single GPU (use distribution=local in your config or override on CLI)
python train.py --config-name=<your_experiment> distribution=local
```

### Option B: SLURM batch script (recommended for real runs)

Create a file like `walrus/run_scripts/finetune_<name>.sh`:

```bash
#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH -C gpu
#SBATCH -A <your_nersc_account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J walrus_ft_<name>
#SBATCH --output=walrus_ft_<name>_%j.log

export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

module load conda
conda activate walrus

cd /global/u2/k/khegazy/projects/pde/walrus/walrus

srun python -u $(which torchrun) \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29500 \
    train.py --config-name=<your_experiment>
```

Submit with:

```bash
sbatch walrus/run_scripts/finetune_<name>.sh
```

### Quick smoke test (single GPU, minimal compute)

Before burning GPU hours, verify everything loads correctly:

```bash
cd walrus
python train.py --config-name=<your_experiment> distribution=local trainer.max_epoch=2 trainer.val_frequency=1 trainer.rollout_val_frequency=1
```

This runs 2 epochs on a single GPU. If it gets through 1 validation + 1 rollout without crashing, your configs are correct.

---

## Step 7: Understanding the Output

### Experiment directory structure

When training runs, it creates:

```
<experiment_dir>/
└── <name>/                         # or auto-generated name
    ├── config.yaml                 # frozen copy of the resolved config
    ├── checkpoints/
    │   ├── sharded_checkpoint_dir/ # FSDP sharded checkpoints
    │   └── best/                   # best checkpoint by validation loss
    └── logs/                       # training logs
```

### What to look for in the training output

- **Training loss** — printed every `log_interval` steps. Should decrease.
- **Validation metrics** — printed every `val_frequency` epochs. Look for `NRMSE` and `PearsonR`.
- **Rollout metrics** — printed every `rollout_val_frequency` epochs. These show autoregressive performance.
- **Checkpoint saved** — printed when a new best checkpoint is saved.

### Common runtime errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError` in `choose_kernel_size_deterministic` | Spatial dims not in `{32, 128, 256, 384, 512, 768, 1024}` | Re-run conversion with `--pad-to-multiple 32` |
| Shape mismatch loading checkpoint | `field_index_map_override` has 68 fields but checkpoint has 67 | Use `full_well_field_index` (67 fields), not `lbm_field_index` |
| `FileNotFoundError` for `stats.yaml` | Using `GlobalRevNormalization` without pre-computed stats | Override `trainer.revin` to `SamplewiseRevNormalization` in your config |
| `FileNotFoundError` for data path | Relative path in data config | Use absolute path for `well_dataset_info.<name>.path` |
| Out of memory | Batch size too large for model size | Reduce `batch_size` to 1, or reduce `n_steps_input` |
| `IndexError` in data loading | `n_steps_input + n_steps_output` > number of timesteps in a split | Reduce `n_steps_input` or increase data size |

---

## Complete Walkthrough: New Dataset End-to-End

Here's the full sequence for a hypothetical new dataset called `my_simulation`:

```bash
# 1. Convert the data
python scripts/convert_lbm_to_well.py \
    --input /global/cfs/cdirs/m4790/Data/kinet/my_simulation/run.h5 \
    --bc-x periodic --bc-y periodic \
    --pad-to-multiple 32 \
    --split --split-ratios 0.8,0.1,0.1

# Note the output path printed by the script, e.g.:
#   /global/cfs/cdirs/m4790/Data/kinet/my_simulation/run_well-format/
```

```bash
# 2. Verify conversion
python scripts/convert_lbm_to_well.py \
    --verify /global/cfs/cdirs/m4790/Data/kinet/my_simulation/run_well-format/data/train/file_00.h5
```

```bash
# 3. Create data config: walrus/configs/data/my_simulation.yaml
cat > walrus/configs/data/my_simulation.yaml << 'EOF'
defaults:
- field_index_map_override: full_well_field_index
well_base_path: null
wandb_data_name: my_simulation
module_parameters:
  _target_: walrus.data.MixedWellDataModule
  batch_size: 2
  n_steps_input: 10
  n_steps_output: 1
  min_dt_stride: 1
  max_dt_stride: 1
  max_samples: 2000
  max_rollout_steps: 200
  well_dataset_info:
    my_simulation:
      include_filters: []
      exclude_filters: []
      path: /global/cfs/cdirs/m4790/Data/kinet/my_simulation/run_well-format
EOF
```

```bash
# 4. Create top-level config: walrus/configs/finetune_my_simulation.yaml
cat > walrus/configs/finetune_my_simulation.yaml << 'EOF'
defaults:
  - trainer: globalnorm
  - optimizer: adam
  - lr_scheduler: inv_sqrt_w_sqrt_ramps_longer
  - model: extended_isotropic
  - data: my_simulation
  - experiment: finetune_example
  - server: local
  - distribution: fsdp
  - logger: none
  - checkpoint: defaults
  - finetuning_mods: all
  - _self_
data_workers: 1
name: finetune_my_simulation
finetune: True
automatic_setup: True
experiment_dir: /global/u2/k/khegazy/projects/pde/walrus/walrus/experiments

folder_override: ""
checkpoint_override: ""

optimizer:
  lr: 1e-4

model:
  drop_path: 0.0
  input_field_drop: 0.0

trainer:
  max_epoch: 51
  grad_acc_steps: 1
  clip_gradient: 10
  val_frequency: 5
  rollout_val_frequency: 5
  skip_spectral_metrics: True
  enable_amp: False
  revin:
    _target_: walrus.trainer.normalization_strat.SamplewiseRevNormalization
    _partial_: True
EOF
```

```bash
# 5. Smoke test on a login node or single GPU
cd walrus
python train.py --config-name=finetune_my_simulation distribution=local \
    trainer.max_epoch=2 trainer.val_frequency=1 trainer.rollout_val_frequency=1
```

```bash
# 6. Run for real on 4 GPUs
cd walrus
torchrun --nnodes=1 --nproc_per_node=4 train.py --config-name=finetune_my_simulation
```

---

## Config Reference Cheat Sheet

### Config groups you can swap via CLI overrides

Any config value can be overridden on the command line. Examples:

```bash
# Use single GPU instead of FSDP
python train.py --config-name=finetune_my_simulation distribution=local

# Change batch size on the fly
python train.py --config-name=finetune_my_simulation data.module_parameters.batch_size=1

# Enable wandb logging
python train.py --config-name=finetune_my_simulation logger=wandb logger.wandb_project_name="my_project"

# Reduce epochs for testing
python train.py --config-name=finetune_my_simulation trainer.max_epoch=5

# Change learning rate
python train.py --config-name=finetune_my_simulation optimizer.lr=5e-5
```

### Key hyperparameters to tune

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `optimizer.lr` | `1e-4` | Learning rate. Lower (1e-5) for very small datasets. |
| `trainer.max_epoch` | `51` | Total training epochs. |
| `trainer.clip_gradient` | `10` | Gradient clipping norm. Lower if training is unstable. |
| `trainer.grad_acc_steps` | `1` | Gradient accumulation. Increase for effective larger batch without more memory. |
| `data.module_parameters.batch_size` | `2` | Per-GPU batch size. Effective batch = batch_size × n_GPUs × grad_acc_steps. |
| `data.module_parameters.n_steps_input` | `10` | Input temporal context. Reduce if dataset has few timesteps. |
| `trainer.val_frequency` | `5` | Epochs between validation. |
| `lr_scheduler.warmup_epochs` | `20` | LR warmup duration. Should be < max_epoch. |

### Valid spatial dimensions

Walrus only accepts these spatial sizes: **32, 128, 256, 384, 512, 768, 1024**. These come from the model's patch dictionary. Any other dimension causes a `KeyError` at runtime.

---

## Resuming a Run

If a run gets interrupted, Walrus auto-resumes if `auto_resume: True` (set by `experiment: finetune_example`). Just re-launch the same command. It will find the latest checkpoint in the experiment directory and continue.

If you want to resume from a specific checkpoint, set:

```bash
python train.py --config-name=finetune_my_simulation \
    checkpoint.load_checkpoint_path=/path/to/sharded_checkpoint_dir/
```

---

## File Inventory

Here's every file involved and its role:

| File | Role |
|------|------|
| `scripts/convert_lbm_to_well.py` | Converts kinet HDF5 → Well format |
| `walrus/configs/data/<name>.yaml` | Data config: paths, batch size, field index |
| `walrus/configs/data/field_index_map_override/full_well_field_index.yaml` | 67-field index map (matches pretrained checkpoint) |
| `walrus/configs/data/field_index_map_override/lbm_field_index.yaml` | 68-field index map (adds vorticity — do NOT use for finetuning) |
| `walrus/configs/<experiment>.yaml` | Top-level config composing all groups |
| `walrus/configs/experiment/finetune_example.yaml` | Experiment config: sets finetune mode, frozen components |
| `walrus/configs/checkpoint/defaults.yaml` | Checkpoint config: loads `walrus/checkpoints/walrus.pt` |
| `walrus/configs/trainer/globalnorm.yaml` | Trainer config (needs `revin` override for custom data) |
| `walrus/configs/finetuning_mods/all.yaml` | Makes RoPE position encoding learnable |
| `walrus/configs/distribution/fsdp.yaml` | Multi-GPU FSDP distribution |
| `walrus/configs/distribution/local.yaml` | Single-GPU, no distribution |
| `walrus/configs/server/local.yaml` | NERSC paths for Well base data |
| `walrus/checkpoints/walrus.pt` | Pretrained Walrus checkpoint |
| `walrus/train.py` | Training entrypoint |
| `walrus/run_scripts/finetune_*.sh` | SLURM / launch scripts |
