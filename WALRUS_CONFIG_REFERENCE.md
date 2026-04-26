# Walrus Configuration Reference: Pretraining vs Finetuning

This document lists every configuration parameter used for Walrus pretraining and finetuning, where each value was found, and how they differ. Use this as a reference when setting up new finetuning or evaluation runs.

---

## Sources

Each parameter is annotated with one or more of these source tags:

| Tag | Description | File |
|-----|-------------|------|
| **[PT-script]** | Pretraining run script (CLI overrides) | `walrus/run_scripts/pretrain_example_distributed_walrus.sh` |
| **[PT-config]** | Frozen pretraining config saved at checkpoint time | `walrus/configs/extended_config.yaml` |
| **[FT-script]** | Authors' finetuning run script (CLI overrides) | `walrus/run_scripts/finetuning_example_distributed_walrus.sh` |
| **[FT-yaml]** | Our sod finetuning top-level config | `walrus/configs/finetune_sod_subsonic.yaml` |
| **[trainer-defaults]** | Trainer defaults config | `walrus/configs/trainer/defaults.yaml` |
| **[trainer-globalnorm]** | Trainer globalnorm config | `walrus/configs/trainer/globalnorm.yaml` |
| **[optimizer-adam]** | Adam optimizer config | `walrus/configs/optimizer/adam.yaml` |
| **[lr-sqrt-longer]** | LR scheduler (longer warmup variant) | `walrus/configs/lr_scheduler/inv_sqrt_w_sqrt_ramps_longer.yaml` |
| **[lr-sqrt]** | LR scheduler (standard variant) | `walrus/configs/lr_scheduler/inv_sqrt_w_sqrt_ramps.yaml` |
| **[exp-finetune]** | Experiment config for finetuning | `walrus/configs/experiment/finetune_example.yaml` |
| **[finetuning-mods-all]** | Finetuning modifications config | `walrus/configs/finetuning_mods/all.yaml` |
| **[checkpoint-defaults]** | Local checkpoint config | `walrus/configs/checkpoint/defaults.yaml` |
| **[checkpoint-finetune]** | Remote checkpoint config | `walrus/configs/checkpoint/finetune.yaml` |
| **[eval-script]** | Authors' evaluation run script | `walrus/run_scripts/eval_onegpu_example_walrus.sh` |

---

## 1. Pretraining Configuration

These are the values used to pretrain the released `walrus.pt` checkpoint. Values come from both the run script (`pretrain_example_distributed_walrus.sh`) and the frozen config (`extended_config.yaml`). When both sources exist, they agree.

### 1.1 Model Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| `model._target_` | `walrus.models.IsotropicModel` | [PT-config]:119 |
| `model.hidden_dim` | 1408 | [PT-script]:37, [PT-config]:120 |
| `model.projection_dim` | 48 | [PT-script]:37, [PT-config]:121 |
| `model.intermediate_dim` | 352 | [PT-script]:37, [PT-config]:122 |
| `model.processor_blocks` | 40 | [PT-script]:38, [PT-config]:123 |
| `model.drop_path` | 0.05 | [PT-script]:38, [PT-config]:124 |
| `model.groups` | 16 | [PT-script]:38, [PT-config]:125 |
| `model.max_d` | 3 | [PT-config]:126 |
| `model.causal_in_time` | True | [PT-script]:39, [PT-config]:129 |
| `model.jitter_patches` | True | [PT-script]:39, [PT-config]:134 |
| `model.use_periodic_fixed_jitter` | True | [PT-script]:43, [PT-config]:136 |
| `model.input_field_drop` | 0.0 | [PT-script]:43, [PT-config]:137 |
| `model.gradient_checkpointing_freq` | 2 | [PT-script]:37, [PT-config]:135 |
| `model.override_dimensionality` | 0 | [PT-script]:42, [PT-config]:133 |
| `model.static_axes` | True | [PT-config]:127 |
| `model.weight_tied_axes` | False | [PT-config]:128 |
| `model.include_d` | [2, 3] | [PT-config]:130-131 |
| Encoder | `SpaceBagAdaptiveDVstrideEncoder` | [PT-config]:45 |
| Decoder | `AdaptiveDVstrideDecoder` | [PT-config]:79 |
| Spatial attention | `FullAttention`, 16 heads | [PT-script]:38, [PT-config]:103-104 |
| Temporal attention | `AxialTimeAttention`, 16 heads, rel bias | [PT-script]:38, [PT-config]:108-110 |
| Norm layer | `RMSGroupNorm` | [PT-config]:118 |

### 1.2 Optimizer

| Parameter | Value | Source |
|-----------|-------|--------|
| `optimizer._target_` | `torch.optim.AdamW` | [PT-config]:32 |
| `optimizer.lr` | **2e-4** | [PT-script]:36, [PT-config]:35 |
| `optimizer.weight_decay` | 1e-4 | [PT-config]:33 |
| `optimizer.eps` | 1e-10 | [PT-config]:34 |

### 1.3 LR Scheduler

| Parameter | Value | Source |
|-----------|-------|--------|
| `lr_scheduler` | `inv_sqrt_w_sqrt_ramps` | [PT-script]:41 |
| `lr_scheduler._target_` | `InverseSqrtLinearWarmupSqrtCooldown` | [lr-sqrt]:1, [PT-config]:37 |
| `warmup_epochs` | **10** | [lr-sqrt]:2, [PT-config]:38 |
| `cooldown_epochs` | 10 | [lr-sqrt]:3, [PT-config]:39 |
| `warmup_lr_factor` | **0.1** | [lr-sqrt]:4, [PT-config]:40 |
| `cooldown_lr_factor` | 0.001 | [lr-sqrt]:5, [PT-config]:41 |

### 1.4 Trainer

| Parameter | Value | Source |
|-----------|-------|--------|
| `trainer` base config | `defaults` | [PT-script]:36 |
| `trainer.max_epoch` | **201** | [PT-script]:42 |
| `trainer.val_frequency` | **10** | [PT-script]:41, [PT-config]:7 |
| `trainer.rollout_val_frequency` | **10** | [PT-script]:41, [PT-config]:8 |
| `trainer.short_validation_length` | 20 | [PT-config]:9 |
| `trainer.max_rollout_steps` | **60** | [PT-script]:40 |
| `trainer.num_time_intervals` | 5 | [PT-config]:11 |
| `trainer.loss_fn` | **MAE** | [trainer-defaults]:10, [PT-config]:14 |
| `trainer.revin` | **SamplewiseRevNormalization** | [trainer-defaults]:15, [PT-config]:18-20 |
| `trainer.prediction_type` | delta | [PT-script]:42, [PT-config]:21 |
| `trainer.grad_acc_steps` | **4** | [PT-script]:36, [PT-config]:22 |
| `trainer.clip_gradient` | **10** | [PT-script]:37, [PT-config]:26 |
| `trainer.log_interval` | **200** | [PT-script]:37, [PT-config]:27 |
| `trainer.loss_multiplier` | 100.0 | [PT-config]:28 |
| `trainer.enable_amp` | False | [PT-script]:37, [PT-config]:12 |
| `trainer.skip_spectral_metrics` | True | [PT-script]:43, [PT-config]:30 |
| `trainer.lr_scheduler_per_step` | False | [PT-config]:29 |
| `trainer.image_validation` | True | [PT-config]:23 |
| `trainer.video_validation` | True | [PT-config]:24 |

### 1.5 Data

| Parameter | Value | Source |
|-----------|-------|--------|
| `data` config | `all_2_3d` | [PT-script]:42 |
| `data.module_parameters.batch_size` | **2** | [PT-script]:37, [PT-config]:215 |
| `data.module_parameters.n_steps_input` | **6** | [PT-script]:37, [PT-config]:216 |
| `data.module_parameters.n_steps_output` | **1** | [PT-script]:37, [PT-config]:217 |
| `data.module_parameters.min_dt_stride` | **1** | [PT-script]:41, [PT-config]:218 |
| `data.module_parameters.max_dt_stride` | **5** | [PT-script]:41, [PT-config]:219 |
| `data.module_parameters.max_samples` | 2000 | [PT-script]:39, [PT-config]:220 |
| `data.field_index_map_override` | 67 fields (indices 0-66) | [PT-config]:139-206 |
| `data.transform.train` | `RandomRotation90` (p=1.0) | [PT-config]:208-210 |
| Number of training scenarios | 19 (2D + 3D) | [PT-config]:221-304 |

**Per-dataset overrides during pretraining** (from [PT-config]):

| Dataset | `step_downsample_factor` | `batch_downsample_factor` | `field_transforms` |
|---------|------------------------|--------------------------|-------------------|
| supernova_explosion_128 | 0.5 | 0.5 | density/temperature: log10 |
| turbulence_gravity_cooling | 0.5 | 0.5 | density/temperature: log10 |
| turbulent_radiative_layer_3D | 0.5 | 0.5 | density/temperature: log10 |
| MHD_64 | 0.5 | 0.5 | none |
| rayleigh_taylor_instability | 0.5 | 0.5 | none |
| acoustic_scattering_* (3 datasets) | none | none | density: zeros_like |
| All other 2D datasets | none | none | none |

Datasets with `step_downsample_factor: 0.5` have an effective `n_steps_input` of 3 (= 6 x 0.5).

### 1.6 Distribution & Infrastructure

| Parameter | Value | Source |
|-----------|-------|--------|
| `distribution` | **hsdp** (local_size=4) | [PT-script]:36, [PT-config]:312-314 |
| Nodes | **24** | [PT-script]: SBATCH line 4 |
| GPUs per node | 4 | [PT-script]: SBATCH line 6 |
| Total GPUs | **96** | 24 nodes x 4 GPUs |
| `data_workers` | **10** | [PT-script]:42, [PT-config]:1 |
| `finetuning_mods` | `defaults` (none) | [PT-script]:43, [PT-config]:327 |
| `checkpoint` | `defaults` | [PT-script]:43 |
| `auto_resume` | True | [PT-script]:42, [PT-config]:305 |

### 1.7 Effective Training Scale

| Metric | Value | Derivation |
|--------|-------|------------|
| Per-GPU batch size | 2 | `batch_size: 2` |
| Gradient accumulation | 4 | `grad_acc_steps: 4` |
| Total GPUs | 96 | 24 nodes x 4 |
| **Effective batch size** | **768** | 2 x 4 x 96 |
| Epochs | 201 | `max_epoch: 201` |
| Samples per epoch | 2000 | `max_samples: 2000` |

---

## 2. Finetuning Configuration (Paper / Authors' Script)

These values come from the authors' finetuning example script (`finetuning_example_distributed_walrus.sh`), which finetunes on `euler_multi_quadrants_openBC`. This represents the paper's finetuning protocol.

### 2.1 Model Architecture

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `model` base | `isotropic_model` | Same architecture | [FT-script]:34 |
| `model.hidden_dim` | 1408 | Same | [FT-script]:36 |
| `model.projection_dim` | 48 | Same | [FT-script]:36 |
| `model.intermediate_dim` | 352 | Same | [FT-script]:36 |
| `model.processor_blocks` | 40 | Same | [FT-script]:36 |
| `model.groups` | 16 | Same | [FT-script]:36 |
| `model.drop_path` | **0.0** | **Changed from 0.05** | [FT-script]:36 |
| `model.input_field_drop` | 0.0 | Same | [FT-script]:41 |
| `model.causal_in_time` | True | Same | [FT-script]:38 |
| `model.jitter_patches` | True | Same | [FT-script]:38 |
| `model.use_periodic_fixed_jitter` | True | Same | [FT-script]:41 |
| `model.gradient_checkpointing_freq` | 2 | Same | [FT-script]:35 |
| `model.override_dimensionality` | 0 | Same | [FT-script]:40 |
| `model.processor.space_mixing` | `full_spatial_attention`, 16 heads | Same | [FT-script]:37 |
| `model.processor.time_mixing` | 16 heads | Same | [FT-script]:37 |

### 2.2 Optimizer

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `optimizer` | `adam` (AdamW) | Same | [FT-script]:34 |
| `optimizer.lr` | **1e-4** | **Halved from 2e-4** | [FT-script]:34 |
| `optimizer.weight_decay` | 1e-4 | Same (inherited from [optimizer-adam]) | [optimizer-adam]:2 |
| `optimizer.eps` | 1e-10 | Same (inherited from [optimizer-adam]) | [optimizer-adam]:3 |
| `trainer.epsilon` | 1e-8 | **New** (added via `++`) | [FT-script]:37 |

### 2.3 LR Scheduler

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `lr_scheduler` | **`inv_sqrt_w_sqrt_ramps_longer`** | **Changed** (was `inv_sqrt_w_sqrt_ramps`) | [FT-script]:39 |
| `warmup_epochs` | **20** | **Doubled from 10** | [lr-sqrt-longer]:2 |
| `cooldown_epochs` | 10 | Same | [lr-sqrt-longer]:3 |
| `warmup_lr_factor` | **0.01** | **Reduced from 0.1** | [lr-sqrt-longer]:4 |
| `cooldown_lr_factor` | 0.001 | Same | [lr-sqrt-longer]:5 |

Effective warmup start LR: 0.01 x 1e-4 = **1e-6** (vs pretraining: 0.1 x 2e-4 = 2e-5).

### 2.4 Trainer

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `trainer` base config | **`globalnorm`** | **Changed** (was `defaults`) | [FT-script]:34 |
| `trainer.max_epoch` | **51** | **Reduced from 201** | [FT-script]:40 |
| `trainer.val_frequency` | **5** | **Reduced from 10** | [FT-script]:39 |
| `trainer.rollout_val_frequency` | **5** | **Reduced from 10** | [FT-script]:39 |
| `trainer.short_validation_length` | 20 | Same | [FT-script]:38 |
| `trainer.max_rollout_steps` | **60** | Same | [FT-script]:38 |
| `trainer.loss_fn` | **MAE** | Same (from [trainer-globalnorm]:10) | [trainer-globalnorm]:10 |
| `trainer.revin` | **GlobalRevNormalization** | **Changed** (was Samplewise) | [trainer-globalnorm]:15 |
| `trainer.prediction_type` | delta | Same | [FT-script]:40 |
| `trainer.grad_acc_steps` | **1** | **Reduced from 4** | [FT-script]:34 |
| `trainer.clip_gradient` | **10** | Same | [FT-script]:35 |
| `trainer.log_interval` | **200** | Same | [FT-script]:35 |
| `trainer.loss_multiplier` | 100.0 | Same (from [trainer-globalnorm]) | [trainer-globalnorm]:24 |
| `trainer.enable_amp` | False | Same | [FT-script]:35 |
| `trainer.skip_spectral_metrics` | True | Same | [FT-script]:41 |

**Important note on `revin`**: The authors' finetuning script uses `trainer: globalnorm` which defaults to `GlobalRevNormalization`. This requires a `stats.yaml` file in the dataset directory. The `euler_multi_quadrants_openBC` dataset from The Well has this file. Custom datasets (like our LBM data) do not, and must override `revin` to `SamplewiseRevNormalization`.

### 2.5 Data

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `data` config | **single dataset** (euler_multi_quadrants_openBC) | **Changed** (was all_2_3d) | [FT-script]:40 |
| `data.module_parameters.batch_size` | **2** | Same | [FT-script]:35 |
| `data.module_parameters.n_steps_input` | **6** | Same | [FT-script]:35 |
| `data.module_parameters.n_steps_output` | **1** | Same | [FT-script]:35 |
| `data.module_parameters.min_dt_stride` | **1** | Same | [FT-script]:39 |
| `data.module_parameters.max_dt_stride` | **1** | **Reduced from 5** | [FT-script]:39 |
| `data.module_parameters.max_samples` | 2000 | Same | [FT-script]:38 |
| `data.module_parameters.start_rollout_valid_output_at_t` | **17** | **New** (not in pretraining) | [FT-script]:42 |

### 2.6 Finetuning-Specific Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| `finetuning_mods` | **`all`** (learnable RoPE, rope_per_axis) | [FT-script]:42 |
| `finetune` | True | implied by `experiment: finetune_example` |
| `experiment` | `finetune_example` | [FT-script]:41 |
| `frozen_components` | `[model]` (model **config** frozen, not weights) | [exp-finetune]:8-9 |
| `checkpoint` | `finetune` (loads remote `walrus_step200.pt`) | [FT-script]:41 |
| `auto_resume` | True | [FT-script]:40 |

### 2.7 Distribution & Infrastructure

| Parameter | Value | Differs from pretraining? | Source |
|-----------|-------|--------------------------|--------|
| `distribution` | **fsdp** | **Changed** (was hsdp) | [FT-script]:34 |
| Nodes | **1** | **Reduced from 24** | [FT-script]: SBATCH line 4 |
| GPUs per node | 4 | Same | [FT-script]: SBATCH line 6 |
| Total GPUs | **4** | **Reduced from 96** | 1 node x 4 GPUs |
| `data_workers` | **10** | Same | [FT-script]:40 |

### 2.8 Effective Training Scale

| Metric | Value | Derivation |
|--------|-------|------------|
| Per-GPU batch size | 2 | `batch_size: 2` |
| Gradient accumulation | 1 | `grad_acc_steps: 1` |
| Total GPUs | 4 | 1 node x 4 |
| **Effective batch size** | **8** | 2 x 1 x 4 |
| Epochs | 51 | `max_epoch: 51` |
| Samples per epoch | 2000 | `max_samples: 2000` |

### 2.9 Evaluation Configuration (Authors' Script)

From `eval_onegpu_example_walrus.sh`, the authors evaluate finetuned models with:

| Parameter | Value | Source |
|-----------|-------|--------|
| `distribution` | local (1 GPU) | [eval-script]:29 |
| `validation_mode` | True | [eval-script]:29 |
| `max_rollout_steps` | 200 | [eval-script]:29 |
| `start_rollout_valid_output_at_t` | **17** | [eval-script]:29 |
| `validation_suite` | NRMSE, VRMSE, PearsonR | [eval-script]:29 |
| `validation_trajectory_metrics` | [] (empty) | [eval-script]:29 |

---

## 3. Pretraining vs Finetuning: All Differences

This section lists every parameter that changes between pretraining and the authors' finetuning protocol.

### 3.1 Intentional Changes (Finetuning Modifications)

These are deliberate changes for the finetuning regime:

| Parameter | Pretraining | Finetuning | Rationale |
|-----------|------------|------------|-----------|
| `optimizer.lr` | 2e-4 | **1e-4** | Lower learning rate to avoid catastrophic forgetting |
| `model.drop_path` | 0.05 | **0.0** | Disable stochastic depth regularization (smaller dataset) |
| `finetuning_mods` | `defaults` (none) | **`all`** (learnable RoPE) | Make positional encoding learnable for domain adaptation |
| `max_epoch` | 201 | **51** | Fewer epochs needed (single domain, not 19) |
| `data` | `all_2_3d` (19 domains) | **single domain** | Finetuning targets one dataset |
| `distribution` | hsdp (24 nodes, 96 GPUs) | **fsdp** (1 node, 4 GPUs) | Smaller scale sufficient for single domain |
| `checkpoint` | `defaults` | **`finetune`** | Load pretrained checkpoint for initialization |
| `experiment` | `defaults` | **`finetune_example`** | Enable finetuning path with auto-resume |
| `val_frequency` | 10 | **5** | Validate more often (shorter training) |
| `rollout_val_frequency` | 10 | **5** | Rollout eval more often |
| `start_rollout_valid_output_at_t` | not set (-1) | **17** | Start rollouts from T=17 for fair comparison |

### 3.2 Changes That Affect Training Dynamics

These changes alter the effective training regime:

| Parameter | Pretraining | Finetuning | Effect |
|-----------|------------|------------|--------|
| `grad_acc_steps` | **4** | **1** | Effective batch size drops from 768 to 8 (96x reduction) |
| `max_dt_stride` | **5** | **1** | No variable temporal stride during finetuning |
| `lr_scheduler` | `inv_sqrt_w_sqrt_ramps` | **`inv_sqrt_w_sqrt_ramps_longer`** | Different warmup schedule |
| `warmup_epochs` | 10 | **20** | Longer warmup (39% of 51 epochs vs 5% of 201 epochs) |
| `warmup_lr_factor` | 0.1 | **0.01** | Start from 10x lower LR during warmup |
| `trainer` base | `defaults` | **`globalnorm`** | Different normalization default + slightly different validation suite |
| `trainer.revin` | SamplewiseRevNorm | **GlobalRevNorm** | Different normalization strategy (requires `stats.yaml`) |

### 3.3 Unchanged Parameters

These remain identical between pretraining and finetuning:

| Parameter | Value |
|-----------|-------|
| Model architecture (hidden_dim, blocks, etc.) | 1408 / 40 / 48 / 352 |
| `n_steps_input` | 6 |
| `n_steps_output` | 1 |
| `batch_size` | 2 |
| `max_samples` | 2000 |
| `clip_gradient` | 10 |
| `loss_fn` | MAE |
| `prediction_type` | delta |
| `loss_multiplier` | 100.0 |
| `enable_amp` | False |
| `jitter_patches` | True |
| `causal_in_time` | True |
| `gradient_checkpointing_freq` | 2 |
| `input_field_drop` | 0.0 |
| `cooldown_epochs` | 10 |
| `weight_decay` | 1e-4 |

---

## 4. Our Sod Finetuning Config vs Authors' Finetuning

This section compares `finetune_sod_subsonic.yaml` + `lbm_sod_subsonic.yaml` against the authors' finetuning script.

### 4.1 Differences

| Parameter | Authors' Finetuning | Our Sod Config | Impact | Source comparison |
|-----------|-------------------|----------------|--------|-------------------|
| **`n_steps_input`** | **6** | **10** | **HIGH** — model pretrained on 6-step context; 10 is out of distribution | [FT-script]:35 vs [FT-yaml] data config:8 |
| **`trainer.revin`** | **GlobalRevNorm** (has stats.yaml) | **SamplewiseRevNorm** (override) | **MEDIUM** — necessary since custom data lacks stats.yaml, but changes normalization behavior | [FT-script]:34 (globalnorm) vs [FT-yaml]:43-45 |
| **`start_rollout_valid_output_at_t`** | **17** | **not set** (defaults to -1) | **MEDIUM** — rollouts start from T=0 instead of T=17; affects metric comparability | [FT-script]:42 vs not in sod data config |
| **`checkpoint`** | **`finetune`** (remote path) | **`defaults`** (local `walrus.pt`) | **LOW** — same checkpoint, different load path | [FT-script]:41 vs [FT-yaml]:11 |
| **`data_workers`** | **10** | **1** | **LOW** — affects I/O throughput, not model behavior | [FT-script]:40 vs [FT-yaml]:14 |
| **`trainer.epsilon`** | **1e-8** (added via `++`) | **not set** | **LOW** — minor numerical stability parameter | [FT-script]:37 |
| **`model` base config** | `isotropic_model` | `extended_isotropic` | **NONE** — `extended_isotropic` should resolve to same architecture; verify by comparing model params | [FT-script]:34 vs [FT-yaml]:4 |

### 4.2 Matches

These parameters correctly match the authors' finetuning protocol:

| Parameter | Value | Source |
|-----------|-------|--------|
| `optimizer.lr` | 1e-4 | [FT-script]:34, [FT-yaml]:26 |
| `lr_scheduler` | `inv_sqrt_w_sqrt_ramps_longer` | [FT-script]:39, [FT-yaml]:3 |
| `model.drop_path` | 0.0 | [FT-script]:36, [FT-yaml]:30 |
| `model.input_field_drop` | 0.0 | [FT-script]:41, [FT-yaml]:31 |
| `grad_acc_steps` | 1 | [FT-script]:34, [FT-yaml]:36 |
| `clip_gradient` | 10 | [FT-script]:35, [FT-yaml]:37 |
| `max_epoch` | 51 | [FT-script]:40, [FT-yaml]:35 |
| `val_frequency` | 5 | [FT-script]:39, [FT-yaml]:38 |
| `rollout_val_frequency` | 5 | [FT-script]:39, [FT-yaml]:39 |
| `n_steps_output` | 1 | [FT-script]:35, data config:9 |
| `min_dt_stride` | 1 | [FT-script]:39, data config:10 |
| `max_dt_stride` | 1 | [FT-script]:39, data config:11 |
| `batch_size` | 2 | [FT-script]:35, data config:7 |
| `max_samples` | 2000 | [FT-script]:38, data config:12 |
| `enable_amp` | False | [FT-script]:35, [FT-yaml]:41 |
| `skip_spectral_metrics` | True | [FT-script]:41, [FT-yaml]:40 |
| `finetuning_mods` | `all` (learnable RoPE) | [FT-script]:42, [FT-yaml]:12 |
| `distribution` | fsdp | [FT-script]:34, [FT-yaml]:9 |
| `experiment` | `finetune_example` | [FT-script]:41, [FT-yaml]:7 |
| `finetune` | True | [FT-script] implied, [FT-yaml]:16 |
| `field_index_map_override` | `full_well_field_index` (67 fields) | inferred from [PT-config], data config:2 |

### 4.3 Recommended Fixes

1. **Change `n_steps_input` from 10 to 6** in `walrus/configs/data/lbm_sod_subsonic.yaml`
   - This is the most impactful fix. The model was pretrained and finetuned with 6 input steps for 2D data.

2. **Add `start_rollout_valid_output_at_t: 17`** to `walrus/configs/data/lbm_sod_subsonic.yaml`
   - Not strictly necessary for custom data (T=17 was chosen for The Well datasets to equalize context across models), but useful if comparing against paper baselines.

3. **Increase `data_workers` to 10** in `walrus/configs/finetune_sod_subsonic.yaml`
   - Matches authors' setting; improves data loading throughput on multi-GPU runs.

---

## 5. Notes and Caveats

### 5.1 What the paper says vs what the repo shows

The paper (arXiv:2511.15684) describes the methodology but not every hyperparameter. The most authoritative sources for exact values are:
- **Pretraining**: `pretrain_example_distributed_walrus.sh` and `extended_config.yaml`
- **Finetuning**: `finetuning_example_distributed_walrus.sh`
- **Evaluation**: `eval_onegpu_example_walrus.sh`

The paper states (from prior reading): finetuning uses 500K additional training samples, learnable RoPE, and rollouts start from T=17. These are confirmed by the run scripts.

### 5.2 GlobalRevNorm vs SamplewiseRevNorm

The authors use `GlobalRevNormalization` for finetuning (via `trainer: globalnorm`), which requires pre-computed per-field statistics (`stats.yaml`) in each dataset directory. The Well datasets ship with this file. Custom datasets (like our LBM data) must either:
- Compute `stats.yaml` from their training data, or
- Override `trainer.revin` to `SamplewiseRevNormalization` (computes stats per-sample at runtime)

The pretrained model was trained with `SamplewiseRevNormalization`. Using `GlobalRevNormalization` during finetuning is a deliberate choice by the authors — the normalization strategy can change between pretraining and finetuning without causing issues, because RevIN normalization is applied and reversed at the input/output boundaries (it doesn't affect the learned weights).

### 5.3 `_self_` ordering

The repo's shipped `config.yaml` has `_self_` as the **first** entry in `defaults:`, which is a Hydra gotcha — later defaults will override the config's own values. The authors work around this by specifying all values as CLI overrides in the run scripts. Our configs correctly place `_self_` **last**.

### 5.4 `frozen_components: [model]`

This freezes the **model config** (architecture settings imported from checkpoint), not the model **weights**. The model weights are still trainable during finetuning. This ensures the finetuned model uses the same architecture as the pretrained checkpoint.
