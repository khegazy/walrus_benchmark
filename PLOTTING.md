# Plotting Forecast Images

## Prerequisites

Run evaluation with prediction dumping enabled (already configured in `test_eval.sh`):

```bash
cd walrus
bash run_scripts/test_eval.sh
```

This saves `ypred_*.npy` and `yref_*.npy` files to:
```
<experiment_dir>/viz/<dataset_name>/full_trajectory_dumps/
```

For example:
```
/pscratch/sd/k/khegazy/projects/pde/walrus/experiments/test/viz/post_neutron_star_merger/full_trajectory_dumps/
```

## Running the plotting script

```bash
cd walrus

# Basic usage — plots all channels at first, middle, and last timesteps
python run_scripts/plot_forecasts.py <path_to_full_trajectory_dumps>

# With field name labels
python run_scripts/plot_forecasts.py <path_to_dumps> \
    --field-names density pressure velocity_x velocity_y

# Specific timesteps (0-indexed into the rollout)
python run_scripts/plot_forecasts.py <path_to_dumps> --timesteps 0 10 19

# Specific channels only
python run_scripts/plot_forecasts.py <path_to_dumps> --channels 0 2

# Custom output directory
python run_scripts/plot_forecasts.py <path_to_dumps> --output-dir /path/to/output

# Select a different sample from the batch (default: 0)
python run_scripts/plot_forecasts.py <path_to_dumps> --sample 1
```

## Output

Each plot is a PNG with three panels: **ground truth**, **prediction**, and **difference** (pred - truth). Plots are saved to `<dump_dir>/plots/` by default.

## Notes

- The saved arrays have padded channels already removed, so channel indices correspond only to active fields in the dataset.
- Field names are not saved automatically. Use `--field-names` to add labels matching the channel order from your data config.
