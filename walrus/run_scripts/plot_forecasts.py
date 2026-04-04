"""Plot 2D forecast snapshots from saved prediction dumps.

Supports both 2D (B, T, H, W, C) and 3D (B, T, H, W, D, C) data.
For 3D data, takes a midplane slice along the last spatial axis by default.

Usage:
    python plot_forecasts.py /path/to/viz/dataset_name/full_trajectory_dumps

    # Specify field names (in channel order after padded fields are removed)
    python plot_forecasts.py /path/to/dumps --field-names density pressure velocity_x velocity_y

    # Plot specific timesteps (0-indexed into the rollout)
    python plot_forecasts.py /path/to/dumps --timesteps 0 5 10 19

    # Plot specific channels only
    python plot_forecasts.py /path/to/dumps --channels 0 2

    # For 3D data: choose slice axis and index
    python plot_forecasts.py /path/to/dumps --slice-axis 2 --slice-index 33
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def extract_2d_slice(arr, slice_axis, slice_index):
    """Extract a 2D slice from a 3D spatial volume.

    Args:
        arr: Array with shape (H, W, D) — the spatial dimensions.
        slice_axis: Which spatial axis to slice along (0=H, 1=W, 2=D).
        slice_index: Index along that axis.
    """
    return np.take(arr, slice_index, axis=slice_axis)


def plot_forecast_snapshot(
    y_pred,
    y_ref,
    timestep,
    channel,
    output_path,
    field_name=None,
    sample=0,
    vmin=None,
    vmax=None,
    slice_axis=None,
    slice_index=None,
):
    """Plot ground truth, prediction, and difference for one field at one timestep."""
    # Index into (B, T, ..spatial.., C) — spatial dims are everything between T and C
    truth = y_ref[sample, timestep, ..., channel]
    pred = y_pred[sample, timestep, ..., channel]

    # For 3D spatial data, take a 2D slice
    if truth.ndim == 3:
        if slice_axis is None:
            slice_axis = 2
        if slice_index is None:
            slice_index = truth.shape[slice_axis] // 2
        truth = extract_2d_slice(truth, slice_axis, slice_index)
        pred = extract_2d_slice(pred, slice_axis, slice_index)

    diff = pred - truth

    if vmin is None:
        vmin = min(truth.min(), pred.min())
    if vmax is None:
        vmax = max(truth.max(), pred.max())

    label = field_name if field_name else f"channel {channel}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(truth, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    abs_max = max(abs(diff.min()), abs(diff.max()))
    if abs_max == 0:
        abs_max = 1.0
    im2 = axes[2].imshow(
        diff, origin="lower", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max
    )
    axes[2].set_title(f"Difference (pred - truth)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    title = f"{label}, t={timestep}"
    if slice_axis is not None and slice_index is not None:
        axis_names = ["H", "W", "D"]
        axis_label = axis_names[slice_axis] if slice_axis < len(axis_names) else str(slice_axis)
        title += f", slice {axis_label}={slice_index}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_dump_pairs(dump_dir):
    """Find matching ypred/yref .npy file pairs in the dump directory."""
    pred_files = sorted(glob.glob(os.path.join(dump_dir, "ypred_*.npy")))
    pairs = []
    for pred_path in pred_files:
        ref_path = pred_path.replace("ypred_", "yref_")
        if os.path.exists(ref_path):
            pairs.append((pred_path, ref_path))
        else:
            print(f"Warning: no matching yref for {pred_path}")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D forecast snapshots from Walrus prediction dumps."
    )
    parser.add_argument(
        "dump_dir", help="Path to full_trajectory_dumps directory"
    )
    parser.add_argument(
        "--field-names",
        nargs="+",
        default=None,
        help="Field names in channel order (e.g., density pressure velocity_x)",
    )
    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=int,
        default=None,
        help="Timesteps to plot (0-indexed). Default: first, middle, last.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="Channel indices to plot. Default: all channels.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Batch sample index to plot (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots. Default: <dump_dir>/plots/",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=None,
        help="For 3D data: spatial axis to slice along (0=H, 1=W, 2=D). Default: 2.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="For 3D data: index along slice axis. Default: midplane.",
    )
    args = parser.parse_args()

    pairs = find_dump_pairs(args.dump_dir)
    if not pairs:
        print(f"No ypred_*.npy / yref_*.npy pairs found in {args.dump_dir}")
        return

    output_dir = args.output_dir or os.path.join(args.dump_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for pred_path, ref_path in pairs:
        basename = os.path.basename(pred_path).replace("ypred_", "").replace(".npy", "")
        print(f"Processing: {basename}")

        y_pred = np.load(pred_path)
        y_ref = np.load(ref_path)

        ndim = y_pred.ndim
        if ndim == 5:
            B, T, H, W, C = y_pred.shape
            print(f"  Shape: {y_pred.shape}  (B, T, H, W, C) — 2D data")
        elif ndim == 6:
            B, T, H, W, D, C = y_pred.shape
            print(f"  Shape: {y_pred.shape}  (B, T, H, W, D, C) — 3D data, will take 2D slices")
        else:
            print(f"  Unexpected shape {y_pred.shape} with {ndim} dims, skipping")
            continue

        # Determine timesteps to plot
        if args.timesteps is not None:
            timesteps = [t for t in args.timesteps if t < T]
        else:
            timesteps = sorted(set([0, T // 2, T - 1]))

        # Determine channels to plot
        channels = args.channels if args.channels is not None else list(range(C))

        for ch in channels:
            if ch >= C:
                print(f"  Skipping channel {ch} (only {C} channels)")
                continue
            field_name = (
                args.field_names[ch]
                if args.field_names and ch < len(args.field_names)
                else None
            )
            for t in timesteps:
                fname = f"{basename}_ch{ch}"
                if field_name:
                    fname = f"{basename}_{field_name}"
                fname = f"{fname}_t{t}.png"
                out_path = os.path.join(output_dir, fname)
                plot_forecast_snapshot(
                    y_pred,
                    y_ref,
                    timestep=t,
                    channel=ch,
                    output_path=out_path,
                    field_name=field_name,
                    sample=args.sample,
                    slice_axis=args.slice_axis,
                    slice_index=args.slice_index,
                )
            print(f"  Channel {ch}{f' ({field_name})' if field_name else ''}: "
                  f"{len(timesteps)} plots saved")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
