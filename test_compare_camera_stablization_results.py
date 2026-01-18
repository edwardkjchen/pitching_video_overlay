import pandas as pd
import matplotlib.pyplot as plt

def compare_motion_data(ground_truth_csv, estimated_raw_csv, estimated_smoothed_csv, output_image_path):
    """
    Reads, plots, and compares ground truth motion with raw and smoothed estimated motion.
    """
    try:
        # Read the CSV files
        gt_df = pd.read_csv(ground_truth_csv)
        est_raw_df = pd.read_csv(estimated_raw_csv)
        est_smoothed_df = pd.read_csv(estimated_smoothed_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all required CSV files exist.")
        return

    # Convert the ground truth position to frame-to-frame motion
    gt_motion_df = gt_df.diff().fillna(0)

    # Ensure all dataframes have the same length for fair comparison
    min_len = min(len(gt_motion_df), len(est_raw_df), len(est_smoothed_df))
    gt_motion_df = gt_motion_df.iloc[:min_len]
    est_raw_df = est_raw_df.iloc[:min_len]
    est_smoothed_df = est_smoothed_df.iloc[:min_len]

    # Create a figure and axes for the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Ground Truth vs. Raw vs. Smoothed Estimated Motion', fontsize=16)

    # Plot dx (horizontal motion)
    ax1.plot(gt_motion_df.index, gt_motion_df['dx'], label='Ground Truth', color='blue', linewidth=2)
    ax1.plot(est_raw_df.index, est_raw_df['dx'], label='Estimated (Raw)', color='red', linestyle='--', alpha=0.6)
    ax1.plot(est_smoothed_df.index, est_smoothed_df['dx'], label='Estimated (Smoothed)', color='purple', linewidth=2.5)
    ax1.set_ylabel('Horizontal Motion (dx)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('Horizontal Motion (dx)')

    # Plot dy (vertical motion)
    ax2.plot(gt_motion_df.index, gt_motion_df['dy'], label='Ground Truth', color='green', linewidth=2)
    ax2.plot(est_raw_df.index, est_raw_df['dy'], label='Estimated (Raw)', color='orange', linestyle='--', alpha=0.6)
    ax2.plot(est_smoothed_df.index, est_smoothed_df['dy'], label='Estimated (Smoothed)', color='cyan', linewidth=2.5)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Vertical Motion (dy)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Vertical Motion (dy)')

    # Save the plot to a file
    plt.savefig(output_image_path)
    print(f"Comparison plot saved to {output_image_path}")

if __name__ == '__main__':
    compare_motion_data(
        'motion.csv',
        'calculated_motion_raw.csv',
        'calculated_motion_smoothed.csv',
        'motion_comparison.png'
    )
