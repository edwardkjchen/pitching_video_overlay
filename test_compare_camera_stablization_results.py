import pandas as pd
import matplotlib.pyplot as plt

def compare_motion_data(ground_truth_csv, estimated_raw_csv, estimated_smoothed_csv, output_image_path):
    """
    Reads, plots, and compares ground truth motion data with the raw and
    smoothed motion data estimated by the stabilization algorithm.

    This function generates a plot that visually contrasts the actual motion
    (ground truth) with the motion detected by the algorithm before and after
    smoothing. This is crucial for evaluating the accuracy of the stabilization.

    Args:
        ground_truth_csv (str): Path to the CSV file with the actual motion data.
        estimated_raw_csv (str): Path to the CSV with the raw, unfiltered estimated motion.
        estimated_smoothed_csv (str): Path to the CSV with the smoothed, filtered estimated motion.
        output_image_path (str): Path to save the output comparison plot image.
    """
    try:
        # Load the motion data from the respective CSV files into pandas DataFrames
        gt_df = pd.read_csv(ground_truth_csv)
        est_raw_df = pd.read_csv(estimated_raw_csv)
        est_smoothed_df = pd.read_csv(estimated_smoothed_csv)
    except FileNotFoundError as e:
        # Handle cases where one or more of the CSV files are not found
        print(f"Error: {e}. Please ensure all required CSV files exist.")
        return

    # The ground truth data might be positional. Convert it to frame-to-frame motion
    # by calculating the difference between consecutive frames. `diff()` computes this.
    # `fillna(0)` handles the first row, which will have a NaN value after diffing.
    gt_motion_df = gt_df.diff().fillna(0)

    # To ensure a fair and accurate comparison, truncate all dataframes to the
    # length of the shortest one. This prevents errors from mismatched indices.
    min_len = min(len(gt_motion_df), len(est_raw_df), len(est_smoothed_df))
    gt_motion_df = gt_motion_df.iloc[:min_len]
    est_raw_df = est_raw_df.iloc[:min_len]
    est_smoothed_df = est_smoothed_df.iloc[:min_len]

    # Create a figure with two subplots stacked vertically (2 rows, 1 column).
    # `figsize` controls the plot dimensions, and `sharex=True` makes both plots share the same x-axis.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Ground Truth vs. Raw vs. Smoothed Estimated Motion', fontsize=16)

    # --- Plot 1: Horizontal Motion (dx) ---
    ax1.plot(gt_motion_df.index, gt_motion_df['dx'], label='Ground Truth', color='blue', linewidth=2)
    ax1.plot(est_raw_df.index, est_raw_df['dx'], label='Estimated (Raw)', color='red', linestyle='--', alpha=0.6)
    ax1.plot(est_smoothed_df.index, est_smoothed_df['dx'], label='Estimated (Smoothed)', color='purple', linewidth=2.5)
    ax1.set_ylabel('Horizontal Motion (dx)')
    ax1.legend() # Display the legend
    ax1.grid(True, linestyle='--', alpha=0.6) # Add a grid for readability
    ax1.set_title('Horizontal Motion (dx)')

    # --- Plot 2: Vertical Motion (dy) ---
    ax2.plot(gt_motion_df.index, gt_motion_df['dy'], label='Ground Truth', color='green', linewidth=2)
    ax2.plot(est_raw_df.index, est_raw_df['dy'], label='Estimated (Raw)', color='orange', linestyle='--', alpha=0.6)
    ax2.plot(est_smoothed_df.index, est_smoothed_df['dy'], label='Estimated (Smoothed)', color='cyan', linewidth=2.5)
    ax2.set_xlabel('Frame Number') # Label for the shared x-axis
    ax2.set_ylabel('Vertical Motion (dy)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Vertical Motion (dy)')

    # Save the entire figure (both subplots) to an image file
    plt.savefig(output_image_path)
    print(f"Comparison plot saved to {output_image_path}")

# This block executes when the script is run directly
if __name__ == '__main__':
    # Define the paths for the input CSVs and the output plot image
    compare_motion_data(
        'motion.csv',
        'calculated_motion_raw.csv',
        'calculated_motion_smoothed.csv',
        'motion_comparison.png'
    )
