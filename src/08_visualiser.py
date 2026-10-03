import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
from pathlib import Path

# CONCEPT: We import our central configuration to maintain consistency across the project.
# This prevents magic numbers and ensures we use the exact colors and styles defined.
from src.config import PLOT_CONFIG, PATHS, PROJECT_ROOT

# DESIGN NOTE: We set the matplotlib style globally to ensure all plots have a consistent,
# publication-ready look. The 'seaborn-v0_8-whitegrid' style provides a clean background
# with faint gridlines, which helps readers easily trace data points to the axes.
plt.style.use(PLOT_CONFIG['style'])

def _add_significance_stars(ax: plt.Axes, p_value: float, x1: float, x2: float, y: float, h: float = 0.05):
    """
    Helper function to draw statistical significance stars between two groups.
    """
    # CONCEPT: We determine the number of stars based on standard academic thresholds.
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'  # Not significant
        
    # CONCEPT: We draw a bracket (a line that goes up, across, and down) to visually connect the two groups.
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    # CONCEPT: We place the text (stars) centered above the horizontal part of the bracket.
    ax.text((x1+x2)*.5, y+h, stars, ha='center', va='bottom', color='black', fontsize=PLOT_CONFIG['font_size'])

def plot_jitter_comparison(agg_data_a: pd.DataFrame, agg_data_b: pd.DataFrame, joint_name: str, model_name: str, p_value: Optional[float] = None, save_path: Optional[Path] = None):
    """
    Creates a side-by-side box plot comparing Set A vs Set B jitter for a specific joint.
    """
    # CONCEPT: fig, ax = plt.subplots() creates a figure (the canvas) and axes (the actual plot area).
    # DESIGN NOTE: figsize=(6, 6) creates a square plot, good for single-joint comparisons in a column.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Combine data for plotting using copies to avoid mutating original data
    df_a = agg_data_a.copy()
    df_b = agg_data_b.copy()
    
    df_a['Condition'] = 'Set A (Control)'
    df_b['Condition'] = 'Set B (Experimental)'
    df_combined = pd.concat([df_a, df_b])
    
    # CONCEPT: seaborn's boxplot automatically handles quartiles, medians, and outliers.
    sns.boxplot(
        data=df_combined,
        x='Condition',
        y=joint_name,
        hue='Condition',
        legend=False,
        ax=ax,
        palette=[PLOT_CONFIG['color_set_a'], PLOT_CONFIG['color_set_b']],
        width=0.4
    )
    
    # DESIGN NOTE: Clear, descriptive axis labels are essential for publication.
    ax.set_ylabel('Median Spatial Jitter (normalised units)', fontsize=PLOT_CONFIG['font_size'])
    ax.set_xlabel('Condition', fontsize=PLOT_CONFIG['font_size'])
    ax.set_title(f'Spatial Jitter: {joint_name} ({model_name})', fontsize=PLOT_CONFIG['font_size'] + 2)
    ax.tick_params(labelsize=PLOT_CONFIG['font_size'])
    
    # Add significance stars if p-value is provided
    if p_value is not None:
        y_max = df_combined[joint_name].max()
        # Calculate a nice height for the bracket above the maximum data point
        y_bracket = y_max + (y_max * 0.05)
        _add_significance_stars(ax, p_value, 0, 1, y_bracket, h=y_max*0.02)
        # Adjust y-limit to fit the stars
        ax.set_ylim(top=y_bracket + (y_max * 0.1))
        
    plt.tight_layout()
    
    # CONCEPT: Saving the figure.
    # DESIGN NOTE: We use the DPI specified in config for high-resolution output.
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_jitter_all_joints(agg_data_a: pd.DataFrame, agg_data_b: pd.DataFrame, model_name: str, p_values: Optional[Dict[str, float]] = None, save_path: Optional[Path] = None):
    """
    Creates a grouped box plot showing ALL joints side by side.
    This is the main figure for Paper 1.
    """
    # Data preparation: melt the dataframes to long format for seaborn grouping
    df_a = agg_data_a.melt(var_name='Joint', value_name='Jitter')
    df_a['Condition'] = 'Set A (Control)'
    
    df_b = agg_data_b.melt(var_name='Joint', value_name='Jitter')
    df_b['Condition'] = 'Set B (Experimental)'
    
    df_combined = pd.concat([df_a, df_b])
    
    # DESIGN NOTE: Wider figure (12, 6) to accommodate multiple joints side-by-side.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # CONCEPT: hue='Condition' creates grouped boxes for each joint.
    sns.boxplot(
        data=df_combined,
        x='Joint',
        y='Jitter',
        hue='Condition',
        ax=ax,
        palette=[PLOT_CONFIG['color_set_a'], PLOT_CONFIG['color_set_b']],
        width=0.6
    )
    
    ax.set_ylabel('Median Spatial Jitter (normalised units)', fontsize=PLOT_CONFIG['font_size'])
    ax.set_xlabel('Joint', fontsize=PLOT_CONFIG['font_size'])
    ax.set_title(f'Spatial Jitter Comparison Across Joints ({model_name})', fontsize=PLOT_CONFIG['font_size'] + 4)
    ax.tick_params(axis='x', rotation=45, labelsize=PLOT_CONFIG['font_size'])
    ax.tick_params(axis='y', labelsize=PLOT_CONFIG['font_size'])
    
    # CONCEPT: We position the legend cleanly outside the main plot area or in a clear corner.
    ax.legend(title='Condition', fontsize=PLOT_CONFIG['font_size'])
    
    if p_values:
        joints = df_combined['Joint'].unique()
        y_max = df_combined['Jitter'].max()
        for i, joint in enumerate(joints):
            if joint in p_values:
                # Calculate x positions for the two bars of this joint
                # Offset is roughly based on the width of the bars
                x1 = i - 0.2
                x2 = i + 0.2
                # Get max y for this specific joint to place bracket closely
                joint_max = df_combined[df_combined['Joint'] == joint]['Jitter'].max()
                y_bracket = joint_max + (y_max * 0.05)
                _add_significance_stars(ax, p_values[joint], x1, x2, y_bracket, h=y_max*0.02)
                
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_confidence_heatmap(landmarks_df: pd.DataFrame, model_name: str, condition_label: str, save_path: Optional[Path] = None):
    """
    Creates a heatmap (joints on y-axis, frames on x-axis, color = confidence).
    """
    # Assuming landmarks_df has columns for joints and rows for frames, containing confidence scores.
    # We transpose it so joints are on the y-axis.
    data_to_plot = landmarks_df.T
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # CONCEPT: sns.heatmap generates a color-coded matrix.
    # DESIGN NOTE: 'RdYlGn' is a standard diverging colormap where Green is good (high confidence 1.0)
    # and Red is bad (low confidence 0.0). Vmin/vmax lock the scale.
    sns.heatmap(
        data_to_plot,
        ax=ax,
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Confidence Score'}
    )
    
    ax.set_xlabel('Frame Number', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Joint', fontsize=PLOT_CONFIG['font_size'])
    ax.set_title(f'Detection Confidence Over Time ({condition_label} - {model_name})', fontsize=PLOT_CONFIG['font_size'] + 2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_confidence_timeseries(landmarks_df_a: pd.DataFrame, landmarks_df_b: pd.DataFrame, joint_name: str, model_name: str, save_path: Optional[Path] = None):
    """
    Plots confidence score over time for a specific joint, overlaying Set A and Set B.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # CONCEPT: ax.plot creates a standard line graph.
    ax.plot(landmarks_df_a.index, landmarks_df_a[joint_name], label='Set A (Control)', color=PLOT_CONFIG['color_set_a'], alpha=0.8)
    ax.plot(landmarks_df_b.index, landmarks_df_b[joint_name], label='Set B (Experimental)', color=PLOT_CONFIG['color_set_b'], alpha=0.8)
    
    # CONCEPT: ax.axhline draws a horizontal line across the entire plot.
    # DESIGN NOTE: We use a dashed line ('--') in black to clearly mark the 0.50 threshold.
    ax.axhline(y=0.50, color='black', linestyle='--', label='EKF Trigger Threshold (0.50)')
    
    # CONCEPT: ax.fill_between shades an area. We shade the area below 0.50 to emphasize occlusion events.
    ax.fill_between(landmarks_df_a.index, 0, 0.50, color='red', alpha=0.1, label='Low Confidence Zone')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frame Number', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Confidence Score', fontsize=PLOT_CONFIG['font_size'])
    ax.set_title(f'{joint_name} Confidence Over Time ({model_name})', fontsize=PLOT_CONFIG['font_size'] + 2)
    ax.legend(fontsize=PLOT_CONFIG['font_size'], loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_jitter_timeseries(jitter_df_a: pd.DataFrame, jitter_df_b: pd.DataFrame, joint_name: str, model_name: str, save_path: Optional[Path] = None):
    """
    Plots spatial jitter over time for a specific joint, overlaying Set A and Set B.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(jitter_df_a.index, jitter_df_a[joint_name], label='Set A (Control)', color=PLOT_CONFIG['color_set_a'], alpha=0.7)
    ax.plot(jitter_df_b.index, jitter_df_b[joint_name], label='Set B (Experimental)', color=PLOT_CONFIG['color_set_b'], alpha=0.7)
    
    # CONCEPT: Identify frames where Set B spikes. We define a spike heuristically, e.g., > 95th percentile.
    threshold = jitter_df_b[joint_name].quantile(0.95)
    spikes = jitter_df_b[jitter_df_b[joint_name] > threshold]
    
    # CONCEPT: ax.scatter overlays points on the line plot to highlight specific events.
    ax.scatter(spikes.index, spikes[joint_name], color='red', marker='x', s=50, label='Major Set B Spikes', zorder=5)
    
    ax.set_xlabel('Frame Number', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Spatial Jitter (normalised units)', fontsize=PLOT_CONFIG['font_size'])
    ax.set_title(f'{joint_name} Spatial Jitter Over Time ({model_name})', fontsize=PLOT_CONFIG['font_size'] + 2)
    ax.legend(fontsize=PLOT_CONFIG['font_size'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_statistical_summary_table(results_dict: Dict[str, Any], save_path: Optional[Path] = None):
    """
    Creates a formatted table figure summarising all statistical results.
    """
    # Prepare dataframe from the dictionary
    df = pd.DataFrame(results_dict)
    
    # CONCEPT: We turn off axes because we just want to draw a table on the blank canvas.
    fig, ax = plt.subplots(figsize=(12, min(4 + len(df) * 0.3, 10)))
    ax.axis('off')
    ax.axis('tight')
    
    # DESIGN NOTE: Using matplotlib's table function to create publication-style tables.
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    # CONCEPT: Loop through table cells to apply styling (bolding significant rows, setting colors).
    table.auto_set_font_size(False)
    table.set_fontsize(PLOT_CONFIG['font_size'])
    table.scale(1.2, 1.5)
    
    # Find column indices for styling
    p_col_idx = list(df.columns).index('p-value') if 'p-value' in df.columns else -1
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
        elif p_col_idx != -1 and col == p_col_idx:
            # Highlight significant p-values
            try:
                p_val = float(cell.get_text().get_text())
                if p_val < 0.05:
                    cell.set_text_props(weight='bold', color='red')
            except ValueError:
                pass # If parsing fails, ignore

    ax.set_title('Statistical Summary Results', weight='bold', fontsize=PLOT_CONFIG['font_size'] + 4)
    
    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def generate_all_plots(agg_a: pd.DataFrame, agg_b: pd.DataFrame, landmarks_a: pd.DataFrame, landmarks_b: pd.DataFrame, stats_results: Dict[str, Any], output_dir: Path):
    """
    Generates ALL plots for Paper 1 and saves them to the output directory.
    """
    # CONCEPT: We ensure the output directory exists before trying to save files into it.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "MediaPipe"
    joint_name = "wrist"
    p_values_all = {joint: 0.02 for joint in agg_a.columns} # Mock p-values
    
    # 1. Jitter comparison single joint
    plot_jitter_comparison(
        agg_a, agg_b, joint_name, model_name, p_value=0.01, 
        save_path=output_dir / 'jitter_comparison_wrist.png'
    )
    
    # 2. Jitter all joints
    plot_jitter_all_joints(
        agg_a, agg_b, model_name, p_values=p_values_all,
        save_path=output_dir / 'jitter_all_joints.png'
    )
    
    # 3. Confidence Heatmap
    plot_confidence_heatmap(
        landmarks_a, model_name, "Set A",
        save_path=output_dir / 'confidence_heatmap_set_a.png'
    )
    
    # 4. Confidence Timeseries
    plot_confidence_timeseries(
        landmarks_a, landmarks_b, joint_name, model_name,
        save_path=output_dir / 'confidence_timeseries_wrist.png'
    )
    
    # 5. Jitter Timeseries
    plot_jitter_timeseries(
        agg_a, agg_b, joint_name, model_name,
        save_path=output_dir / 'jitter_timeseries_wrist.png'
    )
    
    # 6. Statistical Summary Table
    plot_statistical_summary_table(
        stats_results,
        save_path=output_dir / 'statistical_summary.png'
    )
    
    print(f"Generated 6 plots and saved them to {output_dir}")


if __name__ == '__main__':
    # =============================================================================
    # REAL DATA VISUALISATION DEMO
    # =============================================================================
    print("Generating plots from real metrics data...")
    
    out_dir = PATHS['plots']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load real data
        df_jitter = pd.read_csv(PATHS['metrics'] / 'spatial_jitter.csv')
        df_conf = pd.read_csv(PATHS['metrics'] / 'confidence_volatility.csv')
        
        # We only have 1 video, so let's simulate Set A vs Set B by splitting it in half
        mp_jitter = df_jitter[df_jitter['model'] == 'mediapipe'].copy()
        mp_conf = df_conf[df_conf['model'] == 'mediapipe'].copy()
        
        mid_frame = mp_jitter['frame'].max() // 2
        
        # 1. Prepare Timeseries (Landmarks) Data -> Wide format (frames as index, joints as columns)
        land_a = mp_conf[mp_conf['frame'] <= mid_frame].pivot(index='frame', columns='joint_name', values='confidence')
        land_b = mp_conf[mp_conf['frame'] > mid_frame].pivot(index='frame', columns='joint_name', values='confidence')
        
        # 2. Prepare Aggregated Jitter Data -> Wide format (reps as index, joints as columns)
        # Import aggregator to get reps
        # Import aggregator dynamically to handle the numbered filename
        import importlib.util
        spec_agg = importlib.util.spec_from_file_location('agg', str(PROJECT_ROOT / 'src' / '04_aggregator.py'))
        agg = importlib.util.module_from_spec(spec_agg)
        spec_agg.loader.exec_module(agg)
        
        # Detect all 10 reps across the full simulated video
        bounds_all = agg.detect_repetitions(mp_jitter, 'OIZUKI', n_reps=10)
        agg_long_all = agg.aggregate_per_repetition(mp_jitter, bounds_all, 'jitter')
        
        # Split odd reps into Set A, even reps into Set B to simulate the two conditions
        agg_long_a = agg_long_all[agg_long_all['rep_id'] % 2 != 0].copy()
        agg_long_b = agg_long_all[agg_long_all['rep_id'] % 2 == 0].copy()
        
        # Pivot to wide format for the plotting functions
        agg_a = agg_long_a.pivot(index='rep_id', columns='joint_name', values='median')
        agg_b = agg_long_b.pivot(index='rep_id', columns='joint_name', values='median')
        
        # 3. Mock Stats Dict (real stats require pingouin/statsmodels, keeping this simple for plotting)
        stats_dict = {
            'Joint': ['right_wrist', 'right_elbow'],
            'Model': ['MediaPipe', 'MediaPipe'],
            'Set A Median': [agg_a['right_wrist'].median(), agg_a['right_elbow'].median()],
            'Set B Median': [agg_b['right_wrist'].median(), agg_b['right_elbow'].median()],
            'p-value': [0.04, 0.01],
            'Effect Size': [0.5, 0.7],
            'Interpretation': ['Large', 'Large']
        }
        
        # 4. Generate plots!
        # Pick a joint that exists in our data
        target_joint = 'right_wrist' if 'right_wrist' in agg_a.columns else agg_a.columns[0]
        
        print(f"Generating plots for joint: {target_joint}")
        
        # Generate them manually to inject our target joint
        model_name = "MediaPipe"
        p_values_all = {joint: 0.02 for joint in agg_a.columns}
        
        plot_jitter_comparison(agg_a, agg_b, target_joint, model_name, p_value=0.01, save_path=out_dir / f'jitter_comparison_{target_joint}.png')
        plot_jitter_all_joints(agg_a, agg_b, model_name, p_values=p_values_all, save_path=out_dir / 'jitter_all_joints.png')
        plot_confidence_heatmap(land_a, model_name, "Set A", save_path=out_dir / 'confidence_heatmap_set_a.png')
        plot_confidence_timeseries(land_a, land_b, target_joint, model_name, save_path=out_dir / f'confidence_timeseries_{target_joint}.png')
        
        # Jitter timeseries needs wide format of the raw data, not aggregated data
        jitter_raw_a = mp_jitter[mp_jitter['frame'] <= mid_frame].pivot(index='frame', columns='joint_name', values='jitter')
        jitter_raw_b = mp_jitter[mp_jitter['frame'] > mid_frame].pivot(index='frame', columns='joint_name', values='jitter')
        plot_jitter_timeseries(jitter_raw_a, jitter_raw_b, target_joint, model_name, save_path=out_dir / f'jitter_timeseries_{target_joint}.png')
        
        plot_statistical_summary_table(stats_dict, save_path=out_dir / 'statistical_summary.png')
        
        print(f"\n✅ All plots generated successfully in: {out_dir}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find metrics CSVs. Run 03_metrics.py first!\n({e})")

