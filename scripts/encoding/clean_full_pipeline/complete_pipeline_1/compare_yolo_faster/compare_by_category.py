import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import csv

# ------------------ CONFIG ------------------
# Output directory
output_dir = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/compare_yolo_faster/results"
os.makedirs(output_dir, exist_ok=True)

# Custom pastel color palette
pastel_palette = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']
sns.set_palette(pastel_palette)

# YOLO model CSV paths with descriptive names
yolo_csv_files = {
    'YOLOv8 (102-class)': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/train-202-24classes-yolo-9654-data-splits-weights_per_class_results-1744580263.csv",
}

# Faster R-CNN path with descriptive name
faster_rcnn_dir = '/homes/es314/1-results-only-images/'
faster_rcnn_specific_file = os.path.join(faster_rcnn_dir, 'doremiv2/faster-rcnn/obj-det-2023/results_by_category.csv')
faster_rcnn_experiment = 'may_2023_ex001'
faster_rcnn_name = 'FR-CNN (Inception-ResNet)'  # Changed to a more descriptive name

print(f"Checking if Faster R-CNN file exists: {os.path.exists(faster_rcnn_specific_file)}")

# Symbol groups for aggregation
symbol_groups = {
    'clefs': ['gClef', 'fClef', 'cClef', 'gClef8vb', 'fClef8vb', 'unpitchedPercussionClef1'],
    'noteheads': ['noteheadBlack', 'noteheadHalf', 'noteheadWhole'],
    'rests': ['rest8th', 'rest16th', 'rest32nd', 'restQuarter', 'restHalf', 'restWhole', 'rest128th'],
    'flags': ['flag8thUp', 'flag8thDown', 'flag16thUp', 'flag16thDown', 'flag32ndUp'],
    'dynamics': ['dynamicPiano', 'dynamicForte', 'dynamicFF', 'dynamicPP', 'dynamicMF'],
    'accidentals': ['accidentalSharp', 'accidentalFlat', 'accidentalNatural', 'accidentalDoubleFlat'],
    'barlines': ['barline', 'systemicBarline'],
}

# ------------------ FUNCTIONS ------------------

def save_all_formats(fig, filepath_base):
    """Save figure in multiple formats"""
    fig.savefig(f'{filepath_base}.png', dpi=300)
    fig.savefig(f'{filepath_base}.pdf')
    fig.savefig(f'{filepath_base}.svg')

def load_yolo_results(file_path, experiment_name):
    """Load results from YOLO format CSV"""
    df = pd.read_csv(file_path)
    df['Experiment'] = experiment_name
    # Remove duplicates if any - keep the highest AP value
    df = df.sort_values('AP@0.5', ascending=False)
    df = df.drop_duplicates(subset=['Class Name', 'Experiment'])
    return df[['Class Name', 'AP@0.5', 'Experiment']]

def load_faster_rcnn_results(file_path, experiment_filter, experiment_name_override=None):
    """Load results from Faster R-CNN format CSV with experiment filter"""
    try:
        df = pd.read_csv(file_path)
        # Filter for specific experiment
        df = df[df['experiment'] == experiment_filter]
        # Filter for AP@0.5 metric
        df = df[df['metric'].str.contains('DetectionBoxes_Precision mAP@.50IOU')]
        df = df[df['value'] >= 0.0]
        # Rename columns to match YOLO format
        df['Class Name'] = df['category']
        df['Experiment'] = experiment_name_override if experiment_name_override else f'FRCNN_{experiment_filter}'
        df['AP@0.5'] = df['value']
        # Remove duplicates if any - keep the highest AP value
        df = df.sort_values('AP@0.5', ascending=False)
        df = df.drop_duplicates(subset=['Class Name', 'Experiment'])
        return df[['Class Name', 'AP@0.5', 'Experiment']]
    except Exception as e:
        print(f"Error loading Faster R-CNN results from {file_path}: {e}")
        return pd.DataFrame(columns=['Class Name', 'AP@0.5', 'Experiment'])

# ------------------ LOAD DATA ------------------

# Load YOLO results
yolo_dfs = [load_yolo_results(path, name) for name, path in yolo_csv_files.items()]
yolo_results = pd.concat(yolo_dfs) if yolo_dfs else pd.DataFrame(columns=['Class Name', 'AP@0.5', 'Experiment'])

# Load Faster R-CNN results with specific experiment filter and descriptive name
faster_rcnn_results = load_faster_rcnn_results(faster_rcnn_specific_file, faster_rcnn_experiment, faster_rcnn_name)

# Combine all results
df_all = pd.concat([yolo_results, faster_rcnn_results])

# Additional check for duplicates after concatenation
df_all = df_all.sort_values('AP@0.5', ascending=False)
df_all = df_all.drop_duplicates(subset=['Class Name', 'Experiment'])

# Print class counts for debugging
print("Classes per experiment:")
print(df_all.groupby('Experiment')['Class Name'].nunique())

# Find common classes between experiments
yolo_classes = set(yolo_results['Class Name'].unique())
frcnn_classes = set(faster_rcnn_results['Class Name'].unique())
common_classes = yolo_classes.intersection(frcnn_classes)
print(f"Common classes between experiments: {len(common_classes)}")

# Create comparison dataframe for common classes
if common_classes and len(df_all['Experiment'].unique()) > 1:
    comparison_data = []
    for cls in common_classes:
        cls_data = {}
        cls_data['Class Name'] = cls
        
        for exp in df_all['Experiment'].unique():
            ap_value = df_all[(df_all['Class Name'] == cls) & (df_all['Experiment'] == exp)]['AP@0.5'].values
            if len(ap_value) > 0:
                cls_data[exp] = ap_value[0]
            else:
                cls_data[exp] = 0.0
        
        # Calculate the difference between experiments
        exps = list(df_all['Experiment'].unique())
        if len(exps) >= 2:
            cls_data['Difference'] = cls_data[exps[0]] - cls_data[exps[1]]
        
        comparison_data.append(cls_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_csv_path = os.path.join(output_dir, 'model_comparison_ap50.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison data to {comparison_csv_path}")


# If no data was loaded, exit
if df_all.empty:
    print("No data was loaded. Check the file paths and experiment names.")
    exit(1)

print(f"Loaded data for experiments: {df_all['Experiment'].unique()}")
print(f"Total classes: {len(df_all['Class Name'].unique())}")

# ------------------ PLOTS ------------------

# 1. Heatmap: AP@0.5 per class across experiments
pivot_ap = df_all.pivot(index='Class Name', columns='Experiment', values='AP@0.5')
fig = plt.figure(figsize=(14, 20))
sns.heatmap(pivot_ap, annot=False, cmap=sns.color_palette(pastel_palette, as_cmap=True), linewidths=0.3)
plt.title('AP@0.5 Per Class Across Experiments')
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'heatmap_ap50_per_class'))
plt.close()

# 2. Selected class barplot
selected_classes = ['noteheadBlack', 'accidentalSharp', 'gClef', 'fClef', 'barline', 'restQuarter']
df_subset = df_all[df_all['Class Name'].isin(selected_classes)]

# Check if we have data for these classes
if not df_subset.empty:
    # Ensure we have all combinations for plotting
    common_classes = []
    for cls in selected_classes:
        if len(df_subset[df_subset['Class Name'] == cls]['Experiment'].unique()) == len(df_all['Experiment'].unique()):
            common_classes.append(cls)
    
    if common_classes:
        print(f"Common classes across all experiments: {common_classes}")
        df_subset = df_all[df_all['Class Name'].isin(common_classes)]
        
        # Increase figure height to accommodate legend below
        fig = plt.figure(figsize=(10, 8))
        ax = sns.barplot(data=df_subset, x='Class Name', y='AP@0.5', hue='Experiment')
        
        # Rotate labels and adjust font size if needed
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        plt.title('AP@0.5 for Selected Classes')
        
        # Move legend much further down and ensure it's below all labels
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
        
        # Add more bottom padding to accommodate the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        save_all_formats(fig, os.path.join(output_dir, 'ap50_selected_classes'))
        plt.close()
    else:
        print("No common classes found across experiments for selected class barplot")
else:
    print("No data available for selected class barplot")

# 3. Grouped AP@0.5 by symbol type
grouped_results = []
for group_name, classes in symbol_groups.items():
    for exp in df_all['Experiment'].unique():
        df_sub = df_all[(df_all['Class Name'].isin(classes)) & (df_all['Experiment'] == exp)]
        if not df_sub.empty:  # Only add if we have data for this group/experiment
            grouped_results.append({
                'Group': group_name,
                'Experiment': exp,
                'Mean AP@0.5': df_sub['AP@0.5'].mean()
            })

grouped_df = pd.DataFrame(grouped_results)
if not grouped_df.empty:
    # Increase figure height to accommodate legend below
    fig = plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=grouped_df, x='Group', y='Mean AP@0.5', hue='Experiment')
    
    # Rotate labels and adjust font size if needed
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.title('Mean AP@0.5 per Symbol Group')
    
    # Move legend much further down and ensure it's below all labels
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    
    # Add more bottom padding to accommodate the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    save_all_formats(fig, os.path.join(output_dir, 'grouped_ap50_by_symbol_group'))
    plt.close()
else:
    print("No data available for grouped symbol plot.")

# 4. Overall mean AP@0.5 comparison
overall_means = df_all.groupby('Experiment')['AP@0.5'].mean().reset_index()
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(data=overall_means, x='Experiment', y='AP@0.5')
plt.title('Mean AP@0.5 Across All Classes')
plt.xticks(rotation=45)
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'overall_mean_ap50'))
plt.close()

# 5. Extreme performance differences - where models differ most
if 'comparison_df' in locals() and len(df_all['Experiment'].unique()) > 1:
    exps = list(df_all['Experiment'].unique())
    
    # Top classes where YOLO does better than Faster R-CNN
    yolo_better = comparison_df.sort_values('Difference', ascending=False).head(10)
    if not yolo_better.empty:
        yolo_better_plot = pd.melt(yolo_better, 
                                   id_vars=['Class Name'], 
                                   value_vars=exps,
                                   var_name='Experiment', 
                                   value_name='AP@0.5')
        
        # Increase figure height to accommodate legend below
        fig = plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=yolo_better_plot, x='Class Name', y='AP@0.5', hue='Experiment')
        
        # Rotate labels and adjust font size if needed
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        plt.title(f'Top 10 Classes Where {exps[0]} Outperforms {exps[1]}')
        
        # Move legend much further down and ensure it's below all labels
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        
        # Add more bottom padding to accommodate the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        save_all_formats(fig, os.path.join(output_dir, 'top_classes_yolo_better'))
        plt.close()
    
    # Top classes where Faster R-CNN does better than YOLO
    frcnn_better = comparison_df.sort_values('Difference', ascending=True).head(10)
    if not frcnn_better.empty:
        frcnn_better_plot = pd.melt(frcnn_better, 
                                    id_vars=['Class Name'], 
                                    value_vars=exps,
                                    var_name='Experiment', 
                                    value_name='AP@0.5')
        
        # Increase figure height to accommodate legend below
        fig = plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=frcnn_better_plot, x='Class Name', y='AP@0.5', hue='Experiment')
        
        # Rotate labels and adjust font size if needed
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        plt.title(f'Top 10 Classes Where {exps[1]} Outperforms {exps[0]}')
        
        # Move legend much further down and ensure it's below all labels
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        
        # Add more bottom padding to accommodate the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        save_all_formats(fig, os.path.join(output_dir, 'top_classes_frcnn_better'))
        plt.close()
    
    # Highest absolute difference between models
    comparison_df['Abs_Difference'] = comparison_df['Difference'].abs()
    highest_diff = comparison_df.sort_values('Abs_Difference', ascending=False).head(10)
    if not highest_diff.empty:
        highest_diff_plot = pd.melt(highest_diff, 
                                   id_vars=['Class Name'], 
                                   value_vars=exps,
                                   var_name='Experiment', 
                                   value_name='AP@0.5')
        
        # Increase figure height to accommodate legend below
        fig = plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=highest_diff_plot, x='Class Name', y='AP@0.5', hue='Experiment')
        
        # Rotate labels and adjust font size if needed
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        plt.title('Top 10 Classes with Largest Performance Gap Between Models')
        
        # Move legend much further down and ensure it's below all labels
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        
        # Add more bottom padding to accommodate the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        save_all_formats(fig, os.path.join(output_dir, 'top_classes_largest_difference'))
        plt.close()

print(f"Visualizations saved to {output_dir}")