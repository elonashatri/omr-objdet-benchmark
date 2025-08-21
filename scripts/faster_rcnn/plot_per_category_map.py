import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set output directory
output_dir = '/homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/per_class_analysis_eval'
os.makedirs(output_dir, exist_ok=True)

def save_all_formats(fig, filename_base):
    fig.savefig(f"{filename_base}.png", dpi=300)
    fig.savefig(f"{filename_base}.svg")
    fig.savefig(f"{filename_base}.pdf")
    
# Define symbol groups
symbol_groups = {
    'clefs': ['gClef', 'fClef', 'cClef', 'gClef8vb', 'fClef8vb', 'unpitchedPercussionClef1'],
    'noteheads': ['noteheadBlack', 'noteheadHalf', 'noteheadWhole'],
    'rests': ['rest8th', 'rest16th', 'rest32nd', 'restQuarter', 'restHalf', 'restWhole', 'rest128th'],
    'flags': ['flag8thUp', 'flag8thDown', 'flag16thUp', 'flag16thDown', 'flag32ndUp'],
    'dynamics': ['dynamicPiano', 'dynamicForte', 'dynamicFF', 'dynamicPP', 'dynamicMF'],
    'accidentals': ['accidentalSharp', 'accidentalFlat', 'accidentalNatural', 'accidentalDoubleFlat'],
    'barlines': ['barline', 'systemicBarline'],
}

def load_csv(path, label):
    df = pd.read_csv(path)
    df['Experiment'] = label
    return df

# --- Add your file paths and labels here ---
experiments = {
        'full-no-staff':  "/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-no-staff-output/full-no-staff-output_per_class_results-1449.csv",
    'full+staff': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-with-staff-output/full-with-staff-output_per_class_results-1449.csv",
    'half': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/half-older_config_faster_rcnn_omr_output/half-older_config_faster_rcnn_omr_output_per_class_results-1449.csv",
    'half+staff': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/staff-half-older_config_faster_rcnn_omr_output/staff-half-older_config_faster_rcnn_omr_output_per_class_results-1449.csv"
}

# Load and merge
dfs = [load_csv(path, label) for label, path in experiments.items()]
df_all = pd.concat(dfs)

# Custom pastel color palette (from your image)
pastel_palette = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']
sns.set_palette(pastel_palette)

# --- 1. Heatmap of AP@0.5 ---
pivot_ap = df_all.pivot(index='Class Name', columns='Experiment', values='AP@0.5')
fig = plt.figure(figsize=(14, 20))
sns.heatmap(pivot_ap, annot=False, cmap=sns.color_palette(pastel_palette, as_cmap=True), linewidths=0.3)
plt.title('AP@0.5 Per Class Across Experiments')
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'heatmap_ap50_per_class'))
plt.close()


# --- 2. Selected classes (barplot) ---
selected_classes = ['noteheadBlack', 'accidentalSharp', 'gClef', 'fClef', 'barline', 'restQuarter']
df_subset = df_all[df_all['Class Name'].isin(selected_classes)]

# --- 2. Selected classes (barplot) ---
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=df_subset, x='Class Name', y='AP@0.5', hue='Experiment')
plt.xticks(rotation=45)
plt.title('AP@0.5 for Selected Classes')
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'ap50_selected_classes'))
plt.close()


# --- 3. Grouped metrics (mean AP@0.5 per group) ---
grouped_results = []
for group_name, classes in symbol_groups.items():
    for exp in df_all['Experiment'].unique():
        df_sub = df_all[(df_all['Class Name'].isin(classes)) & (df_all['Experiment'] == exp)]
        grouped_results.append({
            'Group': group_name,
            'Experiment': exp,
            'Mean AP@0.5': df_sub['AP@0.5'].mean(),
            'Mean F1': df_sub['F1 Score'].mean()
        })

grouped_df = pd.DataFrame(grouped_results)


# --- 3. Grouped metrics (mean AP@0.5 per group) ---
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Group', y='Mean AP@0.5', hue='Experiment')
plt.title('Mean AP@0.5 per Symbol Group')
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'grouped_ap50_by_symbol_group'))
plt.close()