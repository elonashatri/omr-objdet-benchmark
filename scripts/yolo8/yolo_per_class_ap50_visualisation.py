import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ CONFIG ------------------

csv_files = {
    'train7': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train7/weights/train7-weights_per_class_results-1744580426.csv",
    'train3-yolo-9654': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train3-yolo-9654-data-splits/weights/train3-yolo-9654-data-splits-weights_per_class_results-1744580238.csv",
    'train-202-24classes-88epochs': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/train-202-24classes-yolo-9654-data-splits-weights_per_class_results-1744574857.csv",
    'train-202-24classes-pt2-160epochs': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/201-24-contuned-train-77epoch/weights/train-weights_per_class_results-1744802049.csv",
    'staffline_extreme-2': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/staffline_extreme-2/weights/staffline_extreme-2-weights_per_class_results-1744580288.csv",
    'staffline_extreme': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/staffline_extreme/weights/staffline_extreme-weights_per_class_results-1744580314.csv",
    'experiment-1-april': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/experiment-1-staffline-enhacment-april/weights/experiment-1-staffline-enhacment-april-weights_per_class_results-1744580343.csv",
    'train11': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train11/weights/train11-weights_per_class_results-1744580390.csv",
    'train8': "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train8/weights/train8-weights_per_class_results-1744580408.csv"
}

output_dir = "/homes/es314/omr-objdet-benchmark/scripts/yolo8/yolo_per_class_analysis"
os.makedirs(output_dir, exist_ok=True)

# Custom pastel color palette
pastel_palette = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']
sns.set_palette(pastel_palette)

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

def load_results(file_path, experiment_name):
    df = pd.read_csv(file_path)
    df['Experiment'] = experiment_name
    return df[['Class Name', 'AP@0.5', 'Experiment']]

def save_all_formats(fig, filepath_base):
    fig.savefig(f'{filepath_base}.png', dpi=300)
    fig.savefig(f'{filepath_base}.pdf')
    fig.savefig(f'{filepath_base}.svg')

# ------------------ LOAD DATA ------------------

dfs = [load_results(path, name) for name, path in csv_files.items()]
df_all = pd.concat(dfs)

# ------------------ PLOTS ------------------

# 1. Heatmap of AP@0.5
pivot_ap = df_all.pivot(index='Class Name', columns='Experiment', values='AP@0.5')
fig = plt.figure(figsize=(14, 20))
sns.heatmap(pivot_ap, annot=False, cmap=sns.color_palette(pastel_palette, as_cmap=True), linewidths=0.3)
plt.title('AP@0.5 Per Class Across Experiments')
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'heatmap_ap50_per_class'))
plt.close()

# 2. Selected Classes Barplot
selected_classes = ['noteheadBlack', 'accidentalSharp', 'gClef', 'fClef', 'barline', 'restQuarter']
df_subset = df_all[df_all['Class Name'].isin(selected_classes)]
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=df_subset, x='Class Name', y='AP@0.5', hue='Experiment')
plt.xticks(rotation=45)
plt.title('AP@0.5 for Selected Classes')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'ap50_selected_classes'))
plt.close()

# 3. Grouped AP@0.5 by Symbol Category
grouped_results = []
for group_name, classes in symbol_groups.items():
    for exp in df_all['Experiment'].unique():
        df_sub = df_all[(df_all['Class Name'].isin(classes)) & (df_all['Experiment'] == exp)]
        grouped_results.append({
            'Group': group_name,
            'Experiment': exp,
            'Mean AP@0.5': df_sub['AP@0.5'].mean()
        })

grouped_df = pd.DataFrame(grouped_results)
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Group', y='Mean AP@0.5', hue='Experiment')
plt.title('Mean AP@0.5 per Symbol Group')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
save_all_formats(fig, os.path.join(output_dir, 'grouped_ap50_by_symbol_group'))
plt.close()
