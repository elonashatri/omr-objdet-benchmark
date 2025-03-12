# Staff Line Detection System for OMR

A comprehensive deep learning-based system for detecting staff lines in musical scores, specifically designed for Optical Music Recognition (OMR).

## System Overview

This staff line detection system uses a specialized neural network architecture with direction-aware convolutions and a staff line constraint module to leverage musical knowledge. The system includes:

1. **Preprocessing**: Converting XML annotations to binary masks
2. **Model Training**: Using a specialized U-Net architecture
3. **Evaluation**: Testing the model's performance
4. **Inference**: Detecting staff lines in new images

## Directory Structure

```
staffline_det_DL/
├── model.py                # Neural network architecture
├── data_loader.py          # Dataset handling
├── preprocessing.py        # XML to mask conversion
├── training_evaluation.py  # Training and evaluation
├── inference_.py           # Prediction and visualization
├── pipeline.py             # End-to-end pipeline
├── run.sh                  # Local execution script
└── slurm_job.sh            # HPC job submission script
```

## Key Features

- **Direction-Aware Convolutions**: Specialized for detecting horizontal staff lines
- **Staff Line Constraint Module**: Enforces musical knowledge about 5-line staff structures
- **Attention Mechanisms**: Help handle the long-range dependencies needed for staff lines
- **Custom Loss Function**: Penalizes horizontal discontinuities in staff lines
- **Post-Processing**: Enhances detection results using classical computer vision techniques

## Installation Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- Matplotlib
- tqdm
- Tensorboard

These are already available in your `omr_benchmark` conda environment.

## Running the System

### Option 1: Using the run.sh script

```bash
bash run.sh
```

### Option 2: Submit as a SLURM job

```bash
sbatch slurm_job.sh
```

### Option 3: Run the pipeline directly

```bash
python pipeline.py \
    --image_dir /path/to/images \
    --xml_dir /path/to/xml_annotations \
    --output_dir /path/to/output
```

## Command Line Arguments

### Data Paths
- `--image_dir`: Directory containing images (required)
- `--xml_dir`: Directory containing XML annotations (required)
- `--output_dir`: Directory to save outputs (default: ./staff_line_output)
- `--test_img_dir`: Directory for test images (optional)

### Preprocessing Options
- `--verify`: Verify dataset after preprocessing
- `--visualize`: Visualize sample images and masks
- `--num_vis_samples`: Number of samples to visualize (default: 5)

### Training Options
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--weights`: Path to pre-trained weights (optional)
- `--no_augment`: Disable data augmentation

### Inference Options
- `--no_post_process`: Disable post-processing in inference

### Hardware Options
- `--gpu_id`: GPU ID to use (default: 0)
- `--num_workers`: Number of data loading workers (default: 4)

### Pipeline Control
- `--skip_preprocess`: Skip preprocessing step
- `--skip_train`: Skip training step
- `--skip_eval`: Skip evaluation step
- `--skip_inference`: Skip inference step

## Example Usage

### Complete Pipeline

```bash
python pipeline.py \
    --image_dir /homes/es314/omr-objdet-benchmark/data/images \
    --xml_dir /homes/es314/omr-objdet-benchmark/data/annotations \
    --output_dir /homes/es314/omr-objdet-benchmark/staff_line_output \
    --verify \
    --visualize
```

### Only Preprocessing

```bash
python pipeline.py \
    --image_dir /homes/es314/omr-objdet-benchmark/data/images \
    --xml_dir /homes/es314/omr-objdet-benchmark/data/annotations \
    --output_dir /homes/es314/omr-objdet-benchmark/staff_line_output \
    --skip_train \
    --skip_eval \
    --skip_inference
```

### Only Inference (with existing model)

```bash
python pipeline.py \
    --image_dir /homes/es314/omr-objdet-benchmark/data/images \
    --xml_dir /homes/es314/omr-objdet-benchmark/data/annotations \
    --output_dir /homes/es314/omr-objdet-benchmark/staff_line_output \
    --weights /path/to/model.pth \
    --skip_preprocess \
    --skip_train \
    --skip_eval
```

## Output Structure

The system generates the following outputs:

```
staff_line_output/
├── preprocessed/
│   ├── masks/           # Binary masks from XML annotations
│   └── visualizations/  # Mask visualizations
├── models/
│   ├── best_model.pth   # Best model based on validation
│   └── final_model.pth  # Final trained model
├── logs/                # Tensorboard logs
└── results/
    ├── masks/           # Predicted staff line masks
    ├── visualizations/  # Overlay visualizations
    └── analysis/        # Staff line analysis text files
```

## How It Works

1. **Preprocessing**: Parses your XML annotations to create binary masks for staff lines
2. **Training**: Trains the specialized neural network on the created masks
3. **Evaluation**: Calculates metrics (Dice score, precision, recall)
4. **Inference**: Applies the model to detect staff lines in images
5. **Analysis**: Extracts staff line information and groups them into staff systems

## Customization

- Adjust the model architecture in `model.py`
- Modify data augmentation strategies in `data_loader.py`
- Change post-processing parameters in `inference_.py`

## Acknowledgements

This implementation draws inspiration from various computer vision techniques:
- U-Net for biomedical image segmentation
- Direction-aware convolutions from lane detection systems
- Attention mechanisms from modern deep learning architectures
- Musical score engraving conventions