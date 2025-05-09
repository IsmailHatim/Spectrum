# Spectrum xAI: Explainability Pipeline for Deep Learning Models

![Pipeline Overview](https://github.com/IsmailHatim/Spectrum/blob/master/data/figures/pipeline.jpg)

## Overview

Spectrum xAI is a comprehensive pipeline for explainability in artificial intelligence. It provides a suite of state-of-the-art explainability methods to analyze and interpret deep learning models. The pipeline supports evaluation metrics such as IoU, F1-Score, AUC, and execution time, along with visualizations for better understanding of model predictions.

## Features

- **Explainability Methods**:
  - Grad-CAM
  - Gradient SHAP
  - Integrated Gradients
  - LIME
  - Saliency Map
  - Score-CAM

- **Evaluation Metrics**:
  - Intersection over Union (IoU)
  - F1-Score
  - Area Under the Curve (AUC)
  - Execution Time

- **Supported Model**:
  - DenseNet (fully implemented)

- **Visualization**:
  - Heatmaps and thresholded masks for each explainability method.
  - Ground truth comparison for qualitative evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IsmailHatim/Spectrum
   cd Spectrum
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the DAGM dataset in the `data/dataset/` directory. You can download it from the [Kaggle Challenge](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)

## Usage

### Training a Model

To train a model (e.g., DenseNet121) on the first class of the DAGM dataset:
```bash
python run_training.py --model_name densenet121 --epochs 10 --batch_size 32 --learning_rate 0.0001 --plot
```

### Running Explainability Methods

To run an explainability method (e.g., Grad-CAM) on a trained model and save figures:
```bash
python run_explanation.py --method gradcam --index 0 --threshold 0.5 --model_name densenet121 --conv_layer_index -2 --save
```
### Running Evaluation

To run the whole evluation using a specific method (e.g., Grad-CAM) on a trained model:
```bash
python run_evaluation.py --method gradcam --threshold 0.5 --model_name densenet121 --conv_layer_index -2
```

## Pipeline Overview

1. **Data Loading**:
   - The pipeline uses the DAGM dataset, which is divided into training and testing splits for each class.

2. **Model Training**:
   - Train models like DenseNet121, on the dataset.
   - Save trained models for later use.

3. **Explainability Methods**:
   - Apply explainability methods to generate heatmaps and thresholded masks for model predictions.

4. **Evaluation**:
   - Evaluate the explainability methods using IoU, F1-Score, AUC, and execution time.

5. **Visualization**:
   - Visualize the input image, heatmap, thresholded mask, and ground truth for qualitative analysis.

## Example Workflow

1. Train a DenseNet121 model on Class 1 of the DAGM dataset:
   ```bash
   python run_training.py --model_name densenet121 --epochs 10 --batch_size 32 --learning_rate 0.0001 --plot
   ```

2. Run Grad-CAM on the trained model:
   ```bash
   python run_explanation.py --method gradcam --index 0 --threshold 0.5 --model_name densenet121 --conv_layer_index -2 --save
   ```

3. Run evluation using Grad-CAM on the whole test set:
   ```bash
   python run_evaluation.py --method gradcam --threshold 0.5 --model_name densenet121 --conv_layer_index -2
   ```

4. Visualize the results:
   - Input image
   - Grad-CAM heatmap
   - Thresholded mask
   - Ground truth
   - Mean and Standard Deviation metrics

## Future Work

- **Explainability Methods**:
  - Extend the pipeline with additional explainability methods and other models.

- **Dataset Support**:
  - Generalize the pipeline to work with other datasets.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Captum](https://captum.ai/)
- [LIME](https://github.com/marcotcr/lime)
- [DAGM Dataset](https://hci.iwr.uni-heidelberg.de/node/3616)
