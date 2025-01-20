# Task 1: Named Entity Recognition for Mountain Names

This task involves training a Named Entity Recognition (NER) model to identify mountain names in text data. The model is based on a pre-trained BERT-based architecture and fine-tuned on a custom dataset of mountain names.

## Table of Contents
- [Project Structure](#project-structure)
- [Task Overview](#task-overview)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Running Inference](#2-running-inference)
  - [3. Demo](#3-demo)
- [Solution](#solution)
  - [1. Data](#1-data)
  - [2. Model Architecture](#2-model-architecture)
- [Requirements](#requirements)

## Project Structure

The directory structure for Task 1 is as follows:

```
task1/
├── data/
│   ├── preds/                # Folder for storing inference predictions
│   ├── test_labels/          # Labeled test data
│   ├── test_text/            # Test text data
│   ├── train/                # Training dataset
│   └── val/                  # Validation dataset
├── data.ipynb                # Jupyter notebook for dataset creation and preprocessing
├── demo.ipynb                # Jupyter notebook with model demo
├── inference.py              # Python script for running inference
├── model.py                  # Model definition and architecture
├── README.md                 # This README file
├── requirements.txt          # List of Python packages required
└── train.py                  # Python script for training the model
```

## Task Overview

In this task, you are required to:
1. **Dataset Creation:** Create or find a dataset containing labeled mountain names.
2. **Model Architecture:** Select a suitable architecture for NER, using BERT-based models as a recommendation.
3. **Model Training:** Fine-tune a pre-trained BERT model for NER on your dataset.
4. **Inference and Demo:** Prepare scripts and notebooks for running inference and demonstrating the model's performance.

### Expected Outputs

The following outputs are expected for this task:
- **Jupyter notebook** ([`data.ipynb`](data.ipynb)): This notebook explains the process of dataset creation and preprocessing.
- **Dataset**: Includes the data for training, validation, and testing.
- **Model Weights**: A link to the trained model weights.
- **Training Script** ([`train.py`](train.py)): Python script for model training.
- **Inference Script** ([`inference.py`](inference.py)): Python script for running inference on new data.
- **Demo Notebook** ([`demo.ipynb`](demo.ipynb)): Jupyter notebook showcasing the model's performance on sample text.

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/hnidan-oleksii/test-task-DL.git
```

### 2. Install Dependencies

Navigate to the `task1/` directory and install the required libraries:

```bash
cd task1
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

To train the model, run the `train.py` script:

```bash
python train.py --train_set_path <path_to_train_data> \
                --val_set_path <path_to_val_data> \
                --model_path <path_to_save_trained_model>
```

### 2. Running Inference

To run inference on new text data, use the `inference.py` script:

```bash
python inference.py --model_path <path_to_trained_model> \
                    --input_path <path_to_input_texts> \
                    --output_path <path_to_save_predictions>
```

The script will output predictions in JSON format:
```
[
  {
    "text": [
      ...each line separated into tokens...
    ],
    "labels": [
      ...labels for each token...
    ],
    "word_label": [
      ...word-label pairs...
    ]
  },
  ......
]
```

### 3. Demo

For a demonstration of the model, refer to the `demo.ipynb` notebook. This notebook provides an interactive walkthrough of running the model on example text data.

## Solution

### 1. Data

Data in this repository is from [this dataset](https://data.mendeley.com/datasets/cdcztymf4k/1). The **EWNERTC fine-grained version without noise reduction** was used (`EWNERTC_TC_Fine Grained NER_No_NoiseReduction.DUMP` file from the original archive).

To form the corpora for this task, only lines with mountain name references were kept. Furthermore, all labels except for mountain-related ones were replaced with the default `O`.

After that, the sentences were split into `train`, `validation`, and `test` files with 70% for `train` and 15% for both `validation` and `test` files. Additionally, `test` data was split into `test_texts` and `test_labels` files, where the former is for inference and the latter for checking results and assessing model accuracy.

Example of data from `train` and `validation` files:
```
B-MOUNTAIN O O O O O O O O O	Deomali is a mountain in Eastern Ghats , India .
```

### 2. Model Architecture

The model used in this task is based on the `DistilBERT` architecture, fine-tuned for token classification. The `DistilBertMountainTokenClassifier` class in `model.py` implements the model structure, training logic, and inference methods, as well as methods for loading and saving model weights.

The weights after fine-tuning are uploaded to [Google Drive](https://drive.google.com/drive/folders/1YDn20U2DcYph12MH3CC0IemkkkSEolct?usp=drive_link).

## Requirements

The project requires Python 3.12.8 and the following libraries:

- `datasets`
- `jupyter` (for running data and demo notebooks)
- `numpy`
- `tensorboard`
- `torch`
- `tqdm`
- `transformers`

For a complete list of dependencies, refer to `requirements.txt`.
