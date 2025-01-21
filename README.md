# Deep Learning Test Tasks

This repository contains solutions for two deep learning tasks:

1. **Named Entity Recognition (NER)**: A model for identifying mountain names in text data using BERT-based architecture.
2. **Satellite Image Matching**: An algorithm for matching Sentinel-2 satellite images across different seasons using LightGlue and DISK.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tasks](#tasks)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project demonstrates implementations of:
- Text processing using transformer-based model for named entity recognition
- Computer vision solution for satellite image feature matching

## Getting Started

### Prerequisites

- Python 3.12.8 ([download](https://www.python.org/downloads/release/python-3128/))
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hnidan-oleksii/test-task-DL.git
   cd test-task-DL
   ```

2. Each task has its own environment and setup instructions. Please refer to the respective README.md files in each task directory:
- [Task 1: Mountain Names NER](task1/README.md)
- [Task 2: Satellite Image Matching](task2/README.md)

## Repository Structure

```
test-task-DL/
├── task1/                     # Named Entity Recognition
│   ├── README.md              # Task 1 specific instructions
│   ├── requirements.txt       # Task 1 dependencies
│   ├── data.ipynb             # Data preprocessing
│   ├── model.py               # NER model implementation
│   ├── inference.py           # Inference script
│   └── demo.ipynb             # Interactive demo
│
├── task2/                     # Satellite Image Matching
│   ├── README.md              # Task 2 specific instructions
│   ├── requirements.txt       # Task 2 dependencies
│   ├── image_processing.ipynb # Image preprocessing
│   ├── matcher.py             # Matching algorithm
│   ├── inference.py           # Inference script
│   ├── demo.ipynb             # Interactive demo
│   └── example.jpg            # Example of image matching
│
├── improvements.pdf           # Report with suggested improvements
└── README.md                  # Main repository documentation
```

## Tasks

### Task 1: Mountain Names NER
- Fine-tuned DistilBERT model for mountain name recognition
- Custom dataset based on EWNERTC
- Detailed documentation in [task1/README.md](task1/README.md)

### Task 2: Satellite Image Matching
- LightGlue + DISK approach
- Handles seasonal variations in Sentinel-2 imagery
- Detailed documentation in [task2/README.md](task2/README.md)

## Requirements

Each task has its own set of dependencies specified in their respective `requirements.txt` files. The core requirements are:
- Python 3.12.8
- PyTorch
- Transformers (Task 1)
- Kornia (Task 2)

For specific dependencies, please refer to:
- [Task 1 Requirements](task1/requirements.txt)
- [Task 2 Requirements](task2/requirements.txt)

## License

This project is licensed under the MIT License - see the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
