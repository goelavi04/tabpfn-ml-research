# TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second


## Abstract

This repository presents a comprehensive implementation and empirical evaluation of TabPFN (Tabular Prior-Data Fitted Networks), a novel transformer-based approach for small tabular classification tasks. TabPFN leverages in-context learning to achieve competitive predictive performance without requiring traditional training or hyperparameter tuning at test time. This project includes both a research-oriented Google Colab implementation and an interactive Streamlit web application for practical deployment and comparative analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Overview](#algorithm-overview)
- [Implementation Architecture](#implementation-architecture)
- [Experimental Setup](#experimental-setup)
- [Results and Analysis](#results-and-analysis)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Introduction

### Motivation

Traditional machine learning approaches for tabular data require extensive hyperparameter tuning, cross-validation, and training time. AutoML systems can take hours to optimize a single model. This project explores TabPFN, a paradigm-shifting approach that performs inference in a single forward pass (~1 second) while maintaining competitive accuracy with state-of-the-art methods.

### Research Paper

- **Title:** TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- **Authors:** Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter
- **Institution:** University of Freiburg, Germany
- **Published:** International Conference on Learning Representations (ICLR) 2023
- **Paper Link:** [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
- **Official Repository:** [automl/TabPFN](https://github.com/automl/TabPFN)

### Project Objectives

1. Implement and validate the TabPFN algorithm for small-scale tabular classification
2. Conduct comparative analysis with traditional ML methods (XGBoost, LightGBM, Random Forest)
3. Evaluate GPU vs. CPU performance characteristics
4. Develop an interactive web interface for real-time experimentation
5. Document findings and provide reproducible research artifacts

## Theoretical Background

### Prior-Data Fitted Networks (PFNs)

PFNs represent a novel class of neural networks that approximate Bayesian inference by training on synthetic data generated from a prior distribution. Unlike traditional neural networks that learn task-specific patterns, PFNs learn to perform inference itself.

**Key Concept:** Instead of training on the target dataset, the model is pre-trained on millions of synthetic classification problems. At test time, the model receives both training and test examples as input and outputs predictions through a single forward pass.

### In-Context Learning

TabPFN employs in-context learning, a mechanism inspired by large language models:

1. **Training Phase (Offline):** Model is trained once on synthetic datasets generated using Structural Causal Models (SCMs)
2. **Inference Phase (Online):** Model receives training examples and test queries simultaneously, producing predictions without gradient updates

This approach eliminates the need for:
- Iterative training on the target dataset
- Hyperparameter optimization
- Cross-validation
- Model selection

### Structural Causal Models (SCMs)

The synthetic training data is generated using SCMs, which encode causal relationships between features and labels. This prior incorporates:

- **Occam's Razor:** Preference for simpler explanations
- **Causal Structure:** Feature dependencies and confounders
- **Bayesian Principles:** Uncertainty quantification through posterior distributions

### Architecture

TabPFN employs a transformer encoder architecture with:
- **12 Transformer Layers**
- **512 Embedding Dimensions**
- **8 Attention Heads**
- **25.82M Parameters**

The architecture processes variable-length training sets and test queries jointly, using self-attention for training examples and cross-attention for test predictions.

## Algorithm Overview

### TabPFN Workflow

```
Input: D_train = {(x₁, y₁), ..., (xₙ, yₙ)}, x_test
Output: p(y_test | x_test, D_train)

1. Encode training examples: E_train = Embed(D_train)
2. Encode test query: E_test = Embed(x_test)
3. Concatenate: E = [E_train; E_test]
4. Apply transformer layers: H = Transformer(E)
5. Extract test representation: h_test = H[-1]
6. Predict: p(y_test | x_test, D_train) = Softmax(Linear(h_test))
```

### Computational Complexity

- **Training (one-time):** O(M × N² × D) where M = synthetic datasets, N = max samples, D = dimensions
- **Inference:** O(N² × D) - single forward pass, quadratic in training set size
- **Memory:** O(N × D) - stores training examples in context

### Constraints and Applicability

TabPFN achieves optimal performance under specific constraints:

| Constraint | Maximum Value | Rationale |
|------------|---------------|-----------|
| Training Samples | ≤ 1,000 | Transformer context length limitations |
| Features | ≤ 100 | Computational efficiency and generalization |
| Classes | ≤ 10 | Multi-class softmax stability |
| Feature Type | Numerical only | Pre-training distribution assumptions |
| Missing Values | None | Requires complete feature matrices |

These constraints align with numerous real-world scenarios in medical diagnosis, materials science, and experimental design where sample collection is expensive.

## Implementation Architecture

### System Components

This implementation consists of two primary components:

#### 1. Google Colab Notebook (`colab_notebook.py`)

**Purpose:** Research-oriented implementation with comprehensive evaluation

**Features:**
- Multi-dataset evaluation pipeline
- Statistical analysis and hypothesis testing
- Publication-quality visualizations
- GPU acceleration benchmarking
- Confusion matrix generation
- Classification reports

**Target Users:** Researchers, ML practitioners conducting experiments

#### 2. Streamlit Web Application (`app.py`)

**Purpose:** Interactive deployment for real-time experimentation

**Features:**
- Dynamic device selection (GPU/CPU)
- Multi-dataset configuration
- Real-time performance metrics
- Interactive visualizations
- CSV export functionality
- Responsive UI with configuration sidebar

**Target Users:** End-users, demonstrations, educational purposes

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Deep Learning | PyTorch | 2.0+ | Model inference and GPU acceleration |
| TabPFN | tabpfn | Latest | Pre-trained TabPFN model |
| Web Framework | Streamlit | 1.28+ | Interactive UI |
| ML Libraries | scikit-learn | 1.3+ | Preprocessing, metrics, baselines |
| Boosting | XGBoost | 2.0+ | Gradient boosting baseline |
| Boosting | LightGBM | 4.0+ | Light gradient boosting baseline |
| Visualization | Matplotlib | 3.7+ | Static plots |
| Visualization | Seaborn | 0.12+ | Statistical graphics |
| Data Processing | Pandas | 2.0+ | Data manipulation |
| Numerical | NumPy | 1.24+ | Array operations |

## Experimental Setup

### Datasets

All datasets satisfy TabPFN's operational constraints and represent diverse classification scenarios:

| Dataset | Samples | Features | Classes | Type | Source |
|---------|---------|----------|---------|------|--------|
| **Iris** | 150 | 4 | 3 | Multi-class | UCI ML Repository |
| **Wine** | 178 | 13 | 3 | Multi-class | UCI ML Repository |
| **Breast Cancer** | 569 | 30 | 2 | Binary | UCI ML Repository |
| **Digits** | 500* | 64 | 10 | Multi-class | scikit-learn |

*Subset used for computational efficiency

### Baseline Models

Comparative analysis includes established ML algorithms:

1. **XGBoost:** State-of-the-art gradient boosting (default hyperparameters)
2. **LightGBM:** Efficient gradient boosting framework
3. **Random Forest:** Ensemble of decision trees (100 estimators)
4. **Logistic Regression:** Linear baseline with L2 regularization
5. **K-Nearest Neighbors:** Instance-based learning (k=5)

### Evaluation Metrics

- **Accuracy:** Primary metric for balanced datasets
- **ROC-AUC:** Area under ROC curve (binary: single value, multi-class: OvR macro-average)
- **Inference Time:** Total time (training + prediction) in seconds
- **GPU Speedup:** Ratio of CPU time to GPU time for TabPFN

### Experimental Protocol

```python
# Standardized evaluation pipeline
for dataset in [Iris, Wine, BreastCancer, Digits]:
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_train, X_test = StandardScaler().fit_transform(X_train, X_test)
    
    for model in [TabPFN_GPU, TabPFN_CPU, XGBoost, LightGBM, ...]:
        t_start = time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        t_elapsed = time() - t_start
        
        record_metrics(accuracy, roc_auc, t_elapsed)
```

## Results and Analysis

### Application Interface

The Streamlit application provides an intuitive interface for experimentation:

![About Tab - Research Paper Information](screenshots/screenshot1.png)
*Figure 1: About tab displaying research paper details, key innovations, and TabPFN constraints*

![Run Experiment Tab - Configuration and Execution](screenshots/screenshot2.png)
*Figure 2: Run Experiment tab showing device selection, dataset configuration, and experiment execution with real-time feedback*

![Results Tab - Performance Metrics](screenshots/screenshot3.png)
*Figure 3: Results tab presenting detailed performance metrics, summary statistics, and comparative analysis across models*

### Key Findings

#### 1. Predictive Performance

**Summary Statistics:**
- **Total Experiments:** 6
- **Best Model (Accuracy):** TabPFN (CPU) - 0.9789
- **Fastest Model:** XGBoost - 0.1019 seconds
- **Average Accuracy:** 0.9725

**Detailed Results:**

| Dataset | Model | Accuracy | ROC-AUC | Time (s) |
|---------|-------|----------|---------|----------|
| Iris | TabPFN (CPU) | 0.9778 | 0.9985 | 10.5882 |
| Iris | XGBoost | 0.9333 | 0.9770 | 0.1150 |
| Wine | TabPFN (CPU) | 1.0000 | 1.0000 | 5.0018 |
| Wine | XGBoost | 1.0000 | 1.0000 | 0.1015 |
| Breast Cancer | TabPFN (CPU) | 0.9591 | 0.9965 | 21.1375 |
| Breast Cancer | XGBoost | 0.9649 | 0.9963 | 0.0892 |

**Average Performance by Model:**

| Model | Accuracy | ROC-AUC | Time (s) |
|-------|----------|---------|----------|
| TabPFN (CPU) | 0.9789 | 0.9983 | 12.2425 |
| XGBoost | 0.9661 | 0.9911 | 0.1019 |

#### 2. Inference Time Analysis

**Observations:**
- TabPFN: 12.24 seconds average (includes forward pass only)
- XGBoost: 0.10 seconds average (includes training + prediction)
- **Trade-off:** TabPFN sacrifices speed for zero-hyperparameter-tuning advantage

**Context:** The paper reports competitive timing against AutoML systems with 1-hour tuning budgets, not against single-run traditional ML methods.

#### 3. GPU vs. CPU Performance

**Note:** Current implementation runs on CPU due to hardware constraints. GPU acceleration expected to provide:
- 5-10× speedup for inference
- Reduced memory footprint
- Parallel batch processing capabilities

#### 4. Model Selection Insights

**When to use TabPFN:**
- Small datasets (< 1,000 samples)
- Numerical features only
- No time for hyperparameter tuning
- Need for uncertainty quantification
- Multiple similar classification tasks

**When to use Traditional ML:**
- Large datasets (> 1,000 samples)
- Mixed feature types (categorical + numerical)
- Production systems requiring millisecond latency
- Hardware without GPU acceleration

## Installation and Usage

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for acceleration

### Installation

#### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/goelavi04/tabpfn-ml-research.git
cd tabpfn-ml-research

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n tabpfn python=3.10
conda activate tabpfn

# Install PyTorch (with CUDA support)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install streamlit tabpfn xgboost lightgbm scikit-learn matplotlib seaborn
```

### Running the Streamlit Application

```bash
# Ensure virtual environment is activated
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Running the Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `colab_notebook.py` or create a new notebook
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4)
4. Run all cells sequentially

### Application Usage Workflow

1. **Configure Device:** Select GPU or CPU in the sidebar
2. **Select Datasets:** Choose from Iris, Wine, Breast Cancer, or Digits
3. **Choose Models:** Select models for comparison (TabPFN, XGBoost, LightGBM, etc.)
4. **Set Parameters:** Adjust test set size (20-50%) and random seed
5. **Run Experiment:** Click "Run Experiment" button
6. **Analyze Results:** Navigate to Results and Visualizations tabs
7. **Export Data:** Download results as CSV for further analysis

## Project Structure

```
tabpfn-ml-research/
│
├── app.py                      # Streamlit web application
├── colab_notebook.py           # Google Colab implementation
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── screenshots/               # Application screenshots
│   ├── screenshot1.png        # About tab
│   ├── screenshot2.png        # Run experiment tab
│   └── screenshot3.png        # Results tab
│
├── venv/                      # Virtual environment (not tracked)
│
└── results/                   # Experimental results (generated)
    └── experiment_results.csv
```

### File Descriptions

- **`app.py`:** Main Streamlit application with interactive UI, device selection, and real-time visualizations
- **`colab_notebook.py`:** Research implementation with comprehensive evaluation, statistical analysis, and publication-quality plots
- **`requirements.txt`:** Complete list of Python package dependencies with version specifications
- **`.gitignore`:** Specifies intentionally untracked files (venv/, `__pycache__/`, etc.)

## Future Work

### Planned Enhancements

1. **Extended Benchmarking:**
   - Include CatBoost, Neural Networks, SVM
   - Test on additional UCI ML Repository datasets
   - Evaluate on imbalanced classification scenarios

2. **Feature Engineering:**
   - Automated feature selection integration
   - Handling of categorical variables (one-hot encoding pipeline)
   - Missing value imputation strategies

3. **Scalability:**
   - Implement data streaming for larger datasets
   - Distributed inference across multiple GPUs
   - Memory-efficient batching strategies

4. **Advanced Analysis:**
   - SHAP value integration for model interpretability
   - Uncertainty quantification visualization
   - Statistical significance testing (Wilcoxon signed-rank)

5. **Production Deployment:**
   - REST API development using FastAPI
   - Docker containerization
   - Cloud deployment (AWS/GCP/Azure)

### Open Research Questions

1. Can transfer learning improve TabPFN's performance on domain-specific tasks?
2. How does TabPFN perform with engineered features vs. raw features?
3. What is the optimal training set size for maximum accuracy vs. speed trade-off?
4. Can TabPFN be extended to regression tasks?

## References

### Primary Reference

```bibtex
@inproceedings{hollmann2023tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://arxiv.org/abs/2207.01848}
}
```

### Additional References

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
4. Müller, S., et al. (2021). Transformers can do Bayesian inference. ICLR.

### Datasets

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences.
- scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

## Acknowledgments

### Research Community

- **AutoML Research Group, University of Freiburg:** For developing and open-sourcing TabPFN
- **Streamlit Team:** For providing an excellent framework for rapid ML application development
- **PyTorch Team:** For the foundational deep learning framework

### Educational Support

This project was developed as part of the ML Lab Mini Project curriculum. I would like to thank:

- **Course Instructor:** For guidance on research paper implementation methodologies
- **Teaching Assistants:** For technical support and code review feedback
- **Peer Reviewers:** For constructive feedback on experimental design

### Open Source

This project leverages numerous open-source libraries. Special thanks to the maintainers and contributors of:
- PyTorch, NumPy, pandas, scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, Streamlit

tails.

### Usage Terms

- **Academic Use:** Free for research and educational purposes with proper citation
- **Commercial Use:** Allowed under Apache 2.0 terms
- **Modification:** Permitted with attribution to original authors
- **Distribution:** Allowed with license inclusion





