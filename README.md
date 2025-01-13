# Leveraging Periodicity for Tabular Deep Learning

This repository contains the complete codebase and resources accompanying our paper **Leveraging Periodicity for Tabular Deep Learning**. The project explores the incorporation of periodic features into deep learning models for tabular data, demonstrating how leveraging periodicity can enhance model performance on various tasks.

The codebase supports experiments with both traditional machine learning models and novel periodicity-aware deep learning architectures. Experiments can be conducted on custom datasets located in the `dataset/` directory or on OpenML benchmarks, particularly those discussed in the paper [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/1906.01784)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Grid Search for Traditional Models](#running-grid-search-for-traditional-models)
  - [Evaluating Periodicity-Aware Models](#evaluating-periodicity-aware-models)
- [Configuration](#configuration)
- [Models Overview](#models-overview)
  - [Traditional Machine Learning Models](#traditional-machine-learning-models)
  - [Periodicity-Aware Deep Learning Models](#periodicity-aware-deep-learning-models)
- [Results and Logging](#results-and-logging)
- [Dataset Information](#dataset-information)
- [License](#license)
- [References](#references)

## Features

- **Grid Search Cross-Validation** for traditional machine learning models: Random Forest, ElasticNet, MLP, XGBoost, LightGBM, CatBoost.
- **Periodicity-Aware Deep Learning Models**: Implementations of FNet, OpNet, PNPNet, AutoPNPNet, and FTTransformer.
- **Automatic Periodicity Detection**: Includes a `PeriodicityDetector` to identify periodic and non-periodic features.
- **Flexible Data Handling**: Supports custom datasets and OpenML benchmarks.
- **Comprehensive Logging and Metrics**: Detailed experiment logs and performance metrics are stored for easy analysis.

## Installation

### Prerequisites

- Python 3.10
- [PyTorch](https://pytorch.org/) (version compatible with your CUDA version if using GPU)
- Other Python packages listed in `requirements.txt`

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/matteo-rizzo/brewery-ml.git
   cd brewery-ml
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the datasets:**

   - Place your custom datasets in the `dataset/` directory.
   - For OpenML benchmarks, no action is needed; datasets will be downloaded automatically.

## Docker Instructions

This guide provides step-by-step instructions to build and run the Docker container for the `brewery-ml` project.

### Prerequisites

Before building and running the Docker container, ensure the following are installed on your system:

- Docker ([Download and Install](https://docs.docker.com/get-docker/))
- (Optional) Docker Compose, if you plan to use it.

### Directory Structure

The project directory should follow this structure:

```
.
├── docker
│   └── Dockerfile
├── prod
│   └── run_inference.py
├── dataset
│   └── beer-fermentation.csv
├── models
│   └── tabautopnpnet.pth
├── requirements.txt
└── src
    └── classes
        └── ...
```

Ensure all required files, including the `Dockerfile`, datasets, and model files, are in their respective directories.

### Build the Docker Image

To build the Docker image, navigate to the project root directory and run the following command:

```bash
docker build -t brewery-ml -f docker/Dockerfile .
```

#### Explanation:
- `-t brewery-ml`: Tags the Docker image with the name `brewery-ml`.
- `-f docker/Dockerfile`: Specifies the path to the Dockerfile in the `docker` directory.
- `.`: Uses the current directory as the build context.

#### Common Issues:
If you encounter issues during the build (e.g., missing files or directories), ensure:
- All required directories (e.g., `models`, `dataset`) exist and contain the necessary files.
- The `.dockerignore` file does not exclude any required files or directories.

### Run the Docker Container

Once the image is built, you can run it using the following command:

```bash
docker run --rm -it brewery-ml
```

#### Explanation:
- `--rm`: Automatically removes the container once it stops.
- `-it`: Runs the container in interactive mode with a terminal.
- `brewery-ml`: The name of the image built in the previous step.

#### Running with Custom Arguments
If you need to pass arguments to the `run_inference.py` script (e.g., a custom config file):

```bash
docker run --rm -it brewery-ml --config-file custom_config.yaml --verbose
```

#### Mounting a Volume
To bind a local volume to the container (e.g., to store the output predictions locally):

```bash
docker run --rm -it -v $(pwd):/app brewery-ml --verbose
```

### Debugging and Testing Inside the Container

If you want to debug or explore the container environment, start a shell session:

```bash
docker run --rm -it brewery-ml bash
```

#### Verify Files Inside the Container
Once inside the container, verify the expected files are present:

```bash
ls /app/dataset
ls /app/models
```

### Clean Up

To remove unused Docker containers and images:

```bash
docker system prune -af
```

### Example Workflow

1. Build the Docker image:
   ```bash
   docker build -t brewery-ml -f docker/Dockerfile .
   ```

2. Run the Docker container with a specific configuration file:
   ```bash
   docker run --rm -it brewery-ml --config-file prod/config.yaml --verbose
   ```

3. Debug inside the container (optional):
   ```bash
   docker run --rm -it brewery-ml bash
   ```

## Usage

### Running Grid Search for Traditional Models

The `grid_search.py` script runs grid search cross-validation for traditional machine learning models.

**Example command:**

```bash
python grid_search.py --dataset beer-fermentation
```

**Available arguments:**

- `--dataset`: Specifies the dataset ID to use. Defaults to the `DATASET_ID` defined in `config.py`.

### Evaluating Periodicity-Aware Models

The `evaluate_periodicity_models.py` script evaluates periodicity-aware deep learning models.

**Example command:**

```bash
python evaluate_periodicity_models.py --model tabautopnpnet --dataset beer-fermentation
```

**Available arguments:**

- `--model`: Specifies the periodicity-aware model to use. Defaults to the `MODEL` defined in `config.py`.
- `--dataset`: Specifies the dataset ID to use. Defaults to the `DATASET_ID` defined in `config.py`.

**Note:** To use OpenML benchmarks, set `BENCHMARK = True` in `config.py` and specify the `SUITE_ID`.

## Configuration

The `config.py` file contains all the configurations for the experiments. Key parameters include:

- **Common Settings:**

  ```python
  BASE_LOG_DIR = "logs"
  RANDOM_SEED = 0
  NUM_FOLDS = 5
  TEST_SIZE = 0.1
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  DATASET_ID = "beer-fermentation"
  BENCHMARK = True
  ```

- **Benchmark Suite:**

  ```python
  SUITE_ID = 337  # OpenML suite (e.g., classification on numerical features)
  CLASSIFICATION = BENCHMARK and (SUITE_ID in [334, 337])  # Whether the dataset is for a classification task (defaults to regression for custom datasets) 
  ```

- **Periodicity Settings:**

  ```python
  MODEL = "tabautopnpnet"
  CAT_HIDDEN_SIZE = 256
  NUM_FOURIER_FEATURES = 100
  MAX_POLY_TERMS = 5
  POLY_TYPE = "chebyshev"  # 'chebyshev', 'legendre', 'hermite', 'laguerre'
  ```

- **Training Settings:**

  ```python
  EPOCHS = 1000
  LR = 0.05
  BATCH_SIZE = 64
  PATIENCE = 100
  ```

- **Grid Search Settings:**

  ```python
  CV_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
  MAX_ITERATIONS = 300
  ```

- **Reproducibility:**

  ```python
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  ```

## Models Overview

### Traditional Machine Learning Models

We provide grid search cross-validation for the following Sklearn models:

- **Random Forest (RF)**
- **ElasticNet**
- **Multi-Layer Perceptron (MLP)**
- **XGBoost**
- **LightGBM**
- **CatBoost**

These models are tuned using a comprehensive grid search over hyperparameters defined in `ModelConfigFactory.py`.

### Periodicity-Aware Deep Learning Models

Our proposed models are designed to handle periodic and non-periodic features effectively:

- **BASELINE FTTransformer:** A transformer-based model adapted for tabular data.
- **FourierNet:** Uses Fourier encoding to capture periodicity in features.
- **OrthogonalPolynomialNet:** Employs orthogonal polynomial encodings (Chebyshev polynomials) to model periodic patterns.
- **PNPNet:** Requires periodic and non-periodic features identified a priori using `PeriodicityDetector`. It processes features using separate Fourier and Chebyshev branches.
- **AutoPNPNet:** An extension of PNPNet that automatically detects periodic and non-periodic features using an integrated MLP.

**Model Implementation Details:**

- **Fourier-Based Models:** Implemented in `classes/periodicity/models/fourier/`.
- **Orthogonal Polynomial Models:** Implemented in `classes/periodicity/models/orthogonal_poly/`.
- **PNP and AutoPNP Models:** Implemented in `classes/periodicity/models/pnp/` and `classes/periodicity/models/autopnp/`.

## Results and Logging

Experiment results, metrics, and logs are stored in the directory specified by `BASE_LOG_DIR` in `config.py`. Each experiment creates a unique subdirectory named based on the dataset and model used.

**Visualization:**

- Use `Plotter.py` in `classes/utils/` to generate plots of training metrics.
- Metrics can be analyzed to compare model performance across different configurations.

## Dataset Information

- **Custom Datasets:**

  - Place your datasets in the `dataset/` directory.
  - Ensure datasets are properly formatted (e.g., CSV files with headers).
  - Add the dataset configuration at `dataset/config.json`

**Note:** Each dataset configuration must include;
- Dataset ID
- `n_num_features`: number of numerical features
- `cat_cols`: name of categorical columns
- `cat_cards`: cardinality of each categorical feature, as a list. All categorical features are one-hot encoded, thus it is sufficient to provide a list of `2`s as long as the number of categorical features in the dataset (this is required for FTTransformer)
- `target`: name of the target column 

- **OpenML Benchmarks:**
  
  - More info about the benchmark can be found in the official author's [repo](https://github.com/LeoGrin/tabular-benchmark).
  - Set `BENCHMARK = True` in `config.py`.
  - Specify `SUITE_ID` to select the benchmark suite.
  - Datasets are automatically downloaded and preprocessed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- **Our Paper:** *Leveraging Periodicity for Tabular Deep Learning* (Add link when available)
- **Benchmark Paper:** Prokhorenkova, L., & Gusev, G. (2019). *Why do tree-based models still outperform deep learning on typical tabular data?* arXiv preprint arXiv:1906.01784.
