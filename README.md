# Sentiment Analysis MLOps Project

This project implements an MLOps pipeline for sentiment analysis of movie reviews (IMDB dataset). My goal is to implement an ML project following good MLOps practices. It features:

- **Experiment tracking** with MLflow
- **Model versioning** and registry
- **High-performance REST API** with FastAPI
- **Containerization** with Docker
- **Monitoring** with Prometheus and Grafana
- **Infrastructure as Code** ready for cloud deployment
- **Modern dependency management** with uv

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Monitoring](#monitoring)
- [Development](#development)


## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────┐
│  FastAPI Application (Port 8000)    │
│  - /predict (single & batch)        │
│  - /health                          │
│  - /metrics (Prometheus)            │
└──────┬──────────────────────────────┘
       │
       ├─→ Prometheus (Port 9090) ─→ Grafana (Port 3000)
       │
       └─→ MLflow Server (Port 5000)
```

### Workflow

1. **Training**: Model is trained with MLflow tracking
2. **Versioning**: Model is versioned in MLflow Model Registry
3. **Deployment**: API loads the model and serves predictions
4. **Monitoring**: Prometheus collects metrics, Grafana visualizes them


## Project Structure

```
sentiment-mlops/
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment_model.py       # Model class
│   ├── train/
│   │   ├── __init__.py
│   │   └── train.py                 # Training script with MLflow
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  # FastAPI application
│   └── monitoring/
│       └── __init__.py
│
├── models/
│   ├── sentiment_model.joblib       # Trained model
│   ├── vectorizer.joblib            # TF-IDF vectorizer
│   ├── sample_predictions.csv       # Sample predictions
│   └── important_features.json      # Important features
│
├── data/
│   ├── raw/                         # Raw data (gitignored)
│   └── processed/                   # Processed data
│
├── notebooks/
│   └── 01_data_exploration.ipynb    # Dataset exploration
│
├── monitoring/
│   ├── prometheus.yml               # Prometheus configuration
│   └── grafana/
│       ├── datasources/
│       │   └── prometheus.yml       # Prometheus datasource
│       └── dashboards/
│           └── dashboard.yml        # Dashboard configuration
│
├── tests/                           # Unit and integration tests
│   └── __init__.py
│
├── mlruns/                          # MLflow data (gitignored)
├── mlartifacts/                     # MLflow artifacts (gitignored)
│
├── Dockerfile                       # API Docker image
├── docker-compose.yml               # Multi-service stack
├── .dockerignore                    # Files excluded from Docker build
│
├── pyproject.toml                   # Project configuration (uv)
├── uv.lock                          # Dependency lockfile
├── Makefile                         # Useful commands
│
├── .gitignore
└── README.md
```


## Installation

### Prerequisites

- **Python 3.13**
- **Docker** and Docker Compose
- **uv**

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 2. Clone the project

```bash
git clone git@github.com:abenki/sentiment-mlops.git
cd sentiment-mlops
```

### 3. Install dependencies

```bash
# Sync virtual environment
uv sync

# Install package in editable mode
uv pip install -e .
```


## Usage

### Option 1: With Docker

#### 1. Train a model (once)

```bash
make train
```

#### 2. Launch the complete stack

```bash
# Build images
make docker-build

# Start all services
make docker-up
```

**Available services:**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

#### 3. Test

```bash
# Automated tests
make test

# View logs
make docker-logs

# Logs for specific service
docker logs -f sentiment-api
```

#### 4. Stop services

```bash
make docker-down
```

### Option 2: Local Development

#### 1. Train a model

```bash
# Quick training with sample (5000 samples, ~2 seconds)
make train

# Or with custom parameters
uv run python src/train/train.py \
    --sample-size 5000 \
    --max-features 10000 \
    --C 0.5 \
    --run-name "baseline-v1"

# Full training (25000 samples, ~2-3 minutes)
make train-full
```

**What happens:**
- Model is trained with specified hyperparameters
- All metrics are tracked in MLflow
- Model is saved in `models/`
- Model is versioned in MLflow Model Registry

#### 2. View results in MLflow

```bash
# Start MLflow UI
make mlflow

# Or directly
uv run mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 to:
- Compare different runs
- Visualize metrics
- Explore artifacts
- Manage model versions

#### 3. Run the API locally

```bash
# Start API in development mode
make api

# Or directly
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great film!", "Terrible movie."]}'

# Interactive documentation
open http://localhost:8000/docs
```


## Monitoring

### Available Prometheus Metrics

The API exposes custom metrics for monitoring:

#### Counters
- **`sentiment_predictions_total{sentiment="positive|negative"}`**: Total predictions by sentiment
- **`sentiment_errors_total{error_type="..."}`**: Total errors by type

#### Histograms
- **`sentiment_prediction_latency_seconds`**: Prediction latency (buckets: 0.01s to 5s)
- **`sentiment_prediction_confidence`**: Confidence score distribution

#### Gauges
- **`sentiment_model_load_time_seconds`**: Model loading time

### Accessing Metrics

```bash
# Raw metrics
curl http://localhost:8000/metrics

# Prometheus UI
open http://localhost:9090

# Grafana dashboards
open http://localhost:3000
```

### Example Prometheus Queries

```promql
# Prediction rate per minute
rate(sentiment_predictions_total[1m])

# Average latency (P50)
histogram_quantile(0.5, rate(sentiment_prediction_latency_seconds_bucket[5m]))

# P95 latency
histogram_quantile(0.95, rate(sentiment_prediction_latency_seconds_bucket[5m]))

# Error rate
rate(sentiment_errors_total[5m])
```


## Development

### Useful Commands (Makefile)

```bash
make help           # Show all available commands
make install        # Install dependencies
make train          # Train with sample
make train-full     # Train with full dataset
make api            # Run API locally
make mlflow         # Run MLflow UI
make docker-build   # Build Docker images
make docker-up      # Start Docker stack
make docker-down    # Stop Docker stack
make docker-logs    # View logs
make test           # Test API
make clean          # Clean generated files
```

### Experimenting with Different Models

```bash
# Vary number of features
uv run python src/train/train.py --max-features 5000 --run-name "5k-features"
uv run python src/train/train.py --max-features 10000 --run-name "10k-features"

# Vary regularization
uv run python src/train/train.py --C 0.1 --run-name "high-reg"
uv run python src/train/train.py --C 10.0 --run-name "low-reg"

# Compare in MLflow UI
make mlflow
```
