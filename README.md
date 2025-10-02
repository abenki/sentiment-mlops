# Sentiment Analysis MLOps Project

A complete MLOps pipeline for sentiment analysis.

## Tech Stack

- **Package Management**: uv
- **ML Framework**: scikit-learn / Transformers
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Containerization**: Docker
- **IaC**: Terraform
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

## Project Structure
```
sentiment-mlops/
├── data/             # Data storage
├── src/
│   ├── train/        # Training scripts
│   ├── api/          # FastAPI application
│   └── monitoring/   # Monitoring utilities
├── notebooks/        # Exploratory notebooks
├── tests/            # Unit and integration tests
├── terraform/        # Infrastructure as Code
├── docker/           # Dockerfiles
└── .github/          # CI/CD workflows
```

## Quick Start
```bash
# Install uv (if not already done)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
git clone git@github.com:abenki/sentiment-mlops.git
cd sentiment-mlops
uv sync

# Run training
uv run python src/train/train.py

# Start API
uv run uvicorn src.api.main:app --reload
````
