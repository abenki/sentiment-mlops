# Makefile
.PHONY: help install train api docker-build docker-up docker-down docker-logs test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make train        - Train model with sample data"
	@echo "  make train-full   - Train model with full dataset"
	@echo "  make api          - Run API locally"
	@echo "  make mlflow       - Run MLflow UI"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start Docker Compose stack"
	@echo "  make docker-down  - Stop Docker Compose stack"
	@echo "  make docker-logs  - Show Docker logs"
	@echo "  make test         - Test API endpoints"
	@echo "  make clean        - Clean generated files"

install:
	uv sync
	uv pip install -e .

train:
	uv run python src/train/train.py --sample-size 5000 --run-name "quick-train"

train-full:
	uv run python src/train/train.py --run-name "full-dataset"

api:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	uv run mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "Services started!"
	@echo "  - API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart

test:
	@echo "Testing API endpoints..."
	curl -s http://localhost:8000/health | jq
	@echo "\nTesting prediction..."
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "This movie was fantastic!"}' | jq

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache

clean-all: clean
	rm -rf .venv
	rm -rf mlruns
	rm -rf mlartifacts
	docker compose down -v
