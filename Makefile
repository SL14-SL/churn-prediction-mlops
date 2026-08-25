include .env
export

# --- Configuration ---
.DEFAULT_GOAL := help
PYTHON_VERSION := 3.12.9
PREFECT_API_URL := http://localhost:4200/api
PREFECT_POOL ?= local-pool

.PHONY: help setup dev-up dev-down dev train train-force test lint clean \
        ui-prefect ui-mlflow prefect-status wait-prefect logs refresh-api \
        prefect-pool prefect-setup prefect-worker auto-retrain \
		predict-test demo-churn-lifecycle \
		train-bootstrap

# --- Main Entry Point ---

all: setup dev-up wait-prefect prefect-pool prefect-setup train-force test ## Run the complete pipeline
	@echo "✨ Full build successful! API, MLflow, and Prefect are running."

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Environment Setup ---

setup: ## Initialize local virtual environment using uv
	@echo "🚀 Initializing project with Python $(PYTHON_VERSION)..."
	uv venv --python $(PYTHON_VERSION) --allow-existing
	uv pip install -e .
	@echo "✅ Setup complete. Use 'source .venv/bin/activate' to start."

# --- Docker & Infrastructure ---

dev-up: ## Spin up the full stack (DB, MLflow, API, Prefect) in detached mode
	@echo "🐳 Starting container stack..."
	mkdir -p mlruns
	mkdir -p models
	mkdir -p results
	mkdir -p prefect_data
	mkdir -p data/raw/new_batches
	mkdir -p data/raw/quarantine
	mkdir -p data/features
	mkdir -p data/splits
	mkdir -p data/validation
	mkdir -p data/predictions/archive
	mkdir -p data/monitoring
	GIT_COMMIT_SHA=$$(git rev-parse HEAD 2>/dev/null || echo local) \
	UID=$$(id -u) \
	GID=$$(id -g) \
	docker compose up -d --build	
	@echo "✅ Services are live: API (8000), MLflow (5000), Prefect (4200), Grafana (3000), Prometheus(9090)"
	#@uv run --active prefect config set PREFECT_API_URL=$(PREFECT_API_URL)

dev-down: ## Stop all containers and remove networks
	@echo "🛑 Shutting down services..."
	docker compose down

dev: dev-up wait-prefect prefect-pool prefect-setup ## Start local stack and register Prefect deployment
	@echo "✅ Dev environment ready. Start worker with 'make prefect-worker'."

logs: ## Follow logs from the API service
	docker compose logs -f api

refresh-api: ## Restart or recreate API service using Docker Compose
	@echo "🔄 Refreshing API..."
	docker compose up -d api

# --- Prefect Specifics ---

prefect-status: ## Check if Prefect server is responding
	@echo "🔍 Checking Prefect server status..."
	@uv run --active prefect config view
	@curl -s http://localhost:4200/api/health || echo "⚠️ Prefect server is not reachable. Run 'make dev-up'."

wait-prefect: ## Wait until Prefect server is reachable
	@echo "⏳ Waiting for Prefect server (http://localhost:4200/api/health)..."
	@until curl -s http://localhost:4200/api/health > /dev/null; do \
		sleep 2; \
		echo "Prefect not ready yet..."; \
	done
	@echo "✅ Prefect is online!"

prefect-pool: wait-prefect ## Create local Prefect work pool if missing
	@echo "🏊 Ensuring Prefect work pool '$(PREFECT_POOL)' exists..."
	@uv run --active prefect work-pool inspect $(PREFECT_POOL) > /dev/null 2>&1 || \
		uv run --active prefect work-pool create --type process $(PREFECT_POOL)

prefect-setup: wait-prefect prefect-pool ## Register/update Prefect deployment for auto retraining
	@echo "🧭 Registering Prefect deployment..."
	@uv run --active python scripts/setup_prefect.py

prefect-worker: wait-prefect ## Start Prefect worker for local pool
	@echo "👷 Starting Prefect worker for pool '$(PREFECT_POOL)'..."
	uv run --active prefect worker start --pool $(PREFECT_POOL)

# --- UI Quicklinks ---

ui-prefect: ## Open Prefect UI in the browser
	@python3 -m webbrowser http://localhost:4200

ui-mlflow: ## Open MLflow UI in the browser
	@python3 -m webbrowser http://localhost:5000


COMPOSE_RUN_API=docker compose exec -T \
	-e APP_ENV=dev \
	-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	-e PREFECT_API_URL=http://prefect:4200/api \
	-e PREDICTION_API_URL=http://api:8080/predict \
	api

# --- ML Pipeline Tasks ---

train: wait-prefect ## Execute the training flow inside the API container
	@echo "🧠 Starting training flow inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py

train-force: wait-prefect ## Execute forced training flow inside the API container
	@echo "🧠 Starting forced training flow inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py --force

train-bootstrap: wait-prefect ## Create the initial Champion in an empty registry
	@echo "🌱 Bootstrapping initial Champion..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py --force --bootstrap

auto-retrain: wait-prefect ## Run auto retrain flow once manually inside the API container
	@echo "🤖 Running auto retrain flow once inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/auto_retrain_flow.py

predict-test:
	@echo "🧪 Sending test request..."
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-H "X-API-KEY: $(API_KEY)" \
		-d '{"context":{"request_id":"make-predict-test"},"inputs":[{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":"29.85"}]}' \
		| jq .

demo-churn-lifecycle:
	docker compose exec api uv run --no-sync python scripts/run_churn_demo.py --batch-size 50 --max-days 5 --label-delay-days 1

# --- Quality Assurance ---

test: ## Run unit and integration tests
	@echo "🧪 Running pytest suite..."
	uv run --active pytest tests/

lint: ## Check code style and quality (Ruff)
	@echo "✨ Linting code..."
	uv run --active ruff check .

# --- Cleanup ---

clean: ## Remove temporary files and caches
	@echo "🧹 Cleaning up Python caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✨ Workspace is clean."

clean-venv: ## Remove the virtual environment
	@echo "🗑️ Removing .venv..."
	rm -rf .venv

clean-data: ## Remove local data folders
	@echo "📂 Removing local runtime data folders..."
	rm -rf ./prefect_data ./mlruns ./models 
	@echo "✅ Runtime data folders removed."

clean-all: clean dev-down clean-venv clean-data ## Deep clean everything
	@echo "🐳 Pruning Docker system..."
	docker system prune -f
	docker volume prune -f
	@echo "🧼 Deep clean finished. System is fresh."

	
reset-demo: ## Reset generated artifacts and runtime state, keep only raw input data
	@echo "♻️ Resetting demo state (keeping data/raw)..."
	docker compose down -v
	rm -rf ./mlruns
	rm -rf ./mlruns_artifacts
	rm -rf ./models
	rm -rf ./data/features/*
	rm -rf ./data/splits/*
	rm -rf ./data/validation/*
	rm -rf ./data/predictions/*
	rm -rf ./data/monitoring/*
	rm -rf ./data/versioning/*
	rm -f ./data/raw/simulation_ground_truth.csv || true
	find ./data/raw/new_batches -mindepth 1 -delete
	find ./data/raw/quarantine -mindepth 1 -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f ./mlflow.db
	docker run --rm -v "$$(pwd):/workspace" alpine sh -c "rm -rf /workspace/prefect_data"
	@echo "✅ Demo state reset complete. Raw source data remains in data/raw/."