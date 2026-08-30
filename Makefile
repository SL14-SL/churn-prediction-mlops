include .env
export

# --- Configuration ---
.DEFAULT_GOAL := help
PYTHON_VERSION := 3.12.9

LOCAL_MLFLOW_TRACKING_URI := http://localhost:5000
LOCAL_PREFECT_API_URL := http://localhost:4200/api
PREFECT_POOL ?= local-pool
PREFECT_PROJECT_DIR ?= $(CURDIR)

.PHONY: help setup all dev dev-up dev-down logs refresh-api \
	ui-prefect ui-mlflow prefect-status wait-prefect prefect-pool \
	prefect-setup prefect-worker train train-force train-bootstrap \
	auto-retrain predict-test demo-churn-lifecycle reset-local-stack \
	check-prod-env debug-prod-env prepare-mlflow-prod-demo upload-raw-prod \
	train-bootstrap-prod train-force-prod verify-prod bootstrap-and-verify-prod \
	test lint clean clean-venv clean-data clean-all reset-demo \
	ui-dashboard churn-retraining-comparison churn-retraining-comparison-smoke \
	churn-cohort-shift-comparison-plot churn-concept-drift-comparison \
	churn-concept-drift-comparison-smoke churn-concept-drift-comparison-plot \
	churn-retraining-comparison-plot

# --- Main Entry Point ---

all: setup dev-up wait-prefect prefect-pool prefect-setup train-bootstrap test ## Run the complete local bootstrap pipeline
	@echo "✨ Full build successful! API, MLflow and Prefect are running."

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

# --- Environment Setup ---

setup: ## Initialize local virtual environment using uv
	@echo "🚀 Initializing project with Python $(PYTHON_VERSION)..."
	uv venv --python $(PYTHON_VERSION) --allow-existing
	uv pip install -e .
	@echo "✅ Setup complete. Use 'source .venv/bin/activate' to start."

# --- Docker & Infrastructure ---

dev-up: ## Start the complete local container stack
	@echo "🐳 Starting container stack..."
	mkdir -p mlruns
	mkdir -p mlruns_artifacts
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
	@echo "✅ Services are live: API (8000), MLflow (5000), Prefect (4200), Grafana (3000), Prometheus (9090)"

dev-down: ## Stop all containers and remove networks
	@echo "🛑 Shutting down services..."
	docker compose down

dev: dev-up wait-prefect prefect-pool prefect-setup ## Start local stack and register Prefect deployment
	@echo "✅ Dev environment ready. Start the worker with 'make prefect-worker'."

logs: ## Follow logs from the API service
	docker compose logs -f api

refresh-api: ## Restart or recreate the local API service
	@echo "🔄 Refreshing API..."
	docker compose up -d api

reset-local-stack: ## Delete local runtime state for a clean bootstrap (requires CONFIRM_RESET=1)
	@if [ "$(CONFIRM_RESET)" != "1" ]; then \
		echo "❌ This deletes local MLflow, Prefect and serving state."; \
		echo "Run: make reset-local-stack CONFIRM_RESET=1"; \
		exit 1; \
	fi
	@echo "🧹 Stopping containers and removing local database volumes..."
	docker compose down -v --remove-orphans
	@echo "🗑️ Removing local runtime artifacts..."
	rm -rf \
		./models \
		./mlruns \
		./mlruns_artifacts \
		./prefect_data
	@echo "📁 Recreating empty runtime directories..."
	mkdir -p \
		./models \
		./mlruns \
		./mlruns_artifacts \
		./prefect_data
	@echo "✅ Local runtime state reset."
	@echo "Next: make dev-up && make wait-prefect && make train-bootstrap"

# --- Local Prefect ---

prefect-status: ## Check local Prefect server and configuration
	@echo "🔍 Checking local Prefect server status..."
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		uv run --active prefect config view
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		uv run --active prefect work-pool ls
	@curl -fsS "$(LOCAL_PREFECT_API_URL)/health" || \
		echo "⚠️ Prefect server is not reachable. Run 'make dev-up'."

wait-prefect: ## Wait until the local Prefect server is reachable
	@echo "⏳ Waiting for Prefect server ($(LOCAL_PREFECT_API_URL)/health)..."
	@until curl -fsS "$(LOCAL_PREFECT_API_URL)/health" > /dev/null; do \
		sleep 2; \
		echo "Prefect not ready yet..."; \
	done
	@echo "✅ Prefect is online!"

prefect-pool: wait-prefect ## Create local Prefect work pool if missing
	@echo "🏊 Ensuring Prefect work pool '$(PREFECT_POOL)' exists..."
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		uv run --active prefect work-pool inspect "$(PREFECT_POOL)" \
		> /dev/null 2>&1 || \
		PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		uv run --active prefect work-pool create \
			--type process \
			"$(PREFECT_POOL)"

prefect-setup: wait-prefect prefect-pool ## Register or update local Prefect deployment
	@echo "🧭 Registering local Prefect deployment..."
	@APP_ENV=dev \
		PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		MLFLOW_TRACKING_URI="$(LOCAL_MLFLOW_TRACKING_URI)" \
		uv run --active python scripts/setup_prefect.py

prefect-worker: wait-prefect prefect-pool ## Start Prefect worker for the local pool
	@echo "👷 Starting Prefect worker for pool '$(PREFECT_POOL)'..."
	PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
	PREFECT_API_KEY= \
	uv run --active prefect worker start --pool "$(PREFECT_POOL)"

# --- UI Quicklinks ---

ui-dashboard: ## Start the local Streamlit monitoring dashboard
	docker compose exec \
		-e HOME=/tmp \
		-e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
		api \
		uv run --no-sync streamlit run \
			src/monitoring/dashboard.py \
			--server.address 0.0.0.0 \
			--server.port 8501 \
			--browser.gatherUsageStats false
			
ui-prefect: ## Open local Prefect UI in the browser
	@python3 -m webbrowser http://localhost:4200

ui-mlflow: ## Open local MLflow UI in the browser
	@python3 -m webbrowser http://localhost:5000

# Container-internal local service addresses.
COMPOSE_RUN_API = docker compose exec -T \
	-e APP_ENV=dev \
	-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	-e PREFECT_API_URL=http://prefect:4200/api \
	-e PREFECT_API_KEY= \
	-e PREDICTION_API_URL=http://api:8080/predict \
	api

# --- Local ML Pipeline Tasks ---

train: wait-prefect ## Execute normal training flow inside the API container
	@echo "🧠 Starting training flow inside API container..."
	$(COMPOSE_RUN_API) uv run --no-sync python flows/training_flow.py

train-force: wait-prefect ## Execute forced training flow inside the API container
	@echo "🧠 Starting forced training flow inside API container..."
	$(COMPOSE_RUN_API) uv run --no-sync python flows/training_flow.py --force

train-bootstrap: wait-prefect ## Create initial Champion in an empty local registry
	@echo "🌱 Bootstrapping initial local Champion..."
	$(COMPOSE_RUN_API) uv run --no-sync python flows/training_flow.py --force --bootstrap

auto-retrain: wait-prefect ## Run auto-retraining flow once inside the API container
	@echo "🤖 Running auto-retraining flow once inside API container..."
	$(COMPOSE_RUN_API) uv run --no-sync python flows/auto_retrain_flow.py

predict-test: ## Send a local sample churn prediction request
	@echo "🧪 Sending test prediction request..."
	@curl -fsS -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-H "X-API-KEY: $(API_KEY)" \
		-d '{"context":{"request_id":"make-predict-test"},"inputs":[{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":"29.85"}]}' \
		| jq .

demo-churn-lifecycle: wait-prefect ## Run local churn lifecycle demo
	@echo "📈 Running churn lifecycle demo inside API container..."
	$(COMPOSE_RUN_API) uv run --no-sync python scripts/run_churn_demo.py \
		--batch-size 50 \
		--max-days 5 \
		--label-delay-days 1

churn-retraining-comparison: ## Run controlled churn comparison with and without retraining
	docker compose exec -T \
		-e APP_ENV=dev \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		-e PREFECT_API_URL=http://prefect:4200/api \
		-e PREFECT_API_KEY= \
		api \
		uv run --no-sync python \
		scripts/run_controlled_retraining_experiment.py

churn-retraining-comparison-smoke: ## Smoke-test both controlled comparison branches
	docker compose exec -T \
		-e APP_ENV=dev \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		-e PREFECT_API_URL=http://prefect:4200/api \
		-e PREFECT_API_KEY= \
		api \
		uv run --no-sync python \
		scripts/run_controlled_retraining_experiment.py \
		--smoke-test

churn-concept-drift-comparison: ## Run controlled concept-drift comparison
	docker compose exec -T api \
		uv run --no-sync python \
		scripts/run_controlled_retraining_experiment.py \
		--scenario concept_drift \
		--max-days 28

churn-concept-drift-comparison-smoke: ## Smoke-test controlled concept drift
	docker compose exec -T api \
		uv run --no-sync python \
		scripts/run_controlled_retraining_experiment.py \
		--scenario concept_drift \
		--smoke-test

churn-cohort-shift-comparison-plot: ## Plot the controlled customer-cohort shift experiment
	docker compose exec -T api \
		uv run --no-sync python \
		scripts/plot_churn_retraining_comparison.py \
		--experiment-directory \
		results/churn_retraining_comparison/controlled_cohort_shift

churn-concept-drift-comparison-plot: ## Plot the controlled concept-drift experiment
	docker compose exec -T api \
		uv run --no-sync python \
		scripts/plot_churn_retraining_comparison.py \
		--experiment-directory \
		results/churn_retraining_comparison/controlled_concept_drift

churn-retraining-comparison-plot: \
	churn-cohort-shift-comparison-plot \
	churn-concept-drift-comparison-plot

# --- Production Helpers ---
PRODUCTION_API_BASE_URL = $(patsubst %/predict,%,$(PREDICTION_API_URL))

check-prod-env: ## Validate required production environment variables
	@echo "🔎 Validating production environment..."
	@test -n "$(GCP_PROJECT_ID)" || \
		(echo "❌ GCP_PROJECT_ID is missing." && exit 1)
	@test -n "$(GCP_REGION)" || \
		(echo "❌ GCP_REGION is missing." && exit 1)
	@test -n "$(GCP_BUCKET_NAME)" || \
		(echo "❌ GCP_BUCKET_NAME is missing." && exit 1)
	@test -n "$(MLFLOW_UI_URL)" || \
		(echo "❌ MLFLOW_UI_URL is missing." && exit 1)
	@test -n "$(PREDICTION_API_URL)" || \
		(echo "❌ PREDICTION_API_URL is missing." && exit 1)
	@test -n "$(PREFECT_API_URL)" || \
		(echo "❌ PREFECT_API_URL is missing." && exit 1)
	@test -n "$(PREFECT_API_KEY)" || \
		(echo "❌ PREFECT_API_KEY is missing." && exit 1)
	@test -n "$(API_KEY)" || \
		(echo "❌ API_KEY is missing." && exit 1)
	@case "$(MLFLOW_UI_URL)" in \
		http://localhost*|http://mlflow*) \
			echo "❌ MLFLOW_UI_URL points to a local service."; \
			exit 1 ;; \
	esac
	@case "$(PREDICTION_API_URL)" in \
		http://localhost*|http://api*) \
			echo "❌ PREDICTION_API_URL points to a local service."; \
			exit 1 ;; \
	esac
	@case "$(PREFECT_API_URL)" in \
		http://localhost*|http://prefect*) \
			echo "❌ PREFECT_API_URL points to a local service."; \
			exit 1 ;; \
	esac
	@echo "✅ Production environment is valid."

debug-prod-env: ## Show non-secret production values loaded by Make
	@echo "PREFECT_API_URL=$(PREFECT_API_URL)"
	@echo "MLFLOW_UI_URL=$(MLFLOW_UI_URL)"
	@echo "PREDICTION_API_URL=$(PREDICTION_API_URL)"
	@echo "GCP_PROJECT_ID=$(GCP_PROJECT_ID)"
	@echo "GCP_REGION=$(GCP_REGION)"
	@echo "GCP_BUCKET_NAME=$(GCP_BUCKET_NAME)"
	@echo "PREFECT_API_KEY configured: $$(test -n "$(PREFECT_API_KEY)" && echo yes || echo no)"
	@echo "API_KEY configured: $$(test -n "$(API_KEY)" && echo yes || echo no)"

prepare-mlflow-prod-demo: check-prod-env ## Prepare one warm MLflow instance for the ephemeral demo
	@echo "🔥 Preparing ephemeral MLflow production demo..."
	@gcloud run services update \
		mlflow-server \
		--project "$(GCP_PROJECT_ID)" \
		--region "$(GCP_REGION)" \
		--update-env-vars "^@^MLFLOW_BACKEND_STORE_URI=sqlite:////tmp/mlflow.db@MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=$(MLFLOW_UI_URL)" \
		--memory 4Gi \
		--min 1 \
		--max 1 \
		--quiet
	@echo "⏳ Waiting for MLflow health endpoint..."
	@until curl -fsS "$(MLFLOW_UI_URL)/health" > /dev/null; do \
		echo "MLflow is not ready yet..."; \
		sleep 2; \
	done
	@echo "✅ MLflow demo instance is ready."

upload-raw-prod: check-prod-env ## Upload raw churn data to production GCS bucket
	@echo "☁️ Uploading raw churn data to gs://$(GCP_BUCKET_NAME)/data/raw/..."
	gcloud storage cp \
		data/raw/Telco-Customer-Churn.csv \
		gs://$(GCP_BUCKET_NAME)/data/raw/
	@echo "✅ Raw data uploaded."
	gcloud storage ls gs://$(GCP_BUCKET_NAME)/data/raw/

predict-test-prod: check-prod-env ## Send a sample prediction request to the production API
	@echo "🧪 Sending production test prediction request..."
	@curl -fsS -X POST "$(PREDICTION_API_URL)" \
		-H "Content-Type: application/json" \
		-H "X-API-KEY: $(API_KEY)" \
		-d '{"context":{"request_id":"make-predict-test-prod"},"inputs":[{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":"29.85"}]}' \
		| jq .

train-bootstrap-prod: prepare-mlflow-prod-demo upload-raw-prod ## Bootstrap initial production Champion
	@echo "🌱 Bootstrapping initial production Champion..."
	@set -eu; \
		curl -fsS "$(MLFLOW_UI_URL)/health" > /dev/null; \
		( \
			while true; do \
				curl -fsS "$(MLFLOW_UI_URL)/health" \
					> /dev/null 2>&1 || true; \
				sleep 10; \
			done \
		) & \
		heartbeat_pid=$$!; \
		cleanup() { \
			kill "$$heartbeat_pid" 2>/dev/null || true; \
			wait "$$heartbeat_pid" 2>/dev/null || true; \
		}; \
		trap cleanup EXIT INT TERM; \
		PYTHONPATH=. \
		APP_ENV=prod \
		PREFECT_API_URL="$(PREFECT_API_URL)" \
		PREFECT_API_KEY="$(PREFECT_API_KEY)" \
		MLFLOW_TRACKING_URI="$(MLFLOW_UI_URL)" \
		PREDICTION_API_URL="$(PREDICTION_API_URL)" \
		GCP_BUCKET_NAME="$(GCP_BUCKET_NAME)" \
		GCP_PROJECT_ID="$(GCP_PROJECT_ID)" \
		API_KEY="$(API_KEY)" \
		uv run --active python flows/training_flow.py --force --bootstrap

train-force-prod: check-prod-env ## Execute forced training against production cloud services
	@echo "🧠 Starting forced production training flow..."
	PYTHONPATH=. \
	APP_ENV=prod \
	PREFECT_API_URL="$(PREFECT_API_URL)" \
	PREFECT_API_KEY="$(PREFECT_API_KEY)" \
	MLFLOW_TRACKING_URI="$(MLFLOW_UI_URL)" \
	PREDICTION_API_URL="$(PREDICTION_API_URL)" \
	GCP_BUCKET_NAME="$(GCP_BUCKET_NAME)" \
	GCP_PROJECT_ID="$(GCP_PROJECT_ID)" \
	API_KEY="$(API_KEY)" \
	uv run --active python flows/training_flow.py --force

verify-prod: check-prod-env ## Verify production liveness, readiness, lineage and prediction
	@echo "🔍 Verifying production deployment..."
	@PYTHONPATH=. \
	APP_ENV=prod \
	PRODUCTION_API_URL="$(PRODUCTION_API_BASE_URL)" \
	MLFLOW_TRACKING_URI="$(MLFLOW_UI_URL)" \
	PREDICTION_API_URL="$(PREDICTION_API_URL)" \
	GCP_BUCKET_NAME="$(GCP_BUCKET_NAME)" \
	GCP_PROJECT_ID="$(GCP_PROJECT_ID)" \
	API_KEY="$(API_KEY)" \
	uv run --active python \
		scripts/verify_production_deployment.py

bootstrap-and-verify-prod: train-bootstrap-prod verify-prod ## Bootstrap and verify a fresh production demo
	@echo "✅ Production bootstrap and semantic verification completed."

# --- Quality Assurance ---

test: ## Run unit and integration tests
	@echo "🧪 Running pytest suite..."
	uv run --active pytest tests/

lint: ## Check code style and quality with Ruff
	@echo "✨ Linting code..."
	uv run --active ruff check .

# --- Cleanup ---

clean: ## Remove Python caches
	@echo "🧹 Cleaning up Python caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✨ Workspace is clean."

clean-venv: ## Remove local virtual environment
	@echo "🗑️ Removing .venv..."
	rm -rf .venv

clean-data: ## Remove local runtime data folders
	@echo "📂 Removing local runtime data folders..."
	rm -rf ./prefect_data ./mlruns ./mlruns_artifacts ./models
	@echo "✅ Runtime data folders removed."

clean-all: clean dev-down clean-venv clean-data ## Deep-clean local environment and Docker state
	@echo "🐳 Pruning Docker system..."
	docker system prune -f
	docker volume prune -f
	@echo "🧼 Deep clean finished. System is fresh."

reset-demo: ## Reset generated demo state while retaining raw input data
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
