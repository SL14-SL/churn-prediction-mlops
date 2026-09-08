# --- Google Cloud Provider ---
provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Enable Required Google APIs ---
resource "google_project_service" "base_services" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com"
  ])

  service            = each.key
  disable_on_destroy = false
}

# --- Dummy resource to wait for APIs ---
resource "null_resource" "wait_for_apis" {
  depends_on = [
    google_project_service.base_services
  ]
}

# --- Service Account for MLOps ---
resource "google_service_account" "mlops_sa" {
  account_id   = "mlops-api-sa"
  display_name = "Service Account for MLOps API and Training"

  depends_on = [null_resource.wait_for_apis]
}

# --- Cloud Storage Bucket for Artifacts ---
resource "google_storage_bucket" "artifacts_bucket" {
  name                        = "mlops-churn-artifacts-${var.project_id}"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  depends_on = [null_resource.wait_for_apis]
}

# --- Artifact Registry ---
resource "google_artifact_registry_repository" "mlops_repo" {
  location      = var.region
  repository_id = "mlops-repo"
  format        = "DOCKER"

  depends_on = [null_resource.wait_for_apis]
}

# --- Persistent MLflow PostgreSQL Backend ---

resource "random_password" "mlflow_database" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "mlflow" {
  name             = "mlflow-postgres-dev"
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = false

  settings {
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 10
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = false
    }

    ip_configuration {
      ipv4_enabled = true
    }
  }

  depends_on = [
    null_resource.wait_for_apis
  ]
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
}

resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
  password = random_password.mlflow_database.result
}

resource "google_secret_manager_secret" "mlflow_database_password" {
  secret_id = "mlflow-database-password"

  replication {
    auto {}
  }

  depends_on = [
    null_resource.wait_for_apis
  ]
}

resource "google_secret_manager_secret_version" "mlflow_database_password" {
  secret      = google_secret_manager_secret.mlflow_database_password.id
  secret_data = random_password.mlflow_database.result
}

# --- IAM: Permissions for the Service Account ---
resource "google_storage_bucket_iam_member" "sa_storage_admin" {
  bucket = google_storage_bucket.artifacts_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# --- SERVICE 1: MLflow Tracking Server ---
resource "google_cloud_run_v2_service" "mlflow_server" {
  name                = "mlflow-server"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  depends_on = [
    google_sql_database.mlflow,
    google_sql_user.mlflow,
    google_secret_manager_secret_version.mlflow_database_password,
    google_secret_manager_secret_iam_member.mlflow_database_password_access,
    google_project_iam_member.sa_cloud_sql_client
  ]

  template {
    service_account = google_service_account.mlops_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    volumes {
      name = "cloudsql"

      cloud_sql_instance {
        instances = [
          google_sql_database_instance.mlflow.connection_name
        ]
      }
    }

    containers {
      image = "gcr.io/cloudrun/hello"

      resources {
        cpu_idle = true

        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      ports {
        container_port = 8080
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      env {
        name  = "MLFLOW_ARTIFACT_ROOT"
        value = "gs://${google_storage_bucket.artifacts_bucket.name}/mlruns"
      }

      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*"
      }

      env {
        name  = "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"
        value = "*"
      }

      env {
        name  = "MLFLOW_DB_NAME"
        value = google_sql_database.mlflow.name
      }

      env {
        name  = "MLFLOW_DB_USER"
        value = google_sql_user.mlflow.name
      }

      env {
        name  = "CLOUD_SQL_CONNECTION_NAME"
        value = google_sql_database_instance.mlflow.connection_name
      }

      env {
        name = "MLFLOW_DB_PASSWORD"

        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlflow_database_password.secret_id
            version = "latest"
          }
        }
      }
    }
  }

}

resource "google_cloud_run_v2_service_iam_member" "public_mlflow" {
  location = google_cloud_run_v2_service.mlflow_server.location
  name     = google_cloud_run_v2_service.mlflow_server.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- SERVICE 2: Prediction API ---
resource "google_cloud_run_v2_service" "prediction_api" {
  name                = "churn-prediction-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  lifecycle {
    ignore_changes = [
      template,
    ]
  }

  depends_on = [null_resource.wait_for_apis]

  template {
    service_account = google_service_account.mlops_sa.email

    containers {
      image = "gcr.io/cloudrun/hello"

      ports {
        container_port = 8080
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCS_BUCKET_NAME"
        value = google_storage_bucket.artifacts_bucket.name
      }
    }
  }

}

# --- Workload Identity Federation ---

resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions Pool"
  description               = "Identity pool for GitHub Actions automation"

  depends_on = [null_resource.wait_for_apis]
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "attribute.repository == '${var.github_repo}' && assertion.ref == 'refs/heads/main'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "wif_impersonation" {
  service_account_id = google_service_account.mlops_sa.name
  role               = "roles/iam.workloadIdentityUser"

  member = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}

resource "google_project_iam_member" "sa_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

resource "google_project_iam_member" "sa_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

resource "google_project_iam_member" "sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

resource "google_project_iam_member" "sa_cloud_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "mlflow_database_password_access" {
  secret_id = google_secret_manager_secret.mlflow_database_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# --- Public Access ---
resource "google_cloud_run_v2_service_iam_member" "public_api" {
  location = google_cloud_run_v2_service.prediction_api.location
  name     = google_cloud_run_v2_service.prediction_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}