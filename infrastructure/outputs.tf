output "project_id" {
  value = var.project_id
}

output "artifacts_bucket_name" {
  value = google_storage_bucket.artifacts_bucket.name
}

# Full path to the repository for Docker login/push
output "container_registry_uri" {
  description = "Full URI for Docker push"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops_repo.name}"
}

output "mlflow_url" {
  value = google_cloud_run_v2_service.mlflow_server.uri
}

output "service_account_email" {
  value = google_service_account.mlops_sa.email
}

output "prediction_api_url" {
  value = google_cloud_run_v2_service.prediction_api.uri
}

# Output the provider name for GitHub Secrets
output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github_provider.name
}

output "mlflow_database_instance_name" {
  description = "Cloud SQL instance used by the MLflow backend"
  value       = google_sql_database_instance.mlflow.name
}

output "mlflow_database_connection_name" {
  description = "Cloud SQL connection name used by the MLflow Cloud Run service"
  value       = google_sql_database_instance.mlflow.connection_name
}

output "mlflow_database_name" {
  description = "PostgreSQL database used by MLflow"
  value       = google_sql_database.mlflow.name
}
