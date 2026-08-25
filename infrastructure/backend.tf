terraform {
  backend "gcs" {
    bucket = "mlops-terraform-state-churn-prediction-mlops"
    prefix = "terraform/state"
  }
}
