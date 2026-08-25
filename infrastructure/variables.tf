# infrastructure/variables.tf

variable "project_id" {
  description = "The ID of the GCP project where resources are managed"
  type        = string
}

variable "region" {
  description = "The GCP region for all resources"
  type        = string
  default     = "europe-west1"
}

# environment variable to distinguish between dev, staging, and prod later
variable "environment" {
  description = "The environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "github_repo" {
  description = "GitHub repository in format owner/repo"
  type        = string
}