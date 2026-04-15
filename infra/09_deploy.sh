#!/bin/bash
# ============================================================
# SCRIPT: infra/09_deploy.sh
# ------------------------------------------------------------
# PROJECT      : ReadmitIQ
# AUTHOR       : Dr. Nikki
# CREATED      : 2026-04-08
#
# PURPOSE
# -------
# Automates the AWS deployment pipeline:
#   1. Builds the Docker image
#   2. Pushes it to AWS ECR (Elastic Container Registry)
#   3. Forces a new ECS deployment so Fargate picks up the new image
#
# PREREQUISITES
# -------------
#   - AWS CLI installed and configured (aws configure)
#   - Docker Desktop running
#   - ECR repository already created (do this once manually in AWS console)
#   - ECS cluster and service already created (do this once manually)
#
# USAGE
# -----
#   chmod +x infra/09_deploy.sh
#   ./infra/09_deploy.sh
#
# ENVIRONMENT VARIABLES (set before running)
# ------------------------------------------
#   AWS_REGION       — e.g., us-east-1
#   AWS_ACCOUNT_ID   — Your 12-digit AWS account ID
#   ECR_REPO_NAME    — e.g., readmitiq
#   ECS_CLUSTER      — Your ECS cluster name
#   ECS_SERVICE      — Your ECS service name
# ============================================================

set -e   # Exit immediately if any command fails

# ---------------------------------------------------------
# Configuration — set these to match your AWS environment
# ---------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?ERROR: Set the AWS_ACCOUNT_ID environment variable}"
ECR_REPO_NAME="${ECR_REPO_NAME:-readmitiq}"
ECS_CLUSTER="${ECS_CLUSTER:-readmitiq-cluster}"
ECS_SERVICE="${ECS_SERVICE:-readmitiq-service}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_TAG=$(date +%Y%m%d%H%M%S)   # Tag images with a timestamp for traceability

echo "======================================================"
echo "ReadmitIQ Deployment"
echo "======================================================"
echo "  Region      : ${AWS_REGION}"
echo "  Account     : ${AWS_ACCOUNT_ID}"
echo "  ECR URI     : ${ECR_URI}"
echo "  Image tag   : ${IMAGE_TAG}"
echo "  ECS Cluster : ${ECS_CLUSTER}"
echo "  ECS Service : ${ECS_SERVICE}"
echo "======================================================"

# Step 1: Authenticate Docker to ECR
echo ""
echo "[1/4] Authenticating Docker with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Step 2: Build the Docker image
# Run this from the project root so the Dockerfile can COPY src/ and api/
echo ""
echo "[2/4] Building Docker image..."
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" -t "${ECR_REPO_NAME}:latest" .

# Step 3: Tag and push to ECR
echo ""
echo "[3/4] Pushing image to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag "${ECR_REPO_NAME}:latest"       "${ECR_URI}:latest"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

# Step 4: Force ECS to deploy the new image
echo ""
echo "[4/4] Triggering ECS deployment..."
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --force-new-deployment \
  --region "${AWS_REGION}" \
  --output text \
  --query "service.deployments[0].status"

echo ""
echo "======================================================"
echo "Deployment triggered. Image: ${ECR_URI}:${IMAGE_TAG}"
echo "Monitor at: https://console.aws.amazon.com/ecs"
echo "======================================================"
