# Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Multi-Modal Sentiment Analysis for Crypto Market Prediction system in various environments. It covers both development and production deployments.

## Prerequisites

- Docker
- Docker Compose
- Access to a container registry (e.g., Docker Hub, Google Container Registry)
- Kubernetes cluster (for production deployment)
- Helm (for Kubernetes deployments)

## Development Deployment

1. Clone the repository:
   ```
   git clone https://github.com/selimozten/mmsag.git
   cd mmsag
   ```

2. Create and configure the necessary files:
   ```
   cp example.config.yml config.yml
   cp example.credentials.yml credentials.yml
   ```
   Edit these files with your specific configuration and API credentials.

3. Build the Docker image:
   ```
   docker build -t mmsag:dev .
   ```

4. Run the container:
   ```
   docker run -d -p 8501:8501 --name mmsag-dev mmsag:dev
   ```

5. Access the dashboard at `http://localhost:8501`

## Production Deployment

### 1. Prepare the Environment

1. Set up a Kubernetes cluster (e.g., using Google Kubernetes Engine or Amazon EKS)
2. Install Helm:
   ```
   curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
   ```

### 2. Build and Push the Docker Image

1. Build the production image:
   ```
   docker build -t mmsag:prod .
   ```

2. Tag the image for your container registry:
   ```
   docker tag mmsag:prod gcr.io/your-project/mmsag:prod
   ```

3. Push the image:
   ```
   docker push gcr.io/your-project/mmsag:prod
   ```

### 3. Deploy using Helm

1. Create a Helm chart for the application (if not already created)

2. Update the `values.yaml` file in the Helm chart with the appropriate values for your environment

3. Deploy the application:
   ```
   helm install mmsag ./mmsag-chart
   ```

4. Monitor the deployment:
   ```
   kubectl get pods
   kubectl get services
   ```

### 4. Set Up Monitoring and Logging

1. Deploy Prometheus and Grafana for monitoring:
   ```
   helm install prometheus stable/prometheus
   helm install grafana stable/grafana
   ```

2. Set up log aggregation using Elasticsearch, Fluentd, and Kibana (EFK stack) or a managed logging service

### 5. Configure Auto-scaling

1. Set up Horizontal Pod Autoscaler (HPA):
   ```
   kubectl autoscale deployment mmsag --cpu-percent=80 --min=3 --max=10
   ```

### 6. Set Up Continuous Deployment

1. Configure your CI/CD pipeline (e.g., Jenkins, GitLab CI, or GitHub Actions) to automatically build, test, and deploy changes to your staging environment

2. Implement a blue-green or canary deployment strategy for production updates

## Security Considerations

- Ensure all sensitive data (API keys, credentials) are stored as Kubernetes secrets
- Use network policies to restrict communication between pods
- Regularly update and patch all components of the system
- Implement role-based access control (RBAC) for Kubernetes resources

## Backup and Disaster Recovery

1. Regularly backup all persistent data, including databases and model files
2. Set up a disaster recovery plan, including procedures for failover to a secondary region if using cloud services

## Performance Tuning

- Monitor application performance and resource usage
- Adjust resource requests and limits in the Kubernetes deployment as needed
- Consider using a caching layer (e.g., Redis) for frequently accessed data

## Troubleshooting

- Check pod logs: `kubectl logs <pod-name>`
- Describe pods for events and status: `kubectl describe pod <pod-name>`
- Use port-forwarding for debugging services: `kubectl port-forward <pod-name> 8080:8080`

For any deployment-related issues or questions, please contact the DevOps team.