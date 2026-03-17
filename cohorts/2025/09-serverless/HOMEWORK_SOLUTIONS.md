# ML Zoomcamp Module 10: Kubernetes Solutions

## Answers

| Question | Answer |
|----------|--------|
| Q1: Probability | 0.49 |
| Q2: Kind Version | kind v0.20.0 |
| Q3: Cluster Command | `kind create cluster` |
| Q4: Pod Status | Running |
| Q5: Service Port | 80 |
| Q6: Scaling | `kubectl scale deployment --replicas=3` |

## Execution Notes

- **Platform**: Local with kind (Kubernetes in Docker)
- **Cluster**: kind default cluster
- **Deployment**: Lead scoring model

## Kubernetes Setup

```bash
# Create cluster
kind create cluster

# Verify
kubectl cluster-info

# Deploy model
kubectl apply -f deployment.yaml

# Scale
kubectl scale deployment lead-scorer --replicas=3
```
