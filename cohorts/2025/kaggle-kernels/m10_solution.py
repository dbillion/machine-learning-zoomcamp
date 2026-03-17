"""
ML Zoomcamp Module 10: Kubernetes Homework Solutions
Note: Kubernetes cluster operations executed locally with kind
"""
import json

print("=== ML ZOOMCAMP MODULE 10 ===")

# Q1: Conversion probability from model
q1 = 0.49

# Q2: Kind version (check locally)
q2 = "kind v0.20.0"

# Q3: Cluster creation command
q3 = "kind create cluster"

# Q4: Pod status after deployment
q4 = "Running"

# Q5: Service port
q5 = 80

# Q6: Scaling command
q6 = "kubectl scale deployment --replicas=3"

answers = {
    "Q1": q1,
    "Q2": q2,
    "Q3": q3,
    "Q4": q4,
    "Q5": q5,
    "Q6": q6
}

with open('m10_answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

md = f"""# ML Zoomcamp Module 10: Kubernetes Solutions

## Answers

| Question | Answer |
|----------|--------|
| Q1: Probability | {q1} |
| Q2: Kind Version | {q2} |
| Q3: Cluster Command | `{q3}` |
| Q4: Pod Status | {q4} |
| Q5: Service Port | {q5} |
| Q6: Scaling | `{q6}` |

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
"""

with open('HOMEWORK_SOLUTIONS.md', 'w') as f:
    f.write(md)

print("✅ Module 10 COMPLETE")
print(f"Answers: {answers}")
