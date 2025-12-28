# Homework 10: Kubernetes Fundamentals

In this homework, I deployed a lead scoring model (from Homework 5) to a local Kubernetes cluster using `kind`.

## Files

- `q6_test.py`: Script to test the deployed model.
- `q6_predict.py`: The application code (from HW5).
- `deployment.yaml`: Kubernetes Deployment configuration.
- `service.yaml`: Kubernetes Service configuration.
- `Dockerfile_full`: Dockerfile used to build the image.
- `load_test.py`: Script used for simulating load for autoscaling.

## Setup & Running

### Prerequisites

- Docker
- Kind
- Kubectl
- Uv (for python dependency management)

### Steps

1. **Build the Image**

```bash
docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
```

2. **Create Cluster**

```bash
kind create cluster
```

3. **Load Image into Kind**

```bash
kind load docker-image zoomcamp-model:3.13.10-hw10
```

4. **Deploy**

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

5. **Forward Port**

```bash
kubectl port-forward service/subscription 9696:80
```

6. **Test**

```bash
uv run --with requests q6_test.py
```
