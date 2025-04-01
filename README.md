# About
## 1. Node Specification

| 노드 이름    | 역할               | CPU 코어 | 메모리(GB) | 스토리지(GB) | 운영체제         |
| -------- | ---------------- | ------ | ------- | -------- | ------------ |
| minikube | 마스터 및 Compute 노드 | 4      | 16      | 50       | Ubuntu 22.04 |

## 2. PV 구성

- **StorageClass**
  - 이름: standard
  - 프로비저너: kubernetes.io/minikube-hostpath

- **PV/PVC**
	- Dataset
		- Name: mnist-pvc-standard
		- Capacity: 1Gi
		- Access Modes: ReadWriteOnce
	- Model Storage
		- Name: model-pvc
		- StorageClass: Manual
		- Capacity: 1Gi
		- Access Modes: ReadWriteOnce

## 3. 로깅
Fluent-bit side-car 패턴으로 로그 수집 및 Loki로 전송
Grafana를 통한 시각화

## 4. 레퍼런스
- https://github.com/triton-inference-server/server
- https://github.com/triton-inference-server/client


## 5. 실행 커맨드

```
minikube start --cpus=4 --memory=8192 --disk-size=75g

eval $(minikube docker-env)
```

```
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm upgrade --install loki grafana/loki-stack --namespace logging --create-namespace --set promtail.enabled=false --set grafana.enabled=true
```

```
git clone
```
### Training Job
```
cd ./train

docker build -t resnet/train-resnet18-mnist:0.4.3 .

kubectl apply -f dataset-pvc.yaml
kubectl apply -f train-job.yaml
```
### Triton server Deployment
```
cd ./triton

minikube ssh
mkdir -p /data/models/resnet18-mnist_0.4.3/1
chown -R 1000:1000 /data

exit

minikube cp ./config.pbtxt minikube:/data/models/resnet18-mnist_0.4.3/

cp /tmp/hostpath-provisioner/default/dataset-pvc/model.onnx /data/models/resnet18-mnist_0.4.3/1/

exit

kubectl apply -f model-pvc.yaml
kubectl apply -f triton-deploy.yaml
kubectl apply -f triton-svc.yaml
```
### gRPC Client Pod
```
cd ./client

docker build -t resnet/tritonclient:0.1.3 .

kubectl apply -f fluent-bit-cm.yaml
kubectl apply -f tritonclient-pod.yaml
```


