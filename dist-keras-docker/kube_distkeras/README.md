# DistKeras on Docker on Kubernetes

This directory contains Kubernetes setting files to run DistKeras on Docker on Kubernetes cluster.

## Prerequisite
You need to deploy Kubernetes cluster with more than 2 nodes: one for master and one for worker.
Refer to the following for Kubernetest deployment.
https://kubernetes.io/docs/getting-started-guides/kubeadm/
https://kubernetes.io/docs/getting-started-guides/centos/centos_manual_config/

## How to use
1. git clone the repository and build docker images for Distkeras_master and Distkeras_slave.

```
git clone https://github.com/shibuiwilliam/distkeras-docker.git

cd ~/distkeras-docker/dist-keras-docker/kube_distkeras/master_docker
docker build -t distkeras_master_kube:1.4 .

cd ~/distkeras-docker/dist-keras-docker/kube_distkeras/slave_docker
docker build -t distkeras_slave_kube:1.4 .
```

You will have distkeras_master_kube:1.4 and distkeras_slave_kube:1.4 for docker images.

2. Once the docker images are built, use the service setting yml and replication controller yml to deploy distkeras_master and distkeras_slave.

```
kubectl create -f sv_master.yml
kubectl create -f rc_master.yml
kubectl create -f sv_slave.yml
kubectl create -f rc_slave.yml

kubectl get all
```

You will find one master pod and one slave pod deployed.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/3b88be72-190e-5c6c-6b99-54660865e36a.jpeg>
![24.JPG](https://qiita-image-store.s3.amazonaws.com/0/55384/3b88be72-190e-5c6c-6b99-54660865e36a.jpeg)

3. 
