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

This is the deployed image of DistKeras on Docker on Kubernetes.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/e202e816-f25b-5f0a-3359-d91c4774dd6d.jpeg>


With the current setting yml, you will find one master pod and one slave pod deployed.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/3b88be72-190e-5c6c-6b99-54660865e36a.jpeg>

You are also able to access Spark console for http://<distkeras master ip address>:30080 and able to see there are two slaves.
One slave is colocated in the master.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/8ee4eff9-eb3a-5b72-8d18-2528d0cfe23b.jpeg>

3. You can rescale or rebalance the slave pods with replication controller.
You can run `kubectl scale rc/distkeras-rc-slave --replicas=2` to scale out slave from 1 pod to 2 pods.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/909fa6e9-9790-980e-9226-9f41752082cb.jpeg>

The Spark worker is scaled out as well.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/cd400f42-f100-f3af-2b60-8e499cab2576.jpeg>


You can also stop a slave host server or pod and still Kubernetes will manage to rebalance with existing servers.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/491d3056-ef9b-18ff-c905-ede1ab07583b.jpeg>

Spark worker is added as well.
<img src=https://qiita-image-store.s3.amazonaws.com/0/55384/25b58f85-bab9-341c-3c30-756a366d85f9.jpeg>


4. Run Keras on Spark program.
You can find a sample program here as a Jupyter Notebook.
https://github.com/shibuiwilliam/distkeras-docker/blob/master/dist-keras-docker/kube_distkeras/master_docker/mnist_mod.ipynb

