# distkeras-docker
Dockerfile for Dist-keras

The aim of this docker file is to run spark and dist-keras on docker container.
Dist-Keras is a distributed deep learning framework built op top of Apache Spark and Keras, with a focus on "state-of-the-art" distributed optimization algorithms.
Read the followings for more on dist-keras.

https://db-blog.web.cern.ch/blog/joeri-hermans/2017-01-distributed-deep-learning-apache-spark-and-keras
https://github.com/cerndb/dist-keras


The distkeras-docker file installs Spark2.1.0, Keras2.0.2 and jupyter notebook on the CentOS:latest docker image.
The dockerfile also provides some settings for running Spark on Docker container.


## How to start
To use distkeras-docker, first git clone the repository and build docker file.

```
git clone https://github.com/shibuiwilliam/distkeras-docker.git
# for spark master docker image
cd distkeras-docker/distkeras_master
docker build -t distkeras_master:1.0 .

# for spark slave docker image
cd distkeras-docker/distkeras_slave
docker build -t distkeras_slave:1.0 .
```

You will get docker image for spark master and another one for spark worker.

After the docker images are successfully built, run docker containers.
Number of containers depend on how many Spark workers you need to add.
The script below deploys one Spark master with worker container and two worker-only containers.

```
# docker dist-keras for spark master and slave
docker run -it -p 18080:8080 -p 17077:7077 -p 18888:8888 -p 18081:8081 -p 14040:4040 -p 17001:7001 -p 17002:7002 \
 -p 17003:7003 -p 17004:7004 -p 17005:7005 -p 17006:7006 --name spmaster -h spmaster distkeras_master:1.0 /bin/bash

# docker dist-keras for spark slave1
docker run -it --link spmaster:master -p 28080:8080 -p 27077:7077 -p 28888:8888 -p 28081:8081 -p 24040:4040 -p 27001:7001 \
-p 27002:7002 -p 27003:7003 -p 27004:7004 -p 27005:7005 -p 27006:7006 --name spslave1 -h spslave1 distkeras_slave:1.0 /bin/bash

# docker dist-keras for spark slave2
docker run -it --link spmaster:master -p 38080:8080 -p 37077:7077 -p 38888:8888 -p 38081:8081 -p 34040:4040 -p 37001:7001 \
-p 37002:7002 -p 37003:7003 -p 37004:7004 -p 37005:7005 -p 37006:7006 --name spslave2 -h spslave2 distkeras_slave:1.0 /bin/bash
```

On each container, run shellscripts, added during docker build, to start Spark cluster.

```
# for Spark master
# Spark master and worker start
cd /opt/
sh spark_master.sh

# for Spark worker
# Spark worker starts and added to Spark cluster
cd /opt/
sh spark_slave.sh
```

#### Silent mode
To run docker images silently, use this command.

```
docker run -it -p 18080:8080 -p 17077:7077 -p 18888:8888 -p 18081:8081 -p 14040:4040 -p 17001:7001 -p 17002:7002 \
-p 17003:7003 -p 17004:7004 -p 17005:7005 -p 17006:7006 --name spm -h spm -d distkeras_master:1.0

docker run -it --link spm:master -p 28080:8080 -p 27077:7077 -p 28888:8888 -p 28081:8081 -p 24040:4040 -p 27001:7001 \
-p 27002:7002 -p 27003:7003 -p 27004:7004 -p 27005:7005 -p 27006:7006 --name sps1 -h sps1 -d distkeras_slave:1.0

```


Now you are ready to use Dist-Keras on Docker

## Running MNIST example
Some sample programmes are provided with Dist-Keras; just the same ones as below.
https://github.com/cerndb/dist-keras/tree/master/examples

In order to run an example script, you have to edit some portions of it.
For example on the mnist.py, you have to modify these.

1. Add "from pyspark.sql import SparkSession" to somewhere in the initial import section.

2. Modify parameter:
```
# Modify these variables according to your needs.
application_name = "Distributed Keras MNIST"
using_spark_2 = True  # changed False to True
local = True  # changed False to True
path_train = "data/mnist_train.csv"
path_test = "data/mnist_test.csv"
if local:
    # Tell master to use local resources.
#     master = "local[*]"   comment out
    master = "spark://spmaster:7077"  # add
    num_processes = 1
    num_executors = 3  # changed 1 to 3
else:
    # Tell master to use YARN.
    master = "yarn-client"
    num_executors = 20
    num_processes = 1
```

Now you are ready run mnist.py.

```
python mnist.py
```

Here's the log for running on 3 Spark workers cluster:

```
Using TensorFlow backend.
Ivy Default Cache set to: /root/.ivy2/cache
The jars for the packages stored in: /root/.ivy2/jars
:: loading settings :: url = jar:file:/opt/spark-2.1.0-bin-hadoop2.7/jars/ivy-2.4.0.jar!/org/apache/ivy/core/settings/ivysettings.xml
com.databricks#spark-csv_2.10 added as a dependency
:: resolving dependencies :: org.apache.spark#spark-submit-parent;1.0
        confs: [default]
        found com.databricks#spark-csv_2.10;1.4.0 in central
        found org.apache.commons#commons-csv;1.1 in central
        found com.univocity#univocity-parsers;1.5.1 in central
:: resolution report :: resolve 287ms :: artifacts dl 6ms
        :: modules in use:
        com.databricks#spark-csv_2.10;1.4.0 from central in [default]
        com.univocity#univocity-parsers;1.5.1 from central in [default]
        org.apache.commons#commons-csv;1.1 from central in [default]
        ---------------------------------------------------------------------
        |                  |            modules            ||   artifacts   |
        |       conf       | number| search|dwnlded|evicted|| number|dwnlded|
        ---------------------------------------------------------------------
        |      default     |   3   |   0   |   0   |   0   ||   3   |   0   |
        ---------------------------------------------------------------------
:: retrieving :: org.apache.spark#spark-submit-parent
        confs: [default]
        0 artifacts copied, 3 already retrieved (0kB/10ms)
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
17/04/08 11:26:59 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
cp_mnist.py:154: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), padding="valid", input_shape=(28, 28, 1...)`
  input_shape=input_shape))
cp_mnist.py:156: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3))`
  convnet.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Number of desired executors: 3
Number of desired processes / executor: 1
Total number of workers: 3
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
_________________________________________________________________
activation_1 (Activation)    (None, 26, 26, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 32)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0
_________________________________________________________________
dense_1 (Dense)              (None, 225)               1037025
_________________________________________________________________
activation_3 (Activation)    (None, 225)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2260
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
=================================================================
Total params: 1,048,853.0
Trainable params: 1,048,853.0
Non-trainable params: 0.0
_________________________________________________________________
60000
Training time: 1497.86584091
Accuracy: 0.9897
Number of parameter server updates: 3751
```


## running on multi-host docker container
In order to run the training in better performance, it is highly recommended to run docker on multi-host.
I used overlay network to build multi-host docker, and this is how.
#### for more about multi-host docker, take a look at this documentation
https://docs.docker.com/engine/userguide/networking/get-started-overlay/

I deployed two host servers: host1 and host2.
On host1, install and start etcd for key-value store.

```
yum -y install etcd

vi /etc/etcd/etcd.conf
systemctl enable etcd
systemctl start etcd
```

Next, edit docker network config on host1 and host2.

```
# edit docker-network file
vi /etc/sysconfig/docker-network

# for host1
DOCKER_NETWORK_OPTIONS='--cluster-store=etcd://<host1>:2379 --cluster-advertise=<host1>:2376'

# for host2
DOCKER_NETWORK_OPTIONS='--cluster-store=etcd://<host1>:2379 --cluster-advertise=<host2>:2376'

# from host2 to ensure network connection to host1 etcd is available
curl -L http://<host1>:2379/version
{"etcdserver":"3.1.3","etcdcluster":"3.1.0"}

```

Now you are ready to connect docker on multi-host.
Create docker network on host1.
Here I created test1 network with subnet 10.0.1.0/24

```
# for host1
docker network create --subnet=10.0.1.0/24 -d overlay test1
```

Run `docker network ls` to see test1 network is added to docker network.

```
NETWORK ID          NAME                DRIVER              SCOPE
feb90a5a5901        bridge              bridge              local
de3c98c59ba6        docker_gwbridge     bridge              local
d7bd500d1822        host                host                local
d09ac0b6fed4        none                null                local
9d4c66170ea0        test1               overlay             global
```

Then add docker containers on test1 network.

```
# for host1 as spark master
docker run -it --net=test1 --ip=10.0.1.10 -p 18080:8080 -p 17077:7077 -p 18888:8888 -p 18081:8081 -p 14040:4040 -p 17001:7001 -p 17002:7002 \
-p 17003:7003 -p 17004:7004 -p 17005:7005 -p 17006:7006 --name spm -h spm distkeras_master:1.0 /bin/bash

# for host2 as spark slave
docker run -it --net=test1 --ip=10.0.1.20 --link=spm:master -p 28080:8080 -p 27077:7077 -p 28888:8888 -p 28081:8081 -p 24040:4040 -p 27001:7001 \
-p 27002:7002 -p 27003:7003 -p 27004:7004 -p 27005:7005 -p 27006:7006 --name sps1 -h sps1 distkeras_slave:1.0 /bin/bash
```

Now the docker containers are running on host1 and host2 with connection to test1 network.
You are now ready to run mnist.py as did in the previous section.
