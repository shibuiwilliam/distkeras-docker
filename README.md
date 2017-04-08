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
cd distkeras-docker
docker build -t distkeras .
```

After the docker image is successfully built, run docker containers.
Number of containers depend on how many Spark workers you need to add.
The script below deploys one Spark master with worker container and two worker-only containers.

```
# docker dist-keras for spark master and slave
docker run -it -p 18080:8080 -p 17077:7077 -p 18888:8888 -p 18081:8081 -p 14040:4040 -p 17001:7001 -p 17002:7002 \
 -p 17003:7003 -p 17004:7004 -p 17005:7005 -p 17006:7006 --name spmaster -h spmaster distkeras /bin/bash

# docker dist-keras for spark slave1
docker run -it --link spmaster:master -p 28080:8080 -p 27077:7077 -p 28888:8888 -p 28081:8081 -p 24040:4040 -p 27001:7001 \
-p 27002:7002 -p 27003:7003 -p 27004:7004 -p 27005:7005 -p 27006:7006 --name spslave1 -h spslave1 distkeras /bin/bash

# docker dist-keras for spark slave2
docker run -it --link spmaster:master -p 38080:8080 -p 37077:7077 -p 38888:8888 -p 38081:8081 -p 34040:4040 -p 37001:7001 \
-p 37002:7002 -p 37003:7003 -p 37004:7004 -p 37005:7005 -p 37006:7006 --name spslave2 -h spslave2 distkeras /bin/bash
```

On each container, run shellscripts to start Spark cluster.

```
# for Spark master
# Spark master and worker start
sh spark_master.sh

# for Spark worker
# Spark worker starts and added to Spark cluster
sh spark_slave.sh
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
    master = "spark://spm:7077"  # add
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


