#!/bin/bash
# run this on spark master container

HOSTNAME=192.168.10.10
MASTER=spark://${HOSTNAME}:${SPARK_MASTER_PORT}

${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-slave.sh -c 1 -m 2G ${MASTER}

export PYSPARK_SUBMIT_ARGS="--master ${MASTER}"

jupyter notebook --NotebookApp.token=''  --allow-root
