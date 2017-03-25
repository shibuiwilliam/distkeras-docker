#!/bin/bash

HOSTNAME=`hostname`
MASTER=spark://${HOSTNAME}:${SPARK_MASTER_PORT}
CORES_PER_WORKER=1

${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-slave.sh -c ${CORES_PER_WORKER} -m 3G ${MASTER}

export PYSPARK_SUBMIT_ARGS="--master ${MASTER}"

nohup jupyter notebook &
