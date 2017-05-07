#!/bin/bash
# run this script on spark slave container

MASTER_IP=192.168.10.10
MASTER_PORT=7077
MASTER=spark://${MASTER_IP}:${MASTER_PORT}

${SPARK_HOME}/sbin/start-slave.sh -c 1 -m 2G ${MASTER}

export PYSPARK_SUBMIT_ARGS="--master ${MASTER}"

tail -f /dev/null
