#!/bin/bash
# run this script on spark slave container

MASTER=spark://${MASTER_PORT_7077_TCP_ADDR}:${MASTER_ENV_SPARK_MASTER_PORT}

${SPARK_HOME}/sbin/start-slave.sh -c 1 -m 3G ${MASTER}

export PYSPARK_SUBMIT_ARGS="--master ${MASTER}"

tail -f /dev/null
