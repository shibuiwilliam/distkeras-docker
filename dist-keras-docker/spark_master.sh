#!/bin/bash
# run this script on spark master container

HOSTNAME=`hostname`
MASTER=spark://${HOSTNAME}:${SPARK_MASTER_PORT}

${SPARK_HOME}/sbin/start-master.sh
${SPARK_HOME}/sbin/start-slave.sh -c 1 -m 3G ${MASTER}

export PYSPARK_SUBMIT_ARGS="--master ${MASTER}"

nohup jupyter notebook --NotebookApp.token=''  --allow-root &
