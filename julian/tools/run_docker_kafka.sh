#!/bin/bash
source ./tools/kafka_common.sh
source ./tools/docker_common.sh
#source ./tools/env_julian.sh

docker-current run \
    --name julian-broker\
    --net bridge \
    --env-file tools/env_julian.sh \
    --link julian-zk \
    -p 17811:17811 \
    -d \
    -t $local_tag \
    /opt/julian/tools/start_kafka.sh
