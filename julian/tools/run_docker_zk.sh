#!/bin/bash
source ./tools/kafka_common.sh
source ./tools/docker_common.sh
#source ./tools/env_julian.sh

docker-current run \
    --name julian-zk\
    --net bridge \
    --env-file tools/env_julian.sh \
    -p 17310:17310 \
    -d \
    -t $local_tag \
    /opt/julian/tools/start_zk.sh
