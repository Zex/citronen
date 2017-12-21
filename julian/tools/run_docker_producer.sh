#!/bin/bash
source ./tools/kafka_common.sh
source ./tools/docker_common.sh
#source ./tools/env_julian.sh


docker-current run \
    --name julian-input \
    --env-file ./tools/env_julian.sh \
    -it $local_tag \
    /opt/julian/tools/run_input.sh
