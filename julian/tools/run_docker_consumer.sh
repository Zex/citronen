#!/bin/bash
source ./tools/kafka_common.sh
source ./tools/docker_common.sh
#source ./tools/env_julian.sh


docker-current run \
    --name julian-output \
    --env-file ./tools/env_julian.sh \
    --link julian-broker \
    -it $local_tag \
    /opt/julian/tools/start_output.sh
