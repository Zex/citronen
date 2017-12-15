#!/bin/bash

source ./tools/docker_common.sh

cat > version.txt <<< "$local_tag"
pre_build
build_img $local_tag 
#push_remote $local_tag $remote_tag
