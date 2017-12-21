#!/bin/bash

branch=`git rev-parse --abbrev-ref HEAD`
revision=`git rev-parse --short=6 HEAD`
version=0.0.1-$branch-$revision
project=julian
local_tag="$project:$version"
remote_tag="local-dtr.zhihuiya.com/360/$local_tag"


function pre_build() {
  find . -name "__pycache__" -exec rm -rf {} \; &> /dev/null
}


function build_img() {
  tag=$1
  docker build . -t $tag
}


function push_remote() {
  local_tag=$1
  remote_tag=$2
  docker push $local_tag $remote_tag
}

