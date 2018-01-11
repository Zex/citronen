#!/bin/bash
source tools/docker_common.sh

build_base=$project-`awk -F- '{print $1}' <<<  $version`
packages="julian
setup.py
tests
README.rst
requirements.txt"


function create_tar() {
  mkdir -p $build_base
  for p in $packages; do
    cp -ra $p $build_base
  done
  cat > $build_base/version.txt <<< $local_tag
  tar cfz $build_base.tar.gz $build_base
  rm -rf $build_base 
}

pre_build
create_tar
