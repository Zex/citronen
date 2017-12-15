#!/bin/bash

source ./tools/kafka_common.sh

install_pkgs
if [ $? -eq 0 ] ;then 
  cleanup 
fi
