#!/bin/bash

pushd ../zookeeper-3.5.3-beta/bin
zkServer.sh start
popd

sudo ./bin/kafka-server-start.sh config/server.properties
