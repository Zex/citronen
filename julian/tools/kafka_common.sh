#!/bin/bash

export pkg_base="/opt"

export grandle_src="https://downloads.gradle.org/distributions/gradle-4.4-bin.zip"
export grandle_target="$pkg_base/gradle-4.4-bin.zip"
export grandle_root="$pkg_base/gradle-4.4"

export kafka_src="http://mirrors.hust.edu.cn/apache/kafka/1.0.0/kafka1.0.0-src.tgz"
export kafka_target="$pkg_base/kafka1.0.0-src.tgz"
export kafka_root="$pkg_base/kafka1.0.0-src"

export zk_src="http://mirrors.shuosc.org/apache/zookeeper/zookeeper-3.5.3-beta/zookeeper-3.5.3-beta.tar.gz"
export zk_target="$pkg_base/zookeeper-3.5.3-beta.tar.gz"
export zk_root="$pkg_base/zookeeper-3.5.3-beta"
export clean_list=""

function install_grandle() {
  if [ -f $grandle_root/bin/grandle ] ;then 
    echo "grandle installed in $grandle_root"
    return 0
  fi

  if [ ! -f $grandle_target ] ;then
    echo "$grandle_target not found, downloading"
    curl -o "$grandle_target" "$grandle_src" 
    if [ ! -f $grandle_target ] ;then echo "$grandle_target not found"; return 1;fi
  fi

  unzip -x $grandle_target -d $pkg_base
  if [ ! -f $grandle_root ] ;then echo "$grandle_root not found"; return 1;fi
  export clean_list="$clean_list $grandle_target $grandle_root"
  return 0
}


function install_kafka() {
  if [ -f $kafka_root/bin/kafka-server-start.sh ] ;then
    echo "kafka installed in $kafka_root"
    return 0
  fi

  if [ ! -f $kafka_target ] ;then
    echo "$kafka_target not found, downloading"
    curl -o "$kafka_target" "$kafka_src"
    if [ ! -f $kafka_target ] ;then echo "$kafka_target not found"; return 1;fi
  fi

  tar xf $kafka_target -C $pkg_base
  if [ ! -f $kafka_target ] ;then echo "$kafka_target not found"; return 1;fi
  pushd $kafka_root
  $grandle_root/bin/grandle jar
  popd
  export clean_list="$clean_list $kafka_target"
  return 0
}


function install_zookeeper() {
  if [ -f $zk_root/bin/zkServer.sh ] ;then
    echo "zk installed in $zk_root"
    return 0
  fi

  if [ ! -f $zk_target ] ;then
    echo "$zk_target not found, downloading"
    curl -o "$zk_target" "$zk_src"
    if [ ! -f $zk_target ] ;then echo "$zk_target not found"; return 1;fi
  fi

  tar xf $zk_target -C $pkg_base
  if [ ! -f $zk_target ] ;then echo "$zk_target not found"; return 1;fi
  pushd $zk_root
  $grandle_root/bin/grandle jar
  popd
  export clean_list="$clean_list $zk_target"
  return 0
}


function install_pkgs() {
  install_grandle
  if [ ! $? -eq 0 ] ;then echo "failed to install grandle"; return 1; fi

  install_kafka
  if [ ! $? -eq 0 ] ;then echo "failed to install kafka"; return 1; fi

  install_zk
  if [ ! $? -eq 0 ] ;then echo "failed to install zk"; return 1; fi

  return 0
}


function cleanup() {
  for f in "$clean_list"; do
    if [ -f "$f" ] ;then rm -f "$f"; fi
    if [ -d "$f" ] ;then rm -rf "$f"; fi
  done
}


function start_kafka() {
  $kafka_root/bin/zookeeper-server-start.sh -daemon config/zookeeper.properties
  $kafka_root/bin/kafka-server-start.sh -daemon config/server.properties
}
