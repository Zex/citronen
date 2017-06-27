#!/bin/bash
get_bbc_fulltext() {
	if [ ! -d 'data/bbc' ] ;then
    pkg="bbc-fulltext.zip"
  	pushd data
  	curl http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip -o $pkg
  	unzip $pkg
    rm -f $pkg
  	popd
	fi
}
get_bbc_fulltext
