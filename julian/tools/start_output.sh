#!/bin/bash

cat $JULIAN_ROOT/version.txt

python3 $JULIAN_ROOT/julian/output/DTConsumer.py
