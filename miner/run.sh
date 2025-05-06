#!/bin/bash

mkdir -p logs

./rnnlogic \
  -data-path ../data/semmed \
  -max-length 3 \
  -threads 40 \
  -lr 0.01 \
  -wd 0.0005 \
  -temp 100 \
  -iterations 1 \
  -top-n 0 \
  -top-k 0 \
  -top-n-out 0 \
  -output-file mined_rules.txt \d
