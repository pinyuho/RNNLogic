#!/bin/bash

mkdir -p logs

# 寫進 stdout 檔（log）＋ stderr 檔（error）
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
  # > logs/rule_mining.out.log \
  # 2> logs/rule_mining.err.log

# echo "✅ Done! stdout → rule_mining.out.log / stderr → rule_mining.err.log"
