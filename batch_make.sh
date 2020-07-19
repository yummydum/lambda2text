#!/bin/bash
make ccg2lambda file=mnli_3.txt gpu=0 > /dev/null 2>&1 &
make ccg2lambda file=mnli_1.txt gpu=1 > /dev/null 2>&1 &
make ccg2lambda file=mnli_2.txt gpu=2 > /dev/null 2>&1 &

