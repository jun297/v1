#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 9501 train.py --distributed "$@"
