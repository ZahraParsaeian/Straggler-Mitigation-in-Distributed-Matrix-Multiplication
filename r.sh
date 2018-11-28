#!/bin/bash

for i in range{1..15}
do
mpirun -np 18 python main.py
done
