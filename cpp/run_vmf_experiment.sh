#!/bin/bash

time mpirun -v -np 3 ~/barycenter_mpi -e vmf -i 500000 -N 10000 -d 10000 -a 1 -w 100000 -m 0.00003 -o output-vmf-drift/
