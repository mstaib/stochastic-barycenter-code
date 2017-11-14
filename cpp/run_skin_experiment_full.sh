#!/bin/bash

time mpirun -np 2 ~/barycenter_mpi -e skin -i 100000 -N 100000 -d 1000000 -w 0 -b 100000 -k 5 -a 0.05 -f true -o output-skin/
