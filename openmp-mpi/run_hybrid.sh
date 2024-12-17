#!/bin/bash

# mpirun -np <num_processes> ./mandelbrot_openmp-mpi <max_iter> <processes> <threads_per_process>

if [ -z "$3" ]
then
    threads_per_process=2
else
    threads_per_process=$3
fi

mpirun -np $2 ./mandelbrot_openmp-mpi $1 $2 $threads_per_process