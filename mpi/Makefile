build:
	mpic++ -o mandelbrot_mpi mandelbrot_mpi.cpp `pkg-config --cflags --libs opencv4` -pg -O3

clean:
	rm -f mandelbrot_mpi

# mpirun -np 4 ./mandelbrot_mpi 1000