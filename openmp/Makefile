build:
	g++ -o mandelbrot_openmp mandelbrot_openmp.cpp `pkg-config --cflags --libs opencv4` -pg -O3 -fopenmp

clean:
	rm -f mandelbrot_openmp