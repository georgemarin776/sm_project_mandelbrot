# Mandelbrot Set

The Mandelbrot set algorithm iterates over a grid of complex numbers, applying the equation `z(n+1) = z(n)^2 + c` to each point, checking if it escapes to infinity or remains bounded.

# Main Algorithm

The main algorithm is implemented in the `computeMandelbrotPoints` function. This function takes the width and height of the grid, the minimum and maximum values for the real and imaginary parts of the complex numbers, the maximum number of iterations, and a 2D vector to store the number of iterations for each point. The function iterates over the grid, computes the complex number for each point, and applies the Mandelbrot set algorithm to determine the number of iterations for each point.

The focus of the parallelization is on the outer loops of the function, which was done using the following strategies:
- OpenMP
- MPI
- Pthreads
- OpenMP / MPI hybrid

```cpp

void computeMandelbrotPoints(int width, int height, double xmin, double xmax, double ymin, double ymax, int max_iter, std::vector<std::vector<int>>& mandelbrot_points) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double real = xmin + (x / (double)width) * (xmax - xmin);
            double imag = ymin + (y / (double)height) * (ymax - ymin);
            std::complex<double> c(real, imag);

            std::complex<double> z = 0;
            int n = 0;
            while (abs(z) <= 2.0 && n < max_iter) {
                z = z * z + c;
                n++;
            }

            mandelbrot_points[y][x] = n;
        }
    }
}

```

By far, the easiest to implement was the OpenMP version, which required only a few lines of code to parallelize the outer loops of the function. The other versions required some changes to the algorithm to distribute the work among the threads or processes - basically splitting the grid into smaller chunks and assigning each chunk to a thread or process. The hybrid version combined OpenMP and MPI to parallelize the outer loops using OpenMP and the grid distribution using MPI.

Performance metrics are included in each folder to compare the performance of the different parallelization strategies.

# How to Run

Performance metrics were collected using a series of python scripts. To better ilustrate how to run the algorithms, python snippets will be used.

## Serial

```python

iteration_values = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
compute_times = []
save_times = []

for iterations in iteration_values:
    compute_time_runs = []
    save_time_runs = []
    
    for _ in range(10):
        result = subprocess.run([f"./mandelbrot_serial", str(iterations)], capture_output=True, text=True) # <--------------------------------
        output = result.stdout

        compute_time = int(re.search(r"Time to compute Mandelbrot set: (\d+) ms", output).group(1))
        save_time = int(re.search(r"Time to save image: (\d+) ms", output).group(1))
        
        compute_time_runs.append(compute_time)
        save_time_runs.append(save_time)
    
    compute_times.append(np.mean(compute_time_runs))
    save_times.append(np.mean(save_time_runs))
```

## OpenMP

```python
        result = subprocess.run([f"./mandelbrot_openmp", str(iterations), str(threads)], capture_output=True, text=True)
```

## MPI

```python
        result = subprocess.run([f"mpirun -np {str(threads)} ./mandelbrot_mpi {str(iterations)} {str(threads)}"], capture_output=True, text=True, shell=True)
```

## Pthreads

```python
        result = subprocess.run([f"./mandelbrot_pthreads", str(iterations), str(threads)], capture_output=True, text=True)
```

## OpenMP / MPI Hybrid

```python
        result = subprocess.run([f"mpirun -np {str(processes)} ./mandelbrot_openmp-mpi {str(iterations)} {str(processes)} {str(threads)}"], capture_output=True, text=True, shell=True)
```

