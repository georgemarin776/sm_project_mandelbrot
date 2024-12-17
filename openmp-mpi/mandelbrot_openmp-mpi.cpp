#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

void computeMandelbrotPointsHybrid(int width, int height, double xmin, double xmax, 
                                    double ymin, double ymax, int max_iter, 
                                    std::vector<int>& local_mandelbrot_points, 
                                    int start_row, int num_rows) {
    #pragma omp parallel for collapse(2) shared(local_mandelbrot_points)
    for (int y = start_row; y < start_row + num_rows; ++y) {
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

            local_mandelbrot_points[(y - start_row) * width + x] = n;
        }
    }
}

void saveMandelbrotImage(int width, int height, const std::vector<int>& mandelbrot_points, 
                          int max_iter, const std::string& output_image) {
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int n = mandelbrot_points[y * width + x];
            cv::Vec3b color;

            if (n == max_iter) {
                color = cv::Vec3b(0, 0, 0);
            } else {
                int hue = (int)(log(n + 1) * 255 / log(max_iter));
                color = cv::Vec3b(hue, hue, hue);
            }

            #pragma omp critical
            {
                image.at<cv::Vec3b>(y, x) = color;
            }
        }
    }

    cv::Mat image_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_HSV2BGR);

    cv::imwrite(output_image, image_bgr);
    std::cout << "Mandelbrot set generated and saved as " << output_image << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    // Get process rank and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <num_processes> " << argv[0] 
                      << " <max_iter> <processes> <threads_per_process>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Parse parameters
    int max_iter = std::stoi(argv[1]);
    int processes = std::stoi(argv[2]);
    int threads_per_process = std::stoi(argv[3]);
    omp_set_num_threads(threads_per_process);

    int width = 1000;
    int height = 800;
    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;

    // Distribute rows among processes
    int rows_per_process = height / size;
    int extra_rows = height % size;

    // Calculate local rows for this process
    int start_row = rank * rows_per_process + std::min(rank, extra_rows);
    int num_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate local mandelbrot points
    std::vector<int> local_mandelbrot_points(num_rows * width);

    // Compute local mandelbrot points using hybrid approach
    auto start_time = std::chrono::high_resolution_clock::now();
    computeMandelbrotPointsHybrid(width, height, xmin, xmax, ymin, ymax, 
                                   max_iter, local_mandelbrot_points, 
                                   start_row, num_rows);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (rank == 0) {
        std::cout << "Time to compute Mandelbrot set: " << compute_duration.count() << " ms" << std::endl;
        std::cout << "Number of processes: " << processes 
                  << ", Threads per process: " << threads_per_process << std::endl;
    }

    // Gather results to root process
    std::vector<int> global_mandelbrot_points;
    std::vector<int> recvcounts, displs;

    if (rank == 0) {
        global_mandelbrot_points.resize(width * height);
        recvcounts.resize(size);
        displs.resize(size);

        for (int i = 0; i < size; i++) {
            int rows = rows_per_process + (i < extra_rows ? 1 : 0);
            recvcounts[i] = rows * width;
            displs[i] = i * rows_per_process * width + std::min(i, extra_rows) * width;
        }
    }

    // Gather all points to root process
    MPI_Gatherv(local_mandelbrot_points.data(), num_rows * width, MPI_INT,
                global_mandelbrot_points.data(), recvcounts.data(), 
                displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // Save image from root process
    if (rank == 0) {
        std::string output_image = "mandelbrot_hybrid_" + std::to_string(max_iter) + 
                                    "_processes_" + std::to_string(processes) + 
                                    "_threads_" + std::to_string(threads_per_process) + ".png";
        start_time = std::chrono::high_resolution_clock::now();
        saveMandelbrotImage(width, height, global_mandelbrot_points, max_iter, output_image);
        end_time = std::chrono::high_resolution_clock::now();
        auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Time to save image: " << save_duration.count() << " ms" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}