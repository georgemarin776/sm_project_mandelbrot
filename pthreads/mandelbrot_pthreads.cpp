#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <pthread.h>

struct ThreadArgs {
    int start_row, end_row, width, height, max_iter;
    double xmin, xmax, ymin, ymax;
    std::vector<std::vector<int>>* mandelbrot_points;
};

void* computeMandelbrotThread(void* args) {
    auto* thread_args = static_cast<ThreadArgs*>(args);

    for (int y = thread_args->start_row; y < thread_args->end_row; ++y) {
        for (int x = 0; x < thread_args->width; ++x) {
            double real = thread_args->xmin + (x / (double)thread_args->width) * (thread_args->xmax - thread_args->xmin);
            double imag = thread_args->ymin + (y / (double)thread_args->height) * (thread_args->ymax - thread_args->ymin);
            std::complex<double> c(real, imag);

            std::complex<double> z = 0;
            int n = 0;
            while (std::abs(z) <= 2.0 && n < thread_args->max_iter) {
                z = z * z + c;
                n++;
            }

            (*thread_args->mandelbrot_points)[y][x] = n;
        }
    }

    pthread_exit(nullptr);
}

void computeMandelbrotPoints(int width, int height, double xmin, double xmax, double ymin, double ymax, int max_iter, std::vector<std::vector<int>>& mandelbrot_points, int num_threads) {
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> thread_args(num_threads);

    int rows_per_thread = height / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_args[i] = {
            i * rows_per_thread,
            (i == num_threads - 1) ? height : (i + 1) * rows_per_thread,
            width,
            height,
            max_iter,
            xmin,
            xmax,
            ymin,
            ymax,
            &mandelbrot_points
        };
        pthread_create(&threads[i], nullptr, computeMandelbrotThread, &thread_args[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}

void saveMandelbrotImage(int width, int height, const std::vector<std::vector<int>>& mandelbrot_points, int max_iter, const std::string& output_image) {
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int n = mandelbrot_points[y][x];
            cv::Vec3b color;

            if (n == max_iter) {
                color = cv::Vec3b(0, 0, 0);
            } else {
                int hue = (int)(log(n + 1) * 255 / log(max_iter));
                color = cv::Vec3b(hue, hue, hue);
            }

            image.at<cv::Vec3b>(y, x) = color;
        }
    }

    cv::Mat image_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_HSV2BGR);

    cv::imwrite(output_image, image_bgr);
    std::cout << "Mandelbrot set generated and saved as " << output_image << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <max_iter> <num_threads>" << std::endl;
        return 1;
    }

    int max_iter = std::stoi(argv[1]);
    int num_threads = std::stoi(argv[2]);

    int width = 1000;
    int height = 800;

    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;

    std::string output_image = "mandelbrot_pthreads_" + std::to_string(max_iter) + "_threads_" + std::to_string(num_threads) + ".png";
    std::vector<std::vector<int>> mandelbrot_points(height, std::vector<int>(width, 0));

    auto start_time = std::chrono::high_resolution_clock::now();
    computeMandelbrotPoints(width, height, xmin, xmax, ymin, ymax, max_iter, mandelbrot_points, num_threads);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time to compute Mandelbrot set: " << compute_duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    saveMandelbrotImage(width, height, mandelbrot_points, max_iter, output_image);
    end_time = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time to save image: " << save_duration.count() << " ms" << std::endl;

    return 0;
}
