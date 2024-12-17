#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <max_iter>" << std::endl;
        return 1;
    }

    int max_iter = std::stoi(argv[1]);

    int width = 1000;
    int height = 800;

    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;

    std::string output_image = "mandelbrot_serial_" + std::to_string(max_iter) + ".png";

    std::vector<std::vector<int>> mandelbrot_points(height, std::vector<int>(width, 0));

    auto start_time = std::chrono::high_resolution_clock::now();
    computeMandelbrotPoints(width, height, xmin, xmax, ymin, ymax, max_iter, mandelbrot_points);
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
