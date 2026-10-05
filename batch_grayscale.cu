
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = (call);                          \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(error) << "\n";  \
            return 1;                                         \
        }                                                     \
    } while (0)

__global__ void rgbToGrayscale(const unsigned char* input,
                               unsigned char* output,
                               int width,
                               int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int pixel = y * width + x;
    int rgb = pixel * 3;

    unsigned char r = input[rgb];
    unsigned char g = input[rgb + 1];
    unsigned char b = input[rgb + 2];

    output[pixel] = static_cast<unsigned char>(
        0.299f * r + 0.587f * g + 0.114f * b);
}

int main() {
    const std::string inputDirectory = "input_images";
    const std::string outputDirectory = "output_images";

    fs::create_directories(outputDirectory);

    int processed = 0;

    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        if (entry.path().extension() != ".png") {
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Could not read: "
                      << entry.path() << "\n";
            continue;
        }

        int width = image.cols;
        int height = image.rows;

        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

        cv::Mat grayImage(height, width, CV_8UC1);

        size_t rgbSize = width * height * 3 * sizeof(unsigned char);
        size_t graySize = width * height * sizeof(unsigned char);

        unsigned char* deviceInput = nullptr;
        unsigned char* deviceOutput = nullptr;

        CUDA_CHECK(cudaMalloc(&deviceInput, rgbSize));
        CUDA_CHECK(cudaMalloc(&deviceOutput, graySize));

        CUDA_CHECK(cudaMemcpy(
            deviceInput,
            rgbImage.data,
            rgbSize,
            cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        rgbToGrayscale<<<grid, block>>>(
            deviceInput,
            deviceOutput,
            width,
            height);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            grayImage.data,
            deviceOutput,
            graySize,
            cudaMemcpyDeviceToHost));

        std::string outputPath =
            outputDirectory + "/" + entry.path().stem().string()
            + "_gray.png";

        cv::imwrite(outputPath, grayImage);

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        processed++;

        std::cout << "Processed image "
                  << processed << ": "
                  << entry.path().filename() << "\n";
    }

    std::cout << "\nGPU batch processing completed.\n";
    std::cout << "Total images processed: "
              << processed << "\n";

    return 0;
}
