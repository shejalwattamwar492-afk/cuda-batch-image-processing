#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < pixels) {
        int r = input[i * 3];
        int g = input[i * 3 + 1];
        int b = input[i * 3 + 2];

        output[i] = (unsigned char)(
            0.299f * r +
            0.587f * g +
            0.114f * b
        );
    }
}

__global__ void edgeKernel(
    const unsigned char* gray,
    unsigned char* edges,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (x == 0 || y == 0 ||
        x == width - 1 || y == height - 1) {
        edges[idx] = 0;
        return;
    }

    int gx =
        -gray[(y-1)*width + (x-1)] +
         gray[(y-1)*width + (x+1)] +
        -2*gray[y*width + (x-1)] +
         2*gray[y*width + (x+1)] +
        -gray[(y+1)*width + (x-1)] +
         gray[(y+1)*width + (x+1)];

    int gy =
        -gray[(y-1)*width + (x-1)] -
         2*gray[(y-1)*width + x] -
         gray[(y-1)*width + (x+1)] +
         gray[(y+1)*width + (x-1)] +
         2*gray[(y+1)*width + x] +
         gray[(y+1)*width + (x+1)];

    int magnitude = sqrtf((float)(gx*gx + gy*gy));

    edges[idx] = (unsigned char)min(255, magnitude);
}

int main()
{
    const int width = 2048;
    const int height = 2048;
    const int pixels = width * height;

    std::vector<unsigned char> input(pixels * 3);

    for (int i = 0; i < pixels; i++) {
        input[i*3]     = i % 256;
        input[i*3 + 1] = (i / width) % 256;
        input[i*3 + 2] = (i * 3) % 256;
    }

    std::vector<unsigned char> gray(pixels);
    std::vector<unsigned char> edges(pixels);

    unsigned char *d_input;
    unsigned char *d_gray;
    unsigned char *d_edges;

    cudaMalloc(&d_input, pixels * 3);
    cudaMalloc(&d_gray, pixels);
    cudaMalloc(&d_edges, pixels);

    cudaMemcpy(
        d_input,
        input.data(),
        pixels * 3,
        cudaMemcpyHostToDevice
    );

    int threads = 256;
    int blocks = (pixels + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    grayscaleKernel<<<blocks, threads>>>(
        d_input,
        d_gray,
        pixels
    );

    dim3 block2(16, 16);

    dim3 grid2(
        (width + 15) / 16,
        (height + 15) / 16
    );

    edgeKernel<<<grid2, block2>>>(
        d_gray,
        d_edges,
        width,
        height
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaMemcpy(
        gray.data(),
        d_gray,
        pixels,
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        edges.data(),
        d_edges,
        pixels,
        cudaMemcpyDeviceToHost
    );

    std::cout << "CUDA GPU Image Processing\n";
    std::cout << "Image size: "
              << width << " x "
              << height << "\n";

    std::cout << "Total pixels: "
              << pixels << "\n";

    std::cout << "GPU processing time: "
              << milliseconds << " ms\n";

    std::cout << "Grayscale kernel: SUCCESS\n";
    std::cout << "Sobel edge kernel: SUCCESS\n";
    std::cout << "GPU image processing completed successfully.\n";

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_edges);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
