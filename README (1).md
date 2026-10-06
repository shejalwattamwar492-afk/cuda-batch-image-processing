# CUDA GPU Image Processing

## Project Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA.

The program uses CUDA kernels to perform:
- Grayscale image processing
- Sobel edge detection
- Parallel pixel computation on the GPU

## Technologies

- CUDA C++
- NVIDIA CUDA Toolkit
- Google Colab CUDA GPU

## How to Compile

```bash
nvcc -O2 cuda_image_project.cu -o cuda_image_project
