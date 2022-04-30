#ifndef IP
#define IP

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

#define NO_CHANNELS 3
#define MAX_WINDOW_SIZE 121

#define M_PI 3.14159265358979323846

enum ArithmeticOperator {ADD, SUBTRACT, MULTIPLY, DIVIDE};
enum Direction {DIR_X, DIR_Y};

int divUp(int a, int b);


// CUDA Kernels
void startCUDA_Scale(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_CombineArithmetic(cv::cuda::GpuMat& src_1, cv::cuda::GpuMat& src_2, cv::cuda::GpuMat& dst,
                            ArithmeticOperator type, float offset, float scale, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_Convolution_2D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_Convolution_1D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, Direction type, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_Denoise(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float percentage, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_shift_Hue(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float hue, int BLOCKDIM_X, int BLOCKDIM_Y);

void startCUDA_PDE(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float tau, float alpha, int BLOCKDIM_X, int BLOCKDIM_Y);

/************************************Image******************************************************/
class Image {
	public:
		Image();
		Image(std::string filename);
        Image(cv::Mat img);
		~Image();

        Image operator=(const Image &img);

        Image scale_Bilinear(int scale_factor);
	    Image Combine_Arithmetic(Image img, ArithmeticOperator type, float offset, float scale);
        Image Blur(int kernel_size, float sigma);
        Image laplacian_Filter();
        Image gaussian_Separable(int kernel_size, float sigma);
        Image denoise(int kernel_size, float percentage);
        Image shift_Hue(float hue);

        Image PDE(int iter, float tau, float alpha);

        // Display Image
		void display(std::string text);

    private:
        // Initialization Method
		void init(std::string filename);

        cv::cuda::GpuMat get_2D_GaussianKernel(int kernel_size, float sigma) const; // creating gaussian kernel for GPU Memory
        cv::cuda::GpuMat get_1D_GaussianKernel(int kernel_size, float sigma) const; // creating gaussian kernel for GPU Memory

        cv::Mat host_image;
        cv::cuda::GpuMat device_image;
};

#endif