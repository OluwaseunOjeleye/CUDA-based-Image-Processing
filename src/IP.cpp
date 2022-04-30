#include "../include/IP.h"

void Image::init(std::string filename) {
	if (filename != "")
        this->host_image = cv::imread(filename);
    this->device_image.upload(this->host_image);
}

Image::Image() {
	init("img/grenouille.jpg");
}

Image::Image(std::string filename) {
	init(filename);
}

Image::Image(cv::Mat img) {
	img.copyTo(this->host_image);
	init("");
}

Image::~Image() {

}

Image Image::operator=(const Image &img) {
	if (this != &img){
        this->host_image = img.host_image;
        this->device_image = img.device_image;
    }
	return  *this;
}

// Scale
Image Image::scale_Bilinear(int scale_factor){
    int width= this->device_image.cols * scale_factor;
    int height = this->device_image.size().height * scale_factor;

    // Creating GPU Memory Locations
    cv::cuda::GpuMat device_result;

    // Compute with GPU
    int BLOCKDIM_X = 8, BLOCKDIM_Y = 8;
    cv::cuda::resize(this->device_image, device_result,cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_Scale(device_image, device_result, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);

    std::cout<<this->device_image.cols<< " X "<< this->device_image.size().height<<std::endl;
    std::cout<<host_result.cols<< " X "<<host_result.size().height<<std::endl;
	return Image(host_result);
}

// Addition Operator
Image Image::Combine_Arithmetic(Image img, ArithmeticOperator type, float offset, float scale){

    // Creating GPU Memory Locations
    cv::cuda::GpuMat device_result;

    int width= this->device_image.cols;
    int height = this->device_image.size().height;

    // Compute with GPU
    int BLOCKDIM_X = 8, BLOCKDIM_Y = 8;
    cv::cuda::resize(this->device_image, device_result,cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_CombineArithmetic(device_image, img.device_image, device_result, type, offset, scale, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

cv::cuda::GpuMat Image::get_2D_GaussianKernel(int kernel_size, float sigma) const{
    int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    cv::Mat_<float> filter(k_size, k_size);

    int size = floor(k_size / 2);
	float s = 2.0*sigma*sigma, sum = 0;

	for (int i = -size; i <= size; i++) {
		for (int j = -size; j <= size; j++) {
			float r		= -(i*i + j * j) / s;
			float Coeff	= exp(r) / (M_PI * s);
			filter[i+size][j+size] = Coeff;
			sum += Coeff;
		}
	}

	for (int i = 0; i < k_size; i++)
		for (int j = 0; j < k_size; j++)
			filter[i][j] /= sum;

    cv::cuda::GpuMat device_filter;
    device_filter.upload(filter);
    return device_filter;
}

Image Image::Blur(int kernel_size, float sigma){
    cv::cuda::GpuMat gaussian_filter = get_2D_GaussianKernel(kernel_size, sigma);

    cv::cuda::GpuMat device_result;     // Creating GPU Memory Locations

    int width= this->device_image.cols;
    int height = this->device_image.size().height;

    // Compute with GPU
    int BLOCKDIM_X = 8, BLOCKDIM_Y = 8;
    cv::cuda::resize(device_image, device_result,cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_Convolution_2D(device_image, gaussian_filter, device_result, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

// Laplacian Filter
Image Image::laplacian_Filter() {
	// Creating Laplacian Kernel
	int kernel_size = 3;
	cv::Mat_<float> filter(kernel_size, kernel_size);

	for (int i = 0; i < kernel_size; i++)
		for (int j = 0; j < kernel_size; j++)
			filter[i][j] = -1.0;
	int size = floor(kernel_size / 2);
	filter[size][size] = 8.0;

    cv::cuda::GpuMat laplacian_filter;
    laplacian_filter.upload(filter);
    
	// Applying Laplacian Filter
    cv::cuda::GpuMat device_result;     // Creating GPU Memory Locations

    int width= this->device_image.cols;
    int height = this->device_image.size().height;

    // Compute with GPU
    int BLOCKDIM_X = 8, BLOCKDIM_Y = 8;
    cv::cuda::resize(device_image, device_result,cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_Convolution_2D(device_image, laplacian_filter, device_result, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

// 1D Gaussian Kernel Generator
cv::cuda::GpuMat Image::get_1D_GaussianKernel(int kernel_size, float sigma) const{
    int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    cv::Mat_<float> filter(k_size, 1);

    int size = floor(k_size / 2);
	float s = 2.0*sigma*sigma, sum = 0;

	for (int i = -size; i <= size; i++) {
        float r		    = -(i * i) / s;
        float Coeff	    = exp(r) / (M_PI * s);
        filter[i+size][0]  = Coeff;
        sum += Coeff;
	}

	for (int i = 0; i < k_size; i++){
		filter[i][0] /= sum;
    }

    cv::cuda::GpuMat device_filter;
    device_filter.upload(filter);
    return device_filter;
}

// Separable Filter
Image Image::gaussian_Separable(int kernel_size, float sigma){
    cv::cuda::GpuMat gaussian_filter = get_1D_GaussianKernel(kernel_size, sigma);

    cv::cuda::GpuMat temp, device_result;     // Creating GPU Memory Locations

    int width= this->device_image.cols;
    int height = this->device_image.size().height;

    cv::cuda::resize(device_image, temp,cv::Size(width, height), cv::INTER_CUBIC);
    cv::cuda::resize(device_image, device_result,cv::Size(width, height), cv::INTER_CUBIC);
    
    // Compute with GPU
    int BLOCKDIM_X = 8, BLOCKDIM_Y = 8;
    // Along X-axis
    startCUDA_Convolution_1D(device_image, gaussian_filter, temp, DIR_X, BLOCKDIM_X, BLOCKDIM_Y);
    // Along Y-axis
    startCUDA_Convolution_1D(temp, gaussian_filter, device_result, DIR_Y, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

// Denoising
Image Image::denoise(int kernel_size, float percentage){
    cv::cuda::GpuMat device_result;     // Creating GPU Memory Locations

    int width= this->device_image.cols;
    int height = this->device_image.size().height;
    
    // Compute with GPU
    int BLOCKDIM_X = 5, BLOCKDIM_Y = 5;
    cv::cuda::resize(device_image, device_result, cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_Denoise(device_image, device_result, kernel_size, percentage, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

Image Image::shift_Hue(float hue){
    cv::cuda::GpuMat device_result;     // Creating GPU Memory Locations

    int width= this->device_image.cols;
    int height = this->device_image.size().height;
    
    // Compute with GPU
    int BLOCKDIM_X = 9, BLOCKDIM_Y = 9;
    cv::cuda::resize(device_image, device_result, cv::Size(width, height), cv::INTER_CUBIC);
    startCUDA_shift_Hue(device_image, device_result, hue, BLOCKDIM_X, BLOCKDIM_Y);

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

Image Image::PDE(int iter, float tau, float alpha){
    cv::cuda::GpuMat temp, device_result;     // Creating GPU Memory Locations
    this->device_image.copyTo(temp);

    int width= this->device_image.cols;
    int height = this->device_image.size().height;
        
    cv::cuda::resize(device_image, device_result, cv::Size(width, height), cv::INTER_CUBIC);
    int BLOCKDIM_X = 9, BLOCKDIM_Y = 9;
    for (int i = 0; i < iter; i++){
		if (i % 10 == 0)
			std::cout << "Iteration : " << i << std::endl;
		// Compute with GPU
        startCUDA_PDE(temp, device_result, tau, alpha, BLOCKDIM_X, BLOCKDIM_Y);
        //startCUDA_PDE(device_result, temp, tau, alpha);
        device_result.copyTo(temp);
	}       

    // Load GPU Data into CPU Memory
    cv::Mat host_result;
    device_result.download(host_result);
	return Image(host_result);
}

void Image::display(std::string text) {
	cv::imshow(text, this->host_image);
}