#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "../include/IP.h"

int divUp(int a, int b){
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float interpolate(uchar pixel_00, uchar pixel_01, uchar pixel_10, uchar pixel_11, float x_diff, float y_diff){
  float A = float(pixel_00);
	float B = float(pixel_01);
	float C = float(pixel_10);
	float D = float(pixel_11);
	return A * (1 - x_diff) * (1 - y_diff) + B * x_diff*(1 - y_diff) + C * (1 - x_diff) * y_diff + D * x_diff * y_diff;
}

__device__ float3 bilinear_interp(uchar3 pixel_00, uchar3 pixel_01, uchar3 pixel_10, uchar3 pixel_11, float x_diff, float y_diff){
  float3 pixel;
  pixel.x = interpolate(pixel_00.x, pixel_01.x, pixel_10.x, pixel_11.x, x_diff, y_diff);
  pixel.y = interpolate(pixel_00.y, pixel_01.y, pixel_10.y, pixel_11.y, x_diff, y_diff);
  pixel.z = interpolate(pixel_00.z, pixel_01.z, pixel_10.z, pixel_11.z, x_diff, y_diff);
  return pixel;
}

// Scale
__global__ void Scale(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int dst_rows, int dst_cols, int src_rows, int src_cols){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dst_x >= 0 && dst_x < dst_cols && dst_y >= 0 && dst_y < dst_rows){
    float x_ratio = float(src_cols - 1) / dst_cols;
	  float y_ratio = float(src_rows - 1) / dst_rows;

    int x = x_ratio * dst_x;
    int y = y_ratio * dst_y;

    float x_diff = (x_ratio * dst_x) - x;
    float y_diff = (y_ratio * dst_y) - y;

    float3 result = bilinear_interp(src(y, x), src(y, x + 1), src(y + 1, x), src(y + 1, x + 1), x_diff, y_diff);
    
    dst(dst_y, dst_x).x = (unsigned char)result.x;  //B
    dst(dst_y, dst_x).y = (unsigned char)result.y;  //G
    dst(dst_y, dst_x).z = (unsigned char)result.z;  //R
  }
}

void startCUDA_Scale(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Scale<<<numBlocks, threadsPerBlock>>>(src, dst, dst.rows, dst.cols, src.rows, src.cols);
}

// Combine Arithmetic Operator
__global__ void arithmetic_Operator(const cv::cuda::PtrStep<uchar3> src_1, const cv::cuda::PtrStep<uchar3> src_2, cv::cuda::PtrStep<uchar3> dst,
                                    int rows, int cols, ArithmeticOperator type, float offset, float scale){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dst_x < cols && dst_y < rows){
    float3 pixel_1, pixel_2, result;
    pixel_1.x = float(src_1(dst_y, dst_x).x) / 255.0; pixel_2.x = float(src_2(dst_y, dst_x).x) / 255.0;
    pixel_1.y = float(src_1(dst_y, dst_x).y) / 255.0; pixel_2.y = float(src_2(dst_y, dst_x).y) / 255.0;
    pixel_1.z = float(src_1(dst_y, dst_x).z) / 255.0; pixel_2.z = float(src_2(dst_y, dst_x).z) / 255.0;

    if(type == ADD){
      result.x = pixel_1.x + pixel_2.x;
      result.y = pixel_1.y + pixel_2.y;
      result.z = pixel_1.z + pixel_2.z;
    }
    else if(type == SUBTRACT){
      result.x = pixel_1.x - pixel_2.x;
      result.y = pixel_1.y - pixel_2.y;
      result.z = pixel_1.z - pixel_2.z;
    }
    else if(type == MULTIPLY){
      result.x = pixel_1.x * pixel_2.x;
      result.y = pixel_1.y * pixel_2.y;
      result.z = pixel_1.z * pixel_2.z;
    }
    else if(type == DIVIDE){
      result.x = pixel_1.x / pixel_2.x;
      result.y = pixel_1.y / pixel_2.y;
      result.z = pixel_1.z / pixel_2.z;
    }

    result.x += offset;
    result.y += offset;
    result.z += offset;

    result.x *= scale;
    result.y *= scale;
    result.z *= scale;

    dst(dst_y, dst_x).x = (unsigned char)(result.x * 255.0 > 255.0? 255.0: result.x * 255.0);
    dst(dst_y, dst_x).y = (unsigned char)(result.y * 255.0 > 255.0? 255.0: result.y * 255.0);
    dst(dst_y, dst_x).z = (unsigned char)(result.z * 255.0 > 255.0? 255.0: result.z * 255.0);
  }
}

void startCUDA_CombineArithmetic(cv::cuda::GpuMat& src_1, cv::cuda::GpuMat& src_2, cv::cuda::GpuMat& dst,
                                          ArithmeticOperator type, float offset, float scale, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  arithmetic_Operator<<<numBlocks, threadsPerBlock>>>(src_1, src_2, dst, dst.rows, dst.cols, type, offset, scale);
}

// 2D Convolution Operator
__global__ void Convolution_2D(const cv::cuda::PtrStep<uchar3> src, const cv::cuda::PtrStep<float> kernel, cv::cuda::PtrStep<uchar3> dst,
                                    int rows, int cols, int kernel_size){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  const int size = (kernel_size%2 == 0)? kernel_size / 2: (kernel_size-1) / 2;
  if (dst_x < cols && dst_y < rows){
    float pixel_sum[NO_CHANNELS] = {0.0};

    for (int k = -size; k <= size; k++) {
      for (int l = -size; l <= size; l++) {
          pixel_sum[0] += (float)src(dst_y + k, dst_x + l).x * kernel(k + size, l + size);
          pixel_sum[1] += (float)src(dst_y + k, dst_x + l).y * kernel(k + size, l + size);
          pixel_sum[2] += (float)src(dst_y + k, dst_x + l).z * kernel(k + size, l + size);
      }
    }

    dst(dst_y, dst_x).x = (unsigned char)pixel_sum[0];
    dst(dst_y, dst_x).y = (unsigned char)pixel_sum[1];
    dst(dst_y, dst_x).z = (unsigned char)pixel_sum[2];
  }
}

void startCUDA_Convolution_2D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Convolution_2D<<<numBlocks, threadsPerBlock>>>(src, kernel, dst, dst.rows, dst.cols, kernel.rows);
}

// 1D Convolution Operator
__global__ void Convolution_1D(const cv::cuda::PtrStep<uchar3> src, const cv::cuda::PtrStep<float> kernel, cv::cuda::PtrStep<uchar3> dst,
                                    int rows, int cols, int kernel_size, Direction type){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  const int size = (kernel_size%2 == 0)? kernel_size / 2: (kernel_size-1) / 2;
  if (dst_x < cols && dst_y < rows){
    float pixel_sum[NO_CHANNELS] = {0.0};

    if(type == DIR_X){
      for (int k = -size; k <= size; k++) {
        pixel_sum[0] += (float)src(dst_y + k, dst_x).x * kernel(0, k + size);
        pixel_sum[1] += (float)src(dst_y + k, dst_x).y * kernel(0, k + size);
        pixel_sum[2] += (float)src(dst_y + k, dst_x).z * kernel(0, k + size);
      }
    }
    else if(type == DIR_Y){
      for (int l = -size; l <= size; l++) {
        pixel_sum[0] += (float)src(dst_y, dst_x + l).x * kernel(0, l + size);
        pixel_sum[1] += (float)src(dst_y, dst_x + l).y * kernel(0, l + size);
        pixel_sum[2] += (float)src(dst_y, dst_x + l).z * kernel(0, l + size);
      }
    }
   
    dst(dst_y, dst_x).x = (unsigned char)pixel_sum[0];
    dst(dst_y, dst_x).y = (unsigned char)pixel_sum[1];
    dst(dst_y, dst_x).z = (unsigned char)pixel_sum[2];
  }
}

void startCUDA_Convolution_1D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, Direction type, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Convolution_1D<<<numBlocks, threadsPerBlock>>>(src, kernel, dst, dst.rows, dst.cols, kernel.rows, type);
}

// Denoise
__device__ void sort(float *array, int window_size){
	for (int i = 0; i < window_size - 1; i++) {
		for (int j = 0; j < window_size - i - 1; j++) {
			if (array[j] > array[j + 1]) {
				float temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
}

__global__ void Denoise(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, 
                        int kernel_size, float percentage){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < cols && dst_y < rows){
    const int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;  // Odd kernel size
    const int window_size = k_size * k_size;
    const int size = (kernel_size - 1) / 2;

    // Get array when window is convolved over image
    float R_array[MAX_WINDOW_SIZE];
    float G_array[MAX_WINDOW_SIZE];
    float B_array[MAX_WINDOW_SIZE];
    int counter = 0;

    for (int k = -size; k <= size; k++) {
      for (int l = -size; l <= size; l++) {
        B_array[counter] = (float)src(dst_y + k, dst_x + l).x;
        G_array[counter] = (float)src(dst_y + k, dst_x + l).y;
        R_array[counter] = (float)src(dst_y + k, dst_x + l).z;
        counter++;
      }
    }

    // Sort neighbors (RGB arrays) in ascending order
    sort(B_array, window_size);
    sort(G_array, window_size);
    sort(R_array, window_size);

    // Get median - since window_size is always odd, index = window_size/2
    int index = int(window_size / 2);

    // Using either Median Filter or Average Filter
    int length = (window_size * percentage / 100) / 2.0;

    float r = 0.0, g = 0.0, b = 0.0;
    for (int i = index - length; i <= index + length; i++) {
      b += B_array[i];
      g += G_array[i];
      r += R_array[i];
    }
    dst(dst_y, dst_x).x = (unsigned char)(b / float(2 * length + 1));
    dst(dst_y, dst_x).y = (unsigned char)(g / float(2 * length + 1));
    dst(dst_y, dst_x).z = (unsigned char)(r / float(2 * length + 1));
  }
}

void startCUDA_Denoise(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float percentage, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Denoise<<<numBlocks, threadsPerBlock>>>(src, dst, dst.rows, dst.cols, kernel_size, percentage);
}

// Color Transformation
__device__ float F(float v){
  const float E = 0.2068965517241379;
	if (v > powf(E, 3.0)) return powf(v, 1.0 / 3.0);
	return ((v / (3.0*powf(E, 2.0))) + 0.1379310344827586);
}

__device__ float F_inverse(float v){
  const float E = 0.2068965517241379;
	if (v > E) return pow(v, 3.0);
	return (3.0*pow(E, 2.0)) * (v - 0.1379310344827586);
}

// Linearize Color
__device__ float linearize(uchar color){
  float c = (float)color / 255.0;
	if (c <= 0.04045) c = (c / 12.92) * 255.0;
  else c = pow((c + 0.055) / 1.055, 2.4) * 255.0;
  return c > 255.0 ? 255.0: c;
}

__device__ float linearize_inverse(float color){
  float c = (float)color / 255.0;
	if (c <= 0.0031308) c = 12.92*c * 255.0;
	else                c = ((1.055 * pow(c, 1.0 / 2.4)) - 0.055) * 255.0;
  return c > 255.0 ? 255.0: c;
}

// rgb to Lab
__device__ float3 rgb_to_LCh(uchar3 pixel, float hue){
  
  // rgb to XYZ
  float3 l_pixel; //l_pixel.x = pixel.z; l_pixel.y = pixel.y; l_pixel.z = pixel.x; 
  l_pixel.x = linearize(pixel.z); l_pixel.y = linearize(pixel.y); l_pixel.z = linearize(pixel.x);
  
  float3 XYZ;
  XYZ.x = 0.4124564 * l_pixel.x + 0.3575761 * l_pixel.y + 0.1804375 * l_pixel.z;
  XYZ.y = 0.2126729 * l_pixel.x + 0.7151522 * l_pixel.y + 0.0721750 * l_pixel.z;
  XYZ.z = 0.0193339 * l_pixel.x + 0.1191920 * l_pixel.y + 0.9503041 * l_pixel.z;

  // XYZ to Lab
  float Fx = F(XYZ.x / 95.04492182750991);
	float Fy = F(XYZ.y / 100);
	float Fz = F(XYZ.z / 108.89166484304715);

  // Lab to LCh
  float3 lab;
  lab.x = (116.0*Fy) - 16.0;
  lab.y = 500.0*(Fx - Fy);
  lab.z = 200.0*(Fy - Fz);

  float c = sqrtf((lab.y * lab.y) + (lab.z * lab.z));
	float h = (atan2f(lab.z, lab.y) * 180 / M_PI) + hue;
	h = h > 360.0 ? h - 360.0 : h;

  float3 lch; lch.x = lab.x; lch.y = c; lch.z = h*M_PI / 180;
	return lch;
}

__device__ float3 LCh_to_rgb(float3 lch){
  float3 Lab;
  Lab.x = lch.x; Lab.y = lch.y * cosf(lch.z); Lab.z = lch.y * sinf(lch.z); 

	float L_star = (Lab.x + 16.0) / 116.0;
	float a_star = Lab.y / 500.0;
	float b_star = Lab.z / 200.0;

	float3 XYZ; 
  XYZ.x = 95.04492182750991 * F_inverse(L_star + a_star);
	XYZ.y = 100 * F_inverse(L_star);
	XYZ.z = 108.89166484304715 * F_inverse(L_star - b_star);

  float3 il_pixel;
  il_pixel.x = 3.2404542 * XYZ.x - 1.5371385 * XYZ.y - 0.4985314 * XYZ.z;  // R
  il_pixel.y = -0.9692660 * XYZ.x + 1.8760108 * XYZ.y + 0.0415560 * XYZ.z; // G
  il_pixel.z = 0.0556434 * XYZ.x - 0.2040259 * XYZ.y + 1.0572252 * XYZ.z;  // B

  float3 pixel;
  pixel.x = linearize_inverse(il_pixel.z); pixel.y = linearize_inverse(il_pixel.y); pixel.z = linearize_inverse(il_pixel.x); 
  return pixel;
}

__global__ void shift_Hue(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float hue){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dst_x < cols && dst_y < rows){
    float3 result = LCh_to_rgb(rgb_to_LCh(src(dst_y, dst_x), hue));

    dst(dst_y, dst_x).x = (unsigned char)result.x;  //B
    dst(dst_y, dst_x).y = (unsigned char)result.y;  //G
    dst(dst_y, dst_x).z = (unsigned char)result.z;  //R
  }
}

void startCUDA_shift_Hue(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float hue, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  shift_Hue<<<numBlocks, threadsPerBlock>>>(src, dst, dst.rows, dst.cols, hue);
}

__device__ float3 dfX_0(uchar3 pixel_a1_0, uchar3 pixel_s1_0) {
  float3 dFx;
  dFx.x = float(pixel_a1_0.x / 255.0 - pixel_s1_0.x / 255.0) / 2.0;
  dFx.y = float(pixel_a1_0.y / 255.0 - pixel_s1_0.y / 255.0) / 2.0;
  dFx.z = float(pixel_a1_0.z / 255.0 - pixel_s1_0.z / 255.0) / 2.0;
  return dFx;
}

__device__ float3 dfY_0(uchar3 pixel_0_a1, uchar3 pixel_0_s1){
	float3 dFy;
  dFy.x = (float(pixel_0_a1.x) / 255.0f - float(pixel_0_s1.x) / 255.0f) / 2.0;
  dFy.y = (float(pixel_0_a1.y) / 255.0f - float(pixel_0_s1.y) / 255.0f) / 2.0;
  dFy.z = (float(pixel_0_a1.z) / 255.0f - float(pixel_0_s1.z) / 255.0f) / 2.0;
  return dFy;
}

__device__ float3 dfX2(uchar3 pixel_a1_0, uchar3 pixel_0_0, uchar3 pixel_s1_0) {
  float3 dFx2;
	dFx2.x = float(pixel_a1_0.x / 255.0f) - 2.0 * float(pixel_0_0.x / 255.0f) + float(pixel_s1_0.x / 255.0f);
  dFx2.y = float(pixel_a1_0.y / 255.0f) - 2.0 * float(pixel_0_0.y / 255.0f) + float(pixel_s1_0.y / 255.0f);
  dFx2.z = float(pixel_a1_0.z / 255.0f) - 2.0 * float(pixel_0_0.z / 255.0f) + float(pixel_s1_0.z / 255.0f);
  return dFx2;
}

__device__ float3 dfY2(uchar3 pixel_0_a1, uchar3 pixel_0_0, uchar3 pixel_0_s1) {
  float3 dFy2;
	dFy2.x = float(pixel_0_a1.x / 255.0f) - 2.0 * float(pixel_0_0.x / 255.0f) + float(pixel_0_s1.x / 255.0f);
  dFy2.y = float(pixel_0_a1.y / 255.0f) - 2.0 * float(pixel_0_0.y / 255.0f) + float(pixel_0_s1.y / 255.0f);
  dFy2.z = float(pixel_0_a1.z / 255.0f) - 2.0 * float(pixel_0_0.z / 255.0f) + float(pixel_0_s1.z / 255.0f);
  return dFy2;
}

__device__ float3 dfXY(uchar3 pixel_a1_a1, uchar3 pixel_s1_s1, uchar3 pixel_s1_a1, uchar3 pixel_a1_s1) {
	float3 dFxy;
  dFxy.x = ((float(pixel_a1_a1.x / 255.0f) + float(pixel_s1_s1.x / 255.0f) - (float(pixel_s1_a1.x / 255.0f) + float(pixel_a1_s1.x / 255.0f))) / 4.0);
	dFxy.y = ((float(pixel_a1_a1.y / 255.0f) + float(pixel_s1_s1.y / 255.0f) - (float(pixel_s1_a1.y / 255.0f) + float(pixel_a1_s1.y / 255.0f))) / 4.0);
	dFxy.z = ((float(pixel_a1_a1.z / 255.0f) + float(pixel_s1_s1.z / 255.0f) - (float(pixel_s1_a1.z / 255.0f) + float(pixel_a1_s1.z / 255.0f))) / 4.0);
  return dFxy;
}

__device__ float pde_value(float pixel, float Ix, float Iy, float Ixx, float Iyy, float Ixy, float alphag){
  float num = Ixx * Iy * Iy - 2.0*Ixy * Ix * Iy + Iyy * Ix * Ix;
  float denom = 1e-8 + Ix * Ix + Iy * Iy;
  float value = pixel + alphag * num / denom;
  if (value > 1) value = 1;
  else if (value < 0) value = 0;
  return value;
}

__global__ void PDE(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float tau, float alpha){

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dst_x < cols && dst_y < rows){
    float indicator, g11, g22, g12;
    float3 Ix, Iy, Ixx, Iyy, Ixy;
    float value, alphag;

    int j = dst_x;
    int i = dst_y;

    Ix = dfX_0(src(i + 1, j), src(i - 1, j));
	  Iy = dfY_0(src(i, j + 1), src(i, j - 1));
	
	  Ixx = dfX2(src(i + 1, j), src(i, j), src(i - 1, j));
	  Iyy = dfY2(src(i, j + 1), src(i, j), src(i, j - 1));

	  Ixy = dfXY(src(i + 1, j + 1), src(i - 1, j - 1), src(i - 1, j + 1), src(i + 1, j - 1));

	  g11 = 1.0 + Ix.x*Ix.x + Ix.y*Ix.y + Ix.z*Ix.z;
	  g12 = Ix.x*Iy.x + Ix.y*Iy.y + Ix.z*Iy.z;
	  g22 = 1.0 + Iy.x*Iy.x + Iy.y*Iy.y + Iy.z*Iy.z;
	
	  indicator = sqrt((g11-g22)*(g11-g22) + 4.0 *g12*g12);
	
	  value = sqrt(indicator) / tau;
	  value *= -value;
	  alphag = alpha*exp(value) ;

    float3 result;
    result.x = pde_value(float(src(dst_y, dst_x).x)/255.0, Ix.x, Iy.x, Ixx.x, Iyy.x, Ixy.x, alphag);
    result.y = pde_value(float(src(dst_y, dst_x).y)/255.0, Ix.y, Iy.y, Ixx.y, Iyy.y, Ixy.y, alphag);
    result.z = pde_value(float(src(dst_y, dst_x).z)/255.0, Ix.z, Iy.z, Ixx.z, Iyy.z, Ixy.z, alphag);

    dst(dst_y, dst_x).x = (unsigned char)(result.x * 255.0 > 255.0? 255.0: result.x * 255.0);
    dst(dst_y, dst_x).y = (unsigned char)(result.y * 255.0 > 255.0? 255.0: result.y * 255.0);
    dst(dst_y, dst_x).z = (unsigned char)(result.z * 255.0 > 255.0? 255.0: result.z * 255.0);

  }
}

void startCUDA_PDE(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float tau, float alpha, int BLOCKDIM_X, int BLOCKDIM_Y){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  PDE<<<numBlocks, threadsPerBlock>>>(src, dst, dst.rows, dst.cols, tau, alpha);
}