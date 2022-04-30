#include "include/IP.h"

/*
scale 3 img/resize.jpg
add 1.0 0.0 img/im1.png img/im2.jpg
subtract 1.0 0.0 img/im1.png img/im2.jpg
multiply 1.0 0.0 img/im1.png img/im2.jpg
divide 1.0 0.0 img/im1.png img/im2.jpg
blur 5 3.7 img/grenouille.jpg
laplacian img/grenouille.jpg
separable 5 3.7 img/grenouille.jpg
denoise 3 50 img/grenouille.jpg
shift_hue 100 img/grenouille.jpg
pde 50, 0.05, 0.1 img/grenouille.jpg
*/

int main(int argc, char** argv){

  	Image result;
	std::string choice = argv[1];

	if (choice == "scale") {
		Image img(argv[3]); img.display("Image");
		result = img.scale_Bilinear(atoi(argv[2]));
		result.display("Scale");
	}
	else if (choice == "add") {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = img1.Combine_Arithmetic(img2, ADD, atof(argv[3]), atof(argv[2]));
		result.display("Combined Arithmetic Operation: Addition");
	}
	else if (choice == "subtract") {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = img1.Combine_Arithmetic(img2, SUBTRACT, atof(argv[3]), atof(argv[2]));
		result.display("Combined Arithmetic Operation: Subtraction");
	}
	else if (choice == "multiply") {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = img1.Combine_Arithmetic(img2, MULTIPLY, atof(argv[3]), atof(argv[2]));
		result.display("Combined Arithmetic Operation: Multiply");
	}
	else if (choice == "divide") {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = img1.Combine_Arithmetic(img2, DIVIDE, atof(argv[3]), atof(argv[2]));
		result.display("Combined Arithmetic Operation: Division");
	}
	else if (choice == "blur"){
		Image img(argv[4]); img.display("Image");
		result = img.Blur(atoi(argv[2]), atof(argv[3]));
		result.display("Gaussian - Blur");
	}
	else if (choice == "laplacian") {
		Image img(argv[2]); img.display("Image");
		result = img.laplacian_Filter();
		result.display("Laplacian");
	}
	else if (choice == "separable") {
		Image img(argv[4]); img.display("Image");
		result = img.gaussian_Separable(atoi(argv[2]), atof(argv[3]));
		result.display("Separable Filter");
	}
	else if (choice == "denoise") {
		Image img(argv[4]); img.display("Image");
		result = img.denoise(atoi(argv[2]), atof(argv[3]));
		result.display("Denoise");
	}
	else if (choice == "shift_hue") {
		Image img(argv[3]); img.display("Image");
		result = img.shift_Hue(atoi(argv[2]));
		result.display("Shift Hue - LCH Space");
	}
	else if (choice == "pde") {
		Image img(argv[5]); img.display("Image");
		result = img.PDE(atoi(argv[2]), atof(argv[3]), atof(argv[4]));
		result.display("PDE");
	}
	else {
		std::cout << "Wrong Command" << std::endl;
	}
  

	/*
	Image img1("img/im1.png");
	Image img2("img/im2.jpg");
	Image img3;
	Image img4("img/resize.jpg");

	// Image img_scale_Bilinear	= img4.scale_Bilinear(2);  img4.display("Image 4"); img_scale_Bilinear.display("Scale - Bilinear");
	// Image C   = img1.Combine_Arithmetic(img2, SUBTRACT, 0.0, 1.0); C.display("Arithmetic");
	// Image blur	= img3.Blur(16, 3.4); img3.display("Image 3"); blur.display("Blur");
	// Image laplacian	= img3.laplacian_Filter(); img3.display("Image 3"); laplacian.display("Laplacian");
	// Image separable	= img3.gaussian_Separable(16, 3.4); img3.display("Image 3"); separable.display("Separable Filter");
	// Image denoise	= img3.denoise(7, 50); img3.display("Image 3"); denoise.display("Denoise");
	// Image shift_hue	= img3.shift_Hue(0); img3.display("Image 3"); shift_hue.display("Shift Hue");
	// Image pde	= img3.PDE(10, 0.05, 0.1); img3.display("Image 3"); pde.display("PDE");
		*/
	
	cv::waitKey();
	return 0;
}
