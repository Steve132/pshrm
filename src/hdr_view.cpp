#include "SimpleImage.hpp"

#include <CImg.h>
#include<cmath>
using namespace cimg_library;

static double scale(int wv_cum)
{
	return 255.0*exp(static_cast<double>(wv_cum)/10.0);
}
void hdr_view(const SimpleImage<float>& v)
{
	CImg<float> imgToShow=v.toCImg();
	
	int wv_cumulative=0;
	CImg<float> currImg=imgToShow*scale(wv_cumulative);
	currImg.cut(0,255.0f);
	CImgDisplay main_disp(currImg);
	main_disp.set_normalization(0);
	main_disp.set_wheel();
	while(!main_disp.is_closed())
	{
		main_disp.wait(10);
		int wv=main_disp.wheel();
		if(wv)
		{
			wv_cumulative+=wv;
			currImg=imgToShow*scale(wv_cumulative);
			currImg.cut(0,255.0f);
			main_disp=currImg;
			main_disp.set_wheel();
		}
	}
}
