#include<iostream>
#include "SimpleImage.hpp"
#include<algorithm>
#include<iterator>
#include "hdr_view.hpp"
#include "pshrm.hpp"
#include<cmath>
using namespace std;


//idea spherical blind deconvolution using known estimates from atmospheric conditions or from cost functions. (spherical higher order statistics..)
//
float gaussian(float ct2)
{
	double ct=ct2;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	ct*=ct;
	return ct;
}

int main(int argc,char** argv)
{
	SimpleImage<float> input("../../testdata/evening_road_01_2k.hdr");
	input=input.boxreduce(2);
	hdr_view(input);

	std::vector<float> kgaussian=pshrm::pano_build_kernel(gaussian,2*input.height());
	for(size_t i=0;i<kgaussian.size();i++)
	{
		std::cout << kgaussian[i] << std::endl;
	}
	SimpleImage<float> output=pshrm::pano_convolve(input,kgaussian);
	hdr_view(output);
	
	auto avg1=input.boxreduce(input.width(),input.height());
	auto avg2=output.boxreduce(output.width(),output.height());
	
	std::cout << avg1(0,0,0) << "," << avg1(1,0,0) << "," << avg1(2,0,0) << std::endl;
	std::cout << avg2(0,0,0) << "," << avg2(1,0,0) << "," << avg2(2,0,0) << std::endl;
	
	//the scale factor incorrectness is almost *exactly* (540*960)/(4*3.14159)

	return 0;
}
