#include<iostream>
#include "SimpleImage.hpp"
#include<algorithm>
#include<iterator>
#include "hdr_view.hpp"

#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_DEVICE_FISSION
#define CL_HPP_ENABLE_EXCEPTIONS
#include<CL/cl2.hpp>
using namespace std;


//idea spherical blind deconvolution using known estimates from atmospheric conditions or from cost functions. (spherical higher order statistics..)
//
SimpleImage<float> pano_pad_flip(const SimpleImage<float>& inpano)
{
		SimpleImage<float> flipped(inpano.channels(),inpano.width(),inpano.height()*2);
		
		std::copy(inpano.data(),inpano.data()+inpano.size(),flipped.data());

		std::copy(
			std::reverse_iterator<const float*>(inpano.data()+inpano.size()),
			std::reverse_iterator<const float*>(inpano.data()),flipped.data()+inpano.size());
		
		size_t N=inpano.size();
		size_t C=inpano.channels();
		float* citer=flipped.data()+inpano.size();
		for(size_t i=0;i<N;i+=C)
		{
			std::reverse(citer+i,citer+i+C);
		}
		return flipped;
}
std::ostream& operator<<(std::ostream& out,const cl::Platform& plat)
{
	return out << plat.getInfo<CL_PLATFORM_VENDOR>() << ":" << plat.getInfo<CL_PLATFORM_NAME>() << " " << plat.getInfo<CL_PLATFORM_VERSION>();
}
extern const char _raytrace_cl_c[];

SimpleImage<float> doOpenCLTest(const SimpleImage<float>& inpano)
{
	std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
	std::cout << "There are: " << platforms.size() << " platforms." << std::endl;
	std::cout << "Selecting first platform:" << std::endl;
	plat=platforms[0];
	std::cout << "\t" << plat << std::endl;
	cl::Platform::setDefault(plat);
	//TODO: get all devices, compile a program for various devices and contexts.
	//For now just use the default.
	std::string raytrace_source(_raytrace_cl_c);
	cl::Program vectorAddProgram({raytrace_source});
	try {
		vectorAddProgram.build("-cl-std=CL1.2");
	}
	catch (...) {
	// Print build info for all devices
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = vectorAddProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		for (auto &pair : buildInfo) {
			std::cerr << pair.second << std::endl << std::endl;
		}
		throw std::runtime_error("Failed to load program");
	}
	

	return inpano;
}


int main(int argc,char** argv)
{
	system("pwd");
	SimpleImage<float> input("../../testdata/evening_road_01_2k.hdr");
	doOpenCLTest(input);
	//hdr_view(input);
	//input=pano_pad_flip(input);
	return 0;
}
