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
	cl::Program raytraceProgram({raytrace_source});
	try {
		raytraceProgram.build("-cl-std=CL1.2");
	}
	catch (...) {
	// Print build info for all devices
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = raytraceProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		for (auto &pair : buildInfo) {
			std::cerr << pair.second << std::endl << std::endl;
		}
		throw std::runtime_error("Failed to load program");
	}
	cl::Context ctx=cl::Context::getDefault();

	SimpleImage<float> si2=inpano.channel_select(0xF);
	
	cl::ImageFormat fmt(CL_RGBA,CL_FLOAT);
	cl_int err;       
	cl::Image2D im(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,fmt,inpano.width(),inpano.height(),0,(void*)si2.data(),&err);
	if(err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to load image");
	}
	cl::Image2D im2(ctx,CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,fmt,inpano.width(),inpano.height(),0,nullptr,&err);
	if(err != CL_SUCCESS)
	{
		throw  std::runtime_error("Failed to create output buffer");
	}
	SimpleImage<float> final_out(si2.channels(),si2.width(),si2.height());
	std::fill(final_out.data(),final_out.data()+final_out.size(),0.0f);
	cl::KernelFunctor<
            cl::Image2D,
			cl::Image2D,
			std::array<uint32_t,2>,
			std::array<uint32_t,2>
            > raytraceKernel(raytraceProgram, "perpixel");
	
	size_t chunksize=64;
	size_t Nchunks=inpano.height()*inpano.width()/(chunksize*chunksize);
	size_t chunks_so_far=0;
	for(size_t cy=0;cy < inpano.height(); cy+=chunksize)
	for(size_t cx=0;cx < inpano.width(); cx+=chunksize)
	{
		cl::NDRange rnge(inpano.width(),inpano.height());
		auto result=raytraceKernel(
			cl::EnqueueArgs(rnge),
			im,im2,
			{cx,cy},
			{(uint32_t)std::min(cx+chunksize,inpano.width()),(uint32_t)std::min(cy+chunksize,inpano.height())}
		);
		result.wait();
		cl::CommandQueue queue(ctx);
		if(CL_SUCCESS != queue.enqueueReadImage(im2,true,{0,0,0},{inpano.width(),inpano.height(),1},0,0, (void*)si2.data()))
		{
			throw std::runtime_error("Error reading back result image");
		}
		queue.finish();
		
		float* dfin=final_out.data();
		const float* dsi2=si2.data();
		for(size_t i=0;i<final_out.size();i++)
		{
			dfin[i]+=dsi2[i];
		}

		std::cerr << "Did chunk " <<(++chunks_so_far) << "/" << Nchunks << std::endl;
	}
	
	return final_out.channel_select(0x7);
}


int main(int argc,char** argv)
{
	SimpleImage<float> input("../../testdata/evening_road_01_2k.hdr");
	SimpleImage<float> output=doOpenCLTest(input);
	hdr_view(output);
	//input=pano_pad_flip(input);
	return 0;
}
