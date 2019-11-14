#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_DEVICE_FISSION
#define CL_HPP_ENABLE_EXCEPTIONS
#include<CL/cl2.hpp>
#include "pshrm.hpp"
#include<iostream>
#include<vector>
#include<cmath>

static inline std::ostream& operator<<(std::ostream& out,const cl::Platform& plat)
{
	return out << plat.getInfo<CL_PLATFORM_VENDOR>() << ":" << plat.getInfo<CL_PLATFORM_NAME>() << " " << plat.getInfo<CL_PLATFORM_VERSION>();
}
extern const char _raytrace_cl_c[];

namespace pshrm
{
SimpleImage<float> pano_convolve(
	const SimpleImage<float>& inpano,
	const std::vector<float>& kernel,
	size_t outheight,
	std::array<size_t,2> inchunksize,
	std::array<size_t,2> outchunksize
)
{
	std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
	std::cout << "There are: " << platforms.size() << " platforms." << std::endl;
	std::cout << "Selecting first platform:" << std::endl;
	plat=platforms[0];
	std::cout << "\t" << plat << std::endl;
	cl::Platform::setDefault(plat);
	std::vector<cl::Device> devices;
	plat.getDevices(CL_DEVICE_TYPE_GPU,&devices);
	if(outheight==0) outheight=inpano.height();
	//TODO: get all devices, compile a program for various devices and contexts.
	//For now just use the default.
	std::string raytrace_source(_raytrace_cl_c);
	cl::Program raytraceProgram({raytrace_source});
	try {
		raytraceProgram.build(devices,"-cl-std=CL1.2");
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
	size_t outwidth=2*outheight;
	
	cl::ImageFormat fmt(CL_RGBA,CL_FLOAT);
	cl_int err;       
	cl::Image2D im(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,fmt,inpano.width(),inpano.height(),0,(void*)si2.data(),&err);
	if(err != CL_SUCCESS)
	{
		throw std::runtime_error("Failed to load image");
	}
	cl::Image2D im2(ctx,CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,fmt,outwidth,outheight,0,nullptr,&err);
	if(err != CL_SUCCESS)
	{
		throw  std::runtime_error("Failed to create output buffer");
	}
	cl::Image1D kim(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,cl::ImageFormat(CL_LUMINANCE,CL_FLOAT),kernel.size(),(void*)kernel.data(),&err);
	if(err != CL_SUCCESS)
	{
		throw  std::runtime_error("Failed to create kernel image");
	}
	
	SimpleImage<float> final_out(si2.channels(),outwidth,outheight);
	std::fill(final_out.data(),final_out.data()+final_out.size(),0.0f);
	cl::KernelFunctor<
            cl::Image2D,
			cl::Image2D,
			cl::Image1D,
			std::array<uint32_t,2>,
			std::array<uint32_t,2>
            > raytraceKernel(raytraceProgram, "perpixel");
	
	size_t inchunksizex=std::min(inchunksize[0],inpano.width());
	size_t inchunksizey=std::min(inchunksize[1],inpano.height());
	
	size_t outchunksizex=std::min(outchunksize[0],outwidth);
	size_t outchunksizey=std::min(outchunksize[1],outheight);
	
	size_t Noutchunks=outwidth*outheight/(outchunksizex*outchunksizey);
	size_t Ninchunks=inpano.height()*inpano.width()/(inchunksizex*inchunksizey);

	si2=SimpleImage<float>(si2.channels(),outchunksizex,outchunksizey);
	
	size_t Nchunks=Ninchunks*Noutchunks;
	size_t chunks_so_far=0;
	cl::CommandQueue queue=cl::CommandQueue::getDefault();
	for(size_t ocy=0;ocy < outheight; ocy+=outchunksizey)
	for(size_t ocx=0;ocx < outwidth; ocx+=outchunksizex)
	for(size_t cy=0;cy < inpano.height(); cy+=inchunksizey)
	for(size_t cx=0;cx < inpano.width(); cx+=inchunksizex)
	{
		cl::NDRange rnge(outchunksizex,outchunksizey);
		cl::NDRange offset(ocx,ocy);
		auto result=raytraceKernel(
			cl::EnqueueArgs(offset,rnge,cl::NDRange()),
			im,im2,
			kim,
			{cx,cy},
			{(uint32_t)std::min(cx+inchunksizex,inpano.width()),(uint32_t)std::min(cy+inchunksizey,inpano.height())}
		);
		result.wait();
		try
		{
		if(CL_SUCCESS != queue.enqueueReadImage(im2,true,{ocx,ocy,0},{outchunksizex,outchunksizey,1},0,0, (void*)si2.data()))
		{
			throw std::runtime_error("Error reading back result std::for_each()image");
		}
		queue.finish();
		} catch(const cl::Error& er)
		{
			std::cerr << "The error number is " << er.err() << std::endl;
		}
		
		final_out.subimage({ocx,ocy},si2,[](const float a,const float b){ return a+b; });
		std::cerr << "Did chunk " <<(++chunks_so_far) << "/" << Nchunks << std::endl;
	}
	
	return final_out.channel_select(0x7);
}
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


//Int from 0 to 2pi Int from 0 to pi/2 f(cos(phi)) dphi dtheta
//(Int from 0 to 2pi dtheta)(Int from 0 to pi/2 f(cos(phi)) dphi
//u=cos(phi)
//du=-sin(phi)dphi
//u1=cos(0)=1
//u2=cos(pi/2)=0
//Int from 1 to 0 f(u) (-du)
//Int from 0 to 1 f(u) du 
std::vector<float> pano_build_kernel(const std::function<float (float)>& kfunc,size_t phisize,bool renormalize)
{
	size_t Nout=static_cast<double>(phisize/2)*std::acos(1.0/static_cast<double>(phisize/2));
	std::vector<float> cosvec(Nout);
	double smval=0.0;
	double du=1.0/static_cast<double>(Nout);
	for(size_t cti=0;cti<Nout;cti++)
	{
		double u=(static_cast<double>(cti)+0.5)*du;
		float v=kfunc(u);
		cosvec[cti]=v;
		smval+=v*du;
	}
	double nm=2.0*CL_M_PI*smval;
	if(renormalize)
	{
		for(size_t cti=0;cti<Nout;cti++)
		{
			cosvec[cti]/=nm;
		}
	}
	return cosvec;
}

}
