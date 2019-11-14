#ifndef PSHRM_HPP
#define PSHRM_HPP

#include "SimpleImage.hpp"
#include<array>
#include<functional>

namespace pshrm
{
	SimpleImage<float> pano_convolve(
		const SimpleImage<float>& inpano,
		const std::vector<float>& kernel,
		size_t outheight=0,
		std::array<size_t,2> inchunksize={64,64},
		std::array<size_t,2> outchunksize={0xFFFFFFF,0xFFFFFF}
	);
	SimpleImage<float> pano_pad_flip(const SimpleImage<float>& inpano);
	
	std::vector<float> pano_build_kernel(const std::function<float (float)>& kfunc,size_t N,bool renormalize=true);
}

#endif
