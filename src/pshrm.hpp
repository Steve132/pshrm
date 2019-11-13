#ifndef PSHRM_HPP
#define PSHRM_HPP

#include "SimpleImage.hpp"
#include<array>

namespace pshrm
{
	SimpleImage<float> pano_convolve(
		size_t outheight,
		const SimpleImage<float>& inpano,
		std::array<size_t,2> inchunksize={64,64},
		std::array<size_t,2> outchunksize={0xFFFFFFF,0xFFFFFF}
	);
}

#endif
