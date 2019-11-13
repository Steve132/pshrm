#include<iostream>
#include "SimpleImage.hpp"
#include<algorithm>
#include<iterator>
#include "hdr_view.hpp"
#include "pshrm.hpp"
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

int main(int argc,char** argv)
{
	SimpleImage<float> input("../../testdata/evening_road_01_2k.hdr");
	input=input.boxreduce(4);
	hdr_view(input);

	SimpleImage<float> output=pshrm::pano_convolve(input.height(),input);
	hdr_view(output);
	//input=pano_pad_flip(input);
	return 0;
}
