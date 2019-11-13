#ifndef SIMPLE_IMAGE_HPP
#define SIMPLE_IMAGE_HPP

#include<vector>
#include<string>
#include<bitset> //for popcnt and sel
#include<array>
#include<algorithm>
#include<type_traits>

#if  __cplusplus >= 201703L
#define RESULT_INVOKE(F,...) std::invoke_result_t<F,__VA_ARGS__>
#else 
#define RESULT_INVOKE(F,...) typename std::result_of<F(__VA_ARGS__)>::type
#endif

namespace cv{
	class Mat;
}
namespace cimg_library
{
	template<class CDataType>
	class CImg;
};

template<class FType>
class SimpleImage
{
private:
	std::vector<FType> pixdata;
	size_t mwidth;
	size_t mheight;
	size_t mchannels;

	void init(size_t c,size_t w, size_t h, const FType* dat = nullptr)
	{
		if(dat == nullptr)
		{
			pixdata=std::vector<FType>(c*w*h);
		}
		else
		{
			pixdata=std::vector<FType>(dat,dat+(c*w*h));			
		}
		mchannels=c;
		mwidth=w;
		mheight=h;
	}
	size_t index(size_t c,size_t x, size_t y) const { return mchannels*(y * mwidth + x)+c; }
public:
	
	size_t width() const { return mwidth; }
	size_t height() const { return mheight; }
	size_t channels() const {return mchannels; }
	size_t size() const { return pixdata.size(); }

	FType* const data() { return pixdata.data(); }
	const FType* const data() const { return pixdata.data(); }
	
	SimpleImage(size_t c,size_t w, size_t h, const FType* dat = nullptr)
	{
		init(c,w, h, dat);
	}
	SimpleImage(const SimpleImage& d)=default;

	template<class DType>
	SimpleImage(const SimpleImage<DType>& other):pixdata(other.data(),other.data()+other.size()),mwidth(other.width()),mheight(other.height()),mchannels(other.channels())
	{}

	const FType& operator()(size_t c,size_t x, size_t y) const { return pixdata[index(c,x, y)]; }
	FType& operator()(size_t c,size_t x, size_t y) { return pixdata[index(c,x, y)]; }

	void write(const std::string& filename) const;
	SimpleImage(const std::string& filename);
	
	SimpleImage(const cv::Mat& mat);
	void toMat(cv::Mat& out) const;
	
	SimpleImage(const cimg_library::CImg<FType>&);
	cimg_library::CImg<FType> toCImg() const;
	
	SimpleImage<FType> channel_select(const std::bitset<64>& mask) const;
	
	SimpleImage<FType> subimage(const std::array<size_t,2>& origin,const std::array<size_t,2>& sz) const;
	void subimage(const std::array<size_t,2>& origin,const SimpleImage<FType>& other);
	
	template<class BinaryFunction>
	void subimage(const std::array<size_t,2>& origin,const SimpleImage<FType>& other, BinaryFunction bf);

	template<class UnaryFunction>
	SimpleImage<RESULT_INVOKE(UnaryFunction,FType)> apply(UnaryFunction uf) const; 
	
	template<class BinaryFunction,class OtherFType>
	SimpleImage<RESULT_INVOKE(BinaryFunction,FType,OtherFType)> apply(const SimpleImage<OtherFType>& other,BinaryFunction bf) const;
	
	SimpleImage<FType> boxreduce(unsigned int factorx=2,int factory=-1) const;
};


template<class FType>
SimpleImage<FType> SimpleImage<FType>::channel_select(const std::bitset<64>& mask) const
{
	size_t new_channels=mask.count();
	SimpleImage<FType> si2(new_channels,width(),height(),nullptr);
	const FType* din=data();
	FType* dout=si2.data();
	size_t N=size()/channels();
	size_t Co=si2.channels();
	size_t Ci=channels();
	
	for(size_t i=0;i<N;i++)
	{
		const FType* ldin=din+i*Ci;
		FType* ldout=dout+i*Co;
		const FType* ldoute=dout+(i+1)*Co;
		
		for(size_t c=0;c<Ci && ldout!=ldoute;c++)
		{
			if(mask[c]) *(ldout++)=ldin[c];
		}
	}
	return si2;
}
template<class FType>
SimpleImage<FType> SimpleImage<FType>::subimage(const std::array<size_t,2>& origin,const std::array<size_t,2>& sz) const
{
	SimpleImage<FType> outim(channels(),sz[0],sz[1]);
	size_t scanline_N=channels()*sz[0];
	size_t inscanline_N=channels()*width();
	const FType* ul=data()+(origin[1]*outim.width()+origin[0])*channels();
	FType* oul=outim.data();
	for(size_t ri=0;ri<sz[1];ri++)
	{
		std::copy(ul,ul+scanline_N,oul);
		ul+=inscanline_N;
		oul+=scanline_N;
	}
	return outim;
}
template<class FType>
void SimpleImage<FType>::subimage(const std::array<size_t,2>& origin,const SimpleImage<FType>& other)
{
	size_t inscanline_N=channels()*other.width();
	size_t scanline_N=channels()*width();
	FType* oul=data()+(origin[1]*width()+origin[0])*channels();
	const FType* ul=other.data();
	for(size_t ri=0;ri<other.height();ri++)
	{
		std::copy(ul,ul+inscanline_N,oul);
		ul+=inscanline_N;
		oul+=scanline_N;
	}
}
template<class FType>
template<class BinaryFunction>
void SimpleImage<FType>::subimage(const std::array<size_t,2>& origin,const SimpleImage<FType>& other,BinaryFunction bf)
{
	size_t inscanline_N=channels()*other.width();
	size_t scanline_N=channels()*width();
	FType* oul=data()+(origin[1]*width()+origin[0])*channels();
	const FType* ul=other.data();
	for(size_t ri=0;ri<other.height();ri++)
	{
		std::transform(oul,oul+inscanline_N,ul,oul,bf);
		ul+=inscanline_N;
		oul+=scanline_N;
	}
}
template<class FType>
template<class UnaryFunction>
SimpleImage<RESULT_INVOKE(UnaryFunction,FType)> SimpleImage<FType>::apply(UnaryFunction uf) const
{
}

template<class FType>
SimpleImage<FType> SimpleImage<FType>::boxreduce(unsigned int factorx,int factory) const
{
	if(factory <= 0) factory=factorx;
	if(factorx < 1) factorx=1;
	if(factory < 1) factory=1;
	
	SimpleImage<FType> output(channels(),(width()/factorx)+(width() % factorx ? 1 : 0),(height()/factory)+(height() % factory ? 1 : 0));
	std::fill(output.data(),output.data()+output.size(),0.0f);
	//std::vector<FType> scanline(channels()*width());
	size_t Sx=output.width();
	const FType* idata=data();
	size_t N=factorx*factory;
	
	for(size_t y=0;y<height();y++)
	{
		for(size_t x=0;x<width();x++)
		{
				for(size_t c=0;c<channels();c++)
				{
					output(c,x/factorx,y/factory)+=operator()(c,x,y)/N;
				}
		}
	}
	return output;
}


#endif

