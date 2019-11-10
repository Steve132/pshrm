#ifndef SIMPLE_IMAGE_HPP
#define SIMPLE_IMAGE_HPP

#include<vector>
#include<string>

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
};


#endif

