#include "SimpleImage.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include<cstdint>
#include<utility>
#include<stdexcept>
using namespace std;

static inline std::string getExt(std::string fn)
{
	size_t spos = fn.find_last_of("\\") + 1;
	if (spos == std::string::npos) { spos = 0; }
	fn = fn.substr(spos);
	return fn.substr(fn.find_last_of(".") + 1);
}

template<>
SimpleImage<uint8_t>::SimpleImage(const std::string& filename)
{
	int w, h, n;
	unsigned char* data = stbi_load(filename.c_str(), &w, &h, &n, 0);
	if (data)
	{
		init(n, w, h, data);
		stbi_image_free(data);
	}
	else
	{
		throw std::runtime_error("Failed to load file...");
	}
}
template<>
SimpleImage<float>::SimpleImage(const std::string& filename)
{
	std::string ext = getExt(filename);
	int w,h,n;
	if(ext == "hdr" || ext == "HDR" || ext=="rgbe" || ext=="RGBE" || stbi_is_hdr(filename.c_str()))
	{
		float* data=stbi_loadf(filename.c_str(),&w,&h,&n,0);
		if(data) 
		{
			init(n,w,h,data);
			stbi_image_free(data);
		}
		else
		{
			throw std::runtime_error("Failed to load file...");
		}
	}
	else
	{
		SimpleImage<uint8_t> ldrim(filename);
		*this=ldrim;
	}
}
template<>
void SimpleImage<uint8_t>::write(const std::string& filename) const
{
	std::string ext = getExt(filename);
	const unsigned char* dat = pixdata.data();
	int val = -2;
	if (ext == "png" || ext == "PNG")
	{
		val = stbi_write_png(filename.c_str(), mwidth, mheight, mchannels, dat, 0);
	}
	else if (ext == "bmp" || ext == "BMP")
	{
		val = stbi_write_bmp(filename.c_str(), mwidth, mheight, mchannels, dat);
	}
	else if (ext == "tga" || ext == "TGA")
	{
		val = stbi_write_tga(filename.c_str(), mwidth, mheight, mchannels, dat);
	}
	else if (ext == "jpg" || ext == "JPG" || ext=="jpeg" || ext=="JPEG")
	{
		val = stbi_write_jpg(filename.c_str(), mwidth, mheight, mchannels, dat, 0);
	}
	else if (ext == "hdr" || ext == "HDR" || ext=="rgbe" || ext=="RGBE")
	{
		SimpleImage<float> fimg(*this);
		val = stbi_write_hdr(filename.c_str(),mwidth,mheight,mchannels,fimg.data());
	}
	else
	{
		throw std::runtime_error(std::string("Unrecognized file extension")+filename);
	}
	if (val == 0)
	{
		throw std::runtime_error(std::string("Failure writing file: ") + filename);
	}
}
template<>
void SimpleImage<float>::write(const std::string& filename) const
{
	int val = -2;
	std::string ext = getExt(filename);
	if (ext == "hdr" || ext == "HDR" || ext=="rgbe" || ext=="RGBE")
	{
		val = stbi_write_hdr(filename.c_str(),mwidth,mheight,mchannels,pixdata.data());
		if (val == 0)
		{
			throw std::runtime_error(std::string("Failure writing file: ") + filename);
		}
	}
	else
	{
		SimpleImage<uint8_t> ldrim(*this);
		ldrim.write(filename);
	}
}





#ifdef SIMPLE_IMAGE_OPENCV_SUPPORT
#include <opencv2/core.hpp>

static inline void swapBGR(color_t* const data, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		color_t c = data[i];
		std::swap(c.r, c.b);
		data[i] = c;
	}
}

template<>
SimpleImage::SimpleImage(const cv::Mat& mat)
{
	int w = mat.cols;
	int h = mat.rows;
	if (mat.type() != CV_8UC3)
	{
		throw std::runtime_error("invalid number of cvmat types");
	}
	init(w, h, reinterpret_cast<const color_t*>(mat.data));
	swapBGR(mdata, w * h);
	
}
void SimpleImage::toMat(cv::Mat& out) const
{
	out.create(mheight, mwidth, CV_8UC3);
	memcpy(out.data, mdata, mheight * mwidth * 3);
	swapBGR(reinterpret_cast<color_t*>(out.data), mheight * mwidth);
}
#endif

#ifdef SIMPLE_IMAGE_CIMG_SUPPORT
#include <CImg.h>

template class SimpleImage<double>;
template class SimpleImage<float>;
template class SimpleImage<uint8_t>;
template class SimpleImage<uint16_t>;
template class SimpleImage<uint32_t>;
template class SimpleImage<uint64_t>;





template<class FData>
SimpleImage<FData>::SimpleImage(const cimg_library::CImg<FData>& cimg)
{
	cimg_library::CImg<FData> ac = cimg.get_permute_axes("CXYZ");
	init(ac.width(), ac.height(), ac.depth(), ac.data());
}

template<class FData>
cimg_library::CImg<FData> SimpleImage<FData>::toCImg() const
{
	cimg_library::CImg<FData> out(data(),channels(),width(),height(),1); //shared temporary
	out.permute_axes("YZCX");
	return out;
}


#endif
