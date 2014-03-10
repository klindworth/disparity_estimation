/*
Copyright (c) 2013, Kai Klindworth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef GENERICFUNCTIONS_H
#define GENERICFUNCTIONS_H

#include <opencv2/core/core.hpp>
#include <string>
#include <iosfwd>

//saves a matrix to a binary file
void matToFile(const cv::Mat& input , const std::string &filename);
//opens a binary file and constructs a matrix out of it
cv::Mat fileToMat(const std::string &filename);

cv::Mat streamToMat(std::ifstream& istream);
void matToStream(const cv::Mat& input, std::ofstream& ostream);

inline cv::Mat subwindow(const cv::Mat& image, int x, int y, int windowsize)
{
	assert(x >= windowsize/2 && y >= windowsize/2 && x < image.cols- windowsize/2 && y < image.rows-windowsize/2 && windowsize % 2 == 1);
	return cv::Mat(image, cv::Range(y - windowsize/2, y+windowsize/2+1), cv::Range(x-windowsize/2, x+windowsize/2+1));
}

inline cv::Mat subwindow(const cv::Mat& image, int x, int y, int windowsize_x, int windowsize_y)
{
	assert(x >= windowsize_x/2 && y >= windowsize_y/2 && x < image.cols- windowsize_x/2 && y < image.rows-windowsize_y/2 && windowsize_y % 2 == 1 && windowsize_x % 2 == 1 );
	return cv::Mat(image, cv::Range(y - windowsize_y/2, y+windowsize_y/2+1), cv::Range(x-windowsize_x/2, x+windowsize_x/2+1));
}


//sets a range in an array to the given value. like memset, but beware: count must not include the size of T!
template<typename T>
inline void resetRange(T* src, T value, int count)
{
	std::fill(src, src + count, value);
}

template<>
inline void resetRange(unsigned char *src, unsigned char value, int count)
{
	memset(src, value, count);
}

/*template<typename T>
inline void resetRange(T* src, int count)
{
	memset(src, 0, count*sizeof(T));
}*/

//rests the border of a matrix (2d/3d) to an given value
template<typename T>
void resetBorder(cv::Mat& input, int borderwidth, T value = 0)
{
	int thirdfactor = 1;

	if(input.dims == 3)
		thirdfactor = input.size[2];

	resetRange(input.ptr<T>(0), value, input.size[1]*thirdfactor*borderwidth);
	resetRange(input.ptr<T>(input.size[0]-borderwidth-1, 0), value, input.size[1]*thirdfactor*borderwidth);
	for(int i = borderwidth; i < input.size[0] - borderwidth - 2; ++i)
	{
		resetRange(input.ptr<T>(i, 0), value, borderwidth*input.size[2]);
		resetRange(input.ptr<T>(i, input.size[1]-borderwidth-2), value, borderwidth*thirdfactor);
	}
}

//takes an image and saves all possible windows in a row of the given image in a 3d matrix
template<typename T>
cv::Mat serializeRow(const cv::Mat& input, int y, int windowsize, bool enable_padding = true)
{
	int padding = (windowsize*windowsize)%4;
	padding = padding == 0 ? 0 : 4-padding;
	if(!enable_padding)
		padding = 0;

	//int sz[] = {input.rows, input.cols, windowsize*windowsize+padding};
	//cv::Mat result = cv::Mat(3, sz, type);
	cv::Mat_<T> result = cv::Mat_<T>(input.cols, windowsize*windowsize+padding);

	const int x_min = windowsize/2;
	const int x_max = input.cols - windowsize/2;

	for(int x = x_min; x < x_max; ++x)
	{
		cv::Mat src_window  = subwindow(input,  x, y, windowsize);

		T *dst_ptr  = result[x];
		for(int i = 0; i < windowsize; ++i)
		{
			T *window_ptr = src_window.ptr<T>(i);
			memcpy(dst_ptr,  window_ptr,  sizeof(T)*windowsize);
			dst_ptr += windowsize;
		}
		memset(dst_ptr,  0, padding);
	}
	return result;
}


//sets all points of a foreign matrix to zero, when the value is above the threshold
template<typename data_type, typename thres_type>
void foreignThreshold(cv::Mat& image, const cv::Mat& thresValues, thres_type threshold, bool inverted)
{
	assert(image.size == thresValues.size);

	if(!inverted)
	{
		for(int i = 0; i < image.rows; ++i)
		{
			data_type* data = image.ptr<data_type>(i);
			const thres_type* dataThres = thresValues.ptr<thres_type>(i);
			for(int j = 0; j < image.cols; ++j)
			{
				if(*dataThres > threshold)
					*data = 0;
				++data;
				++dataThres;
			}
		}
	}
	else
	{
		for(int i = 0; i < image.rows; ++i)
		{
			data_type* data = image.ptr<data_type>(i);
			const thres_type* dataThres = thresValues.ptr<thres_type>(i);
			for(int j = 0; j < image.cols; ++j)
			{
				if(*dataThres < threshold)
					*data = 0;
				++data;
				++dataThres;
			}
		}
	}
}

template<typename cost_class>
cv::Mat slidingWindow(const cv::Mat& image, unsigned int windowsize)
{
	const int y_min = windowsize/2;
	const int y_max = image.rows - windowsize/2;

	const int x_min = windowsize/2;
	const int x_max = image.cols - windowsize/2;

	cv::Mat result = cv::Mat(image.size(), CV_32FC1, cv::Scalar(0));
	cost_class cost_agg(windowsize);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for default(none) shared(image, result, cost_agg, windowsize) private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cv::Mat_<unsigned char> window = subwindow(image, x_min, y, windowsize, windowsize);
		for(int x = x_min; x < x_max; ++x)
		{
			result.at<float>(y,x) = cost_agg.increm(thread_data, window);
			window.adjustROI(0,0,-1,1);
		}
	}

	return result;
}

template<typename src_t, typename dst_t>
cv::Mat getValueScaledImage(const cv::Mat& image)
{
	double dmin, dmax;
	cv::minMaxIdx(image, &dmin, &dmax);

	src_t min = dmin;
	double scale = std::numeric_limits<dst_t>::max()/(dmax-dmin);

	cv::Mat_<dst_t> result = cv::Mat_<dst_t>(image.size());

	const src_t *src_ptr = image.ptr<src_t>(0);
	dst_t *dst_ptr = result[0];
	for(unsigned int i = 0; i < image.total(); ++i)
	{
		*dst_ptr++ = (*src_ptr++ - min) * scale;
	}
	return result;
}

void derivedMat(const cv::Mat &input, cv::Mat& grad_x, cv::Mat& grad_y, bool blur);
cv::Mat quantizeImage(const cv::Mat &input, int quantizer);
cv::Mat cutImageBorder(const cv::Mat &input, int windowsize);
cv::Mat lowerDimensionality(const cv::Mat &input);
void filterGradientCostmap(cv::Mat& cost_map, int threshold);
cv::Mat lab_to_bgr(const cv::Mat& src);
cv::Mat bgr_to_lab(const cv::Mat& src);



#endif // GENERICFUNCTIONS_H
