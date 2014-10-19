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

#include "genericfunctions.h"

#include <fstream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat cutImageBorder(const cv::Mat& input, int windowsize)
{
	assert(input.dims == 2);
	cv::Mat result = cv::Mat(input, cv::Range(windowsize/2, input.rows-windowsize/2-1), cv::Range(windowsize/2, input.cols-windowsize/2-1));
	return result.clone();
}

cv::Mat lowerDimensionality(const cv::Mat& input)
{
	assert(input.dims == 3);
	cv::Mat result(input.size[0]*input.size[1], input.size[2], input.type());
	int data_size = input.total()*input.elemSize();
	memcpy(result.data, input.data, data_size);

	return result;
}

void matToStream(const cv::Mat& input, std::ofstream& ostream)
{
	int type = input.type();
	ostream.write((char*)&type, sizeof(int));
	ostream.write((char*)&(input.dims), sizeof(int));
	int elems = 1;
	for(int i = 0; i < input.dims; ++i)
	{
		ostream.write((char*)&(input.size[i]), sizeof(int));
		elems *= input.size[i];
	}
	ostream.write(input.ptr<char>(0), elems*input.elemSize());
}

void matToFile(const cv::Mat& input, const std::string& filename)
{
	std::ofstream ostream(filename, std::ofstream::binary);
	matToStream(input, ostream);
	ostream.close();
}

cv::Mat streamToMat(std::ifstream& istream)
{
	assert(istream.is_open());
	int type;
	istream.read((char*)&type, sizeof(int));
	int dims;
	istream.read((char*)&dims, sizeof(int));
	int elems = 1;
	int *sz = new int[dims];
	for(int i = 0; i < dims; ++i)
	{
		//istream >> sz[i];
		istream.read((char*)&(sz[i]), sizeof(int));
		elems *= sz[i];
		//std::cout << sz[i] << std::endl;
	}
	cv::Mat output(dims, sz, type);
	istream.read(output.ptr<char>(0), elems*output.elemSize());//check elemSize
	delete[] sz;
	return output;
}

cv::Mat fileToMat(const std::string& filename)
{
	std::ifstream istream(filename, std::ifstream::binary);
	assert(istream.is_open());
	cv::Mat output = streamToMat(istream);
	istream.close();
	return output;
}

cv::Mat lab_to_bgr(const cv::Mat& src)
{
	cv::Mat temp = src;
	if(src.type() == CV_64FC3)
		src.convertTo(temp, CV_32FC3);
	cv::Mat bgr_float_image;
	cv::cvtColor(temp, bgr_float_image, CV_Lab2BGR);
	cv::Mat result;
	bgr_float_image.convertTo(result, CV_8UC3, 255);

	return result;
}

cv::Mat bgr_to_lab(const cv::Mat& src)
{
	cv::Mat bgr_float_image;
	src.convertTo(bgr_float_image, CV_32FC3, 1/255.0);

	cv::Mat result;
	cv::cvtColor(bgr_float_image,result, CV_BGR2Lab);
	//showMatTable(result);

	return result;
}
