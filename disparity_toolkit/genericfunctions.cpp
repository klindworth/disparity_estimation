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
#include <stdexcept>

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

void mat_to_stream(const cv::Mat& input, std::ostream& ostream)
{
	assert(input.data);

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

void mat_to_file(const cv::Mat& input, const std::string& filename)
{
	std::ofstream ostream(filename, std::ofstream::binary);
	mat_to_stream(input, ostream);
	ostream.close();
}

cv::Mat stream_to_mat(std::istream& istream)
{
	//assert(istream.is_open());
	int type;
	istream.read((char*)&type, sizeof(int));
	int dims;
	istream.read((char*)&dims, sizeof(int));
	int elems = 1;
	std::vector<int> sz(dims);

	for(int i = 0; i < dims; ++i)
	{
		//istream >> sz[i];
		istream.read((char*)&(sz[i]), sizeof(int));
		elems *= sz[i];
		//std::cout << sz[i] << std::endl;
	}
	cv::Mat output(dims, sz.data(), type);
	istream.read(output.ptr<char>(0), elems*output.elemSize());//check elemSize

	return output;
}

cv::Mat file_to_mat(const std::string& filename)
{
	std::ifstream istream(filename, std::ifstream::binary);
	if(!istream.is_open())
		throw std::runtime_error(filename + " couldn't opened");

	cv::Mat output = stream_to_mat(istream);
	istream.close();
	return output;
}

