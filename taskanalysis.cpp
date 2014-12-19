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

#include "taskanalysis.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "stereotask.h"
#include "genericfunctions.h"

/*cv::Mat createFixedDisparity(const cv::Mat& disparity, float scale)
{
	assert(disparity.type() == CV_16SC1);

	double mind;
	double maxd;
	cv::minMaxIdx(disparity, &mind, &maxd);

	if(mind >= 0 && maxd >= 0)
		return disparity * scale;
	else
		return disparity * -scale + std::abs(static_cast<short>(maxd));
}*/

cv::Mat createFixedDisparity(const cv::Mat& disparity, float scale)
{
	assert(disparity.type() == CV_16SC1);

	cv::Mat disparity_image = cv::Mat(disparity.size(), CV_8UC1);
	double mind;
	double maxd;
	cv::minMaxIdx(disparity, &mind, &maxd);

	short mins = mind;
	short maxs = maxd;

	std::cout << mins << std::endl;
	std::cout << maxs << std::endl;

	int maxcounter = disparity.total();
	const short* disparity_ptr = disparity.ptr<short>(0);
	unsigned char* dst_ptr = disparity_image.data;
	if(mins >= 0 && maxs >= 0)
	{
		for(int i = 0; i < maxcounter; ++i)
			*dst_ptr++ = (*disparity_ptr++)*scale;
	}
	else
	{
		for(int i = 0; i < maxcounter; ++i)
			*dst_ptr++ = std::abs(maxs)-(*disparity_ptr++)*scale;
	}

	return disparity_image;
}

TaskAnalysis::TaskAnalysis()
{
	std::fill(error_hist_left.begin(), error_hist_left.end(), 0);
	std::fill(error_hist_right.begin(), error_hist_right.end(), 0);
}

void TaskAnalysis::createInternal(const single_stereo_task& task, const cv::Mat& disparity, cv::Mat& error_mat, std::array<int, maxdiff>& hist, int subsamplingDisparity, unsigned int ignore_border)
{
	if(task.groundTruth.data)
	{
		cv::Mat scaledDisp, scaledGround;
		int commonSubsampling = 1;
		if(subsamplingDisparity != task.groundTruthSampling)
		{
			scaledDisp = disparity / subsamplingDisparity;
			scaledGround = task.groundTruth / task.groundTruthSampling;
		}
		else
		{
			scaledDisp = disparity;
			scaledGround = task.groundTruth;
			commonSubsampling = subsamplingDisparity;
		}

		cv::Mat ndisp = createFixedDisparity(scaledDisp, 1.0f);
		//cv::imshow("ndisp", ndisp);
		//cv::imshow("ground", scaledGround);
		cv::absdiff(ndisp, scaledGround, error_mat);
		if(task.occ.data)
			foreignThreshold<unsigned char, unsigned char>(error_mat, task.occ, 128, true);
		foreignThreshold<unsigned char, unsigned char>(error_mat, task.groundTruth, 1, true);
		//if(ignore_border > 0)
			//resetBorder<unsigned char>(error_mat, ignore_border);

		for(unsigned int y = ignore_border; y < error_mat.rows - ignore_border; ++y)
		{
			for(unsigned int x = ignore_border; x < error_mat.cols - ignore_border; ++x)
			{
				unsigned char idx = std::min((maxdiff-1), error_mat.at<unsigned char>(y,x)/commonSubsampling);
				if(task.occ.data && task.groundTruth.at<unsigned char>(y,x) != 0)
				{
					if(task.occ.at<unsigned char>(y,x) > 128)
						++(hist[idx]);
				}
				else if(task.groundTruth.at<unsigned char>(y,x) != 0)
					++(hist[idx]);
			}
		}
	}
	else
		std::clog << "no ground truth data" << std::endl;
}

TaskAnalysis::TaskAnalysis(const stereo_task& task, const cv::Mat& disparity_left, const cv::Mat& disparity_right, int subsampling, int ignore_border)
{
	std::fill(error_hist_left.begin(), error_hist_left.end(), 0);
	std::fill(error_hist_right.begin(), error_hist_right.end(), 0);

	createInternal(task.forward, disparity_left, this->diff_mat_left, this->error_hist_left, subsampling, ignore_border);
	createInternal(task.backward, disparity_right, this->diff_mat_right, this->error_hist_right, subsampling, ignore_border);
}

/*void TaskAnalysis::write(cv::FileNode& node) const
{
	node << "error_hist" << "[:";
	for(int count : error_hist_left)
		node << count;
	node << "]";

	node << "error_hist_right" << "[:";
	for(int count : error_hist_right)
		node << count;
	node << "]";
}*/

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskAnalysis& analysis)
{
	stream << "error_hist" << "[:";
	for(int count : analysis.error_hist_left)
		stream << count;
	stream << "]";

	stream << "error_hist_right" << "[:";
	for(int count : analysis.error_hist_right)
		stream << count;
	stream << "]";

	return stream;
}
