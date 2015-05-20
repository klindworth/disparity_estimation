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
#include <boost/math/common_factor.hpp>

task_analysis::task_analysis() : error_hist_left(maxdiff, 0), error_hist_right(maxdiff, 0)
{
}

disparity_map create_diff_disparity(const disparity_map& disparity, const disparity_map& ground_truth)
{
	cv::Mat_<short> scaledDisp, scaledGround;
	int common_sampling = boost::math::lcm(disparity.sampling, ground_truth.sampling);
	if(disparity.sampling != ground_truth.sampling)
	{
		scaledDisp = disparity * (common_sampling / disparity.sampling);
		scaledGround = ground_truth * (common_sampling / ground_truth.sampling);
	}
	else
	{
		scaledDisp = disparity;
		scaledGround = ground_truth;
	}

	cv::Mat error_mat_temp;
	cv::absdiff(scaledDisp, scaledGround, error_mat_temp);

	return disparity_map(error_mat_temp, common_sampling);
}

void task_analysis::create_internal(const single_stereo_task& task, const disparity_map& disparity, cv::Mat_<unsigned char>& error_mat, std::vector<int>& hist, unsigned int ignore_border)
{
	if(task.groundTruth.data)
	{
		disparity_map error_mat_temp = create_diff_disparity(disparity, task.groundTruth);
		error_mat_temp.convertTo(error_mat, CV_8UC1);

		if(task.occ.data)
			foreign_threshold(error_mat, task.occ, (unsigned char)128, true);
		foreign_null(error_mat, task.groundTruth); //doesn't work with signed gt

		const int diff_bound = hist.size() - 1;
		for(unsigned int y = ignore_border; y < error_mat.rows - ignore_border; ++y)
		{
			for(unsigned int x = ignore_border; x < error_mat.cols - ignore_border; ++x)
			{
				unsigned char idx = std::min(diff_bound, error_mat(y,x)/error_mat_temp.sampling);
				if(task.occ.data && task.groundTruth(y,x) != 0)
				{
					if(task.occ(y,x) > 128)
						++(hist[idx]);
				}
				else if(task.groundTruth(y,x) != 0)
					++(hist[idx]);
			}
		}
	}
	else
		std::clog << "no ground truth data" << std::endl;
}

task_analysis::task_analysis(const stereo_task& task, const disparity_map& disparity_left, const disparity_map& disparity_right, int ignore_border) : error_hist_left(maxdiff, 0), error_hist_right(maxdiff, 0)
{
	create_internal(task.forward, disparity_left, this->diff_mat_left, this->error_hist_left, ignore_border);
	create_internal(task.backward, disparity_right, this->diff_mat_right, this->error_hist_right, ignore_border);
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

cv::FileStorage& operator<<(cv::FileStorage& stream, const task_analysis& analysis)
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
