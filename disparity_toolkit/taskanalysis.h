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

#ifndef TASKANALYSIS_H
#define TASKANALYSIS_H

#include "disparity_toolkit/disparity_map.h"

#include <opencv2/core/core.hpp>
#include <numeric>

class stereo_task;
class single_stereo_task;

class task_analysis
{
public:
	task_analysis();
	task_analysis(const stereo_task& task, const disparity_map& disparity_left, const disparity_map& disparity_right, int ignore_border = 0);
	//void write(cv::FileNode& node) const;
	static const int maxdiff = 15;
	std::vector<int> error_hist_left, error_hist_right;
	cv::Mat_<unsigned char> diff_mat_left, diff_mat_right;
private:
	void create_internal(const single_stereo_task& task, const disparity_map& disparity, cv::Mat_<unsigned char>& error_mat, std::vector<int>& hist, unsigned int ignore_border = 0);
};

cv::FileStorage& operator<<(cv::FileStorage& stream, const task_analysis& analysis);

template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const task_analysis& analysis)
{
	//std::copy(analysis.error_hist.begin(), analysis.error_hist.end(), std::ostream_iterator<int>(stream, ","));
	int sum = std::accumulate(analysis.error_hist_left.begin(), analysis.error_hist_left.end(), 0);
	float wsum = 0.0f;
	for(std::size_t i = 0; i < analysis.error_hist_left.size(); ++i)
	{
		float percent = (float)analysis.error_hist_left[i]/sum;
		stream << i << ": " << analysis.error_hist_left[i] << " (" << percent*100 << "%)\n";
		wsum += percent*i;
	}
	stream << "mean: " << wsum;
	return stream;
}

#endif // TASKANALYSIS_H
