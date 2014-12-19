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

#ifndef IT_METRICS_H
#define IT_METRICS_H

#include "stereotask.h"
#include "slidingEntropy.h"
#include "costmap_creators.h"

inline cv::Mat quantizeImage(const cv::Mat& input, int quantizer)
{
	//return input/quantizer;
	assert(input.isContinuous());
	int size = input.cols * input.rows;
	cv::Mat result(input.size(), input.type());
	unsigned char *input_ptr = input.data;
	unsigned char *result_ptr = result.data;
	for(int i = 0; i < size; ++i)
	{
		*result_ptr++ = *input_ptr++/quantizer;
	}
	return result;
}

enum class IT_Metric {
	MutualInformation, VariationOfInformation, NormalizedVariationOfInformation, NormalizedInformationDistance
};

class ClassicSearchConfig
{
public:
	int windowsize;
	int quantizer;
	bool soft;
	std::string metric;
};

template<int quantizer>
costmap_creators::entropy::entropies calculateEntropies(single_stereo_task task, bool soft, unsigned int windowsize)
{
	using namespace costmap_creators;

	entropy::entropies result;
	cv::Mat base  = quantizeImage(task.baseGray, quantizer);
	cv::Mat match = quantizeImage(task.matchGray, quantizer);

	if(!soft)
	{
		result.XY = slidingJointWindow<entropy::joint_fixed_windowsize<quantizer> >(base, match, task.dispMin, task.dispMax, windowsize);
		result.X  = slidingWindow<entropy::single_fixed_windowsize<quantizer> >(base,  windowsize);
		result.Y  = slidingWindow<entropy::single_fixed_windowsize<quantizer> >(match, windowsize);
	}
	else
	{
		result.XY = slidingJointWindow<entropy::joint_fixed_windowsize_soft<quantizer> >(base, match, task.dispMin, task.dispMax, windowsize);
		result.X  = slidingWindow<entropy::single_fixed_windowsize_soft<quantizer> >(base,  windowsize);
		result.Y  = slidingWindow<entropy::single_fixed_windowsize_soft<quantizer> >(match, windowsize);
	}

	return result;
}

template<typename T>
class entropy_agg
{
private:
	cv::Mat_<float> m_joint_entropy, m_entropy_x, m_entropy_y;
	int m_dispMin;
	T calculator;
public:
	entropy_agg(cv::Mat& joint_entropy, std::pair<cv::Mat, cv::Mat> data, int dispMin) : m_joint_entropy(joint_entropy), m_entropy_x(data.first), m_entropy_y(data.second), m_dispMin(dispMin)
	{
	}

	inline float operator()(int y, int x, int d)
	{
		//return m_entropy_x(y,x) + m_entropy_y(y,x+d) - m_joint_entropy(y, x, d);
		return calculator(m_joint_entropy(y, x, d-m_dispMin), m_entropy_x(y,x), m_entropy_y(y,x+d));
	}
};

template<typename cost_agg, int quantizer>
cv::Mat InformationTheoreticDistance(single_stereo_task& task, bool soft, unsigned int windowsize)
{
	//long long start = cv::getCPUTickCount();
	auto entropy = calculateEntropies<quantizer>(task, soft, windowsize);

	//start = cv::getCPUTickCount() - start;
	//std::cout << "entropy " << start << std::endl;

	return costmap_creators::calculate_pixelwise<entropy_agg<cost_agg>, std::pair<cv::Mat, cv::Mat> >(entropy.XY, std::make_pair(entropy.X, entropy.Y), task.dispMin, task.dispMax);
}

template<typename T>
class mutual_information_calc
{
public:
	inline T operator()(T joint_entropy, T base_entropy, T match_entropy) const
	{
		return -std::max(base_entropy + match_entropy - joint_entropy,0.0f);
	}

	static inline T upper_bound() {
		return 0;
	}
};

template<typename T>
class variation_of_information_calc
{
public:
	inline T operator()(T joint_entropy, T base_entropy, T match_entropy) const
	{
		return 2*joint_entropy-base_entropy-match_entropy;
	}

	static inline T upper_bound() {
		return 8;
	}
};

template<typename T>
class normalized_variation_of_information_calc
{
public:
	inline T operator()(T joint_entropy, T base_entropy, T match_entropy) const
	{
		return 1.0f-(base_entropy+match_entropy-joint_entropy)/std::max(joint_entropy, std::numeric_limits<T>::min());
	}


	static inline T upper_bound() {
		return 1;
	}
};

template<typename T>
class normalized_information_distance_calc
{
public:
	inline T operator()(T joint_entropy, T base_entropy, T match_entropy) const
	{
		return 1-(base_entropy+match_entropy-joint_entropy)/std::max(std::max(base_entropy, match_entropy), std::numeric_limits<T>::min());
	}

	static inline T upper_bound() {
		return 1;
	}
};

template<int windowsize, int quantizer>
cv::Mat InformationTheoreticDistance(single_stereo_task task, IT_Metric metric, bool soft)
{
	auto entropy = calculateEntropies<windowsize, quantizer>(task, soft);

	switch(metric)
	{
	case IT_Metric::MutualInformation:
		return costmap_creators::calculate_pixelwise<entropy_agg<mutual_information_calc<float>>, std::pair<cv::Mat, cv::Mat> >(entropy.XY, std::make_pair(entropy.X, entropy.Y), task.dispMin, task.dispMax);
	case IT_Metric::VariationOfInformation:
		return costmap_creators::calculate_pixelwise<entropy_agg<variation_of_information_calc<float>>, std::pair<cv::Mat, cv::Mat> >(entropy.XY, std::make_pair(entropy.X, entropy.Y), task.dispMin, task.dispMax);
	case IT_Metric::NormalizedVariationOfInformation:
		return costmap_creators::calculate_pixelwise<entropy_agg<normalized_variation_of_information_calc<float>>, std::pair<cv::Mat, cv::Mat> >(entropy.XY, std::make_pair(entropy.X, entropy.Y), task.dispMin, task.dispMax);
	case IT_Metric::NormalizedInformationDistance:
		return costmap_creators::calculate_pixelwise<entropy_agg<normalized_information_distance_calc<float>>, std::pair<cv::Mat, cv::Mat> >(entropy.XY, std::make_pair(entropy.X, entropy.Y), task.dispMin, task.dispMax);
	}

	assert(false);
	return cv::Mat();
}

#endif // IT_METRICS_H
