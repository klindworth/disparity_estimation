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
#include "sliding_entropy.h"
#include "costmap_creators.h"

inline cv::Mat quantize_image(const cv::Mat& input, int quantizer)
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

class classic_search_config
{
public:
	int windowsize;
	int quantizer;
	bool soft;
	std::string metric;
};

template<typename T, int quantizer, bool soft>
costmap_creators::entropy::entropies<T> calculate_entropies(const single_stereo_task& task, unsigned int windowsize)
{
	using namespace costmap_creators;

	entropy::entropies<T> result;
	cv::Mat_<unsigned char> base  = quantize_image(task.baseGray, quantizer);
	cv::Mat_<unsigned char> match = quantize_image(task.matchGray, quantizer);

	result.XY = sliding_window::joint_fixed_size<entropy::joint_fixed_windowsize<T, quantizer, soft> >(base, match, task.dispMin, task.dispMax, windowsize);
	result.X  = slidingWindow<entropy::single_fixed_windowsize<T, quantizer, soft> >(base,  windowsize);
	result.Y  = slidingWindow<entropy::single_fixed_windowsize<T, quantizer, soft> >(match, windowsize);

	return result;
}

template<typename data_type, template<typename datatype_type> class calc_type>
class entropy_agg
{
private:
	cv::Mat_<data_type> m_joint_entropy, m_entropy_x, m_entropy_y;
	int m_dispMin;

public:
	typedef data_type result_type;

	entropy_agg(cv::Mat& joint_entropy, std::pair<cv::Mat_<data_type>, cv::Mat_<data_type>> data, int dispMin) : m_joint_entropy(joint_entropy), m_entropy_x(data.first), m_entropy_y(data.second), m_dispMin(dispMin)
	{
	}

	inline data_type operator()(int y, int x, int d)
	{
		//return m_entropy_x(y,x) + m_entropy_y(y,x+d) - m_joint_entropy(y, x, d);
		return calc_type<data_type>::calculate(m_joint_entropy(y, x, d-m_dispMin), m_entropy_x(y,x), m_entropy_y(y,x+d));
	}
};

template<typename T>
struct mutual_information_calc
{
	static inline T calculate(T joint_entropy, T base_entropy, T match_entropy)
	{
		return -std::max(base_entropy + match_entropy - joint_entropy, static_cast<T>(0));
	}

	static inline T upper_bound() {
		return 0;
	}
};

template<typename T>
struct variation_of_information_calc
{
	static inline T calculate(T joint_entropy, T base_entropy, T match_entropy)
	{
		return 2*joint_entropy-base_entropy-match_entropy;
	}

	static inline T upper_bound() {
		return 8;
	}
};

template<typename T>
struct normalized_variation_of_information_calc
{
	static inline T calculate(T joint_entropy, T base_entropy, T match_entropy)
	{
		return 1.0f-(base_entropy+match_entropy-joint_entropy)/std::max(joint_entropy, std::numeric_limits<T>::min());
	}


	static inline T upper_bound() {
		return 1;
	}
};

template<typename T>
struct normalized_information_distance_calc
{
	static inline T calculate(T joint_entropy, T base_entropy, T match_entropy)
	{
		return 1-(base_entropy+match_entropy-joint_entropy)/std::max(std::max(base_entropy, match_entropy), std::numeric_limits<T>::min());
	}

	static inline T upper_bound() {
		return 1;
	}
};

#endif // IT_METRICS_H
