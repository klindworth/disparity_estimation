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

#ifndef SLIDINGENTROPY_H
#define SLIDINGENTROPY_H

#include "genericfunctions.h"
#include "fast_array.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace costmap_creators
{

namespace entropy
{


class Entropies
{
public:
	cv::Mat X, Y, XY;
};


template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_joint_entropy_unnormalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;
	for(int i = 1; i <= bins; ++i)
	{
		for(int j = 1; j <= bins; ++j)
		{
			auto ccounter = counter(i,j);
			normalize_counter += ccounter;
			result += entropy_table(ccounter);
		}
	}
	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename data_type, typename result_type, typename counter_type, typename entropytable_type>
inline void joint_entropy_result_reset(result_type& result, unsigned int& normalize_counter, counter_type& counter, const entropytable_type& entropy_table, data_type cleft, data_type cright)
{
	auto ccounter = counter(cleft, cright);
	counter(cleft, cright) = 0;
	result += entropy_table(ccounter);
	normalize_counter += ccounter;
}

template<typename result_type, typename counter_type, typename entropytable_type, typename data_type>
inline result_type calculate_joint_entropy_unnormalized_sparse(counter_type& counter, const entropytable_type& entropy_table, int bins, int len, const data_type* dataLeft, const data_type* dataRight)
{
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;

	//reset borders
	for(int i = 0; i <= bins; ++i)
	{
		counter(0, i) = 0;
		counter(i, 0) = 0;
		counter(bins+1, i) = 0;
		counter(i, bins+1) = 0;
	}

	for(int i = 0; i < len; ++i)
	{
		const data_type cleft  = *dataLeft++;
		const data_type cright = *dataRight++;
		joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft, cright+1);
		joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright);
		joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright+1);
		joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright+2);
		joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+2, cright+1);
	}

	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_entropy_unnormalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	auto *counter_ptr = counter.at(1);
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;
	for(int i = 1; i <= bins; ++i)
	{
		normalize_counter += *counter_ptr;
		result += entropy_table(*counter_ptr++);
	}
	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename result_type, typename counter_type, typename entropytable_type, typename data_type>
inline result_type calculate_joint_entropy_normalized_sparse(counter_type& counter, const entropytable_type& entropy_table, int len, const data_type* dataLeft, const data_type* dataRight)
{
	result_type result = 0.0f;
	for(int i = 0; i < len; ++i)
	{
		const data_type cleft  = *dataLeft++;
		const data_type cright = *dataRight++;
		result += entropy_table(counter(cleft,cright));
		counter(cleft,cright) = 0;
	}

	return result;
}

template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_entropy_normalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	auto *counter_ptr = counter.data;
	result_type result = 0.0f;
	for(int i = 0; i < bins; ++i)
		result += entropy_table(*counter_ptr++);

	return result;
}

template<typename counter_type, typename data_type>
inline void calculate_joint_soft_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
	{
		const data_type cleft  = *dataLeft++;
		const data_type cright = *dataRight++;
		counter(cleft,   cright+1) += 1;
		counter(cleft+1, cright)   += 1;
		counter(cleft+1, cright+1) += 5;
		counter(cleft+1, cright+2) += 1;
		counter(cleft+2, cright+1) += 1;
	}
}

template<typename counter_type, typename data_type>
inline void calculate_joint_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
		counter(*dataLeft++, *dataRight++) += 1;
}

template<typename counter_type, typename data_type>
inline void calculate_histogramm(counter_type& counter, const data_type* data, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
		counter(*data++) += 1;
}

template<typename counter_type, typename data_type>
inline void calculate_soft_histogramm(counter_type& counter, const data_type* data, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
	{
		data_type cdata = *data++;
		counter(cdata) += 1;
		counter(cdata+1) += 7;
		counter(cdata+2) += 1;
	}
}

inline void fill_entropytable_normalized(cv::Mat_<float>& entropy_table, int size)
{
	entropy_table = cv::Mat(size, 1, CV_32FC1);
	float *entropy_table_ptr = entropy_table[0];
	float n = 1.0f/size;

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = -i*n*std::log(i*n);
}

inline void fill_entropytable_unnormalized(cv::Mat_<float>& entropy_table, int size)
{
	assert(size > 0);
	entropy_table = cv::Mat(size, 1, CV_32FC1);
	/*float *entropy_table_ptr = entropy_table[0];

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = i*std::log(i);*/

	entropy_table(0) = 0.0f;
	#pragma omp parallel for
	for(int i = 1; i < size; ++i)
		entropy_table(i) = i*std::log(i);
}

inline void fill_entropytable_unnormalized(cv::Mat_<double>& entropy_table, int size)
{
	assert(size > 0);

	entropy_table = cv::Mat(size, 1, CV_64FC1);
	double *entropy_table_ptr = entropy_table[0];

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = i*std::log(i);
}


template<int quantizer>
class single_fixed_windowsize
{
private:
	static const int bins = 256/quantizer;
	cv::Mat_<float> entropy_table;
	unsigned int windowsize;
public:
	typedef fast_array<unsigned short, bins> thread_type;

	single_fixed_windowsize(unsigned int pwindowsize) : windowsize(pwindowsize) {
		fill_entropytable_normalized(entropy_table, windowsize*windowsize);
	}

	inline float increm(thread_type& thread, cv::Mat& window)
	{
		cv::Mat serWindow = window.clone();
		//creating histogramm
		calculate_histogramm(thread, serWindow.data, windowsize*windowsize);

		//compute probability and log-proability
		return calculate_entropy_normalized<float>(thread, entropy_table, bins);
	}
};

template<int quantizer>
class single_fixed_windowsize_soft
{
private:
	static const int bins = 256/quantizer;
	cv::Mat_<float> entropy_table;
	const unsigned int windowsize;
public:
	typedef fast_array<unsigned short, bins+2> thread_type;

	single_fixed_windowsize_soft(unsigned int pwindowsize) : windowsize(pwindowsize) {
		fill_entropytable_unnormalized(entropy_table, windowsize*windowsize*7);
	}

	inline float increm(thread_type& thread, const cv::Mat& window)
	{
		cv::Mat serWindow = window.clone();
		//creating histogramm
		calculate_soft_histogramm(thread, serWindow.data, windowsize*windowsize);

		//compute probability and log-proability
		return calculate_entropy_unnormalized<float>(thread, entropy_table, bins);
	}
};

template<int bins>
class joint_fixed_windowsize_threaddata
{
public:
	cv::Mat_<unsigned char> m_base;
	cv::Mat_<unsigned char> m_match_table;
	fast_array2d<unsigned short, bins, bins> counter_array;
};

template<int quantizer>
class joint_fixed_windowsize
{
public:
	typedef float prob_table_type;
	static const int bins = 256/quantizer;
	typedef joint_fixed_windowsize_threaddata<bins> thread_type;
	const unsigned int windowsize;
private:
	cv::Mat_<prob_table_type> entropy_table;

public:
	joint_fixed_windowsize(const cv::Mat& /*match*/, int pwindowsize) : windowsize(pwindowsize)
	{
		fill_entropytable_normalized(entropy_table, windowsize*windowsize);
	}

	//prepares a row for calculation
	inline void prepareRow(thread_type& thread, const cv::Mat& match, int y)
	{
		thread.m_match_table = serializeRow<unsigned char>(match, y, windowsize, false);
	}

	inline void prepareWindow(thread_type& thread, const cv::Mat& base)
	{
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();
	}

	//calculates the histogramms for mutual information
	inline prob_table_type increm(thread_type& thread, int x)
	{
		//creating histogramm
		calculate_joint_histogramm(thread.counter_array, thread.m_base[0], thread.m_match_table[x], windowsize*windowsize);

		//compute entropy
		if(windowsize*windowsize < bins*bins)
			return calculate_joint_entropy_normalized_sparse<float>(thread.counter_array, entropy_table, windowsize*windowsize, thread.m_base[0], thread.m_match_table[x]);
		else
			return calculate_entropy_normalized<float>(thread.counter_array, entropy_table, bins*bins);
	}
};

template<int quantizer>
class joint_fixed_windowsize_soft
{
public:
	typedef float prob_table_type;
	static const int bins = 256/quantizer;
	typedef joint_fixed_windowsize_threaddata<bins+2> thread_type;
	const unsigned int windowsize;
private:
	cv::Mat_<prob_table_type> entropy_table;

public:
	joint_fixed_windowsize_soft(const cv::Mat& /*match*/, unsigned int pwindowsize) : windowsize(pwindowsize)
	{
		fill_entropytable_unnormalized(entropy_table, windowsize*windowsize*9);
	}

	//prepares a row for calculation
	inline void prepareRow(thread_type& thread, const cv::Mat& match, int y)
	{
		thread.m_match_table = serializeRow<unsigned char>(match, y, windowsize, false);
	}

	inline void prepareWindow(thread_type& thread, const cv::Mat& base)
	{
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();
	}

	//calculates the histogramms for mutual information
	inline prob_table_type increm(thread_type& thread, int x)
	{

		//creating histogramm
		calculate_joint_soft_histogramm(thread.counter_array, thread.m_base[0], thread.m_match_table[x], windowsize*windowsize);

		//compute entropy
		//return calculate_joint_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);
		if(windowsize*windowsize*6 <= bins*bins)
			return calculate_joint_entropy_unnormalized_sparse<float>(thread.counter_array, entropy_table, bins, windowsize*windowsize, thread.m_base[0], thread.m_match_table[x]);
		else
			return calculate_joint_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);
	}
};

template<int bins>
class flexible_windowsize_threaddata
{
public:
	cv::Mat_<unsigned char> m_base;
	fast_array2d<unsigned short, bins, bins> counter_array;
	int cwindowsizeX;
	int cwindowsizeY;
	int crow;
	float base_entropy;
};

template<typename cost_func, int quantizer>
class flexible_windowsize
{
public:
	typedef float prob_table_type;
	static const int bins = 256/quantizer;
	typedef flexible_windowsize_threaddata<bins+2> thread_type;
private:
	cv::Mat_<prob_table_type> entropy_table;
	cv::Mat m_match;
	cost_func m_evalfunc;
	const int windowsize;

public:
	flexible_windowsize(const cv::Mat& match, unsigned int max_windowsize) : m_match(match), windowsize(max_windowsize)
	{
		fill_entropytable_unnormalized(entropy_table, windowsize*windowsize*9);
	}

	//prepares a row for calculation
	inline void prepareRow(thread_type& thread, const cv::Mat& /*match*/, int y)
	{
		thread.crow = y;
	}

	inline void prepareWindow(thread_type& thread, const cv::Mat& base, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();

		calculate_soft_histogramm(thread.counter_array, thread.m_base[0], thread.m_base.total());
		thread.base_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);
	}

	//calculates the histogramms
	inline prob_table_type increm(thread_type& thread, int x)
	{
		cv::Mat_<unsigned char> match_window = subwindow(m_match, x, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();
		//creating joint histogramm, compute joint entropy
		calculate_joint_soft_histogramm(thread.counter_array, thread.m_base[0], match_window[0], thread.cwindowsizeX*thread.cwindowsizeY);
		float joint_entropy;
		if(thread.cwindowsizeX*thread.cwindowsizeY*6 <= bins*bins)
			joint_entropy = calculate_joint_entropy_unnormalized_sparse<float>(thread.counter_array, entropy_table, bins, thread.cwindowsizeX*thread.cwindowsizeY, thread.m_base[0], match_window[0]);
		else
			joint_entropy = calculate_joint_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);

		calculate_soft_histogramm(thread.counter_array, match_window.data, match_window.total());
		float match_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);

		return m_evalfunc(joint_entropy, thread.base_entropy, match_entropy);
	}
};

}
}

#endif // SLIDINGENTROPY_H
