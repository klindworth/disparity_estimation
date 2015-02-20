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
#include "entropy.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace costmap_creators
{

namespace entropy
{

template<int quantizer>
class single_fixed_windowsize
{
private:
	static const int bins = 256/quantizer;
	cv::Mat_<float> entropy_table;
	unsigned int windowsize;

	using entropy_style = classic<float>;
public:
	typedef fast_array<unsigned short, bins> thread_type;

	single_fixed_windowsize(unsigned int pwindowsize) : windowsize(pwindowsize) {
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize);
	}

	inline float increm(thread_type& thread, cv::Mat& window)
	{
		cv::Mat serWindow = window.clone();
		//creating histogramm
		entropy_style::calculate_histogramm(thread, serWindow.data, windowsize*windowsize);

		//compute probability and log-proability
		return entropy_style::calculate_entropy(thread, entropy_table, bins);
	}
};

template<int quantizer>
class single_fixed_windowsize_soft
{
private:
	using entropy_style = soft<float>;
	static const int bins = 256/quantizer + entropy_style::additional_bins();
	cv::Mat_<float> entropy_table;
	const unsigned int windowsize;

public:
	typedef fast_array<unsigned short, bins> thread_type;

	single_fixed_windowsize_soft(unsigned int pwindowsize) : windowsize(pwindowsize) {
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize*7);
	}

	inline float increm(thread_type& thread, const cv::Mat& window)
	{
		cv::Mat serWindow = window.clone();
		//creating histogramm
		entropy_style::calculate_histogramm(thread, serWindow.data, windowsize*windowsize);

		//compute probability and log-proability
		return entropy_style::calculate_entropy(thread, entropy_table, bins);
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

	using entropy_style = classic<float>;

private:
	cv::Mat_<prob_table_type> entropy_table;

public:
	joint_fixed_windowsize(const cv::Mat& /*match*/, int pwindowsize) : windowsize(pwindowsize)
	{
		fill_entropytable_normalized(entropy_table, windowsize*windowsize);
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, const cv::Mat& match, int y)
	{
		thread.m_match_table = serializeRow<unsigned char>(match, y, windowsize, false);
	}

	inline void prepare_window(thread_type& thread, const cv::Mat& base)
	{
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();
	}

	//calculates the histogramms for mutual information
	inline prob_table_type increm(thread_type& thread, int x)
	{
		//creating histogramm
		entropy_style::calculate_joint_histogramm(thread.counter_array, thread.m_base[0], thread.m_match_table[x], windowsize*windowsize);

		//compute entropy
		if(windowsize*windowsize < bins*bins)
			return entropy_style::calculate_joint_entropy_sparse(thread.counter_array, entropy_table, windowsize*windowsize, thread.m_base[0], thread.m_match_table[x]);
		else
			return entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins*bins);
	}
};

template<int quantizer>
class joint_fixed_windowsize_soft
{
public:
	using entropy_style = soft<float>;
	typedef float prob_table_type;
	static const int bins = 256/quantizer + entropy_style::additional_bins();
	typedef joint_fixed_windowsize_threaddata<bins> thread_type;
	const unsigned int windowsize;

private:
	cv::Mat_<prob_table_type> entropy_table;

public:
	joint_fixed_windowsize_soft(const cv::Mat& /*match*/, unsigned int pwindowsize) : windowsize(pwindowsize)
	{
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize*9);
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, const cv::Mat& match, int y)
	{
		thread.m_match_table = serializeRow<unsigned char>(match, y, windowsize, false);
	}

	inline void prepare_window(thread_type& thread, const cv::Mat& base)
	{
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();
	}

	//calculates the histogramms for mutual information
	inline prob_table_type increm(thread_type& thread, int x)
	{

		//creating histogramm
		entropy_style::calculate_joint_histogramm(thread.counter_array, thread.m_base[0], thread.m_match_table[x], windowsize*windowsize);

		//compute entropy
		//return calculate_joint_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);
		if(windowsize*windowsize*6 <= bins*bins)
			return entropy_style::calculate_joint_entropy_sparse(thread.counter_array, entropy_table, bins, windowsize*windowsize, thread.m_base[0], thread.m_match_table[x]);
		else
			return entropy_style::calculate_joint_entropy(thread.counter_array, entropy_table, bins);
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
	using entropy_style = soft<float>;
	static const int bins = 256/quantizer + entropy_style::additional_bins();
	typedef flexible_windowsize_threaddata<bins> thread_type;


private:
	cv::Mat_<prob_table_type> entropy_table;
	cv::Mat m_match;
	cost_func m_evalfunc;
	const int windowsize;

public:
	flexible_windowsize(const cv::Mat& match, unsigned int max_windowsize) : m_match(match), windowsize(max_windowsize)
	{
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize*9);
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, const cv::Mat& /*match*/, int y)
	{
		thread.crow = y;
	}

	inline void prepare_window(thread_type& thread, const cv::Mat& base, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();

		entropy_style::calculate_histogramm(thread.counter_array, thread.m_base[0], thread.m_base.total());
		thread.base_entropy = entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins);
	}

	//calculates the histogramms
	inline prob_table_type increm(thread_type& thread, int x)
	{
		cv::Mat_<unsigned char> match_window = subwindow(m_match, x, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();
		//creating joint histogramm, compute joint entropy
		entropy_style::calculate_joint_histogramm(thread.counter_array, thread.m_base[0], match_window[0], thread.cwindowsizeX*thread.cwindowsizeY);
		float joint_entropy;
		if(thread.cwindowsizeX*thread.cwindowsizeY*6 <= bins*bins)
			joint_entropy = entropy_style::calculate_joint_entropy_sparse(thread.counter_array, entropy_table, bins, thread.cwindowsizeX*thread.cwindowsizeY, thread.m_base[0], match_window[0]);
		else
			joint_entropy = entropy_style::calculate_joint_entropy(thread.counter_array, entropy_table, bins);

		entropy_style::calculate_histogramm(thread.counter_array, match_window.data, match_window.total());
		float match_entropy = entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins);

		return m_evalfunc(joint_entropy, thread.base_entropy, match_entropy);
	}
};

}
}

#endif // SLIDINGENTROPY_H
