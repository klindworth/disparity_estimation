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

#include "disparity_toolkit/genericfunctions.h"
#include "fast_array.h"
#include "entropy.h"
#include "disparity_toolkit/disparity_range.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace costmap_creators
{

namespace entropy
{

template<typename T, int quantizer, bool soft>
class single_fixed_windowsize
{
private:
	using entropy_style = typename get_entropy_style<T>::type;

	static const int bins = 256/quantizer+entropy_style::additional_bins();
	std::vector<T> entropy_table;
	unsigned int windowsize;


public:
	typedef fast_array<unsigned short, bins> thread_type;
	typedef T result_type;

	single_fixed_windowsize(unsigned int pwindowsize) : windowsize(pwindowsize) {
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize*entropy_style::counter_factor());
	}

	inline T increm(thread_type& thread, cv::Mat& window)
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
	cv::Mat_<unsigned char> m_basewindow;
	cv::Mat_<unsigned char> m_match_table;
	fast_array2d<unsigned short, bins, bins> counter_array;
};

template<typename T, int quantizer, bool soft>
class joint_fixed_windowsize
{
public:
	using entropy_style = typename get_entropy_style<T, soft>::type;

	static const int bins = 256/quantizer + entropy_style::additional_bins();
	typedef joint_fixed_windowsize_threaddata<bins> thread_type;
	typedef T result_type;
	const unsigned int windowsize;

private:
	std::vector<T> entropy_table;
	cv::Mat m_base, m_match;

public:
	joint_fixed_windowsize(const cv::Mat& base, const cv::Mat& match, int pwindowsize) : windowsize(pwindowsize), m_base(base), m_match(match)
	{
		entropy_style::fill_entropytable(entropy_table, windowsize*windowsize*entropy_style::counter_factor());
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, int y)
	{
		thread.m_match_table = serializeRow<unsigned char>(m_match, y, windowsize, false);
	}

	inline void prepare_window(thread_type& thread, const cv::Mat& base)
	{
		//copy the window for L1 Cache friendlieness
		thread.m_basewindow = base.clone();
	}

	//calculates the histogramms for mutual information
	inline T increm(thread_type& thread, int x)
	{
		//creating histogramm
		entropy_style::calculate_joint_histogramm(thread.counter_array, thread.m_basewindow[0], thread.m_match_table[x], windowsize*windowsize);

		//compute entropy
		if(windowsize*windowsize*entropy_style::kernel_size() < bins*bins)
			return entropy_style::calculate_joint_entropy_sparse(thread.counter_array, entropy_table, bins, windowsize*windowsize, thread.m_basewindow[0], thread.m_match_table[x]);
		else
			return entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins*bins);
	}
};

template<int bins>
class flexible_windowsize_threaddata
{
public:
	cv::Mat_<unsigned char> m_basewindow;
	fast_array2d<unsigned short, bins, bins> counter_array;
	int cwindowsizeX;
	int cwindowsizeY;
	int crow;
	float base_entropy;
};

template<typename T, typename cost_func, int quantizer>
class flexible_windowsize
{
public:
	using entropy_style = soft_entropy<T>;
	static const int bins = 256/quantizer + entropy_style::additional_bins();
	typedef flexible_windowsize_threaddata<bins> thread_type;


private:
	std::vector<T> entropy_table;
	cv::Mat m_base, m_match;

public:
	flexible_windowsize(const cv::Mat& base, const cv::Mat& match, const disparity_range&, unsigned int max_windowsize) : m_base(base), m_match(match)
	{
		entropy_style::fill_entropytable(entropy_table, max_windowsize*max_windowsize*entropy_style::counter_factor());
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, int y)
	{
		thread.crow = y;
	}

	inline void prepare_window(thread_type& thread, int x, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		//copy the window for L1 Cache friendlieness
		thread.m_basewindow = subwindow(m_base, x, thread.crow, cwindowsizeX, cwindowsizeY).clone();

		entropy_style::calculate_histogramm(thread.counter_array, thread.m_basewindow[0], thread.m_basewindow.total());
		thread.base_entropy = entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins);
	}

	//calculates the histogramms
	inline T increm(thread_type& thread, int x, int d)
	{
		cv::Mat_<unsigned char> match_window = subwindow(m_match, x+d, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();
		//creating joint histogramm, compute joint entropy
		entropy_style::calculate_joint_histogramm(thread.counter_array, thread.m_basewindow[0], match_window[0], thread.cwindowsizeX*thread.cwindowsizeY);
		T joint_entropy;
		if(thread.cwindowsizeX*thread.cwindowsizeY*entropy_style::kernel_size() <= bins*bins)
			joint_entropy = entropy_style::calculate_joint_entropy_sparse(thread.counter_array, entropy_table, bins, thread.cwindowsizeX*thread.cwindowsizeY, thread.m_basewindow[0], match_window[0]);
		else
			joint_entropy = entropy_style::calculate_joint_entropy(thread.counter_array, entropy_table, bins);

		entropy_style::calculate_histogramm(thread.counter_array, match_window.data, match_window.total());
		T match_entropy = entropy_style::calculate_entropy(thread.counter_array, entropy_table, bins);

		return cost_func::calculate(joint_entropy, thread.base_entropy, match_entropy);
	}
};

}
}

#endif // SLIDINGENTROPY_H
