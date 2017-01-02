#ifndef METRICS_SLIDING_SAD_H_
#define METRICS_SLIDING_SAD_H_

#include "sad_disparitywise.h"
#include "disparity_toolkit/disparity_range.h"

class sliding_sad_threaddata
{
public:
	cv::Mat m_base;
	int cwindowsizeX;
	int cwindowsizeY;
	int crow;
	int cbase_x;
};

class sliding_sad
{
public:
	typedef float prob_table_type;
	typedef sliding_sad_threaddata thread_type;
private:

	cv::Mat m_base, m_match;

public:
	inline sliding_sad(const cv::Mat& base, const cv::Mat& match, const disparity_range&, unsigned int /*max_windowsize*/) : m_base(base), m_match(match)
	{
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
		thread.cbase_x = x;
		thread.m_base = subwindow(m_base, x, thread.crow, cwindowsizeX, cwindowsizeY).clone();
	}

	inline prob_table_type increm(thread_type& thread, int x, int d)
	{
		cv::Mat match_window = subwindow(m_match, x+d, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();

		return cv::norm(thread.m_base, match_window, cv::NORM_L1)/match_window.total()/256;
	}
};

#endif
