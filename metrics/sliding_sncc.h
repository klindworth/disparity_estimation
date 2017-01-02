#ifndef METRICS_SLIDING_SNCC_H_
#define METRICS_SLIDING_SNCC_H_

#include "sliding_sad.h"
#include "metrics/pixelwise/sncc_disparitywise_calculator.h"

class sliding_sncc
{
public:
	typedef float prob_table_type;
	typedef sliding_sad_threaddata thread_type;

private:
	cv::Mat m_base, m_match;
	std::vector<cv::Mat_<float> > results;

public:
	inline sliding_sncc(const cv::Mat& base, const cv::Mat& match, const disparity_range& range, unsigned int) : m_base(base), m_match(match)
	{
		sncc_disparitywise_calculator sncc(base, match);
		results.resize(range.size());
		for(int d = range.start(); d <= range.end(); ++d)
			results[std::abs(d)] = sncc(d);
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
		thread.cbase_x = x;
	}

	inline prob_table_type increm(thread_type& thread, int x, int d)
	{
		int d_offset = d < 0 ? d : 0;
		return cv::norm(subwindow(results[std::abs(d)], x+d_offset, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY))/thread.cwindowsizeX/thread.cwindowsizeY;
	}
};

#endif

