#ifndef DISPARITYWISE_CALCULATOR_H
#define DISPARITYWISE_CALCULATOR_H

#include <opencv2/core/core.hpp>

inline cv::Mat prepare_base(const cv::Mat& src, int d)
{
	if(d > 0)
		return src(cv::Range(0, src.rows), cv::Range(0, src.cols-d));
	else
		return src(cv::Range(0, src.rows), cv::Range(-d, src.cols));
}

inline cv::Mat prepare_match(const cv::Mat& src, int d)
{
	if(d > 0)
		return src(cv::Range(0, src.rows), cv::Range(d, src.cols));
	else
		return src(cv::Range(0, src.rows), cv::Range(0, src.cols+d));
}

inline cv::Rect prepare_result_rect(cv::Size input, int d)
{
	return cv::Rect(std::max(-d, 0), 0, input.width, input.height);
}

inline cv::Rect prepare_border_rect(cv::Size input, int d)
{
	if(d > 0)
		return cv::Rect(input.width, 0, d, input.height);
	else
		return cv::Rect(0,0,-d,input.height);
}

inline cv::Mat prepare_result(cv::Size src, int d, float val)
{
	cv::Rect rect_result = prepare_result_rect(src, d);
	cv::Rect rect_border = prepare_border_rect(src, d);
	cv::Size nsize(rect_result.width + rect_border.width, src.height);

	cv::Mat_<float> result(nsize);
	result(rect_border) = val;

	return result;
}

inline int calculate_row_length(const cv::Mat& src, int d)
{
	return src.cols - std::abs(d);
}

inline int calculate_base_offset(int d)
{
	return std::max(-d, 0);
}

inline int calculate_match_offset(int d)
{
	return std::max(d, 0);
}

#endif
