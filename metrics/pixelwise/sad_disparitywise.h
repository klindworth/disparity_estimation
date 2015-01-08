#ifndef SAD_DISPARITYWISE_H
#define SAD_DISPARITYWISE_H

#include <opencv2/core/core.hpp>

class sad_disparitywise_calculator
{
public:
	typedef unsigned char result_type;

	sad_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch) : base(pbase), match(pmatch)
	{
	}

	cv::Mat operator ()(int d)
	{
		cv::Mat pbase = prepare_base(base, d);
		cv::Mat pmatch = prepare_match(match, d);

		cv::Mat diff;
		cv::absdiff(pbase, pmatch, diff);

		return diff; //TODO: result enlargement?, channel summation
	}

private:
	cv::Mat base, match;
};

#endif
