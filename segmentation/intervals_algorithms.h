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

#ifndef INTERVALS_ALGORITHMS_H
#define INTERVALS_ALGORITHMS_H

#include "intervals.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <functional>

namespace intervals
{

inline bool weakIntervalLess(const region_interval& lhs, const region_interval& rhs, int d)
{
	return (lhs.y < rhs.y) || (lhs.y == rhs.y && lhs.upper <= rhs.lower - d); //TODO: check upper >= lower
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator>
void difference(InputIterator1 base_it, InputIterator1 base_end, InputIterator2 match_it, InputIterator2 match_end, int d, OutputIterator inserterDifferenceBase)
{
	assert(std::is_sorted(base_it, base_end));
	assert(std::is_sorted(match_it, match_end));

	region_interval last(0,0,0);
	bool lastSet = false;
	//sync y
	while(true)
	{
		do
		{
			while(base_it != base_end && (weakIntervalLess(*base_it, *match_it, d) || match_it == match_end ))
			{
				if(lastSet && last.y == base_it->y) //process rest of line
				{
					*inserterDifferenceBase = region_interval(base_it->y, last.upper, base_it->upper);
					++inserterDifferenceBase;
					lastSet = false;
				} else { //alone in a row
					*inserterDifferenceBase = *base_it;
					++inserterDifferenceBase;
					lastSet = false;
				}
				++base_it;
			}

			while(match_it != match_end && weakIntervalLess(*match_it, *base_it, -d))
				++match_it;

			//if((base_it == base_end) || (match_it == match_end))
				//return;
			if(base_it == base_end)
				return;
		} while(weakIntervalLess(*base_it, *match_it, d) || weakIntervalLess(*match_it, *base_it, -d) || match_it == match_end);//while(match_it->y != base_it->y);

		assert(match_it->y == base_it->y);
		assert(match_it != match_end && base_it != base_end);

		//overlapping?
		int lower = std::max(base_it->lower, match_it->lower - d);
		int upper = std::min(base_it->upper, match_it->upper - d);

		if(base_it->y != last.y)
			lastSet = false;

		assert(lower <= upper);
		{
			//process part before the new interval (difference for base)
			//if(base_it->x_lower < lower)// && base_it->x_upper != upper)
			if(lastSet && last.upper < lower)
			{
				*inserterDifferenceBase = region_interval(base_it->y, std::max(last.upper, base_it->lower), std::min(lower, base_it->upper));
				++inserterDifferenceBase;
			}
			else if(!lastSet && base_it->lower < lower)
			{
				*inserterDifferenceBase = region_interval(base_it->y, base_it->lower, std::min(lower, base_it->upper));
				++inserterDifferenceBase;
			}

			last = region_interval(base_it->y, lower, upper);
			lastSet = true;
		}

		if(base_it->upper == upper)
		{
			++base_it;
			lastSet = false;
		}

		if(match_it->upper - d == upper)
			++match_it;
	}
}

template<typename dst_type>
void set_region_value(cv::Mat& dst, const std::vector<region_interval>& pixel_idx, dst_type value)
{
	for(const region_interval& interval : pixel_idx)
	{
		dst_type *ptr = dst.ptr<dst_type>(interval.y, interval.lower);
		std::fill(ptr, ptr + interval.length(), value);
	}
}

template<>
inline void set_region_value(cv::Mat& dst, const std::vector<region_interval>& pixel_idx, unsigned char value)
{
	for(const region_interval& interval : pixel_idx)
		memset(dst.ptr<unsigned char>(interval.y, interval.lower), value, interval.length());
}

/**
 * @brief Calls func for every pixel in an interval. The function must accept the position as parameter
 * @param interval Interval
 * @param func Function that accepts the coordinates as cv::Point
 */
template<typename lambda_type>
inline void foreach_interval_point(const region_interval& interval, lambda_type func)
{
	for(int x = interval.lower; x < interval.upper; ++x)
		func(cv::Point(x, interval.y));
}

/**
 * @brief Calls func for every pixel in a region. The function must accept the position as parameter
 * @param it Start iterator of the range of RegionIntervals
 * @param end End iterator of the range of RegionIntervals
 * @param func Function that accepts the coordinates as cv::Point
 */
template<typename Iterator, typename lambda_type>
inline void foreach_region_point(Iterator it, Iterator end, lambda_type func)
{
	for(; it != end; ++it)
		foreach_interval_point(*it, func);
}

/**
 * Adds within a region in dst the value change. The region is defined by the RegionIntervals in interval_container
 * @param dst Matrix that will be modified
 * @param interval_container Region within the dst matrix that will be modified
 * @param change Value that will be added
 */
template<typename dst_type>
void add_region_value(cv::Mat& dst, const std::vector<region_interval>& interval_container, dst_type change)
{
	foreach_region_point(interval_container.begin(), interval_container.end(), [=,&dst](cv::Point pt){
		dst.at<dst_type>(pt) += change;
	});
}

/**
 * Subtracts within a region in dst the value change. The region is defined by the RegionIntervals in interval_container
 * @param dst Matrix that will be modified
 * @param interval_container Region within the dst matrix that will be modified
 * @param change Value that will be substracted
 */
template<typename dst_type>
void substract_region_value(cv::Mat& dst, const std::vector<region_interval>& interval_container, dst_type change)
{
	foreach_region_point(interval_container.begin(), interval_container.end(), [=,&dst](cv::Point pt){
		dst.at<dst_type>(pt) -= change;
	});
}


/**
 * Calls a factory function every time a different interval occurs in the cv::Mat
 * @param values The matrix with the values in it.
 * @param factory The function is called, every time an interval ends.
 * Could be used to construct a new interval with the information passed to the function
 * (line, start and end of the intervall, value within the interval)
 * @param cmp_func Determines if two values belong to the same interval. The function should return true, if the two values should belong together.
 */
template<typename value_type, typename factory_type, typename cmp_type>
void convert_generic(const cv::Mat& values, factory_type factory, cmp_type cmp_func)
{
	assert(values.dims == 2);

	for(int y = 0; y < values.rows; ++y)
	{
		const value_type *src_ptr = values.ptr<value_type>(y,0);

		std::size_t last_x = 0;
		value_type last_value = *src_ptr;

		for(int x = 0; x < values.cols; ++x)
		{
			value_type cvalue = *src_ptr++;

			if(!cmp_func(last_value, cvalue))
			{
				factory(y, last_x, x, last_value);

				last_x = x;
				last_value = cvalue;
			}
		}

		factory(y, last_x, values.cols, *(src_ptr-1));
	}
}

/**
 * @brief Every time the value changes in the matrix, a new interval will be created.
 * [11122333] will trigger three calls of the factory function for [111],[22] and [333]
 * @param values
 * @param factory Function of type (int x, int lower, int upper, value_type value), that will be called, every time a new interval was discovered
 */
template<typename value_type, typename factory_type>
void convert_differential(const cv::Mat& values, factory_type factory)
{
	auto cmp_func = [](value_type last, value_type current) {
		return (last == current);
	};

	convert_generic<value_type>(values, factory, cmp_func);
}

template<typename value_type, typename InserterIterator>
void convert_mat_to_value(const cv::Mat_<value_type>& values, InserterIterator inserter)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, value_type value) {
		*inserter = value_region_interval<value_type>(y,lower,upper, value);
		++inserter;
	};

	convert_differential<value_type>(values, factory);
}

/**
 * Creates a new interval, each time a range of search_value is discovered in the matrix values.
 * E.g. if the search_value is 3, and values is [1233453332], two intervals will be created [33] and [333], the rest will be ignored,
 * @param inserter Container in which the RegionIntervals will be inserted
 * @param search_value Every range of search_value in values will create a new interval
 */
template<typename value_type, typename InserterIterator>
void turn_value_into_intervals(const cv::Mat_<value_type>& values, InserterIterator inserter, value_type search_value)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, value_type value) {
		if(value == search_value)
		{
			*inserter = region_interval(y, lower, upper);
			++inserter;
		}
	};

	convert_differential<value_type>(values, factory);
}

template<typename value_type, typename InserterIterator>
void convert_minima_ranges(const cv::Mat_<value_type>& values, InserterIterator inserter, value_type threshold)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, value_type value) {
		if(value < threshold)
		{
			*inserter = region_interval(y,lower,upper);
			++inserter;
		}
	};

	auto cmp_func = [=](value_type last, value_type current) {
		return (last > threshold && current > threshold) || (last <= threshold && current <= threshold);
	};

	convert_generic<value_type>(values, factory, cmp_func);
}

}

template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const region_interval& interval)
{
	stream << "(y: " << interval.y << ", x: " << interval.lower << "-" << interval.upper << ") ";
	return stream;
}

template<typename charT, typename traits, typename T>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const value_region_interval<T>& interval)
{
	stream << "(y: " << interval.y << ", x: " << interval.lower << "-" << interval.upper << ", value: " << interval.value << ") ";
	return stream;
}

#endif // INTERVALS_ALGORITHMS_H
