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

#ifndef INTERVALS_H
#define INTERVALS_H

#include <assert.h>

/**
 * @brief The region_interval class represents an interval in x-direction
 * Because it just represents an interval in x direction, it has three fields: The y-coordinate, an the two coordinates in x-direction: lower and upper.
 * Attention: [lower, upper), like in the STL, the end is not part of the interval
 * @author Kai Klindworth
 */
class region_interval
{
public:
	inline void init(int y, int x_lower, int x_upper)
	{
		assert(y >= 0);
		assert(x_lower >= 0);
		assert(x_upper >= 0);
		assert(x_upper >= x_lower);

		this->y = y;
		this->lower = x_lower;
		this->upper = x_upper;
	}

	region_interval() {}
	region_interval(int y, int x_lower, int x_upper)
	{
		init(y, x_lower, x_upper);
	}

	int y;
	int lower;
	int upper;

	//! Moves the interval in x-direction
	inline void move(int offset) {
		lower += offset;
		upper += offset;
	}

	/**
	 * @brief move Moves the interval in x-direction. The values are capped to valid values.
	 * Valid means, the interval is within the image. Therefore you have to pass additionaly the witdth of the image.
	 * @param offset The amount of pixels, which the interval moves in x direction
	 * @param width Width of the image
	 */
	inline void move(int offset, int width) {
		lower = std::min(width, std::max(0, lower + offset));
		upper = std::min(width, std::max(0, upper + offset));
	}

	//! Returns the length (in x direction) of the interval
	inline int length() const
	{
		return upper - lower;
	}

	inline bool operator<(const region_interval& rhs) const
	{
		return (this->y < rhs.y) || (this->lower < rhs.lower && this->y == rhs.y);
	}

	//rhs is bigger (operator< ordered)
	inline bool intersectingOrdered(const region_interval& rhs) const
	{
		return((this->lower <= rhs.lower && this->y == rhs.y) && this->upper > rhs.lower);
	}

	//! Returns, if this interval have an intersection with another interval
	inline bool intersecting(const region_interval& rhs) const
	{
		return intersectingOrdered(rhs) || rhs.intersectingOrdered(*this);
	}
};

/**
 * @brief This is like region_interval but also stores an value associated with the interval.
 * The type of the value is the type that is passed as template parameter
 * @author Kai Klindworth
 */
template<typename T>
class value_region_interval : public region_interval
{
public:
	T value;
	value_region_interval() {}
	value_region_interval(region_interval interval, T value) {
		init(interval.y, interval.lower, interval.upper);
		this->value = value;
	}

	value_region_interval(int y, int x_lower, int x_upper, T value)
	{
		init(y, x_lower, x_upper);
		this->value = value;
	}

	//! Constructs an classic version of this interval without the additional value.
	region_interval to_interval()
	{
		return region_interval(y, lower, upper);
	}
};

inline bool operator==(const region_interval& lhs, const region_interval& rhs)
{
	return (lhs.y == rhs.y) && (lhs.lower == rhs.lower) && (lhs.upper == rhs.upper);
}

#endif // INTERVALS_H
