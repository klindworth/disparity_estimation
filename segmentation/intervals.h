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

class RegionInterval
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

	RegionInterval() {}
	RegionInterval(int y, int x_lower, int x_upper)
	{
		init(y, x_lower, x_upper);
	}

	int y;
	int lower;
	int upper;

	inline void move(int offset) {
		lower += offset;
		upper += offset;
	}

	//ensure that you delete the intervals with no or negative length
	inline void move(int offset, int width) {
		lower += std::max(std::min(offset, width), 0);
		upper += std::max(std::min(offset, width), 0);
	}

	inline int length() const
	{
		return upper - lower;
	}

	inline bool operator<(const RegionInterval& rhs) const
	{
		return (this->y < rhs.y) || (this->lower < rhs.lower && this->y == rhs.y);
	}

	//rhs is bigger (operator< ordered)
	inline bool intersectingOrdered(const RegionInterval& rhs) const
	{
		return((this->lower <= rhs.lower && this->y == rhs.y) && this->upper > rhs.lower);
	}

	inline bool intersecting(const RegionInterval& rhs) const
	{
		return intersectingOrdered(rhs) || rhs.intersectingOrdered(*this);
	}
};

template<typename T>
class ValueRegionInterval : public RegionInterval
{
public:
	T value;
	ValueRegionInterval() {}
	ValueRegionInterval(RegionInterval interval, T value) {
		init(interval.y, interval.lower, interval.upper);
		this->value = value;
	}

	ValueRegionInterval(int y, int x_lower, int x_upper, T value)
	{
		init(y, x_lower, x_upper);
		this->value = value;
	}
	RegionInterval toInterval()
	{
		return RegionInterval(y, lower, upper);
	}
};

inline bool operator==(const RegionInterval& lhs, const RegionInterval& rhs)
{
	return (lhs.y == rhs.y) && (lhs.lower == rhs.lower) && (lhs.upper == rhs.upper);
}

#endif // INTERVALS_H
