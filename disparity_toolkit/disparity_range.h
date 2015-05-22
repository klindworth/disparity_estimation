/*
Copyright (c) 2015, Kai Klindworth
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

#ifndef DISPARITY_RANGE_H
#define DISPARITY_RANGE_H

#include <cassert>
#include <cstddef>
#include <algorithm>

/*class current_disparity
{
public:
	std::size_t index;
	int disparity;

	inline current_disparity& operator++()
	{
		++index;
		++disparity;
		return *this;
	}
};*/

class disparity_range
{
public:
	disparity_range(int start, int end) : _start(start), _end(end), _offset(start) { assert(valid());}
	disparity_range(int start, int end, int offset) : _start(start), _end(end), _offset(offset) {assert(valid());}

	/**
	 * @brief subrange Creates a subrange around a pivot element.
	 * If the pivot element is near the border, closer than delta,
	 * the range will be automatically cut off at the borders of the original range. The offset will nor be affected.
	 * That means, you should call the function if the underlying array is the same like the original range.
	 * @param pivot Pivot element. This is the element in the middle of the new range
	 * @param delta Number of elements before and after the pivot element which should belong to the new range
	 * @return The newly created range
	 */
	disparity_range subrange(int pivot, int delta) const
	{
		assert(pivot >= _start && pivot <= _end);

		const int start = std::max(_start, pivot - delta);
		const int end = std::min(_end, pivot + delta);

		return disparity_range(start, end, _offset);
	}

	/**
	 * @brief subrange_with_subspace Like subrange, but with it's own offset.
	 * Usefull if you have a seperate array with the length of the new range.
	 * @param pivot
	 * @param delta
	 * @return
	 */
	disparity_range subrange_with_subspace(int pivot, int delta) const
	{
		assert(pivot >= _start && pivot <= _end);

		const int start = std::max(_start, pivot - delta);
		const int end = std::min(_end, pivot + delta);
		const int offset = pivot - delta;

		return disparity_range(start, end, offset); //FIXME: old offset ignored
	}

	disparity_range restrict_to_image(int pt, int image_size, int border_size = 0) const
	{
		const int min_pt = border_size;
		const int max_pt = image_size - border_size - 1;

		const int disp_start = std::min(std::max(pt + _start, min_pt), max_pt) - pt;
		const int disp_end   = std::min(std::max(pt + _end,   min_pt), max_pt) - pt;

		return disparity_range(disp_start, disp_end, _offset);
	}

	disparity_range without_offset() const
	{
		return disparity_range(_start, _end);
	}

	inline std::size_t index(int d) const
	{
		assert(valid_disparity(d));

		return d-_offset;
	}

	inline int disparity_at_index(std::size_t index) const
	{
		return index+_offset;
	}

	bool valid_disparity(int d) const
	{
		return d >= _start && d <= _end;
	}

	bool valid() const
	{
		return _start <= _end;
	}

	inline int start() const { return _start; }
	inline int end() const { return _end; }
	inline int size() const { return _end - _start + 1; }
	inline int offset() const { return _offset; }

	/*template<typename lambda_type>
	void foreach_disparity(lambda_type func)
	{
		current_disparity cdisp;
		cdisp.index = start() - offset();
		cdisp.disparity = start();
		int range = size();
		for(int i = 0; i < range; ++i)
		{
			func(cdisp);
			++cdisp;
		}
	}*/

private:
	const int _start, _end, _offset;
};

#endif
