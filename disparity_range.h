#ifndef DISPARITY_RANGE_H
#define DISPARITY_RANGE_H

#include <cassert>
#include <cstddef>
#include <algorithm>

class disparity_range
{
public:
	//disparity_range() : _start(0), _end(0), _offset(0) {}
	disparity_range(int start, int end) : _start(start), _end(end), _offset(start) {}
	disparity_range(int start, int end, int offset) : _start(start), _end(end), _offset(offset) {}

	disparity_range subrange(int pivot, int delta) const
	{
		assert(pivot >= _start && pivot <= _end);

		int start = std::max(_start, pivot - delta);
		int end = std::min(_end, pivot + delta);

		return disparity_range(start, end, _offset);
	}

	std::size_t index(int d) const
	{
		assert(valid(d));

		return d-_offset;
	}

	bool valid(int d) const
	{
		return d >= _start && d <= _end;
	}

	inline int start() const { return _start; }
	inline int end() const { return _end; }
	inline int size() const { return _end - _start + 1; }
	inline int offset() const { return _offset; }

private:
	int _start, _end, _offset;
};

#endif
