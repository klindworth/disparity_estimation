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

#ifndef SPARSE_COUNTER_H
#define SPARSE_COUNTER_H

#include <utility>
#include <algorithm>
#include <map>
#include <unordered_map>

class sparse_counter_compare_yx
{
public:
	inline bool operator()(std::pair<int, int> lhs, std::pair<int, int> rhs) const
	{
		return (lhs.first < rhs.first) || ( (lhs.first == rhs.first) && (lhs.second < rhs.second) );
	}
};

class sparse_counter_compare_val
{
public:
	inline bool operator()(int lhs, int rhs) const
	{
		return lhs < rhs;
	}
};

template<typename index_type_t, typename compare_obj>
class sparse_counter
{
public:
	typedef index_type_t index_type;
	typedef std::map<index_type, int, compare_obj> internal_store;
	//typedef std::unordered_map<index_type, int> internal_store;
	typedef typename internal_store::iterator iterator;
	typedef typename internal_store::const_iterator const_iterator;

	sparse_counter()
	{
		total_counter = 0;
	}

	inline void increment(index_type index)
	{
		++total_counter;

		iterator it = data.find(index);
		if(it != data.end())
			++(it->second);
		else
			data.insert(std::make_pair(index, 1));
	}

	inline void increment(index_type index, int inc)
	{
		total_counter += inc;

		iterator it = data.find(index);
		if(it != data.end())
			it->second += inc;
		else
			data.insert(std::make_pair(index, inc));
	}

	inline int value(index_type index) const
	{
		iterator it = data.find(index);
		if(it != data.end())
			return it->second;
		else
			return 0;
	}

	inline void reset()	{
		total_counter = 0;
		data.clear();
	}

	inline iterator begin()	{
		return data.begin();
	}

	inline iterator end() {
		return data.end();
	}

	inline const_iterator cbegin() const {
		return data.cbegin();
	}

	inline const_iterator cend() const {
		return data.cend();
	}

	inline int total() const {
		return total_counter;
	}

	inline std::size_t size() const {
		return data.size();
	}

private:
	internal_store data;
	int total_counter;
};

typedef sparse_counter<std::pair<int, int>, sparse_counter_compare_yx> sparse_2d_histogramm;
typedef sparse_counter<int, sparse_counter_compare_val> sparse_histogramm;

#endif // SPARSE_COUNTER_H
