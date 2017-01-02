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

#ifndef FAST_ARRAY_H
#define FAST_ARRAY_H

#include <opencv2/core/core.hpp>

template<typename T, int tsize>
class fast_array
{
public:
	static constexpr int size = tsize;

	fast_array() {
		data = static_cast<T*>(cv::fastMalloc(size*sizeof(T)));
		reset();
	}

	inline T& operator()(int x) {
		assert(x >= 0 && x < size);
		return *(data + x);
	}

	inline const T& operator()(int x) const {
		assert(x >= 0 && x < size);
		return *(data + x);
	}

	inline T* ptr() {
		return data;
	}

	inline T* ptr(int x) {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline const T* ptr() const {
		return data;
	}

	inline const T* ptr(int x) const {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline void reset() {
		memset(data, 0, size*sizeof(T));
	}

	~fast_array() {
		cv::fastFree(data);
	}

private:
	T *data;
};

template<typename T, int tsize, int tsize2>
class fast_array2d
{
public:
	static constexpr int size = tsize*tsize2;
	static constexpr int step1 = tsize;
	static constexpr int step2 = tsize2;

	fast_array2d() {
		data = static_cast<T*>(cv::fastMalloc(size*sizeof(T)));
		reset();
	}

	inline T& operator()(int x) {
		assert(x >= 0 && x < size);
		return *(data + x);
	}

	inline T& operator()(int x, int y) {
		assert(x >= 0 && y >= 0 && x < step1 && y < step2);
		return *(data + step1*x + y);
	}

	inline const T& operator()(int x) const {
		assert(x >= 0 && x < size);
		return *(data + x);
	}

	inline const T& operator()(int x, int y) const {
		assert(x >= 0 && y >= 0 && x < step1 && y < step2);
		return *(data + step1*x + y);
	}

	inline T* ptr() {
		return data;
	}

	inline T* ptr(int x) {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline T* ptr(int x, int y) {
		assert(x >= 0 && y >= 0 && x < step1 && y < step2);
		return data + step1*x + y;
	}

	inline const T* ptr() const {
		return data;
	}

	inline const T* ptr(int x) const {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline const T* ptr(int x, int y) const {
		assert(x >= 0 && y >= 0 && x < step1 && y < step2);
		return data + step1*x + y;
	}

	inline void reset() {
		memset(data, 0, size*sizeof(T));
	}

	~fast_array2d() {
		cv::fastFree(data);
	}

private:
	T *data;
};

#endif // FAST_ARRAY_H
