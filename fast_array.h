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
	static const int size = tsize;
	T *data;

	fast_array() {
		data = static_cast<T*>(cv::fastMalloc(size*sizeof(T)));
		reset();
	}

	inline T& operator()(int x) {
		assert(x >= 0 && x < size);
		return *(data + x);
	}

	inline T* at(int x) {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline void reset() {
		memset(data, 0, size*sizeof(T));
	}

	~fast_array() {
		cv::fastFree(data);
	}
};

template<typename T, int tsize, int tsize2>
class fast_array2d
{
public:
	T *data;
	static const int size = tsize*tsize2;
	static const int step1 = tsize;
	static const int step2 = tsize2;

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

	inline T* at(int x) {
		assert(x >= 0 && x < size);
		return data + x;
	}

	inline T* at(int x, int y) {
		assert(x >= 0 && y >= 0 && x < step1 && y < step2);
		return data + step1*x + y;
	}

	inline void reset() {
		memset(data, 0, size*sizeof(T));
	}

	~fast_array2d()	{
		cv::fastFree(data);
	}
};


//TODO: in C++11 change reference_count to atomic for thread safety and align the memory for POD types
template<typename T>
class DataStoreInternal
{
public:
	DataStoreInternal(int size)
	{
		reference_count = 1;
		//data = static_cast<T*>(cv::fastMalloc(size*sizeof(T)));
		data = cv::allocate<T>((size_t)size);
		this->size = size;
	}

	~DataStoreInternal()
	{
		//cv::fastFree(data);
		cv::deallocate<T>(data, size);
	}

	T* data;
	int reference_count;
	int size;
};

//Attention: currently not aligned!!
template<typename T>
class DataStore2D
{
public:
	DataStore2D() : m_rows(0), m_cols(0)
	{
		data = 0;
		m_internal = 0;
	}

	DataStore2D(int rowCount, int colCount, bool setzero = false) : m_rows(rowCount), m_cols(colCount)
	{
		m_internal = new DataStoreInternal<T>(m_rows*m_cols);
		data = m_internal->data;
		if(setzero)
			memset(data, 0, m_rows*m_cols*sizeof(T));
	}

	DataStore2D(const DataStore2D<T>& src) : data(src.data), m_rows(src.m_rows), m_cols(src.m_cols),m_internal(src.m_internal)
	{
		m_internal->reference_count += 1;
	}

	DataStore2D<T>& operator=(const DataStore2D<T>& src)
	{
		if(&src == this)
			return *this;

		if(m_internal)
		{
			m_internal->reference_count -= 1;
			if(m_internal->reference_count == 0)
				delete m_internal;
		}
		m_internal = src.m_internal;
		m_internal->reference_count += 1;
		m_rows = src.rows();
		m_cols = src.cols();
		data = m_internal->data;

		return *this;
	}

	~DataStore2D()
	{
		if(m_internal)
		{
			m_internal->reference_count -= 1;
			if(m_internal->reference_count == 0)
				delete m_internal;
		}
	}

	inline T& operator()(int row)
	{
		assert(row >= 0 && row < m_rows && data);
		return data[row*m_cols];
	}

	inline T& operator()(int row, int col)
	{
		assert(row >= 0 && row < m_rows && col >= 0 && col < m_cols && data);
		return data[row*m_cols+col];
	}

	inline T* ptr(int row, int col)
	{
		assert(row >= 0 && row < m_rows && col >= 0 && col < m_cols && data);
		return data +row*m_cols+col;
	}

	inline int rows() const
	{
		return m_rows;
	}

	inline int cols() const
	{
		return m_cols;
	}

	T* data;

private:
	int m_rows;
	int m_cols;
	DataStoreInternal<T>* m_internal;
};

#endif // FAST_ARRAY_H
