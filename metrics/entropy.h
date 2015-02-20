#ifndef ENTROPY_H
#define ENTROPY_H

/*
Copyright (c) 2014, Kai Klindworth
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

namespace costmap_creators
{

namespace entropy
{

class entropies
{
public:
	cv::Mat X, Y, XY;
};

template<typename data_type, typename result_type, typename counter_type, typename entropytable_type>
inline void joint_entropy_result_reset(result_type& result, unsigned int& normalize_counter, counter_type& counter, const entropytable_type& entropy_table, data_type cleft, data_type cright)
{
	auto ccounter = counter(cleft, cright);
	counter(cleft, cright) = 0;
	result += entropy_table(ccounter);
	normalize_counter += ccounter;
}

template<typename result_type>
struct soft
{
	static constexpr int additional_bins() { return 2; }

	static inline result_type normalize(result_type result, result_type n)
	{
		result /= n;
		result = std::log(n) - result;
		return result;
	}

	template<typename counter_type, typename entropytable_type>
	static inline result_type calculate_entropy(counter_type& counter, const entropytable_type& entropy_table, int bins)
	{
		auto *counter_ptr = counter.ptr(1);
		result_type result = 0.0f;
		unsigned int normalize_counter = 0;
		for(int i = 1; i < bins-1; ++i)
		{
			normalize_counter += *counter_ptr;
			result += entropy_table(*counter_ptr++);
		}
		return normalize(result, normalize_counter);
	}

	template<typename counter_type, typename entropytable_type>
	static inline result_type calculate_joint_entropy(counter_type& counter, const entropytable_type& entropy_table, int bins)
	{
		result_type result = 0.0f;
		unsigned int normalize_counter = 0;
		for(int i = 1; i < bins-1; ++i)
		{
			for(int j = 1; j < bins-1; ++j)
			{
				auto ccounter = counter(i,j);
				normalize_counter += ccounter;
				result += entropy_table(ccounter);
			}
		}
		return normalize(result, normalize_counter);
	}

	template<typename counter_type, typename entropytable_type, typename data_type>
	static inline result_type calculate_joint_entropy_sparse(counter_type& counter, const entropytable_type& entropy_table, int bins, int len, const data_type* dataLeft, const data_type* dataRight)
	{
		result_type result = 0.0f;
		unsigned int normalize_counter = 0;

		//reset borders
		for(int i = 0; i < bins; ++i)
		{
			counter(0, i) = 0;
			counter(i, 0) = 0;
			counter(bins-1, i) = 0;
			counter(i, bins-1) = 0;
		}

		for(int i = 0; i < len; ++i)
		{
			const data_type cleft  = *dataLeft++;
			const data_type cright = *dataRight++;
			joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft, cright+1);
			joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright);
			joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright+1);
			joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+1, cright+2);
			joint_entropy_result_reset<data_type>(result, normalize_counter, counter, entropy_table, cleft+2, cright+1);
		}

		return normalize(result, normalize_counter);
	}

	static inline void fill_entropytable(cv::Mat_<result_type>& entropy_table, int size)
	{
		assert(size > 0);
		entropy_table = cv::Mat_<result_type>(size, 1);
		result_type *entropy_table_ptr = entropy_table[0];

		*entropy_table_ptr++ = 0.0f;
		for(int i = 1; i < size; ++i)
			*entropy_table_ptr++ = i*std::log(i);
	}

	template<typename counter_type, typename data_type>
	inline static void calculate_joint_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
	{
		counter.reset();
		for(int i = 0; i < len; ++i)
		{
			const data_type cleft  = *dataLeft++;
			const data_type cright = *dataRight++;
			counter(cleft,   cright+1) += 1;
			counter(cleft+1, cright)   += 1;
			counter(cleft+1, cright+1) += 5;
			counter(cleft+1, cright+2) += 1;
			counter(cleft+2, cright+1) += 1;
		}
	}

	template<typename counter_type, typename data_type>
	static inline void calculate_histogramm(counter_type& counter, const data_type* data, int len)
	{
		counter.reset();
		for(int i = 0; i < len; ++i)
		{
			data_type cdata = *data++;
			counter(cdata) += 1;
			counter(cdata+1) += 7;
			counter(cdata+2) += 1;
		}
	}
};

template<typename result_type>
struct classic
{
	static constexpr int additional_bins() { return 0; }

	template<typename counter_type, typename entropytable_type, typename data_type>
	static inline result_type calculate_joint_entropy_sparse(counter_type& counter, const entropytable_type& entropy_table, int len, const data_type* dataLeft, const data_type* dataRight)
	{
		result_type result = 0.0f;
		for(int i = 0; i < len; ++i)
		{
			const data_type cleft  = *dataLeft++;
			const data_type cright = *dataRight++;
			result += entropy_table(counter(cleft,cright));
			counter(cleft,cright) = 0;
		}

		return result;
	}

	template<typename counter_type, typename entropytable_type>
	static inline result_type calculate_entropy(counter_type& counter, const entropytable_type& entropy_table, int bins)
	{
		auto *counter_ptr = counter.ptr();
		result_type result = 0.0f;
		for(int i = 0; i < bins; ++i)
			result += entropy_table(*counter_ptr++);

		return result;
	}

	template<typename counter_type, typename data_type>
	static inline void calculate_joint_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
	{
		counter.reset();
		for(int i = 0; i < len; ++i)
			counter(*dataLeft++, *dataRight++) += 1;
	}

	template<typename counter_type, typename data_type>
	static inline void calculate_histogramm(counter_type& counter, const data_type* data, int len)
	{
		counter.reset();
		for(int i = 0; i < len; ++i)
			counter(*data++) += 1;
	}

	static void fill_entropytable(cv::Mat_<result_type>& entropy_table, int size)
	{
		//entropy_table = cv::Mat(size, 1, CV_32FC1);
		entropy_table = cv::Mat_<result_type>(size,1);
		result_type *entropy_table_ptr = entropy_table[0];
		result_type n = 1.0f/size;

		*entropy_table_ptr++ = 0.0f;
		for(int i = 1; i < size; ++i)
			*entropy_table_ptr++ = -i*n*std::log(i*n);
	}
};

//TODO: check bins boundaries in non-soft case: likely to be broken


inline void fill_entropytable_normalized(cv::Mat_<float>& entropy_table, int size)
{
	entropy_table = cv::Mat(size, 1, CV_32FC1);
	float *entropy_table_ptr = entropy_table[0];
	float n = 1.0f/size;

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = -i*n*std::log(i*n);
}

inline void fill_entropytable_unnormalized(cv::Mat_<float>& entropy_table, int size)
{
	assert(size > 0);
	entropy_table = cv::Mat(size, 1, CV_32FC1);
	/*float *entropy_table_ptr = entropy_table[0];

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = i*std::log(i);*/

	entropy_table(0) = 0.0f;
	#pragma omp parallel for
	for(int i = 1; i < size; ++i)
		entropy_table(i) = i*std::log(i);
}

inline void fill_entropytable_unnormalized(cv::Mat_<double>& entropy_table, int size)
{
	assert(size > 0);

	entropy_table = cv::Mat(size, 1, CV_64FC1);
	double *entropy_table_ptr = entropy_table[0];

	*entropy_table_ptr++ = 0.0f;
	for(int i = 1; i < size; ++i)
		*entropy_table_ptr++ = i*std::log(i);
}

}

}

#endif
