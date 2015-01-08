#ifndef ENTROPY_H
#define ENTROPY_H

namespace costmap_creators
{

namespace entropy
{

class entropies
{
public:
	cv::Mat X, Y, XY;
};

//TODO: check bins boundaries in non-soft case: likely to be broken

template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_joint_entropy_unnormalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;
	for(int i = 1; i <= bins; ++i)
	{
		for(int j = 1; j <= bins; ++j)
		{
			auto ccounter = counter(i,j);
			normalize_counter += ccounter;
			result += entropy_table(ccounter);
		}
	}
	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename data_type, typename result_type, typename counter_type, typename entropytable_type>
inline void joint_entropy_result_reset(result_type& result, unsigned int& normalize_counter, counter_type& counter, const entropytable_type& entropy_table, data_type cleft, data_type cright)
{
	auto ccounter = counter(cleft, cright);
	counter(cleft, cright) = 0;
	result += entropy_table(ccounter);
	normalize_counter += ccounter;
}

template<typename result_type, typename counter_type, typename entropytable_type, typename data_type>
inline result_type calculate_joint_entropy_unnormalized_sparse(counter_type& counter, const entropytable_type& entropy_table, int bins, int len, const data_type* dataLeft, const data_type* dataRight)
{
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;

	//reset borders
	for(int i = 0; i <= bins; ++i)
	{
		counter(0, i) = 0;
		counter(i, 0) = 0;
		counter(bins+1, i) = 0;
		counter(i, bins+1) = 0;
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

	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_entropy_unnormalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	auto *counter_ptr = counter.at(1);
	result_type result = 0.0f;
	unsigned int normalize_counter = 0;
	for(int i = 1; i <= bins; ++i)
	{
		normalize_counter += *counter_ptr;
		result += entropy_table(*counter_ptr++);
	}
	result_type n = normalize_counter;
	result /= n;
	result = std::log(n) - result;

	return result;
}

template<typename result_type, typename counter_type, typename entropytable_type, typename data_type>
inline result_type calculate_joint_entropy_normalized_sparse(counter_type& counter, const entropytable_type& entropy_table, int len, const data_type* dataLeft, const data_type* dataRight)
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

template<typename result_type, typename counter_type, typename entropytable_type>
inline result_type calculate_entropy_normalized(counter_type& counter, const entropytable_type& entropy_table, int bins)
{
	auto *counter_ptr = counter.data;
	result_type result = 0.0f;
	for(int i = 0; i < bins; ++i)
		result += entropy_table(*counter_ptr++);

	return result;
}

template<typename counter_type, typename data_type>
inline void calculate_joint_soft_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
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
inline void calculate_joint_histogramm(counter_type& counter, const data_type* dataLeft, const data_type* dataRight, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
		counter(*dataLeft++, *dataRight++) += 1;
}

template<typename counter_type, typename data_type>
inline void calculate_histogramm(counter_type& counter, const data_type* data, int len)
{
	counter.reset();
	for(int i = 0; i < len; ++i)
		counter(*data++) += 1;
}

template<typename counter_type, typename data_type>
inline void calculate_soft_histogramm(counter_type& counter, const data_type* data, int len)
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
