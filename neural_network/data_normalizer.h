#ifndef DATA_NORMALIZER_H
#define DATA_NORMALIZER_H

#include <vector>
#include <iostream>

namespace neural_network
{

template<typename T>
class data_normalizer
{
private:
	template<typename lambda_type>
	void gather_sample_statistic(const std::vector<T>& data, std::vector<T>& sums, int& count, lambda_type func)
	{
		int crange = (data.size() - onetimesize)/ vector_size;
		const T* ptr = data.data();
		for(int k = 0; k < crange; ++k)
		{
			for(int i = 0; i < vector_size; ++i)
				sums[i] += func(*ptr++, i);
			++count;
		}
		for(int k = 0; k < onetimesize; ++k)
			sums[vector_size+k] += func(*ptr++, vector_size+k);

		assert(std::distance(data.data(), ptr) == (int)data.size());
	}

	template<typename lambda_type>
	void gather_statistic(const std::vector<std::vector<T>>& data, std::vector<T>& sums, int& count, lambda_type func)
	{
		assert(sums.size() == vector_size + onetimesize);
		std::fill(sums.begin(), sums.end(), 0);
		count = 0;

		for(const std::vector<T>& cdata :data)
			gather_sample_statistic(cdata, sums, count, func);
	}

	void normalize_statistics(std::vector<T>& sums, int count, std::size_t samples)
	{
		for(int i = 0; i < vector_size; ++i)
			sums[i] /= count;
		for(int i = vector_size; i < vector_size+onetimesize; ++i)
			sums[i] /= samples;
	}

public:

	data_normalizer(int vector_size, int onetimesize)
	{
		this->vector_size = vector_size;
		this->onetimesize = onetimesize;

		mean_normalizers.resize(vector_size+onetimesize);
		stddev_normalizers.resize(vector_size+onetimesize);
	}

	data_normalizer(std::istream& stream)
	{
		this->read(stream);
	}

	void read(std::istream& stream)
	{
		stream >> vector_size;
		stream >> onetimesize;

		mean_normalizers.resize(vector_size+onetimesize);
		stddev_normalizers.resize(vector_size+onetimesize);

		for(auto& cmean : mean_normalizers)
			stream >> cmean;
		for(auto& cstddev : stddev_normalizers)
			stream >> cstddev;
	}

	void write(std::ostream& stream) const
	{
		stream << vector_size << " ";
		stream << onetimesize << " ";

		for(auto& cmean : mean_normalizers)
			stream << cmean;
		for(auto& cstddev : stddev_normalizers)
			stream << cstddev;

		std::copy(mean_normalizers.begin(), mean_normalizers.end(), std::ostream_iterator<T>(stream, " "));
		std::copy(stddev_normalizers.begin(), stddev_normalizers.end(), std::ostream_iterator<T>(stream, " "));
	}

	void gather(const std::vector<std::vector<T>>& data)
	{
		mean_normalizers.resize(vector_size+onetimesize);
		stddev_normalizers.resize(vector_size+onetimesize);

		int mean_count = 0;
		gather_statistic(data, mean_normalizers, mean_count, [](T val, std::size_t) {return val;});
		normalize_statistics(mean_normalizers, mean_count, data.size());

		int std_count = 0;
		gather_statistic(data, stddev_normalizers, std_count, [&](T val, std::size_t n_idx) {
			T submean = val - mean_normalizers[n_idx];
			return submean*submean;
		});
		normalize_statistics(stddev_normalizers, std_count, data.size());

		for(auto& val : stddev_normalizers)
			val = 1.0 / std::sqrt(val);
	}

	void apply(T *ptr, int n)
	{
		int cmax = (n - onetimesize) / vector_size;
		assert((n - onetimesize) % vector_size == 0);
		assert(mean_normalizers.size() == stddev_normalizers.size());
		for(int j = 0; j < cmax; ++j)
		{
			for(int i = 0; i < vector_size; ++i)
			{
				*ptr -= mean_normalizers[i];
				*ptr++ *= stddev_normalizers[i];
			}
		}
		for(int j = 0; j < onetimesize; ++j)
		{
			*ptr -= mean_normalizers[vector_size+j];
			*ptr++ *= stddev_normalizers[vector_size+j];
		}
	}

	void apply(std::vector<T>& data)
	{
		apply(data.data(), data.size());
	}

	int vector_size, onetimesize;

	std::vector<T> mean_normalizers, stddev_normalizers;
};

}

#endif
