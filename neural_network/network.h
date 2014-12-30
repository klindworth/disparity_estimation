#ifndef NEURAL_NETWORK_NETWORK_H
#define NEURAL_NETWORK_NETWORK_H

#include <memory>
#include <cassert>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

#include <iostream>
#include <iterator>

#include <omp.h>
#include <neural_network/blas_wrapper.h>
#include <neural_network/layer.h>


template<typename T>
class neural_network
{
public:
	neural_network(int in_dim) : in_dim(in_dim), out_dim(in_dim)
	{}

	neural_network(int in_dim, int out_dim, const std::initializer_list<int>& list) : in_dim(in_dim), out_dim(in_dim)
	{
		bool dropout = false;
		for(int csize : list)
		{
			emplace_layer<fully_connected_layer>(csize);
			emplace_layer<relu_layer>();
			if(dropout)
				emplace_layer<dropout_layer>();
		}

		emplace_layer<fully_connected_layer>(out_dim);
		emplace_layer<softmax_output_layer>();
		//emplace_layer<tanh_entropy_output_layer>();
	}

	template<template<typename Ti> class layer_type, typename... args_type>
	void emplace_layer(args_type... args)
	{
		bool propagate = layers.size() > 0 ? true : false;
		layers.push_back(std::make_shared<layer_type<T>>(propagate, out_dim, args...));
		out_dim = layers.back()->output_dimension();

		layers.back()->init_weights();
	}

	float test(const std::vector<std::vector<T>>& data, const std::vector<short>& gt)
	{
		assert(data.size() == gt.size());

		int correct = 0;
		int approx_correct = 0;
		int approx_correct2 = 0;

		for(std::size_t i = 0; i < data.size(); ++i)
		{
			int result = this->predict(data[i].data());
			int expected = std::abs(gt[i]);

			if(result == expected)
				++correct;
			if(std::abs(result - expected) < 5)
				++approx_correct;
			if(std::abs(result - expected) < 10)
				++approx_correct2;
		}

		std::cout << "result: " << (float)correct/data.size() << ", approx5: " << (float)approx_correct/data.size() << ", approx10: " << (float)approx_correct2/data.size() << std::endl;
		return (float)correct/data.size();
	}

	std::vector<T> output(const T* data)
	{
		forward_propagation(data);

		std::vector<T> result(out_dim);
		const T* output_data = layers.back()->output();
		std::copy(output_data, output_data + out_dim, result.begin());

		return result;
	}

	int predict(const T* data)
	{
		forward_propagation(data);

		const T* output_data = layers.back()->output();

		int idx = -1;
		T max_output = -std::numeric_limits<T>::max();
		for(int i = 0; i < out_dim; ++i)
		{
			if(output_data[i] > max_output)
			{
				max_output = output_data[i];
				idx = i;
			}
		}

		return idx;
	}

	void update_weights(int batch_size)
	{
		for(auto& clayer : layers)
			clayer->update_weights(batch_size);
	}

	void training_sample(const T* data, const T* gt)
	{
		assert(!layers.empty());

		forward_propagation(data);
		backward_propagation(data, gt);
		//update_weights();
	}

	void training_sample(const T* data, short gt_idx)
	{
		assert(gt_idx >= 0 && gt_idx < out_dim);
		if(gt_idx < 0 || gt_idx >= out_dim)
			throw std::runtime_error("invalid ground truth value");
		std::vector<T> gt(out_dim, 0.0);
		gt[gt_idx] = 1.0;

		training_sample(data, gt.data());
	}

	void end_batch(int batch_size)
	{
		update_weights(batch_size);
	}

	void training(const std::vector<std::vector<T>>& data, const std::vector<short>& gt, std::size_t batch_size)
	{
		for(auto& clayer : layers)
			clayer->set_phase(layer_base<T>::phase::Training);

		assert(data.size() == gt.size());

		std::size_t batch_count = std::ceil((float)data.size() / batch_size);

		for(std::size_t i = 0; i < batch_count; ++i)
		{
			std::size_t offset = i*batch_size;
			std::size_t bound = std::min(offset+batch_size, data.size());
			//std::cout << "length: " << bound - offset << std::endl;
			#pragma omp parallel for
			for(std::size_t j = offset; j < bound; ++j)
			{
				//std::cout << "offset: " << offset << ", j: " << j << std::endl;
				training_sample(data[j].data(), std::abs(gt[j]));
			}
			end_batch(batch_size);
			//std::cout << "i: " << i << std::endl;
		}

		for(auto& clayer : layers)
			clayer->set_phase(layer_base<T>::phase::Testing);
	}

	void training(const std::vector<std::vector<T>>& data, const std::vector<short>& gt, std::size_t batch_size, std::size_t epochs, std::size_t training_error_calculation, bool reset_weights = true)
	{
		if(reset_weights)
			this->reset_weights();
		for(std::size_t i = 0; i < epochs; ++i)
		{
			std::cout << "epoch: " << i << std::endl;
			training(data, gt, batch_size);
			if(training_error_calculation != 0)
			{
				if(i % training_error_calculation == 0)
				{
					float res = test(data, gt);
					if(i > 7 && res < 0.10)
					{
						this->reset_weights();
						i = 0;
					}
				}
			}
		}
	}

	void forward_propagation(const T* bottom_data)
	{
	//	std::cout << "forward" << std::endl;		//execute layers
		const T* current_out = bottom_data;
		for(auto& clayer : layers)
		{
			clayer->forward_propagation(current_out);
			current_out = clayer->output();
		}
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient)
	{
		const T* cgradient = top_gradient;
		const T* cdata = bottom_data;
		for(int i = layers.size() - 1; i >= 0; --i)
		{
			if(i == 0)
				cdata = bottom_data;
			else
				cdata = layers[i-1]->output();

			layers[i]->backward_propagation(cdata, cgradient);
			cgradient = layers[i]->gradient();
		}
	}

	void save_weights(std::ostream& stream) const
	{
		stream.precision(17);
		for(const auto& clayer : layers)
			clayer->save_weights(stream);
	}

	void load_weights(std::istream& stream)
	{
		for(const auto& clayer : layers)
			clayer->load_weights(stream);
	}

	void reset_weights()
	{
		for(const auto& clayer : layers)
			clayer->init_weights();
	}

	std::vector<std::shared_ptr<layer_base<T>>> layers;
	int in_dim, out_dim;
};

template<typename T>
std::ostream& operator<<(std::ostream& stream, const neural_network<T>& net)
{
	net.save_weights(stream);
	return stream;
}

template<typename T>
std::istream& operator>>(std::istream& stream, neural_network<T>& net)
{
	net.load_weights(stream);
	return stream;
}

#endif // NEURAL_NETWORK_NETWORK_H
