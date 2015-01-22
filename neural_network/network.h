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

#ifndef NEURAL_NETWORK_NETWORK_H
#define NEURAL_NETWORK_NETWORK_H

#include <memory>
#include <cassert>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#include <iostream>
#include <iterator>
#include <sstream>

#include <omp.h>
#include <neural_network/blas_wrapper.h>
#include <neural_network/layer.h>

namespace neural_network
{
template<typename T>
class network
{
public:
	network(int in_dim) : in_dim(in_dim), out_dim(in_dim), is_trainable(false)
	{
	}

	network(int in_dim, int out_dim, const std::initializer_list<int>& list) : in_dim(in_dim), out_dim(in_dim), is_trainable(false)
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

	/**
	 * Creates a new layer and passes the first two parameters of the layer's constructor on it's own (propagate down, input dimension)
	 * E.g. if your layer has the following constructor layer_type(bool propagate_down, int input_dimension, int output_dimension) you have to call this function
	 * emplace_layer<layer_type>(output_dimension)
	 */
	template<template<typename Ti> class layer_type, typename... args_type>
	void emplace_layer(args_type... args)
	{
		layers.push_back(std::make_shared<layer_type<T>>(is_trainable, out_dim, args...));
		out_dim = layers.back()->output_dimension();

		layers.back()->init_weights();
		is_trainable |= layers.back()->trainable();
	}

	/**
	 * @brief test Tests the neural network. It measures, how many sample classes will be correctly predicted
	 * @param data Samples
	 * @param gt Ground truth label of the samples. Therefore it must have the same size as the data vector
	 * @return Percentual amount of correctly predicted sample classes
	 */
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

		//std::cout << "result: " << (float)correct/data.size() << ", approx5: " << (float)approx_correct/data.size() << ", approx10: " << (float)approx_correct2/data.size() << std::endl;
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

	/**
	 * @brief predict Runs the prediction for one sample and returns it's class label
	 * @param data Sample
	 * @return Class label
	 */
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

	int predict(const std::vector<T>& data)
	{
		assert((int)data.size() == layers.front()->input_dimension());
		return this->predict(data.data());
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
		std::vector<T> gt(out_dim, 0.0); //for softmax
		//std::vector<T> gt(out_dim, -1.0); //for tanh
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

	void multi_training(const std::vector<std::vector<T>>& data, const std::vector<short>& gt, std::size_t batch_size, std::size_t epochs, std::size_t training_error_calculation)
	{
		float best_res = 0;
		std::stringstream weightstream;

		for(int j = 0; j < 7; ++j)
		{
			float old_res = 0;
			int nothing_learned = 0;
			this->reset_weights();

			for(std::size_t i = 0; i < epochs; ++i)
			{
				training(data, gt, batch_size);
				if(training_error_calculation != 0)
				{
					if(i % training_error_calculation == 0)
					{
						//std::cout << "epoch: " << i << std::endl;
						float res = test(data, gt);
						if(res - old_res < 0.01)
							nothing_learned++;

						if(nothing_learned >= 4)
							break;
						else
							old_res = res;
					}
				}
			}

			float res = test(data, gt);
			if(res > best_res)
			{
				weightstream.str("");
				this->save_weights(weightstream);
				best_res = res;
			}
		}

		this->load_weights(weightstream);
		std::cout << "----------------final---------------------" << std::endl;
		std::cout << "result: " << test(data, gt) << std::endl;
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

	void load_weights(std::istream &stream, int up_to_layer)
	{
		assert(up_to_layer < layers.size() && up_to_layer);
		for(int i = 0; i <= up_to_layer; ++i)
			layers[i]->load_weights(stream);
	}

	void reset_weights()
	{
		for(const auto& clayer : layers)
			clayer->init_weights();
	}

	int output_dimension() const
	{
		return out_dim;
	}

	int input_dimension() const
	{
		return in_dim;
	}

	std::vector<std::shared_ptr<layer_base<T>>> layers;

private:
	int in_dim, out_dim;
	bool is_trainable;
};
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const neural_network::network<T>& net)
{
	net.save_weights(stream);
	return stream;
}

template<typename T>
std::istream& operator>>(std::istream& stream, neural_network::network<T>& net)
{
	net.load_weights(stream);
	return stream;
}

#endif // NEURAL_NETWORK_NETWORK_H
