#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

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

template<typename Iterator>
void uniform_init(Iterator begin, Iterator end, typename Iterator::value_type var)
{
	static std::mt19937 gen;
	std::uniform_real_distribution<> dist(-var, var);

	for(Iterator it = begin; it != end; ++it)
		*it = dist(gen);
}

template<typename T>
struct layer_thread_data
{
	layer_thread_data(int in_dim, int out_dim, int weights) {
		this->output_data.resize(out_dim);
		this->gradient_data.resize(in_dim);
		this->dW.resize(weights,0);
		this->dB.resize(out_dim,0);
	}
	std::vector<T> output_data;
	std::vector<T> gradient_data;
	std::vector<T> dW, dB;
};

template<typename T>
class layer_base
{
public:
	enum phase {Testing, Training};

	layer_base(int in_dim, int out_dim, int weights) : cthread_data(omp_get_max_threads(), layer_thread_data<T>(in_dim, out_dim, weights)){
		this->weights.resize(weights);
		this->in_dim = in_dim;
		this->out_dim = out_dim;
		this->bias.resize(out_dim);
		this->dW.resize(weights,0);
		this->dB.resize(out_dim,0);

		this->Eg_w.resize(weights,0);
		this->Ex_w.resize(weights,0);
		this->Eg_b.resize(out_dim,0);
		this->Ex_b.resize(out_dim,0);

		init_weights();

		this->in_dim = in_dim;
		this->out_dim = out_dim;
		this->current_phase = phase::Testing;
	}

	void set_phase(phase cphase)
	{
		this->current_phase = cphase;
	}

	layer_thread_data<T>& thread_data()
	{
		return cthread_data[thread_num()];
	}

	const layer_thread_data<T>& thread_data() const
	{
		return cthread_data[thread_num()];
	}

	inline int thread_num() const
	{
		return omp_get_thread_num();
	}

	inline int thread_max() const
	{
		return omp_get_max_threads();
	}

	const T* output() const
	{
		return thread_data().output_data.data();
	}

	const T* gradient() const
	{
		return thread_data().gradient_data.data();
	}

	std::vector<T>& get_weights()
	{
		return weights;
	}

	std::vector<T>& get_bias()
	{
		return bias;
	}

	std::vector<T>& get_weights_diff()
	{
		return thread_data().dW;
	}

	std::vector<T>& get_bias_diff()
	{
		return thread_data().dB;
	}

	int output_dimension() const
	{
		return out_dim;
	}

	int input_dimension() const
	{
		return in_dim;
	}

	void update_weights(int count = 1)
	{
		for(layer_thread_data<T>& cdata : cthread_data)
		{
			for(std::size_t i = 0; i < cdata.dW.size(); ++i)
				dW[i] += cdata.dW[i];
			for(std::size_t i = 0; i < cdata.dB.size(); ++i)
				dB[i] += cdata.dB[i];

			std::fill(cdata.dW.begin(), cdata.dW.end(), 0.0f);
			std::fill(cdata.dB.begin(), cdata.dB.end(), 0.0f);
		}

		update_weights_internal(count);
	}

	void init_weights()
	{
		const T weight_base = 0.5 / std::sqrt(in_dim);
		uniform_init(weights.begin(), weights.end(), weight_base);
		uniform_init(bias.begin(), bias.end(), weight_base);
	}

	virtual void forward_propagation(const T* bottom_data) = 0;
	virtual void backward_propagation(const T* bottom_data, const T* top_gradient) = 0;


protected:

	void update_weights_internal(int count = 1)
	{
		T rho = 0.95;
		T epsilon =1e-6;

		auto rms = [=](T val) {
			return std::sqrt(val+epsilon);
		};

		auto adadelta = [=](std::vector<T>& Eg, std::vector<T>& Ex, std::vector<T>& W, std::vector<T>& dW, T factor)
		{
			assert((Eg.size() == Ex.size()) && (Ex.size() == W.size()));
			int iterations = Eg.size();
			for(int i = 0; i < iterations; ++i)
			{
				T cdW = dW[i] * factor;
				dW[i] = 0;

				Eg[i] = rho * Eg[i] + (1-rho) * cdW * cdW;
				T dx = -rms(Ex[i]) / rms(Eg[i]) * cdW;
				Ex[i] = rho * Ex[i]+(1-rho)*dx*dx;
				W[i] += dx;
			}
		};

		adadelta(Eg_w, Ex_w, weights, dW, 1.0/count);
		adadelta(Eg_b, Ex_b, bias, dB, 1.0/count);
	}

	phase current_phase;
	int in_dim;
	int out_dim;
	std::vector<T> bias;
	std::vector<T> weights;
	std::vector<T> Eg_w, Ex_w, Eg_b, Ex_b;
	std::vector<T> dW, dB;

	std::vector<layer_thread_data<T>> cthread_data;
};

template<typename T>
class fully_connected_layer : public layer_base<T>
{
public:
	fully_connected_layer(int in_dim, int out_dim) : layer_base<T>(in_dim, out_dim, in_dim*out_dim)
	{std::cout << "fully connected layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		//int weight_idx = 0;
		const T* cweight = this->weights.data();
		for(int i = 0; i < this->out_dim; ++i)
		{
			T sum = 0;
			for(int j = 0; j < this->in_dim; ++j)
				sum += bottom_data[j]* *cweight++;//this->weights[weight_idx++];
			cdata.output_data[i] = sum + this->bias[i];
		}
		//std::copy(this->output_data.begin(), this->output_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();
		/*std::cout << "top: ";
		std::copy(top_gradient, top_gradient + this->out_dim, std::ostream_iterator<T>(std::cout, ", "));
		std::cout << std::endl;
		std::cout << "bottom: ";
		std::copy(bottom_data, bottom_data + this->in_dim, std::ostream_iterator<T>(std::cout, ", "));
		std::cout << std::endl;*/
		for(int i = 0; i < this->in_dim; ++i)
		{
			const T* cweights = &(this->weights[i]);
			T sum = 0;
			for(int j = 0; j < this->out_dim; ++j)
			{
				sum += *cweights * top_gradient[j];
				cweights += this->in_dim;
			}
			cdata.gradient_data[i] = sum;
		}

		T* cdw = cdata.dW.data();
		for(int j = 0; j < this->out_dim; ++j)
		{
			T cgradient = top_gradient[j];
			for(int i = 0; i < this->in_dim; ++i)
				*cdw++ += cgradient * bottom_data[i];
		}

		for(int i = 0; i < this->out_dim; ++i)
			cdata.dB[i] += top_gradient[i];

		//std::copy(this->gradient_data.begin(), this->gradient_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}
};

template<typename T>
class relu_layer : public layer_base<T>
{
public:
	relu_layer(int dim) : layer_base<T>(dim, dim, 0) {std::cout << "relu-layer" << std::endl;}

	void forward_propagation(const T *bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();
		/*std::cout << "bottom: ";
		std::copy(bottom_data, bottom_data + this->in_dim, std::ostream_iterator<T>(std::cout, ", "));
		std::cout << std::endl;*/

		for(int i = 0; i < this->in_dim; ++i)
			cdata.output_data[i] = std::max(bottom_data[i], static_cast<T>(0));
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		for(int i = 0; i < this->in_dim; ++i)
			cdata.gradient_data[i] = bottom_data[i] > 0 ? top_gradient[i] : 0;
	}
};

template<typename T>
class dropout_layer : public layer_base<T>
{
public:
	typedef layer_base<T> Base;
	dropout_layer(int dim) : layer_base<T>(dim, dim, 0), dropout_rate(0.25), mask(this->thread_max(), std::vector<unsigned char>(dim, 1)), rng(this->thread_max())
	{std::cout << "dropout" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();
		std::vector<unsigned char>& cmask = mask[this->thread_num()];
		std::mt19937& crng = rng[this->thread_num()];

		if(this->current_phase == Base::phase::Training)
		{
			//init mask
			std::bernoulli_distribution dist(1.0-dropout_rate);
			for(auto& centry : cmask)
				centry = dist(crng);

			/*std::copy(bottom_data, bottom_data + this->in_dim, std::ostream_iterator<T>(std::cout, ", "));
			std::cout << std::endl;

			std::copy(mask.begin(), mask.end(), std::ostream_iterator<int>(std::cout, ", "));
			std::cout << std::endl;*/

			//fprop
			for(int i = 0; i < this->in_dim; ++i)
				cdata.output_data[i] = bottom_data[i] * cmask[i];
			//std::copy(cdata.output_data.begin(), cdata.output_data.end(), std::ostream_iterator<T>(std::cout, ", "));
			//std::cout << std::endl;
		}
		else
		{
			double factor = 1.0-dropout_rate;

			for(int i = 0; i < this->in_dim; ++i)
				cdata.output_data[i] = bottom_data[i] * factor;
		}
	}

	void backward_propagation(const T*, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();
		const std::vector<unsigned char>& cmask = mask[this->thread_num()];

		for(int i = 0; i < this->in_dim; ++i)
			cdata.gradient_data[i] = top_gradient[i] * cmask[i];
	}

protected:
	T dropout_rate;
	std::vector<std::vector<unsigned char>> mask;
	std::vector<std::mt19937> rng;
};

template<typename T>
class softmax_output_layer : public layer_base<T>
{
public:
	softmax_output_layer(int in_dim) : layer_base<T>(in_dim, in_dim, 0), temp(this->thread_max(), std::vector<T>(in_dim))
	{std::cout << "softmax output layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();
		std::vector<T>& ctemp = temp[this->thread_num()];

		T current_max = *(std::max_element(bottom_data, bottom_data + this->out_dim));

		for(int i = 0; i < this->out_dim; ++i)
			ctemp[i] = std::exp(std::min(bottom_data[i] - current_max, static_cast<T>(340)));

		//add final activation (softmax)
		T sum = 0.0;
		for(int i = 0; i < this->out_dim; ++i)
			sum += ctemp[i];

		for(int i = 0; i < this->out_dim; ++i)
			cdata.output_data[i] = ctemp[i]/sum;
	}

	void backward_propagation(const T*, const T* gt) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		T error_sum = 0;
		//std::cout << "backprop softmax" << std::endl;
		for(int i = 0; i < this->out_dim; ++i)
		{
			//std::cout << cdata.output_data[i] << " vs " << gt[i] << std::endl;
			cdata.gradient_data[i] = cdata.output_data[i] - gt[i];
			error_sum += std::abs(cdata.gradient_data[i]);
		}

		//std::cout << "error_sum: " << error_sum << std::endl;

		//std::copy(this->gradient_data.begin(), this->gradient_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	std::vector<std::vector<T>> temp;
};

template<typename T>
class tanh_entropy_output_layer : public layer_base<T>
{
public:
	tanh_entropy_output_layer(int in_dim) : layer_base<T>(in_dim, in_dim, 0), temp(this->thread_max(), std::vector<T>(in_dim))
	{std::cout << "softmax output layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		for(int i = 0; i < this->out_dim; ++i)
			cdata.output_data[i] = std::tanh(bottom_data[i]);
	}

	void backward_propagation(const T* bottom_data, const T* gt) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		T error_sum = 0;
		//std::cout << "backprop softmax" << std::endl;
		for(int i = 0; i < this->out_dim; ++i)
		{
			//std::cout << "[" << i << "]: " << bottom_data[i] - gt[i] << std::endl;
			std::cout << bottom_data[i] << " vs " << gt[i] << std::endl;
			cdata.gradient_data[i] = bottom_data[i] - gt[i];
			error_sum += std::abs(cdata.gradient_data[i]);
		}

		std::cout << "error_sum: " << error_sum << std::endl;

		//std::copy(this->gradient_data.begin(), this->gradient_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	std::vector<std::vector<T>> temp;
};

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
		layers.push_back(std::make_shared<layer_type<T>>(out_dim, args...));
		out_dim = layers.back()->output_dimension();
	}

	void test(const std::vector<std::vector<T>>& data, const std::vector<short>& gt)
	{
		assert(data.size() == gt.size());

		int correct = 0;

		for(std::size_t i = 0; i < data.size(); ++i)
		{
			int result = this->predict(data[i].data());
			int expected = gt[i];

			if(result == expected)
				++correct;
		}

		std::cout << "result: " << (float)correct/data.size() << std::endl;
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
			//#pragma omp parallel for
			for(std::size_t j = offset; j < bound; ++j)
			{
				//std::cout << "offset: " << offset << ", j: " << j << std::endl;
				training_sample(data[j].data(), gt[j]);
			}
			end_batch(batch_size);
			//std::cout << "i: " << i << std::endl;
		}

		for(auto& clayer : layers)
			clayer->set_phase(layer_base<T>::phase::Testing);
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

	std::vector<std::shared_ptr<layer_base<T>>> layers;
	int in_dim, out_dim;


};

#endif // SIMPLE_NN_H
