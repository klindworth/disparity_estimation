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

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <random>
#include <vector>
#include <neural_network/blas_wrapper.h>
#include <stdexcept>
#include <omp.h>
#include <iterator>
#include <algorithm>
#include <cassert>

namespace neural_network
{

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
	layer_thread_data(int in_dim, int out_dim, int weights, int bias) {
		this->output_data.resize(out_dim);
		this->gradient_data.resize(in_dim);
		this->dW.resize(weights,0);
		this->dB.resize(bias,0);
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

	layer_base(bool propagate_down, int in_dim, int out_dim, int weights, int bias_dim) : cthread_data(omp_get_max_threads(), layer_thread_data<T>(in_dim, out_dim, weights, bias_dim)) {
		this->in_dim = in_dim;
		this->out_dim = out_dim;
		this->propagate_down = propagate_down;
		this->weights.resize(weights);
		this->weights_transposed.resize(weights);
		this->bias.resize(bias_dim);
		this->dW.resize(weights,0);
		this->dB.resize(bias_dim,0);

		this->Eg_w.resize(weights,0);
		this->Ex_w.resize(weights,0);
		this->Eg_b.resize(bias_dim,0);
		this->Ex_b.resize(bias_dim,0);

		//init_weights();
		this->current_phase = phase::Testing;
		regularize = false;
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
		//regularize_weights();
		transpose_weights();
	}

	void init_weights()
	{
		if(!weights.empty())
		{
			const T weight_base = 0.5 / std::sqrt(in_connectivity());
			uniform_init(weights.begin(), weights.end(), weight_base);
			transpose_weights();
		}
		uniform_init(bias.begin(), bias.end(), 0.5);
		/*const T weight_base = 0.5 / std::sqrt(in_dim);
		uniform_init(weights.begin(), weights.end(), weight_base);
		transpose_weights();
		uniform_init(bias.begin(), bias.end(), weight_base);*/
	}

	void save_weights(std::ostream& stream) const
	{
		//std::cout << "weights: " << this->weights.size() << ", bias: " << this->bias.size() << std::endl;
		stream << this->name() << " " << this->weights.size() << " " << this->bias.size() << " ";
		std::copy(this->weights.begin(), this->weights.end(), std::ostream_iterator<T>(stream, " "));
		std::copy(this->bias.begin(), this->bias.end(), std::ostream_iterator<T>(stream, " "));
		/*for(T cweight : this->weights)
			stream << cweight << " ";
		for(T cbias : this->bias)
			stream << cbias << " ";*/
	}

	void load_weights(std::istream& stream)
	{
		std::size_t weight_count, bias_count;
		std::string layername;
		stream >> layername;
		stream >> weight_count;
		stream >> bias_count;

		if( (weight_count != this->weights.size()) || (bias_count != this->bias.size()) || (layername != this->name()))
			throw std::runtime_error("weightsfile doesn't fit");

		for(T& cweight : this->weights)
			stream >> cweight;
		for(T& cbias : this->bias)
			stream >> cbias;
	}

	virtual void forward_propagation(const T* bottom_data) = 0;
	virtual void backward_propagation(const T* bottom_data, const T* top_gradient) = 0;

	virtual bool trainable() const
	{
		return true;
	}

	virtual std::string name() const
	{
		return "layer_base";
	}

	virtual void regularize_weights() {}
	virtual int in_connectivity() { return in_dim; }

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
			#pragma omp parallel for
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

	void transpose_weights()
	{
		if((int)this->weights.size() != (this->in_dim * this->out_dim) )
			return;
		T* dst = this->weights_transposed.data();
		for(int i = 0; i < this->in_dim; ++i)
		{
			const T* weight_col = &(this->weights[i]);
			for(int j = 0; j < out_dim; ++j)
			{
				*dst++ = *weight_col;
				weight_col += this->in_dim;
			}
		}
	}

	void abs_renorm(T* data, int n, T desired, int stride = 1)
	{
		T sum = 0;
		for(int i = 0; i < n; ++i)
			sum += std::abs(data[i*stride]);

		sum /= n;

		if(sum > desired)
			blas::scale(desired/sum, data, n, stride);
	}

	void max_renorm(T* data, int n, T allowed, int stride = 1)
	{
		T current = 0;
		for(int i = 0; i < n; ++i)
			current = std::max(std::abs(data[i*stride]), current);

		if(allowed > current)
			blas::scale(allowed/current, data, n, stride);
	}

	phase current_phase;
	int in_dim;
	int out_dim;
	std::vector<T> bias;
	std::vector<T> weights;
	std::vector<T> weights_transposed;
	std::vector<T> Eg_w, Ex_w, Eg_b, Ex_b;
	std::vector<T> dW, dB;

	std::vector<layer_thread_data<T>> cthread_data;

	bool propagate_down;
	bool regularize;
};

template<typename T>
class fully_connected_layer : public layer_base<T>
{
public:
	fully_connected_layer(bool propagate_down, int in_dim, int out_dim) : layer_base<T>(propagate_down, in_dim, out_dim, in_dim*out_dim, out_dim)
	{}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		blas::gemv(cdata.output_data.data(), this->weights.data(), false, this->out_dim, this->in_dim, bottom_data);
		for(int i = 0; i < this->out_dim; ++i)
		{
			cdata.output_data[i] += this->bias[i];
		}
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		//propagate down
		if(this->propagate_down)
		{
			//blas_gemv(cdata.gradient_data.data(), this->weights.data(), true, this->out_dim, this->in_dim, top_gradient);
			blas::gemv(cdata.gradient_data.data(), this->weights_transposed.data(), false, this->in_dim, this->out_dim, top_gradient);
		}

		blas::ger(cdata.dW.data(), top_gradient, this->out_dim, bottom_data, this->in_dim);

		for(int i = 0; i < this->out_dim; ++i)
			cdata.dB[i] += top_gradient[i];
	}

	std::string name() const override
	{
		return "fully_connected_layer";
	}

	void regularize_weights() override
	{
		for(int i = 0; i < this->out_dim; ++i)
			this->abs_renorm(this->weights.data() + i*this->in_dim, this->in_dim, 1.0/this->in_dim);
	}
};



template<typename T>
class relu_layer : public layer_base<T>
{
public:
	relu_layer(bool propagate_down, int dim) : layer_base<T>(propagate_down, dim, dim, 0, 0) {}

	void forward_propagation(const T *bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		for(int i = 0; i < this->in_dim; ++i)
			cdata.output_data[i] = std::max(bottom_data[i], static_cast<T>(0));
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		if(this->propagate_down)
		{
			layer_thread_data<T>& cdata = this->thread_data();

			for(int i = 0; i < this->in_dim; ++i)
				cdata.gradient_data[i] = bottom_data[i] > 0 ? top_gradient[i] : 0;
		}
	}

	bool trainable() const override
	{
		return false;
	}

	std::string name() const override
	{
		return "relu_layer";
	}
};

template<typename T>
class dropout_layer : public layer_base<T>
{
public:
	typedef layer_base<T> Base;
	dropout_layer(bool propagate_down, int dim) : layer_base<T>(propagate_down, dim, dim, 0, 0), dropout_rate(0.25), mask(this->thread_max(), std::vector<unsigned char>(dim, 1)), rng(this->thread_max())
	{}

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
		if(this->propagate_down)
		{
			layer_thread_data<T>& cdata = this->thread_data();
			const std::vector<unsigned char>& cmask = mask[this->thread_num()];

			for(int i = 0; i < this->in_dim; ++i)
				cdata.gradient_data[i] = top_gradient[i] * cmask[i];
		}
	}

	bool trainable() const override
	{
		return false;
	}

	std::string name() const override
	{
		return "dropout_layer";
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
	softmax_output_layer(bool propagate_down, int in_dim) : layer_base<T>(propagate_down, in_dim, in_dim, 0, 0), temp(this->thread_max(), std::vector<T>(in_dim))
	{}

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

		//T error_sum = 0;
		//std::cout << "backprop softmax" << std::endl;
		for(int i = 0; i < this->out_dim; ++i)
		{
			//std::cout << cdata.output_data[i] << " vs " << gt[i] << std::endl;
			cdata.gradient_data[i] = cdata.output_data[i] - gt[i];
			//error_sum += std::abs(cdata.gradient_data[i]);
		}

		//std::cout << "error_sum: " << error_sum << std::endl;

		//std::copy(this->gradient_data.begin(), this->gradient_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	bool trainable() const override
	{
		return false;
	}

	std::string name() const override
	{
		return "softmax_output_layer";
	}

private:
	std::vector<std::vector<T>> temp;
};

template<typename T>
class tanh_output_layer : public layer_base<T>
{
public:
	tanh_output_layer(bool propagate_down, int in_dim) : layer_base<T>(propagate_down, in_dim, in_dim, 0, 0)
	{}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		for(int i = 0; i < this->out_dim; ++i)
		{
			//const T exp2 = std::exp(2*std::min(static_cast<T>(300),bottom_data[i]));
			//cdata.output_data[i] = (exp2 - 1)/(exp2 + 1);
			cdata.output_data[i] = std::tanh(bottom_data[i]);
		}
	}

	void backward_propagation(const T*, const T* gt) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		//T error_sum = 0;
		//std::cout << "backprop tanh" << std::endl;
		for(int i = 0; i < this->out_dim; ++i)
		{
			//std::cout << cdata.output_data[i] << " vs " << gt[i] << std::endl;
			cdata.gradient_data[i] = (cdata.output_data[i] - gt[i]) * (1- cdata.output_data[i]*cdata.output_data[i]);
			//std::cout << i << ": " << (cdata.output_data[i] - gt[i]) * (1- cdata.output_data[i]*cdata.output_data[i]) << std::endl;
			//error_sum += std::abs(cdata.gradient_data[i]);
		}

		//std::cout << "error_sum: " << error_sum << std::endl;

		//std::copy(this->gradient_data.begin(), this->gradient_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	bool trainable() const override
	{
		return false;
	}

	std::string name() const override
	{
		return "tanh_output_layer";
	}
};

template<typename T>
class transpose_vector_connected_layer : public layer_base<T>
{
public:
	transpose_vector_connected_layer(bool propagate_down, int in_dim, int out_dim, int vectorsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/vectorsize*out_dim + passthrough, vectorsize*out_dim, out_dim), channels_out(out_dim), vectorsize(vectorsize), passthrough(passthrough)
	{}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		blas::gemm(cdata.output_data.data(), this->weights.data(), false, channels_out, vectorsize, bottom_data, true, channels_in, vectorsize);
		T* coutdata = cdata.output_data.data();// + channels_out*channels_in;
		for(int j = 0; j < channels_out; ++j)
		{
			T cbias = this->bias[j];
			for(int i = 0; i < channels_in; ++i)
			{
				*coutdata++ += cbias;
			}
		}

		const T* in_data = &(bottom_data[channels_in*vectorsize]);
		std::copy(in_data, in_data + passthrough, coutdata);
		//std::copy(this->output_data.begin(), this->output_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		if(this->propagate_down)
			blas::gemm(cdata.gradient_data.data(), top_gradient, true, channels_out, channels_in, this->weights.data(), false, channels_out, vectorsize);
		blas::gemm(cdata.dW.data(), top_gradient, false, channels_out, channels_in, bottom_data, false, channels_in, vectorsize, 1.0, 1.0);

		const T* cgradient = top_gradient;
		for(int i = 0; i < this->channels_out; ++i)
		{
			T sum = 0;
			for(int j = 0; j < channels_in; ++j)
				sum += *cgradient++;
			cdata.dB[i] += sum;
		}
	}

	std::string name() const override
	{
		return "transpose_vector_connected_layer";
	}

	int in_connectivity() override
	{
		return vectorsize;
	}

	void regularize_weights() override
	{
		for(int i = 0; i < channels_out; ++i)
			this->abs_renorm(this->weights.data() + i*vectorsize, vectorsize, 1.0/vectorsize);
	}

protected:
	int channels_out, vectorsize, passthrough;
};

/**
 * Splts the input values into vectors oof size vectorsize. Each vector will have out_dim neurons. The weights of the neurons will be shared between the different vectors
 */
template<typename T>
class vector_connected_layer : public layer_base<T>
{
public:
	vector_connected_layer(bool propagate_down, int in_dim, int out_dim, int vectorsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/vectorsize*out_dim + passthrough, vectorsize*out_dim, out_dim), channels_out(out_dim), vectorsize(vectorsize), passthrough(passthrough)
	{}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		//blas_gemm(cdata.output_data.data(), this->weights.data(), false, channels_out, vectorsize, bottom_data, true, channels_in, vectorsize);
		blas::gemm(cdata.output_data.data(), bottom_data, false, channels_in, vectorsize, this->weights.data(), false, vectorsize, channels_out);
		T* coutdata = cdata.output_data.data();// + channels_out*channels_in;
		for(int i = 0; i < channels_in; ++i)
		{
			for(int j = 0; j < channels_out; ++j)
				*coutdata++ += this->bias[j];
		}

		const T* in_data = &(bottom_data[channels_in*vectorsize]);
		std::copy(in_data, in_data + passthrough, coutdata);
		//std::copy(this->output_data.begin(), this->output_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		if(this->propagate_down)
			//blas_gemm(cdata.gradient_data.data(), top_gradient, true, channels_out, channels_in, this->weights.data(), false, channels_out, vectorsize);
			blas::gemm(cdata.gradient_data.data(), top_gradient, false, channels_in, channels_out, this->weights.data(), true, vectorsize, channels_out);
		//blas_gemm(cdata.dW.data(), top_gradient, false, channels_out, channels_in, bottom_data, false, channels_in, vectorsize, 1.0, 1.0);
		blas::gemm(cdata.dW.data(), bottom_data, true, channels_in, vectorsize, top_gradient, false, channels_in, channels_out, 1.0, 1.0);

		const T* cgradient = top_gradient;
		for(int j = 0; j < channels_in; ++j)
		{
			for(int i = 0; i < this->channels_out; ++i)
				cdata.dB[i] += *cgradient++;
		}
	}

	std::string name() const override
	{
		return "vector_connected_layer";
	}

	void regularize_weights() override
	{
		for(int i = 0; i < channels_out; ++i)
			this->abs_renorm(this->weights.data() + i, vectorsize, 1.0/vectorsize, channels_out);
	}

	int in_connectivity() override
	{
		return vectorsize;
	}

protected:
	int channels_out, vectorsize, passthrough;
};

/**
 * Splits the input values into rows. You can specifiy the number of output values per row. Each of those neurons only connects to the input values of a row. The total output dimension is: neurons per row * rows
 * E.g input: 100, rowsize: 20, neurons per row: 10, means that there are five rows, each of them has ten neurons
 */
template<typename T>
class row_connected_layer : public layer_base<T>
{
public:
	row_connected_layer(bool propagate_down, int in_dim, int neurons_per_row, int rowsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/rowsize*neurons_per_row + passthrough, (in_dim - passthrough)*neurons_per_row, (in_dim - passthrough)/rowsize*neurons_per_row), rowsize(rowsize), passthrough(passthrough), per_row_output(neurons_per_row)
	{}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		const int row_count = (this->in_dim - passthrough)/rowsize;
		const int regular_output = per_row_output * row_count;

		for(int i = 0; i < row_count; ++i)
		{
			int offset = i*rowsize;
			blas::gemv(cdata.output_data.data() + i*per_row_output, this->weights.data() + offset*per_row_output, false, per_row_output, rowsize, bottom_data+offset);
		}


		for(int i = 0; i < regular_output; ++i)
		{
			cdata.output_data[i] += this->bias[i];
		}

		const T* in_data = &(bottom_data[regular_output]);
		std::copy(in_data, in_data + passthrough, cdata.output_data.data() + regular_output);
		//std::copy(this->output_data.begin(), this->output_data.end(), std::ostream_iterator<T>(std::cout, ", "));
		//std::cout << std::endl;
	}

	void backward_propagation(const T* bottom_data, const T* top_gradient) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		const int row_count = (this->in_dim - passthrough)/rowsize;
		const int regular_output = per_row_output * row_count;

		//propagate down
		if(this->propagate_down)
		{
			for(int i = 0; i < row_count; ++i)
			{
				int offset = i * rowsize;
				blas::gemv(cdata.gradient_data.data()+offset, this->weights.data()+offset*per_row_output, true, per_row_output, rowsize, top_gradient + i*per_row_output);
			}
		}

		for(int i = 0; i < row_count; ++i)
		{
			int offset = i * rowsize;
			blas::ger(cdata.dW.data() + offset*per_row_output, top_gradient+i*per_row_output, per_row_output, bottom_data + offset, rowsize);
		}

		for(int i = 0; i < regular_output; ++i)
			cdata.dB[i] += top_gradient[i];
	}

	std::string name() const override
	{
		return "row_connected_layer";
	}

	int in_connectivity() override
	{
		return rowsize;
	}

	void regularize_weights() override
	{
		const int row_count = (this->in_dim - passthrough)/rowsize;
		const int neurons = per_row_output * row_count;
		for(int i = 0; i < neurons; ++i)
			this->abs_renorm(this->weights.data() + i*rowsize, rowsize, 1.0/rowsize);
	}

protected:
	int rowsize, passthrough, per_row_output;
};

template<typename T>
class vector_extension_layer : public layer_base<T>
{
public:
	vector_extension_layer(bool propagate_down, int in_dim, int old_vectorsize, int vec_extension) :
		layer_base<T>(propagate_down, in_dim, (in_dim - vec_extension)/old_vectorsize*(old_vectorsize+vec_extension), 0, 0), old_vectorsize(old_vectorsize), vector_extension(vec_extension)
	{assert(!propagate_down);}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int vector_count = (this->in_dim - vector_extension)/old_vectorsize;
		const T* extension_data = bottom_data + vector_count*old_vectorsize;
		const T* old_data = bottom_data;
		T* new_data = cdata.output_data.data();

		for(int i = 0; i < vector_count; ++i)
		{
			std::copy(old_data, old_data + old_vectorsize, new_data);
			new_data += old_vectorsize;
			std::copy(extension_data, extension_data + vector_extension, new_data);
			new_data += vector_extension;
			old_data += old_vectorsize;
		}
	}

	void backward_propagation(const T*, const T*) override
	{
	}

	bool trainable() const override
	{
		return false;
	}

	std::string name() const override
	{
		return "vector_extension_layer";
	}

protected:
	int old_vectorsize, vector_extension;
};
}

#endif //NEURAL_NETWORK_LAYER_H
