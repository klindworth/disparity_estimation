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
#include <cblas.h>

inline void blas_gemm(double* Y, const double* A, bool transposeA, int rowsA, int colsA, const double* B, bool transposeB, int rowsB, int colsB, double alpha = 1.0, double beta = 0.0)
{
	cblas_dgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans, transposeA ? colsA : rowsA, transposeB ? rowsB : colsB, transposeB ? colsB : rowsB, alpha, A, colsA, B, colsB, beta, Y, transposeB ? rowsB : colsB);
}

inline void blas_gemm(float* Y, const float* A, bool transposeA, int rowsA, int colsA, const float* B, bool transposeB, int rowsB, int colsB, float alpha = 1.0f, float beta = 0.0f)
{
	cblas_sgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans, transposeA ? colsA : rowsA, transposeB ? rowsB : colsB, transposeB ? colsB : rowsB, alpha, A, colsA, B, colsB, beta, Y, transposeB ? rowsB : colsB);
}

inline void blas_gemv(double* Y, const double* A, bool transposeA, int rowsA, int colsA, const double* x, double alpha = 1.0, double beta = 0.0)
{
	cblas_dgemv(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, rowsA, colsA, alpha, A, colsA, x, 1, beta, Y, 1);
}

inline void blas_gemv(float* Y, const float* A, bool transposeA, int rowsA, int colsA, const float* x, float alpha = 1.0, float beta = 0.0)
{
	cblas_sgemv(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, rowsA, colsA, alpha, A, colsA, x, 1, beta, Y, 1);
}

inline void blas_ger(double* Y, const double * A, int rowsA, const double* X, int rowsX, double alpha = 1.0)
{
	cblas_dger(CblasRowMajor, rowsA, rowsX, alpha, A, 1, X, 1, Y, rowsX);
}

inline void blas_ger(float* Y, const float * A, int rowsA, const float* X, int rowsX, float alpha = 1.0)
{
	cblas_sger(CblasRowMajor, rowsA, rowsX, alpha, A, 1, X, 1, Y, rowsX);
}

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
		regularize_weights();
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
		std::cout << "weights: " << this->weights.size() << ", bias: " << this->bias.size() << std::endl;
		std::copy(this->weights.begin(), this->weights.end(), std::ostream_iterator<T>(stream, " "));
		std::copy(this->bias.begin(), this->bias.end(), std::ostream_iterator<T>(stream, " "));
		/*for(T cweight : this->weights)
			stream << cweight << " ";
		for(T cbias : this->bias)
			stream << cbias << " ";*/
	}

	void load_weights(std::istream& stream)
	{
		for(T& cweight : this->weights)
			stream >> cweight;
		for(T& cbias : this->bias)
			stream >> cbias;
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

	template<typename Iterator>
	void mul_range(Iterator it, Iterator end, T factor)
	{
		for(; it != end; ++it)
			*it *= factor;
	}

	template<typename Iterator>
	void abs_renorm(Iterator start, Iterator end, T desired)
	{
		T sum = 0;
		for(auto it = start; it != end; ++it)
			sum += std::abs(*it);

		sum /= std::distance(start, end);

		if(sum > desired)
			mul_range(start, end, desired/sum);
	}

	template<typename Iterator>
	void max_renorm(Iterator start, Iterator end, T allowed)
	{
		T current = 0;
		for(auto it = start; it != end; ++it)
			current = std::max(std::abs(+it), current);

		if(allowed > current)
			mul_range(start, end, allowed/current);
	}

	virtual void regularize_weights() {}
	virtual int in_connectivity() { return in_dim; }

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
	{std::cout << "fully connected layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		blas_gemv(cdata.output_data.data(), this->weights.data(), false, this->out_dim, this->in_dim, bottom_data);
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
			blas_gemv(cdata.gradient_data.data(), this->weights_transposed.data(), false, this->in_dim, this->out_dim, top_gradient);
		}

		blas_ger(cdata.dW.data(), top_gradient, this->out_dim, bottom_data, this->in_dim);

		for(int i = 0; i < this->out_dim; ++i)
			cdata.dB[i] += top_gradient[i];
	}
};



template<typename T>
class relu_layer : public layer_base<T>
{
public:
	relu_layer(bool propagate_down, int dim) : layer_base<T>(propagate_down, dim, dim, 0, 0) {std::cout << "relu-layer" << std::endl;}

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
};

template<typename T>
class dropout_layer : public layer_base<T>
{
public:
	typedef layer_base<T> Base;
	dropout_layer(bool propagate_down, int dim) : layer_base<T>(propagate_down, dim, dim, 0, 0), dropout_rate(0.25), mask(this->thread_max(), std::vector<unsigned char>(dim, 1)), rng(this->thread_max())
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
		if(this->propagate_down)
		{
			layer_thread_data<T>& cdata = this->thread_data();
			const std::vector<unsigned char>& cmask = mask[this->thread_num()];

			for(int i = 0; i < this->in_dim; ++i)
				cdata.gradient_data[i] = top_gradient[i] * cmask[i];
		}
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
		bool propagate = layers.size() > 0 ? true : false;
		layers.push_back(std::make_shared<layer_type<T>>(propagate, out_dim, args...));
		out_dim = layers.back()->output_dimension();

		layers.back()->init_weights();
	}

	void test(const std::vector<std::vector<T>>& data, const std::vector<short>& gt)
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

	void training(const std::vector<std::vector<T>>& data, const std::vector<short>& gt, std::size_t batch_size, std::size_t epochs, std::size_t training_error_calculation)
	{
		this->reset_weights();
		for(std::size_t i = 0; i < epochs; ++i)
		{
			std::cout << "epoch: " << i << std::endl;
			training(data, gt, batch_size);
			if(training_error_calculation != 0)
			{
				if(i % training_error_calculation == 0)
					test(data, gt);
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

template<typename T>
class transpose_vector_connected_layer : public layer_base<T>
{
public:
	transpose_vector_connected_layer(bool propagate_down, int in_dim, int out_dim, int vectorsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/vectorsize*out_dim + passthrough, vectorsize*out_dim, out_dim), channels_out(out_dim), vectorsize(vectorsize), passthrough(passthrough)
	{std::cout << "transpose vector connected layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		blas_gemm(cdata.output_data.data(), this->weights.data(), false, channels_out, vectorsize, bottom_data, true, channels_in, vectorsize);
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
			blas_gemm(cdata.gradient_data.data(), top_gradient, true, channels_out, channels_in, this->weights.data(), false, channels_out, vectorsize);
		blas_gemm(cdata.dW.data(), top_gradient, false, channels_out, channels_in, bottom_data, false, channels_in, vectorsize, 1.0, 1.0);

		const T* cgradient = top_gradient;
		for(int i = 0; i < this->channels_out; ++i)
		{
			T sum = 0;
			for(int j = 0; j < channels_in; ++j)
				sum += *cgradient++;
			cdata.dB[i] += sum;
		}
	}

protected:
	int in_connectivity() override
	{
		return vectorsize;
	}

	void regularize_weights() override
	{
		for(int i = 0; i < channels_out; ++i)
			this->abs_renorm(this->weights.begin() + i*vectorsize, this->weights.begin() + (i+1)*vectorsize, 1.0/vectorsize);
	}

	int channels_out, vectorsize, passthrough;
};

template<typename T>
class vector_connected_layer : public layer_base<T>
{
public:
	vector_connected_layer(bool propagate_down, int in_dim, int out_dim, int vectorsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/vectorsize*out_dim + passthrough, vectorsize*out_dim, out_dim), channels_out(out_dim), vectorsize(vectorsize), passthrough(passthrough)
	{std::cout << "vector connected layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		int channels_in = (this->in_dim - passthrough)/vectorsize;

		//blas_gemm(cdata.output_data.data(), this->weights.data(), false, channels_out, vectorsize, bottom_data, true, channels_in, vectorsize);
		blas_gemm(cdata.output_data.data(), bottom_data, false, channels_in, vectorsize, this->weights.data(), false, vectorsize, channels_out);
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
			blas_gemm(cdata.gradient_data.data(), top_gradient, false, channels_in, channels_out, this->weights.data(), true, vectorsize, channels_out);
		//blas_gemm(cdata.dW.data(), top_gradient, false, channels_out, channels_in, bottom_data, false, channels_in, vectorsize, 1.0, 1.0);
		blas_gemm(cdata.dW.data(), bottom_data, true, channels_in, vectorsize, top_gradient, false, channels_in, channels_out, 1.0, 1.0);

		const T* cgradient = top_gradient;
		for(int j = 0; j < channels_in; ++j)
		{
			for(int i = 0; i < this->channels_out; ++i)
				cdata.dB[i] += *cgradient++;
		}
	}

protected:
	int in_connectivity() override
	{
		return vectorsize;
	}

	void regularize_weights() override
	{
		//for(int i = 0; i < channels_out; ++i)
			//this->abs_renorm(this->weights.begin() + i*vectorsize, this->weights.begin() + (i+1)*vectorsize, 1.0/vectorsize);
	}

	int channels_out, vectorsize, passthrough;
};

template<typename T>
class row_connected_layer : public layer_base<T>
{
public:
	row_connected_layer(bool propagate_down, int in_dim, int out_dim, int vectorsize, int passthrough) :
		layer_base<T>(propagate_down, in_dim, (in_dim - passthrough)/vectorsize*out_dim + passthrough, (in_dim - passthrough)*out_dim, (in_dim - passthrough)/vectorsize*out_dim), vectorsize(vectorsize), passthrough(passthrough), per_row_output(out_dim)
	{std::cout << "row connected layer" << std::endl;}

	void forward_propagation(const T* bottom_data) override
	{
		layer_thread_data<T>& cdata = this->thread_data();

		const int row_count = (this->in_dim - passthrough)/vectorsize;
		const int regular_output = per_row_output * row_count;

		for(int i = 0; i < row_count; ++i)
		{
			int offset = i*vectorsize;
			blas_gemv(cdata.output_data.data() + i*per_row_output, this->weights.data() + offset*per_row_output, false, per_row_output, vectorsize, bottom_data+offset);
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

		const int row_count = (this->in_dim - passthrough)/vectorsize;
		const int regular_output = per_row_output * row_count;

		//propagate down
		if(this->propagate_down)
		{
			for(int i = 0; i < row_count; ++i)
			{
				int offset = i * vectorsize;
				blas_gemv(cdata.gradient_data.data()+offset, this->weights.data()+offset*per_row_output, true, per_row_output, vectorsize, top_gradient + i*per_row_output);
			}
		}

		for(int i = 0; i < row_count; ++i)
		{
			int offset = i * vectorsize;
			blas_ger(cdata.dW.data() + offset*per_row_output, top_gradient+i*per_row_output, per_row_output, bottom_data + offset, vectorsize);
		}

		for(int i = 0; i < regular_output; ++i)
			cdata.dB[i] += top_gradient[i];
	}

protected:
	int in_connectivity() override
	{
		return vectorsize;
	}

	void regularize_weights() override
	{
		//for(int i = 0; i < channels_out; ++i)
			//this->abs_renorm(this->weights.begin() + i*vectorsize, this->weights.begin() + (i+1)*vectorsize, 1.0/vectorsize);
	}

	int vectorsize, passthrough, per_row_output;
};

#endif // SIMPLE_NN_H
