#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <vector>
#include <neural_network/blas_wrapper.h>

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
			current = std::max(std::abs(it), current);

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
			blas::gemv(cdata.output_data.data() + i*per_row_output, this->weights.data() + offset*per_row_output, false, per_row_output, vectorsize, bottom_data+offset);
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
				blas::gemv(cdata.gradient_data.data()+offset, this->weights.data()+offset*per_row_output, true, per_row_output, vectorsize, top_gradient + i*per_row_output);
			}
		}

		for(int i = 0; i < row_count; ++i)
		{
			int offset = i * vectorsize;
			blas::ger(cdata.dW.data() + offset*per_row_output, top_gradient+i*per_row_output, per_row_output, bottom_data + offset, vectorsize);
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

template<typename T>
class vector_extension_layer : public layer_base<T>
{
public:
	vector_extension_layer(bool propagate_down, int in_dim, int old_vectorsize, int vec_extension) :
		layer_base<T>(propagate_down, in_dim, (in_dim - vec_extension)/old_vectorsize*(old_vectorsize+vec_extension), 0, 0), old_vectorsize(old_vectorsize), vector_extension(vec_extension)
	{std::cout << "vector extension layer" << std::endl; assert(!propagate_down);}

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

protected:
	int old_vectorsize, vector_extension;
};

#endif //NEURAL_NETWORK_LAYER_H
