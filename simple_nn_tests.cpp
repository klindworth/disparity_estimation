#include "simple_nn.h"

#include <gtest/gtest.h>

template<typename T>
T get_loss(const std::vector<T>& output, const std::vector<T>& gt)
{
	T error_sum = 0;
	for(std::size_t i = 0; i < output.size(); ++i)
	{
		//std::cout << "loss_calc: " << output[i] << " vs " << gt[i] << std::endl;
		//std::cout << "loss[" << i << "] " << gt[i]*output[i] << std::endl;
		error_sum += -gt[i]*std::log(output[i]);
	}

	return error_sum;
}

template<typename T>
T calc_numeric_gradient(neural_network<T>& net, const std::vector<std::vector<T> >& in, const std::vector<std::vector<T>>& v, T& w, T delta)
{
	auto calc_delta = [&](T temp_w) {
		std::swap(w, temp_w);

		T f = 0.0;
		std::size_t data_size = in.size();
		for (std::size_t i = 0; i < data_size; i++)
			f += get_loss(net.output(in[i].data()), v[i]);

		std::swap(w, temp_w);
		return f;
	};

	T f_p = calc_delta(w + delta);
	T f_m = calc_delta(w - delta);

	T delta_by_numerical = (f_p - f_m) / (2.0 * delta);

	return delta_by_numerical;
}

template<typename T>
bool gradient_check(neural_network<T>& net, const std::vector<std::vector<T>>& in, const std::vector<short>& t, float_t eps, T delta)
{
	std::vector<std::vector<T> > v(t.size(), std::vector<T>(net.out_dim, 0.0));
	for(std::size_t i = 0; i < t.size(); ++i)
		v[i][t[i]] = 1.0f;

	for(std::size_t j = 0; j < net.layers.size() - 1; ++j)
	{
		//std::cout << "check layer " << j << std::endl;
		layer_base<T>& current = *(net.layers[j]);
		auto& w = current.get_weights();
		auto& b = current.get_bias();
		auto& dw = current.get_weights_diff();
		auto& db = current.get_bias_diff();

		// calculate dw/db by bprop
		std::fill(dw.begin(), dw.end(), 0.0);
		std::fill(db.begin(), db.end(), 0.0);

		for (std::size_t i = 0; i < in.size(); i++)
		{
			net.forward_propagation(in[i].data());
			net.backward_propagation(in[i].data(), v[i].data());
		}

		for (std::size_t i = 0; i < w.size(); i++)
		{
			T delta_by_numeric = calc_numeric_gradient(net, in, v, w[i], delta);
			T diff = std::abs(delta_by_numeric - dw[i]);
			bool res = diff < eps;
			if(!res)
			{
				std::cout.precision(17);
				std::cout << "failed: " << delta_by_numeric << " vs " << dw[i] << std::endl;
			}
			assert(res);
		}

		for (std::size_t i = 0; i < b.size(); i++) {
			assert(std::abs(calc_numeric_gradient(net, in, v, b[i], delta) - db[i]) < eps);
		}
	}
	return true;
}

template<typename data_type>
bool test_gradient_check_internal(data_type delta, data_type eps)
{
	neural_network<data_type> net(2,2, {12,12});

	std::vector<data_type> input1 {-0.8, 0.8};
	std::vector<data_type> input2 {0.7, -0.7};
	std::vector<data_type> input3 {-0.2, 0.2};
	std::vector<data_type> input4 {0.2, -0.3};
	std::vector<data_type> input5 {-0.5, 0.6};
	std::vector<data_type> input6 {0.5, -0.6};
	short output1 = 0;
	short output2 = 1;
	short output3 = 0;
	short output4 = 1;
	short output5 = 0;
	short output6 = 1;

	std::vector<std::vector<data_type>> input {input1, input2, input3, input4, input5, input6};
	std::vector<short> output {output1, output2, output3, output4, output5, output6};

	return gradient_check(net, input, output, eps, delta);
}

TEST(SimpleNN, GradientDouble)
{
	ASSERT_TRUE(test_gradient_check_internal<double>(1e-8, 1e-6));
}

TEST(SimpleNN, GradientFloat)
{
	ASSERT_TRUE(test_gradient_check_internal<float>(1e-3, 1e-3));
}

/*int main(int, char **)
{
	typedef double data_type;
	neural_network<data_type> net(2,2, {12,12});

	std::vector<data_type> input1 {-0.8, 0.8};
	std::vector<data_type> input2 {0.7, -0.7};
	std::vector<data_type> input3 {-0.2, 0.2};
	std::vector<data_type> input4 {0.2, -0.3};
	std::vector<data_type> input5 {-0.5, 0.6};
	std::vector<data_type> input6 {0.5, -0.6};
	short output1 = 0;
	short output2 = 1;
	short output3 = 0;
	short output4 = 1;
	short output5 = 0;
	short output6 = 1;

	std::vector<std::vector<data_type>> input {input1, input2, input3, input4, input5, input6};
	std::vector<short> output {output1, output2, output3, output4, output5, output6};

	test_gradient_check();

	for(int i = 0; i < 15; ++i)
	{
		std::cout << "epoch: " << i << std::endl;
		net.training(input, output,2);
		net.test(input, output);
	}

	return 0;
}*/

