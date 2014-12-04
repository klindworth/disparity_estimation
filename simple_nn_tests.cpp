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

TEST(SimpleNN, GradientDoubleVectorLayer)
{
	typedef double data_type;

	std::vector<data_type> input1 {-0.8, 0.8, -0.8, 0.8, 0.2};
	std::vector<data_type> input2 {0.7, -0.7, 0.7, -0.7, 0.3};
	std::vector<data_type> input3 {-0.2, 0.2, -0.2, 0.2, 0.2};
	std::vector<data_type> input4 {0.2, -0.3, 0.2, -0.3, 0.1};
	std::vector<data_type> input5 {-0.5, 0.6, -0.5, 0.6, 0.2};
	std::vector<data_type> input6 {0.5, -0.6, 0.5, -0.6, 0.3};
	short output1 = 0;
	short output2 = 1;
	short output3 = 0;
	short output4 = 1;
	short output5 = 0;
	short output6 = 1;

	std::vector<std::vector<data_type>> input {input1, input2, input3, input4, input5, input6};
	std::vector<short> output {output1, output2, output3, output4, output5, output6};

	neural_network<data_type> net(5);
	net.emplace_layer<vector_connected_layer>(2,4,1);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(2);
	net.emplace_layer<softmax_output_layer>();

	for(int i = 0; i < 15; ++i)
	{
		std::cout << "epoch: " << i << std::endl;
		net.training(input, output,2);
		net.test(input, output);
	}

	ASSERT_TRUE(gradient_check(net, input, output, 1e-5, 1e-5));
}

void vector_connected_layer_test()
{
	vector_connected_layer<double> test(7,2,2,1);

	std::vector<double> test_input {1,2,3,4,5,6,7};
	std::vector<double> expected {3,7,11, 6, 14, 22, 7};
	std::vector<double> weights {1,1,2,2};

	std::cout << test.get_bias().size() << std::endl;
	assert(test.get_bias().size() == 2);
	std::fill(test.get_bias().begin(), test.get_bias().end(), 0.0);

	assert(test.get_weights().size() == weights.size());
	std::copy(weights.begin(), weights.end(), test.get_weights().begin());
	test.forward_propagation(test_input.data());

	std::copy(test.output(), test.output() + 7, std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl;
}

void compare_double_vector(const std::vector<double>& actual, const std::vector<double>& desired)
{
	ASSERT_EQ(actual.size(), desired.size());
	for(std::size_t i = 0; i < actual.size(); ++i)
		ASSERT_DOUBLE_EQ(actual[i], desired[i]);
}

TEST(SimpleNNBlas, GEMM)
{
	//every resulting matrix is two elements bigger than needed for recognizing, if the blas calls try to write behind the resulting matrix


	//A*B
	std::vector<double> Y(6,-1);
	std::vector<double> A {1,2,3,4,5,6};
	std::vector<double> B {7,8,9,10,11,12};
	blas_gemm(Y.data(), A.data(), false, 2, 3, B.data(), false, 3, 2);
	std::vector<double> Y_desired {58,64,139,154, -1, -1};
	compare_double_vector(Y, Y_desired);

	//B'*A'
	std::vector<double> Y2(6,-1);
	blas_gemm(Y2.data(), B.data(), true, 3, 2, A.data(), true, 2, 3);
	std::vector<double> Y2_desired {58,139,64,154, -1, -1};
	compare_double_vector(Y2, Y2_desired);

	//A'*B'
	std::vector<double> Y3(11,-1);
	blas_gemm(Y3.data(), A.data(), true, 2, 3, B.data(), true, 3, 2);
	std::vector<double> Y3_desired {39, 49, 59, 54, 68, 82, 69, 87, 105,-1,-1};
	compare_double_vector(Y3, Y3_desired);

	//A2'*B (A2'=A)
	std::vector<double> Y4(6,-1);
	std::vector<double> A2 {1, 4, 2, 5, 3, 6};
	blas_gemm(Y4.data(), A2.data(), true, 3, 2, B.data(), false, 3, 2);
	std::vector<double> Y4_desired = Y_desired;
	compare_double_vector(Y4, Y4_desired);

	//A*B2' (B2'=B)
	std::vector<double> Y5(6,-1);
	std::vector<double> B2 {7,9,11,8,10,12};
	blas_gemm(Y5.data(), A.data(), false, 2, 3, B2.data(), true, 2, 3);
	std::vector<double> Y5_desired = Y_desired;
	compare_double_vector(Y5, Y5_desired);

	//A3*B3
	std::vector<double> Y6(14,-1);
	std::vector<double> A3 {1,2,3,4,5,6,7,8};
	std::vector<double> B3 {9,10,11,12,13,14};
	blas_gemm(Y6.data(), A3.data(), false, 4, 2, B3.data(), false, 2, 3);
	std::vector<double> Y6_desired {33, 36, 39, 75, 82, 89, 117, 128, 139, 159, 174, 189,-1,-1};
	compare_double_vector(Y6, Y6_desired);

	//A4'*B3 (A4'=A3)
	std::vector<double> A4 {1,3,5,7,2,4,6,8};
	std::vector<double> Y7(14,-1);
	blas_gemm(Y7.data(), A4.data(), true, 2, 4, B3.data(), false, 2, 3);
	compare_double_vector(Y7, Y6_desired);

	//A3*B4' (B4'=B3)
	std::vector<double> B4 {9,12,10,13,11,14};
	std::vector<double> Y8(14,-1);
	blas_gemm(Y8.data(), A3.data(), false, 4, 2, B4.data(), true, 3, 2);
	compare_double_vector(Y8, Y6_desired);

	//A4'*B4'
	std::vector<double> Y9(14,-1);
	blas_gemm(Y9.data(), A4.data(), true, 2, 4, B4.data(), true, 3, 2);
	compare_double_vector(Y9, Y6_desired);


}

TEST(SimpleNNBlas, GEMV)
{
	//GEMV
	std::vector<double> Y10(5, -1);
	std::vector<double> A5 {1,2,3,4,5,6};
	std::vector<double> X5 {7,8};
	blas_gemv(Y10.data(), A5.data(), false, 3,2, X5.data());
	std::vector<double> Y10_desired {23, 53, 83,-1,-1};
	compare_double_vector(Y10, Y10_desired);

	std::vector<double> Y11(5,-1);
	std::vector<double> A6 {1,3,5,2,4,6};
	blas_gemv(Y11.data(), A6.data(), true, 2,3, X5.data());
	std::vector<double> Y11_desired = Y10_desired;
	compare_double_vector(Y11, Y11_desired);
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

