/*
Copyright (c) 2013, Kai Klindworth
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

#include "configrun.h"

#include <ctime>
#include <chrono>

#include "initial_disparity.h"
#include "disparity_region.h"
#include "disparity_toolkit/genericfunctions.h"
#include "debugmatstore.h"
#include "disparity_toolkit/disparity_utils.h"
#include "disparity_toolkit/taskanalysis.h"
#include "disparity_toolkit/task_collection.h"
#include <segmentation/segmentation.h>
#include "it_metrics.h"
#include "refinement.h"
#include <opencv2/highgui/highgui.hpp>
#include <functional>
#include <cvio/hdf5wrapper.h>
#include <cvio/hdf5internals.h>

#include "optimizer/ml_region_optimizer.h"
#include "optimizer/manual_region_optimizer.h"

#include "metrics/pixelwise/gradient_disparitywise.h"
#include "metrics/pixelwise/sncc_disparitywise_calculator.h"

std::string dateString()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer [11];

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);
	std::strftime(buffer, 11, "%Y-%m-%d", timeinfo);

	return std::string(buffer);
}

std::string timestampString()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer [9];

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);
	std::strftime(buffer, 9, "%H-%M-%S", timeinfo);

	return std::string(buffer);
}

void save_disparity(cvio::hdf5file& hfile, const disparity_map& disp, const std::string& path)
{
	hfile.save(disp, path, false);
	cvio::hdf5dataset dataset(hfile, path);
	dataset.set_attribute("sampling", disp.sampling);
}

void write_logged_data(cv::FileStorage& fs, const std::string& filename, const std::pair<disparity_map, disparity_map>& disparity, const stereo_task& task, int total_runtime, int ignore_border = 0)
{
	bool logging = true;

	task_analysis analysis(task, disparity.first, disparity.second, ignore_border);

	if(logging)
	{
		fs << "taskname" << task.name;
		fs << "total_runtime" << total_runtime;
		fs << task;
		fs << analysis;

		cvio::hdf5file hfile(filename + ".hdf5");

		save_disparity(hfile, disparity.first,  "/results/" + task.name + "/left_disparity");
		save_disparity(hfile, disparity.second, "/results/" + task.name + "/right_disparity");
	}

	if(task.groundLeft.data)
	{
		cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_left);
		matstore.add_mat(err_image, "ground-diff-left");
	}
	if(task.groundRight.data)
	{
		cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_right);
		matstore.add_mat(err_image, "ground-diff-right");
	}
}

class single_time_logger
{
public:
	single_time_logger() : ended(false) {
		start();
	}

	std::chrono::milliseconds::rep end() {
		if(!ended)
		{
			endpoint = std::chrono::system_clock::now();
			ended = true;
		}
		return std::chrono::duration_cast<std::chrono::milliseconds>(endpoint - startpoint).count();
	}

	~single_time_logger() {
		unsigned int total_runtime = static_cast<unsigned int>(end());
		std::cout << "runtime: " << total_runtime << " ms" << std::endl;
	}

private:
	void start() {
		startpoint = std::chrono::system_clock::now();
	}

	std::chrono::system_clock::time_point startpoint, endpoint;
	bool ended;
};

struct time_logger_entry
{
	std::chrono::system_clock::time_point start, end;
	std::string name;
};

class time_logger
{
public:
	time_logger() {}

	void start(const char* name) {
		current.start = std::chrono::system_clock::now();
		current.name = name;
	}

	void end() {
		current.end = std::chrono::system_clock::now();
		entries.push_back(current);
	}

private:
	time_logger_entry current;
	std::vector<time_logger_entry> entries;
};

class time_logger_function
{
public:
	time_logger_function(time_logger& logger, const char* name) : _logger(logger) {
		_logger.start(name);
	}

	~time_logger_function() {
		_logger.end();
	}

private:
	time_logger& _logger;
};

std::pair<disparity_map, disparity_map> single_logged_run(stereo_task& task, disparity_estimator_algo& disparity_estimator, cv::FileStorage& fs, const std::string& filename)
{
	single_time_logger tlog;

	std::pair<disparity_map, disparity_map> disparity = disparity_estimator(task);

	write_logged_data(fs, filename, disparity, task, tlog.end());

	return disparity;
}

void logged_run(stereo_task& task, initial_disparity_config& config, refinement_config& refconfig)
{
	TaskTestSet testset;
	testset.name = task.name;
	testset.tasks.push_back(task);
	logged_run(testset, config, refconfig);
}


void logged_run(task_collection& testset, disparity_estimator_algo& disparity_estimator)
{
	std::string filename = "results/" + dateString() + "_" + timestampString();
	cv::FileStorage fs(filename + ".yml", cv::FileStorage::WRITE);

	fs << "testset" << testset.name;
	fs << "analysis" << "[";
	for(stereo_task& ctask : testset.tasks)
	{
		std::cout << "----------------------\nstart task: " << ctask.name << "\n-------------------" << std::endl;
		if(!ctask.valid())
		{
			std::cerr << "failed to load images: " << ctask.name << std::endl;
			return;
		}

		//matstore.startNewTask(ctask.name, ctask);
		fs << "{:";
		single_logged_run(ctask, disparity_estimator, fs, filename);
		fs << "}";
	}
	fs << "]";

	disparity_estimator.writeConfig(fs);
	//fs << config;
	//fs << refconfig;
}

void logged_run(task_collection& testset, initial_disparity_config& config, refinement_config& refconfig)
{
	std::shared_ptr<region_optimizer> optimizer;
	if(config.optimizer.optimizer_type == "manual")
		optimizer = std::make_shared<manual_region_optimizer>();
	else
		optimizer = std::make_shared<ml_region_optimizer>();

	initial_disparity_algo algo(config, refconfig, optimizer);

	logged_run(testset, algo);
}

template<int quantizer>
std::vector<disparity_map> AllInformationTheoreticDistance(const single_stereo_task& task, bool soft, unsigned int windowsize)
{
	typedef float calc_type;
	long long start = cv::getCPUTickCount();
	costmap_creators::entropy::entropies<calc_type> entropy = soft ? calculate_entropies<calc_type, quantizer, true>(task, windowsize) : calculate_entropies<calc_type, quantizer, false>(task, windowsize);

	start = cv::getCPUTickCount() - start;
	std::cout << "entropy " << start << std::endl;

	auto data_single = std::make_pair(entropy.X, entropy.Y);
	int sampling = task.groundTruth.sampling;

	return std::vector<disparity_map> {disparity::wta_disparity_sampling<entropy_agg<calc_type, mutual_information_calc>>(entropy.XY, data_single, task.range, sampling),
										disparity::wta_disparity_sampling<entropy_agg<calc_type, variation_of_information_calc>>(entropy.XY, data_single, task.range, sampling),
										disparity::wta_disparity_sampling<entropy_agg<calc_type, normalized_variation_of_information_calc>>(entropy.XY, data_single, task.range, sampling),
										disparity::wta_disparity_sampling<entropy_agg<calc_type, normalized_information_distance_calc>>(entropy.XY, data_single, task.range, sampling)};
}

template<int quantizer>
void it_both_sides(std::vector<disparity_map>& resultLeft, std::vector<disparity_map>& resultRight, const stereo_task& task, const classic_search_config& config)
{
	resultLeft = AllInformationTheoreticDistance<quantizer>(task.forward, config.soft, config.windowsize);
	resultRight = AllInformationTheoreticDistance<quantizer>(task.backward, config.soft, config.windowsize);
}

/*std::pair<cv::Mat_<short>> gradient_both(const stereo_task& task, int windowsize)
{
	disparity::create_from_costmap(sliding_gradient(task.forward, windowsize), task.forward.range.start(), 1);
	disparity::create_from_costmap(sliding_gradient(task.backward, windowsize), task.backward.range.start(), 1);
}*/

template<int quantization = 16>
void it_both_sides_runner(std::vector<disparity_map>& result_left, std::vector<disparity_map>& result_right, const stereo_task& task, const classic_search_config& config, int pquantization)
{
	if(quantization != pquantization)
		it_both_sides_runner<quantization/2>(result_left, result_right, task, config, pquantization);
	else
		it_both_sides<quantization>(result_left, result_right, task, config);
}

template<>
void it_both_sides_runner<0>(std::vector<disparity_map>& , std::vector<disparity_map>& , const stereo_task& , const classic_search_config&, int)
{
	throw std::invalid_argument("invalid quantization (valid arguments: 1,2,4,8,16)");
}

void singleClassicRun(const stereo_task& task, const classic_search_config& config, const std::string& filename, std::vector<std::unique_ptr<cv::FileStorage>>& fs, const std::vector<std::string>& names)
{
	std::vector<disparity_map> resultLeft, resultRight;

	single_time_logger tlog;

	it_both_sides_runner(resultLeft, resultRight, task, config, config.quantizer);

	int sampling = task.ground_truth_sampling;

	resultLeft.push_back(disparity::create_from_costmap(sliding_gradient(task.forward, config.windowsize), task.forward.range.start(), sampling));
	resultRight.push_back(disparity::create_from_costmap(sliding_gradient(task.backward, config.windowsize), task.backward.range.start(), sampling));

	sncc_disparitywise_calculator sncc_f(task.leftGray, task.rightGray);
	resultLeft.push_back(disparity::create_from_costmap(simple_window_disparitywise_calculator(sncc_f, cv::Size(config.windowsize, config.windowsize), task.left.size(), task.forward.range),task.forward.range.start(), sampling));
	sncc_disparitywise_calculator sncc_b(task.rightGray, task.leftGray);
	resultRight.push_back(disparity::create_from_costmap(simple_window_disparitywise_calculator(sncc_b, cv::Size(config.windowsize, config.windowsize), task.right.size(), task.backward.range),task.backward.range.start(), sampling));

	int total_runtime = tlog.end();

	for(std::size_t i = 0; i < resultLeft.size(); ++i)
	{
		std::string fullfilename = filename + names[i];
		std::pair<disparity_map, disparity_map> disparity {resultLeft[i], resultRight[i]};

		write_logged_data(*(fs[i]), fullfilename, disparity, task, total_runtime, config.windowsize/2);
	}
}

void classicLoggedRun(task_collection& taskset, classic_search_config& config)
{
	std::vector<std::string> names {"_mi", "_vi", "_nvi", "_ndi", "_grad", "_sncc"};

	std::vector<std::unique_ptr<cv::FileStorage>> fs;
	std::string filename = "results/" + dateString() + "_" + timestampString();
	for(std::string& cname : names)
	{
		fs.push_back(std::make_unique<cv::FileStorage>(filename + cname + ".yml", cv::FileStorage::WRITE));
		*(fs.back()) << "testset" << taskset.name;
		*(fs.back()) << "analysis" << "[";
	}

	for(stereo_task& ctask : taskset.tasks)
	{
		std::cout << "----------------------\nstart task: " << ctask.name << "\n-------------------" << std::endl;
		if(!ctask.valid())
		{
			std::cerr << "failed to load images: " << ctask.name << std::endl;
			return;
		}

		matstore.start_new_task(ctask.name, ctask);
		for(std::unique_ptr<cv::FileStorage>& cfs : fs)
			*cfs << "{:";
		singleClassicRun(ctask, config, filename, fs, names);
		for(std::unique_ptr<cv::FileStorage>& cfs : fs)
			*cfs << "}";
	}

	for(std::unique_ptr<cv::FileStorage>& cfs : fs)
	{
		*cfs << "]";
		*cfs << "windowsize" << config.windowsize;
		*cfs << "quantizer" << config.quantizer;
		*cfs << "soft_hist" << config.soft;
		//cfs->release();
		//delete cfs;
	}
}



