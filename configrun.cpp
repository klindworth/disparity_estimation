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

#include "initial_disparity.h"
#include "disparity_region.h"
#include "genericfunctions.h"
#include "debugmatstore.h"
#include "disparity_utils.h"
#include "taskanalysis.h"
#include <segmentation/segmentation.h>
#include "it_metrics.h"
#include "refinement.h"
#include <opencv2/highgui/highgui.hpp>
#include <functional>
#include <cvio/hdf5wrapper.h>
#include <cvio-b/hdf5internals.h>

#include "optimizer/ml_region_optimizer.h"
#include "optimizer/manual_region_optimizer.h"

#include "metrics/pixelwise/gradient_disparitywise.h"

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

void write_logged_data(cv::FileStorage& fs, const std::string& filename, std::pair<cv::Mat, cv::Mat>& disparity, const stereo_task& task, int total_runtime, int ignore_border = 0)
{
	bool logging = true;
	int subsampling = 1; //TODO: avoid this

	task_analysis analysis(task, disparity.first, disparity.second, subsampling, ignore_border);
	cv::Mat disp_left  = disparity::create_image(disparity.first);
	cv::Mat disp_right = disparity::create_image(disparity.second);
	if(logging)
	{
		fs << "taskname" << task.name;
		fs << "total_runtime" << total_runtime;
		fs << task;
		fs << analysis;

		cvio::hdf5file hfile(filename + ".hdf5");
		hfile.save(disparity.first, "/results/left_disparity", false);
		hfile.save(disparity.second, "/results/right_disparity", false);
	}

	matstore.add_mat(disp_left,  "disp_left");
	//matstore.add_mat(disparity.first, "disp_left_org");
	matstore.add_mat(disp_right, "disp_right");

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

std::pair<cv::Mat, cv::Mat> single_logged_run(stereo_task& task, disparity_estimator_algo& disparity_estimator, cv::FileStorage& fs, const std::string& filename)
{

	std::time_t starttime;
	std::time(&starttime);

	std::pair<cv::Mat, cv::Mat> disparity = disparity_estimator(task);

	std::time_t endtime;
	std::time(&endtime);
	int total_runtime = std::difftime(endtime, starttime);
	std::cout << "runtime: " << total_runtime << std::endl;
	std::cout << "finished" << std::endl;

	write_logged_data(fs, filename, disparity, task, total_runtime);

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
		single_logged_run(ctask, disparity_estimator, fs, filename + "_" + ctask.name);
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
std::vector<cv::Mat_<short>> AllInformationTheoreticDistance(const single_stereo_task& task, bool soft, unsigned int windowsize)
{
	typedef float calc_type;
	long long start = cv::getCPUTickCount();
	costmap_creators::entropy::entropies<calc_type> entropy = soft ? calculate_entropies<calc_type, quantizer, true>(task, windowsize) : calculate_entropies<calc_type, quantizer, false>(task, windowsize);

	start = cv::getCPUTickCount() - start;
	std::cout << "entropy " << start << std::endl;

	auto data_single = std::make_pair(entropy.X, entropy.Y);

	return std::vector<cv::Mat_<short>> {disparity::wta_disparity<entropy_agg<calc_type, mutual_information_calc>>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<calc_type, variation_of_information_calc>>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<calc_type, normalized_variation_of_information_calc>>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<calc_type, normalized_information_distance_calc>>(entropy.XY, data_single, task.dispMin, task.dispMax)};
}

template<int quantizer>
void it_both_sides(std::vector<cv::Mat_<short>>& resultLeft, std::vector<cv::Mat_<short>>& resultRight, const stereo_task& task, const classic_search_config& config)
{
	resultLeft = AllInformationTheoreticDistance<quantizer>(task.forward, config.soft, config.windowsize);
	resultRight = AllInformationTheoreticDistance<quantizer>(task.backward, config.soft, config.windowsize);
}

/*std::pair<cv::Mat_<short>> gradient_both(const stereo_task& task, int windowsize)
{
	disparity::create_from_costmap(sliding_gradient(task.forward, windowsize), task.forward.range.start(), 1);
	disparity::create_from_costmap(sliding_gradient(task.backward, windowsize), task.backward.range.start(), 1);
}*/

void singleClassicRun(const stereo_task& task, const classic_search_config& config, const std::string& filename, std::vector<std::unique_ptr<cv::FileStorage>>& fs, const std::vector<std::string>& names)
{
	std::vector<cv::Mat_<short> > resultLeft, resultRight;

	std::time_t starttime;
	std::time(&starttime);

	if(config.quantizer == 1)
		it_both_sides<1>(resultLeft, resultRight, task, config);
	else if(config.quantizer == 2)
		it_both_sides<2>(resultLeft, resultRight, task, config);
	else if(config.quantizer == 4)
		it_both_sides<4>(resultLeft, resultRight, task, config);
	else if(config.quantizer == 8)
		it_both_sides<8>(resultLeft, resultRight, task, config);
	else if(config.quantizer == 16)
		it_both_sides<16>(resultLeft, resultRight, task, config);
	else
		std::cerr << "invalid quantizer" << std::endl;

	//resultLeft.push_back(disparity::create_from_costmap(sliding_gradient(task.forward, config.windowsize), task.forward.range.start(), 1));
	//resultRight.push_back(disparity::create_from_costmap(sliding_gradient(task.backward, config.windowsize), task.backward.range.start(), 1));

	std::time_t endtime;
	std::time(&endtime);
	int total_runtime = std::difftime(endtime, starttime);

	for(std::size_t i = 0; i < resultLeft.size(); ++i)
	{
		std::string fullfilename = filename + names[i] + "_" + task.name;
		std::pair<cv::Mat, cv::Mat> disparity {resultLeft[i], resultRight[i]};

		write_logged_data(*(fs[i]), fullfilename, disparity, task, total_runtime, config.windowsize/2);
	}
}

void classicLoggedRun(task_collection& taskset, classic_search_config& config)
{
	std::vector<std::string> names {"_mi", "_vi", "_nvi", "_ndi"/*, "_grad"*/};

	std::vector<std::unique_ptr<cv::FileStorage>> fs;
	std::string filename = "results/" + dateString() + "_" + timestampString();
	for(std::string& cname : names)
	{
		fs.push_back(std::unique_ptr<cv::FileStorage>(new cv::FileStorage (filename + cname + ".yml", cv::FileStorage::WRITE)));
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

cv::FileStorage& operator<<(cv::FileStorage& stream, const initial_disparity_config& config)
{
	stream << "metric_type" << config.metric_type;
	stream << "configname" << config.name;
	stream << "dilate" << (int)config.dilate << "refinement" << config.enable_refinement;
	stream << "verbose" << config.verbose;
	stream << "dilate_grow" << config.dilate_grow << "dilate_step" << config.dilate_step;
	stream << "region_refinement_delta" << config.region_refinement_delta;
	stream << "region_refinement_rounds" << config.region_refinement_rounds;

	stream << config.segmentation;
	stream << config.optimizer;
	return stream;
}

void readInitialDisparityConfig(const cv::FileNode& stream, initial_disparity_config& config)
{
	int dilate;
	stream["metric_type"] >> config.metric_type;
	stream["configname"] >> config.name;
	stream["dilate_step"] >> config.dilate_step;
	stream["dilate_grow"] >> config.dilate_grow;
	stream["dilate"] >> dilate;
	stream["region_refinement_delta"] >> config.region_refinement_delta;
	stream["region_refinement_rounds"] >> config.region_refinement_rounds;

	config.dilate = dilate;
	stream["refinement"] >> config.enable_refinement;

	stream["verbose"] >> config.verbose;

	stream >> config.segmentation;
	stream >> config.optimizer;
}

cv::FileStorage& operator>>(cv::FileStorage& stream, initial_disparity_config& config)
{
	readInitialDisparityConfig(stream.root(), config);
	return stream;
}

