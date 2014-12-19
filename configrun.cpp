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

std::pair<cv::Mat, cv::Mat> singleLoggedRun(stereo_task& task, disparity_estimator_algo& disparity_estimator, cv::FileStorage& fs, const std::string& filename)
{
	bool logging = true;
	int subsampling = 1; //TODO: avoid this
	std::time_t starttime;
	std::time(&starttime);

	std::pair<cv::Mat, cv::Mat> disparity = disparity_estimator(task);

	std::time_t endtime;
	std::time(&endtime);
	int total_runtime = std::difftime(endtime, starttime);
	std::cout << "runtime: " << total_runtime << std::endl;
	std::cout << "finished" << std::endl;

	TaskAnalysis analysis(task, disparity.first, disparity.second, subsampling);
	cv::Mat disp_left  = disparity::create_image(disparity.first);
	cv::Mat disp_right = disparity::create_image(disparity.second);
	if(logging)
	{
		fs << "taskname" << task.name;
		fs << "total_runtime" << total_runtime;
		fs << task;
		fs << analysis;

		mat_to_file(disparity.first, filename + "-left.cvmat");
		mat_to_file(disparity.second, filename + "-right.cvmat");

		cv::imwrite(filename + "-left.png",  disp_left);
		cv::imwrite(filename + "-right.png", disp_right);
	}

	matstore.addMat(disp_left,  "disp_left");
	matstore.addMat(disp_right, "disp_right");

	if(task.groundLeft.data)
	{
		cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_left);
		if(logging)
			cv::imwrite(filename + "_error-left.png", err_image);
		matstore.addMat(err_image, "ground-diff-left");
		matstore.addMat(task.groundLeft, "groundLeft");
	}
	if(task.groundRight.data)
	{
		cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_right);
		if(logging)
			cv::imwrite(filename + "_error-right.png", err_image);
		matstore.addMat(err_image, "ground-diff-right");
	}

	return disparity;
}

void loggedRun(stereo_task& task, InitialDisparityConfig& config, RefinementConfig& refconfig)
{
	TaskTestSet testset;
	testset.name = task.name;
	testset.tasks.push_back(task);
	loggedRun(testset, config, refconfig);
}


void loggedRun(task_collection& testset, disparity_estimator_algo& disparity_estimator)
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
		singleLoggedRun(ctask, disparity_estimator, fs, filename + "_" + ctask.name);
		fs << "}";
	}
	fs << "]";

	disparity_estimator.writeConfig(fs);
	//fs << config;
	//fs << refconfig;
}

void loggedRun(task_collection& testset, InitialDisparityConfig& config, RefinementConfig& refconfig)
{
	initial_disparity_algo algo(config, refconfig);

	loggedRun(testset, algo);
}

template<int quantizer>
std::vector<cv::Mat_<short>> AllInformationTheoreticDistance(single_stereo_task& task, bool soft, unsigned int windowsize)
{
	typedef std::pair<cv::Mat, cv::Mat> data_type;
	long long start = cv::getCPUTickCount();
	auto entropy = calculateEntropies<quantizer>(task, soft, windowsize);

	start = cv::getCPUTickCount() - start;
	std::cout << "entropy " << start << std::endl;

	auto data_single = std::make_pair(entropy.X, entropy.Y);

	return std::vector<cv::Mat_<short>> {disparity::wta_disparity<entropy_agg<mutual_information_calc<float>>, data_type>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<variation_of_information_calc<float>>, data_type>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<normalized_variation_of_information_calc<float>>, data_type>(entropy.XY, data_single, task.dispMin, task.dispMax),
										disparity::wta_disparity<entropy_agg<normalized_information_distance_calc<float>>, data_type>(entropy.XY, data_single, task.dispMin, task.dispMax)};
}

void singleClassicRun(stereo_task& task, ClassicSearchConfig& config, std::string filename, std::vector<cv::FileStorage*>& fs)
{
	std::vector<cv::Mat_<short> > resultLeft, resultRight;

	std::time_t starttime;
	std::time(&starttime);

	if(config.quantizer == 1)
	{
		resultLeft = AllInformationTheoreticDistance<1>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<1>(task.backward, config.soft, config.windowsize);
	}
	else if(config.quantizer == 2)
	{
		resultLeft = AllInformationTheoreticDistance<2>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<2>(task.backward, config.soft, config.windowsize);
	}
	else if(config.quantizer == 4)
	{
		resultLeft = AllInformationTheoreticDistance<4>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<4>(task.backward, config.soft, config.windowsize);
	}
	else if(config.quantizer == 8)
	{
		resultLeft = AllInformationTheoreticDistance<8>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<8>(task.backward, config.soft, config.windowsize);
	}
	else if(config.quantizer == 16)
	{
		resultLeft = AllInformationTheoreticDistance<16>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<16>(task.backward, config.soft, config.windowsize);
	}
	else if(config.quantizer == 32)
	{
		resultLeft = AllInformationTheoreticDistance<32>(task.forward, config.soft, config.windowsize);
		resultRight = AllInformationTheoreticDistance<32>(task.backward, config.soft, config.windowsize);
	}
	else
		std::cerr << "invalid quantizer" << std::endl;

	std::time_t endtime;
	std::time(&endtime);
	int total_runtime = std::difftime(endtime, starttime);

	std::vector<std::string> names {"_mi", "_vi", "_nvi", "_ndi"};

	for(std::size_t i = 0; i < resultLeft.size(); ++i)
	{
		std::string fullfilename = filename + names[i] + "_" + task.name;
		TaskAnalysis analysis(task, resultLeft[i], resultRight[i], 1, config.windowsize/2);

		*(fs[i]) << "taskname" << task.name;
		*(fs[i]) << "total_runtime" << total_runtime;
		*(fs[i]) << task;
		*(fs[i]) << analysis;

		mat_to_file(resultLeft[i], fullfilename + "-left.cvmat");
		mat_to_file(resultRight[i], fullfilename + "-right.cvmat");
		cv::imwrite(fullfilename + "-left.png",  disparity::create_image(resultLeft[i]));
		cv::imwrite(fullfilename + "-right.png", disparity::create_image(resultRight[i]));
		if(task.groundLeft.data)
		{
			cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_left);
			cv::imwrite(fullfilename + "_error-left.png", err_image);
			matstore.addMat(err_image, "ground-diff-left");
		}
		if(task.groundRight.data)
		{
			cv::Mat err_image = value_scaled_image<unsigned char, unsigned char>(analysis.diff_mat_right);
			cv::imwrite(fullfilename + "_error-right.png", err_image);
			matstore.addMat(err_image, "ground-diff-right");
		}
	}
}

void classicLoggedRun(task_collection& taskset, ClassicSearchConfig& config)
{
	std::vector<std::string> names {"_mi", "_vi", "_nvi", "_ndi"};

	std::vector<cv::FileStorage*> fs;
	std::string filename = "results/" + dateString() + "_" + timestampString();
	for(std::string& cname : names)
	{
		fs.push_back(new cv::FileStorage (filename + cname + ".yml", cv::FileStorage::WRITE));
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

		matstore.startNewTask(ctask.name, ctask);
		for(cv::FileStorage *cfs : fs)
			*cfs << "{:";
		singleClassicRun(ctask, config, filename, fs);
		for(cv::FileStorage *cfs : fs)
			*cfs << "}";
	}

	for(cv::FileStorage *cfs : fs)
	{
		*cfs << "]";
		*cfs << "windowsize" << config.windowsize;
		*cfs << "quantizer" << config.quantizer;
		*cfs << "soft_hist" << config.soft;
		cfs->release();
		delete cfs;
	}
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const InitialDisparityConfig& config)
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

void readInitialDisparityConfig(const cv::FileNode& stream, InitialDisparityConfig& config)
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

cv::FileStorage& operator>>(cv::FileStorage& stream, InitialDisparityConfig& config)
{
	readInitialDisparityConfig(stream.root(), config);
	return stream;
}

