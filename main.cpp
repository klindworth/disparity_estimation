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

#include <iostream>
#include <algorithm>
#include <cmath>
#include <QApplication>

#include "regionwindow.h"
#include "debugmatstore.h"

#include "stereotask.h"
#include "imagestore.h"
#include "initial_disparity.h"
#include "costmap_utils.h"

#include "region_optimizer.h"
#include "disparity_region.h"
#include "disparity_region_algorithms.h"
#include "configrun.h"

#include "it_metrics.h"
#include "disparity_utils.h"
#include "refinement.h"

#include "ml_region_optimizer.h"
#include "manual_region_optimizer.h"

#include <opencv2/highgui/highgui.hpp>


template<typename region_type, typename InsertIterator>
void region_ground_truth2(const std::vector<region_type>& regions, cv::Mat_<unsigned char> gt, InsertIterator it)
{
	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		int sum = 0;
		int count = 0;

		intervals::foreach_region_point(regions[i].lineIntervals.begin(), regions[i].lineIntervals.end(), [&](cv::Point pt){
			unsigned char value = gt(pt);
			if(value != 0)
			{
				sum += value;
				++count;
			}
		});

		*it = count > 0 ? std::round(sum/count) : 0;
		++it;
	}
}

cv::Mat_<unsigned char> region_ground_truth_image(const cv::Mat_<unsigned char>& disp, std::shared_ptr<segmentation_image<region_descriptor>> seg_image)
{
	std::vector<unsigned char> averages;
	region_ground_truth2(seg_image->regions, disp, std::back_inserter(averages));
	cv::Mat_<unsigned char> avg_image(disp.size());

	for(std::size_t i = 0; i < seg_image->regions.size(); ++i)
		intervals::set_region_value(avg_image, seg_image->regions[i].lineIntervals, averages[i]);

	return avg_image;
}

int main(int argc, char *argv[])
{
	/*cv::Mat image = cv::imread("test.png");
	cv::Mat image_right = cv::imread("test_right.png");

	segmentation_settings seg_settings;
	seg_settings.algorithm = "superpixel";
	seg_settings.superpixel_size = 225;
	seg_settings.superpixel_compactness = 20.0f;
	std::shared_ptr<segmentation_algorithm> segm = getSegmentationClass(seg_settings);
	std::shared_ptr<segmentation_image<RegionDescriptor>> seg_image = segm->getSegmentationImage<segmentation_image<RegionDescriptor>>(image);
	std::shared_ptr<segmentation_image<RegionDescriptor>> seg_image_right = segm->getSegmentationImage<segmentation_image<RegionDescriptor>>(image_right);

	cv::Mat test = getWrongColorSegmentationImage(*seg_image);
	cv::Mat_<unsigned char> disp = cv::imread("disp2.png", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat_<unsigned char> dispImage = region_ground_truth_image(disp, seg_image);

	cv::Mat dispWarped = warpDisparity<unsigned char>(disp, 1.0f);
	cv::Mat_<unsigned char> dispImage_right = region_ground_truth_image(dispWarped, seg_image_right);

	cv::imshow("right", dispImage_right);
	cv::imshow("imright", image_right);
	cv::imshow("avg", dispImage);
	cv::imshow("image", image);
	cv::imshow("test", test);
	cv::waitKey();

	return 0;*/

	QApplication app(argc, argv);

	auto prop_eval = [](const disparity_region& baseRegion, const region_container& base, const region_container& match, int disparity, const stat_t& cstat) {
		const std::vector<corresponding_region>& other_regions = baseRegion.corresponding_regions[disparity-base.task.dispMin];
		float e_other = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.optimization_energy(-disparity-match.task.dispMin);});
		float e_base = baseRegion.optimization_energy(disparity-base.task.dispMin);
		float confidence = std::max(corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cstat.confidence2;}), std::numeric_limits<float>::min());

		return (cstat.confidence2 *e_base+confidence*e_other) / (confidence + cstat.confidence2);
	};

	auto prop_eval2 = [](const disparity_region& baseRegion, const region_container& base, const region_container& match, int disparity, const stat_t& cstat) {
		const std::vector<corresponding_region>& other_regions = baseRegion.corresponding_regions[disparity-base.task.dispMin];
		float disp_pot = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return (float)std::min(std::abs(disparity+cregion.disparity), 10);});

		float e_other = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.optimization_energy(-disparity-match.task.dispMin);});
		float e_base = baseRegion.optimization_energy(disparity-base.task.dispMin);

		float confidence = std::max(corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cstat.confidence2;}), std::numeric_limits<float>::min());

		return (cstat.confidence2 *e_base+confidence*e_other) / (confidence + cstat.confidence2) + disp_pot/2.5f;
	};

	//RGB tasks
	//StereoTask testset("tasks/im2rgb");
	//folder_testset testset("tasks/kitti-training_small");
	//folder_testset testset("tasks/kitti-training_small-valid");
	folder_testset testset("tasks/kitti-training_debug");
	//StereoTask testset("tasks/kit3"); //5, 3 neuer problemfall?

	//simulated (hard) tasks
	//stereo_task testset("tasks/2im2");

	//vl/ir
	//StereoTask testset("tasks/ir-vl");

	//multispectral
	//StereoTask testset("tasks/multi1-big");
	//StereoTask testset("tasks/multi2-big");
	//TaskTestSet testset("tasks/debugset");
	//TaskTestSet testset("tasks/smallset2");
	//TaskTestSet testset("tasks/bigset");
	//TaskTestSet testset("tasks/localset");
	//TaskTestSet testset("tasks/rgbset");



	//std::string configfile = "tasks/config-irvl.yml";
	//std::string configfile = "tasks/config-big-msslic-refine.yml";
	//std::string configfile = "tasks/config-big-msslic.yml";
	//std::string configfile = "tasks/config-small-msslic-refine.yml";
	//std::string configfile = "tasks/config-debug.yml";
	//std::string configfile = "tasks/config-rgbl-msslic.yml";
	//std::string configfile = "tasks/config-kit-refine2.yml";
	std::string configfile = "tasks/config-kit3.yml";
	//std::string configfile = "tasks/config-kit-it.yml";
	//std::string configfile = "tasks/config-rgb-slic.yml";
	//std::string configfile = "tasks/config-small-woopt.yml";

	/*if(!task.valid())
	{
		std::cerr << "failed to load images" << std::endl;
		return 1;
	}*/

	classic_search_config clas_config;
	clas_config.windowsize = 11;
	clas_config.soft = false;
	clas_config.quantizer = 8;

	//classicLoggedRun(testset, clas_config);

	cv::FileStorage fs(configfile, cv::FileStorage::READ);
	if(!fs.isOpened())
	{
		std::cerr << "failed to open config" << std::endl;
		return 1;
	}

	initial_disparity_config config;
	fs >> config;
	config.optimizer.prop_eval = prop_eval;
	config.optimizer.prop_eval2 = prop_eval2;
	//config.optimization_rounds = 3;
	//config.verbose = true;

	refinement_config refconfig;
	fs >> refconfig;

	if(config.verbose)
		std::cout << "warning: verbose activated" << std::endl;

	//loggedRun(testset, config, refconfig);
	std::shared_ptr<region_optimizer> optimizer;
	if(config.optimizer.optimizer_type == "manual")
		optimizer = std::make_shared<manual_region_optimizer>();
	else
		optimizer = std::make_shared<ml_region_optimizer>();
	initial_disparity_algo algo(config, refconfig, optimizer);

	bool training = false;
	if(training)
	{
		algo.train(testset.tasks);
	}
	else
	{
		for(stereo_task& ctask : testset.tasks)
			algo(ctask);
	}

	//algo(testset);

	ImageStore imviewer;
	imviewer.refreshList(matstore);
	imviewer.show();


	if(config.verbose)
	{
		RegionWindow *viewer = new RegionWindow();
		viewer->setStore(&matstore, &config);
		viewer->setData(matstore.tasks.back().left, matstore.tasks.back().right);
		viewer->show();
	}

	return app.exec();
}

