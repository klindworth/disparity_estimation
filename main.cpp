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

#include "disparity_toolkit/stereotask.h"
#include "imagestore.h"
#include "initial_disparity.h"
#include "costmap_utils.h"

#include "region_optimizer.h"
#include "disparity_region.h"
#include "disparity_region_algorithms.h"
#include "configrun.h"

#include "it_metrics.h"
#include "disparity_toolkit/disparity_utils.h"
#include "refinement.h"

#include "ml_region_optimizer.h"
#include "manual_region_optimizer.h"
#include "disparity_toolkit/task_collection.h"

#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	auto prop_eval = [](const disparity_region& baseRegion, const region_container& base, const region_container& match, int disparity, const stat_t& cstat, const std::vector<stat_t>& other_stats) {
		const std::vector<corresponding_region>& other_regions = baseRegion.corresponding_regions[disparity-base.task.range.start()];
		float e_other = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.optimization_energy(-disparity-match.task.range.start());});
		float e_base = baseRegion.optimization_energy(disparity-base.task.range.start());
		//float confidence = std::max(corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cstat.confidence2;}), std::numeric_limits<float>::min());
		float confidence = std::max(corresponding_regions_average_by_index(other_regions, [&](std::size_t idx){ return other_stats[idx].confidence2;}), std::numeric_limits<float>::min());

		return (cstat.confidence2 *e_base+confidence*e_other) / (confidence + cstat.confidence2);
	};

	auto prop_eval2 = [](const disparity_region& baseRegion, const region_container& base, const region_container& match, int disparity, const stat_t& cstat, const std::vector<stat_t>& other_stats) {
		const std::vector<corresponding_region>& other_regions = baseRegion.corresponding_regions[disparity-base.task.range.start()];
		float disp_pot = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return (float)std::min(std::abs(disparity+cregion.disparity), 10);});

		float e_other = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.optimization_energy(-disparity-match.task.range.start());});
		float e_base = baseRegion.optimization_energy(disparity-base.task.range.start());

		//float confidence = std::max(corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cstat.confidence2;}), std::numeric_limits<float>::min());
		float confidence = std::max(corresponding_regions_average_by_index(other_regions, [&](std::size_t idx){ return other_stats[idx].confidence2;}), std::numeric_limits<float>::min());

		return (cstat.confidence2 *e_base+confidence*e_other) / (confidence + cstat.confidence2) + disp_pot/2.5f;
	};

	//RGB tasks
	//stereo_task testset("tasks/im2rgb");
	//folder_testset testset("tasks/kitti-training_small");
	folder_testset testset("tasks/kitti-training1");
	//folder_testset testset("tasks/kitti-training1-valid");
	//folder_testset testset("tasks/kitti-training_small-valid");
	//folder_testset testset("tasks/kitti-training_debug");
	//stereo_task testset("tasks/kit3"); //5, 3 neuer problemfall?
	//TaskTestSet testset("tasks/rgbset");

	//simulated (hard) tasks
	//stereo_task testset("tasks/2im2");
	//TaskTestSet testset("tasks/smallset2");

	//vl/ir
	//stereo_task testset("tasks/ir-vl");
	//stereo_task testset("tasks/bb");

	//multispectral
	//stereo_task testset("tasks/multi1-big");
	//stereo_task testset("tasks/multi2-big");
	//TaskTestSet testset("tasks/debugset");
	//TaskTestSet testset("tasks/smallset2");
	//TaskTestSet testset("tasks/bigset");
	//TaskTestSet testset("tasks/localset");




	//std::string configfile = "tasks/config-irvl.yml";
	//std::string configfile = "tasks/config-big-msslic-refine.yml";
	//std::string configfile = "tasks/config-big-msslic.yml";
	//std::string configfile = "tasks/config-small-msslic-current-sncc.yml";
	//std::string configfile = "tasks/config-small-msslic.yml";
	//std::string configfile = "tasks/config-debug.yml";
	//std::string configfile = "tasks/config-kit-refine2.yml";
	std::string configfile = "tasks/config-kit2.yml";
	//std::string configfile = "tasks/config-kit-it.yml";
	//std::string configfile = "tasks/config-rgb-slic.yml";
	//std::string configfile = "tasks/config-small-woopt.yml";

	/*if(!task.valid())
	{
		std::cerr << "failed to load images" << std::endl;
		return 1;
	}*/

	/*classic_search_config clas_config;
	clas_config.windowsize = 13;
	clas_config.soft = true;
	clas_config.quantizer = 8;

	classicLoggedRun(testset, clas_config);*/

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
	//config.enable_refinement = false;

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

	bool training = true;
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
	//logged_run(testset, config, refconfig);

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

