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

#include "costmapviewer.h"
#include "matstoreviewer.h"
#include "regionwindow.h"
#include "debugmatstore.h"

#include "stereotask.h"
#include "imagestore.h"
#include "initial_disparity.h"

#include "region_optimizer.h"
#include "region.h"
#include "configrun.h"

#include "it_metrics.h"
#include "disparity_utils.h"
#include "refinement.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	auto base_eval = [](const disparity_hypothesis& prop_stat) {
		return (float) prop_stat.costs * 4.0f + prop_stat.occ_avg;
	};

	auto base_eval2 = [](const disparity_hypothesis& prop_stat) {
		return (float) prop_stat.costs * 4.0f + prop_stat.occ_avg + prop_stat.neighbor_color_pot*0.2f + prop_stat.lr_pot * 0.4f;
	};

	auto prop_eval = [](const SegRegion& baseRegion, const RegionContainer& base, const RegionContainer& match, int disparity) {

		const std::vector<MutualRegion>& other_regions = baseRegion.other_regions[disparity-base.task.dispMin];
		float e_other = getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return cregion.optimization_energy(-disparity-match.task.dispMin);});
		float e_base = baseRegion.optimization_energy(disparity-base.task.dispMin);
		float confidence = std::max(getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return cregion.stats.confidence2;}), std::numeric_limits<float>::min());

		return (baseRegion.stats.confidence2 *e_base+confidence*e_other) / (confidence + baseRegion.stats.confidence2);
	};

	auto prop_eval2 = [](const SegRegion& baseRegion, const RegionContainer& base, const RegionContainer& match, int disparity) {

		const std::vector<MutualRegion>& other_regions = baseRegion.other_regions[disparity-base.task.dispMin];
		float disp_pot = getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return (float)std::min(std::abs(disparity+cregion.disparity), 10);});

		float e_other = getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return cregion.optimization_energy(-disparity-match.task.dispMin);});
		float e_base = baseRegion.optimization_energy(disparity-base.task.dispMin);

		float confidence = std::max(getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return cregion.stats.confidence2;}), std::numeric_limits<float>::min());

		return (baseRegion.stats.confidence2 *e_base+confidence*e_other) / (confidence + baseRegion.stats.confidence2) + disp_pot/2.5f;
	};

	//RGB tasks
	//StereoTask testset("tasks/im2rgb");

	//simulated (hard) tasks
	//StereoTask testset("tasks/2im2");

	//vl/ir
	//StereoTask testset("tasks/ir-vl");

	//multispectral
	//StereoTask testset("tasks/multi1-big");
	//StereoTask testset("tasks/multi2-big");
	//TaskTestSet testset("tasks/debugset");
	//TaskTestSet testset("tasks/smallset2");
	//TaskTestSet testset("tasks/bigset");
	//TaskTestSet testset("tasks/localset");
	TaskTestSet testset("tasks/rgbset");
	//StereoTask testset("tasks/kit5");

	//std::string configfile = "tasks/config-irvl.yml";
	//std::string configfile = "tasks/config-big-msslic-refine.yml";
	//std::string configfile = "tasks/config-big-msslic.yml";
	//std::string configfile = "tasks/config-small-msslic-refine.yml";
	//std::string configfile = "tasks/config-debug.yml";
	//std::string configfile = "tasks/config-rgbl-msslic.yml";
	//std::string configfile = "tasks/config-kit-refine2.yml";
	std::string configfile = "tasks/config-rgb-crslic.yml";
	//std::string configfile = "tasks/config-small-woopt.yml";

	/*if(!task.valid())
	{
		std::cerr << "failed to load images" << std::endl;
		return 1;
	}*/

	ClassicSearchConfig clas_config;
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

	InitialDisparityConfig config;
	fs >> config;
	config.optimizer.prop_eval = prop_eval;
	config.optimizer.base_eval = base_eval;
	config.optimizer.prop_eval2 = prop_eval2;
	config.optimizer.base_eval2 = base_eval2;
	/*config.optimizer.prop_eval_refine = prop_eval_refine;
	config.optimizer.base_eval_refine = base_eval_refine;*/
	config.optimizer.prop_eval_refine = prop_eval2;
	config.optimizer.base_eval_refine = base_eval2;
	//config.enable_regionsplit = true;
	//config.optimization_rounds = 3;
	config.verbose = true;

	RefinementConfig refconfig;
	fs >> refconfig;

	if(config.verbose)
		std::cout << "warning: verbose activated" << std::endl;

	loggedRun(testset, config, refconfig);

	MatStoreViewer sviewer;
	sviewer.refreshList(matstore);
	sviewer.show();

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

