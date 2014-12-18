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

#include "regionwindow.h"
#include "ui_regionwindow.h"

#include "disparity_utils.h"
#include "disparity_region.h"
#include "regionwidget.h"
#include <QTableWidgetItem>
#include <segmentation/intervals.h>
#include "debugmatstore.h"
#include "region_optimizer.h"
#include "initial_disparity.h"
#include "manual_region_optimizer.h"
#include "disparity_region_algorithms.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

RegionWindow::RegionWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::RegionWindow)
{
	optimizer = new manual_region_optimizer();

	ui->setupUi(this);

	ui->regionLeft->setInverted(true);
	connect(ui->disparityLeft, SIGNAL(mouseClicked(int,int)), this, SLOT(selectPointOnLeftDisparity(int,int)));
	connect(ui->disparityRight, SIGNAL(mouseClicked(int,int)), this, SLOT(selectPointOnRightDisparity(int,int)));
	connect(ui->regionLeft, SIGNAL(matchRegionSelected(int)), this, SLOT(selectRightRegion(int)));
	connect(ui->regionRight, SIGNAL(matchRegionSelected(int)), this, SLOT(selectLeftRegion(int)));
	connect(ui->regionLeft, SIGNAL(baseRegionSelected(int)), this, SLOT(selectLeftRegion(int)));
	connect(ui->regionRight, SIGNAL(baseRegionSelected(int)), this, SLOT(selectRightRegion(int)));

	ui->hsZoom->setValue(3);
}

void RegionWindow::setData(std::shared_ptr<region_container>& left, std::shared_ptr<region_container>& right)
{
	if(left && right)
	{
		m_left = left;
		m_right = right;

		optimizer->reset(*m_left, *m_right);

		QStringList headers;
		headers << "Nr" << "Disparity" << "Pixelcount";
		ui->treeSegments->setHeaderLabels(headers);

		ui->treeSegments->clear();
		for(std::size_t i = 0; i < m_left->regions.size(); ++i)
		{
			disparity_region& baseRegion =  m_left->regions[i];

			QStringList currentItem;
			currentItem << QString::number(i);
			currentItem << QString::number(baseRegion.disparity);
			currentItem << QString::number(baseRegion.m_size);

			ui->treeSegments->addTopLevelItem(new QTreeWidgetItem(currentItem));

		}

		cv::Mat disparityLeft = disparity_by_segments(*m_left);
		ui->disparityLeft->setCVMat(create_disparity_image(disparityLeft));

		cv::Mat warpedLeft = warp_disparity<short>(disparityLeft, 1.0f);
		//cv::Mat warpedLeft = warpImage<short, short>(disparityLeft, disparityLeft, 1.0f);
		ui->warpedLeft->setCVMat(create_disparity_image(warpedLeft));

		cv::Mat disparityRight = disparity_by_segments(*m_right);
		ui->disparityRight->setCVMat(create_disparity_image(disparityRight));

		cv::Mat warpedRight = warp_disparity<short>(disparityRight, 1.0f);
		//cv::Mat warpedRight = warpImage<short, short>(disparityRight, disparityRight, 1.0f);
		ui->warpedRight->setCVMat(create_disparity_image(warpedRight));
	}
}

void RegionWindow::setStore(DebugMatStore* store, InitialDisparityConfig *config)
{
	m_config = config;
	m_store = store;
	ui->cbTask->clear();
	for(TaskStore& ctask : m_store->tasks)
	{
		ui->cbTask->addItem(ctask.name.c_str());
	}
}

void markRegionColor(cv::Mat& color_image, const std::vector<RegionInterval>& pixel_idx, int d)
{
	std::vector<RegionInterval> filtered = filtered_region(color_image.cols, pixel_idx, d);

	for(const RegionInterval& cinterval : filtered)
	{
		for(int x = cinterval.lower; x < cinterval.upper; ++x)
		{
			cv::Vec3b modcolor = color_image.at<cv::Vec3b>(cinterval.y, x + d);
			modcolor[0] = std::max((int)modcolor[0]*2, 255);
			modcolor[1] = 255;
			modcolor[2] /= 2;
			color_image.at<cv::Vec3b>(cinterval.y, x + d) = modcolor;
		}
	}
}

cv::Mat createModifiedImage(cv::Mat& gray_image, std::vector<disparity_region>& regions, bool disparityApplied)
{
	cv::Mat color_image;
	cv::cvtColor(gray_image, color_image, CV_GRAY2BGR);

	for(disparity_region& cregion : regions)
	{
		//markRegionColor(color_image, cregion.pixel_idx, disparityApplied ? cregion.disparity : 0);
		markRegionColor(color_image, cregion.lineIntervals, disparityApplied ? cregion.disparity : 0);
	}

	return color_image;
}

cv::Mat createModifiedImage(cv::Mat& gray_image, std::vector<disparity_region>& regionsLeft, bool disparityApplied, std::vector<disparity_region>& regionsRight, bool disparityAppliedRight)
{
	cv::Mat color_image;
	cv::cvtColor(gray_image, color_image, CV_GRAY2BGR);

	for(disparity_region& cregion : regionsLeft)
		markRegionColor(color_image, cregion.lineIntervals, disparityApplied ? cregion.disparity : 0);

	for(disparity_region& cregion : regionsRight)
		markRegionColor(color_image, cregion.lineIntervals, disparityAppliedRight ? cregion.disparity : 0);

	return color_image;
}

void RegionWindow::refreshImages(std::vector<disparity_region> markLeft, bool markLeftOnRight, std::vector<disparity_region> markRight, bool markRightOnLeft)
{
	cv::Mat disparityLeft = disparity_by_segments(*m_left);
	cv::Mat dispImageLeft = create_disparity_image(disparityLeft);
	cv::Mat colorLeft;
	if(!markRightOnLeft)
		colorLeft = createModifiedImage(dispImageLeft,markLeft, false);
	else
		colorLeft = createModifiedImage(dispImageLeft,markLeft, false, markRight, true);
	ui->disparityLeft->setCVMat(colorLeft);

	cv::Mat warpedLeft = warp_disparity<short>(disparityLeft, 1.0f);
	cv::Mat warpedImageLeft = create_disparity_image(warpedLeft);
	cv::Mat colorWarpedLeft;
	if(!markRightOnLeft)
		colorWarpedLeft = createModifiedImage(warpedImageLeft,markLeft, true);
	else
		colorWarpedLeft = createModifiedImage(warpedImageLeft,markLeft, true, markRight, false);
	ui->warpedLeft->setCVMat(colorWarpedLeft);


	//mark same on diff image
	cv::Mat disparityRight = disparity_by_segments(*m_right);
	cv::Mat dispImageRight = create_disparity_image(disparityRight);
	cv::Mat colorRight;
	if(!markLeftOnRight)
		colorRight = createModifiedImage(dispImageRight,markRight, false);
	else
		colorRight = createModifiedImage(dispImageRight,markRight, false, markLeft, true);
	ui->disparityRight->setCVMat(colorRight);

	cv::Mat warpedRight = warp_disparity<short>(disparityRight, 1.0f);
	cv::Mat warpedImageRight = create_disparity_image(warpedRight);
	cv::Mat colorWarpedRight;
	if(!markLeftOnRight)
		colorWarpedRight = createModifiedImage(warpedImageRight, markRight, true);
	else
		colorWarpedRight = createModifiedImage(warpedImageRight, markRight, true, markLeft, false);
	ui->warpedRight->setCVMat(colorWarpedRight);
}

/*void RegionViewer::showAboutToChangeRegions()
{
	std::vector<SegRegion> regions;

	for(SegRegion& cregion : m_left)
	{
		for(int i = 0; i < cregion.optimization_energy.total(); ++i)
		{
			float e_base = cregion.optimization_energy(i);
			float e_other =
	}
}*/

void RegionWindow::selectLeftRegion(int index)
{
	std::vector<disparity_region> regions{m_left->regions[index]};
	refreshImages(regions, true, {}, false);

	ui->regionLeft->setData(m_left, m_right, index, m_config, true);
}

void RegionWindow::selectRightRegion(int index)
{
	std::vector<disparity_region> regions{m_right->regions[index]};
	refreshImages({}, false, regions, true);

	ui->regionRight->setData(m_right, m_left, index, m_config, true);
}

void RegionWindow::selectPointOnLeftDisparity(int x, int y)
{
	int index = m_left->labels.at<int>(y,x);
	selectLeftRegion(index);
}

void RegionWindow::selectPointOnRightDisparity(int x, int y)
{
	int index = m_right->labels.at<int>(y,x);
	selectRightRegion(index);
}

void RegionWindow::on_treeSegments_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	//no parent => root item => left image
	if(!item->parent())
		selectLeftRegion(item->text(0).toInt());
	else
		qDebug("right");

}


void RegionWindow::on_hsZoom_valueChanged(int value)
{
	ui->disparityLeft->setScaling(value/10.f);
	ui->disparityRight->setScaling(value/10.f);
	ui->warpedLeft->setScaling(value/10.f);
	ui->warpedRight->setScaling(value/10.f);
	ui->lblZoom->setText(QString::number(value*10)+"%");
}

void RegionWindow::on_cbTask_currentIndexChanged(int index)
{
	setData(m_store->tasks[index].left, m_store->tasks[index].right);
}

RegionWindow::~RegionWindow()
{
	delete ui;
}

disparity_hypothesis_weight_vector RegionWindow::get_weight_vector() const
{
	disparity_hypothesis_weight_vector wv;
	wv.costs = ui->spCostsAbs->value();
	wv.lr_pot = ui->spDisp->value();
	wv.occ_avg = ui->spOcc->value();
	wv.neighbor_pot = ui->spNeighbor->value();
	//TODO: color_neighbor_pot
	return wv;
}

void RegionWindow::on_pbOptimize_clicked()
{
	disparity_hypothesis_weight_vector wv = get_weight_vector();

	int choosen = 0;
	if(ui->rbConfidence2->isChecked())
		choosen = 1;
	else if(ui->rbMI->isChecked())
		choosen = 2;
	else if(ui->rbFixed->isChecked())
		choosen = 3;
	else if(ui->rbBase->isChecked())
		choosen = 4;
	else if(ui->rbConfidence3->isChecked())
		choosen = 5;

	double pot_factor = ui->spDispFinal->value();

	auto prop_eval = [=](const disparity_region& baseRegion, const region_container& base, const region_container& match, int disparity) {

		const std::vector<corresponding_region>& other_regions = baseRegion.corresponding_regions[disparity-base.task.dispMin];
		float disp_pot = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return (float)std::min(std::abs(disparity+cregion.disparity), 10);});
		//float stddev = getOtherRegionsAverage(match.regions, other_regions, [](const DisparityRegion& cregion){return cregion.stats.stddev;});

		float e_other = corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.optimization_energy(-disparity-match.task.dispMin);});
		float e_base = baseRegion.optimization_energy(disparity-base.task.dispMin);

		float confidence = std::max(corresponding_regions_average(match.regions, other_regions, [&](const disparity_region& cregion){return cregion.stats.confidence2;}), std::numeric_limits<float>::min());
		//float mi_confidence = getOtherRegionsAverage(match.regions, other_regions, [&](const SegRegion& cregion){return cregion.confidence(-disparity-match.task.dispMin);});

		//float stddev_sum = stddev + baseRegion.stats.stddev;

		//float own_mi_confidence = std::max(baseRegion.confidence(disparity-base.task.dispMin), std::numeric_limits<float>::min());

		//float rating = disp_pot * baseRegion.stats.stddev/stddev + e_base;
		//float rating = e_other + e_base;
		float rating;
		if(choosen == 1)
			rating = (baseRegion.stats.confidence2 *e_base+confidence*e_other) / (confidence + baseRegion.stats.confidence2)+pot_factor*disp_pot;
		//else if(choosen == 2)
			//rating = (own_mi_confidence * e_base + mi_confidence * e_other)/(own_mi_confidence+mi_confidence)+pot_factor*disp_pot;
		else if(choosen == 3)
			rating = e_base+e_other+pot_factor*disp_pot;
		else if(choosen == 4)
			rating = e_base +pot_factor*disp_pot;
		return rating;
	};

	std::vector<unsigned char> left_damping_history(m_left->regions.size(), 0);
	std::vector<unsigned char> right_damping_history(m_right->regions.size(), 0);
	std::vector<std::vector<float>> left_optimization_vectors, right_optimization_vectors;

	std::cout << "optimization" << std::endl;
	optimizer->optimize(left_damping_history, left_optimization_vectors, right_optimization_vectors, *m_left, *m_right, wv, prop_eval, 0);
	generateRegionInformation(*m_left, *m_right);
	optimizer->optimize(right_damping_history, right_optimization_vectors, left_optimization_vectors, *m_right, *m_left, wv, prop_eval, 0);
	generateRegionInformation(*m_left, *m_right);
	std::cout << "finished" << std::endl;
	setData(m_left, m_right);
}

void RegionWindow::on_pbRefreshBase_clicked()
{
	disparity_hypothesis_weight_vector wv = get_weight_vector();
	std::vector<std::vector<float>> optimization_vectors(m_left->regions.size());
	refreshOptimizationBaseValues(optimization_vectors, *m_left, *m_right, wv, 0);
}

void resetContainerDisparities(region_container& container)
{
	short dispMin = container.task.dispMin;
	short dispMax = container.task.dispMax;
	for(disparity_region& cregion : container.regions)
	{
		float minCost = std::numeric_limits<float>::max();
		short minD = dispMin;
		for(int d = dispMin; d <= dispMax; ++d)
		{
			float cost = cregion.disparity_costs(d-dispMin);

			if(minCost > cost)
			{
				minD = d;
				minCost = cost;
			}
		}
		cregion.disparity = minD;
	}
}

void RegionWindow::on_pbResetOptimization_clicked()
{
	optimizer->reset(*m_left, *m_right);

	resetContainerDisparities(*m_left);
	resetContainerDisparities(*m_right);
	generateRegionInformation(*m_left, *m_right);
	setData(m_left, m_right);
}
