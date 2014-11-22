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

#include "regionwidget.h"
#include "ui_regionwidget.h"
#include "costmap_utils.h"
#include "genericfunctions.h"
#include "disparity_utils.h"
#include "region.h"
#include <segmentation/intervals.h>

#include "region_optimizer.h"
#include "configrun.h"
#include <segmentation/intervals_algorithms.h>
#include "initial_disparity.h"

#include <opencv2/highgui/highgui.hpp>

RegionWidget::RegionWidget(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::RegionWidget)
{
	ui->setupUi(this);
}


void RegionWidget::setInverted(bool inverted)
{
	ui->plot->setInverted(inverted);
}

void RegionWidget::warpTree(int index, DisparityRegion& baseRegion, std::vector<DisparityRegion>& other_regions , QTreeWidget *tree, int dispMin, int currentDisparity)
{
	tree->clear();
	QStringList headers;
	headers << "Nr" << "Disparity-Dev" << "Disparity" << "Pixelcount" << "% (base)" << "% (match)" << "stddev" << "e_base";
	tree->setHeaderLabels(headers);
	tree->setColumnCount(headers.size());

	double disp_average = 0.0f;
	double disp_lr_norm = 0.0f;
	double disp_non_lr_average = 0.0f;
	std::vector<std::pair<double,double>> non_lr_regions;
	//for(auto it = baseRegion.other_labels.begin(); it != baseRegion.other_labels.end(); ++it)
	for(MutualRegion& cregion : baseRegion.other_regions[currentDisparity-dispMin])
	{
		DisparityRegion& matchRegion = other_regions[cregion.index];
		double mutual_percent = cregion.percent;//(double)it->second / baseRegion.other_labels.total();
		disp_average += mutual_percent * matchRegion.disparity;
		if(std::abs(matchRegion.disparity + currentDisparity) < 4)
			disp_lr_norm += mutual_percent;
		else
		{
			disp_non_lr_average += mutual_percent * matchRegion.disparity;
			non_lr_regions.push_back(std::make_pair(mutual_percent, (double)matchRegion.disparity));
		}
		QStringList matchItem;
		matchItem << QString::number(cregion.index);
		matchItem << QString::number(std::abs(matchRegion.disparity + currentDisparity));
		matchItem << QString::number(matchRegion.disparity);
		matchItem << QString::number(matchRegion.m_size);
		matchItem << QString::number(mutual_percent*100, 'f', 2) + "%";
		matchItem << QString::number(matchRegion.getMutualRegion(index, -currentDisparity-m_matchDispMin).percent*100, 'f', 2) + "%";
		matchItem << QString::number(matchRegion.stats.stddev/baseRegion.stats.stddev);
		if(matchRegion.optimization_energy.data)
			matchItem << QString::number(matchRegion.optimization_energy(-currentDisparity-m_matchDispMin));
		matchItem << "noop";

		tree->addTopLevelItem(new QTreeWidgetItem(matchItem));
		/*std::cout << regionsLeft[i].disparity_costs.at<float>(regionsLeft[i].disparity - dispMin) << " vs " << regionsRight[it->first].disparity_costs.at<float>(regionsRight[it->first].disparity) << std::endl; //TODO: dispMax*/
	}

	double mean_non_lr = disp_non_lr_average / (1.0f-disp_lr_norm);
	double stddev = 0.0f;
	for(std::pair<double,double> cpair : non_lr_regions)
		stddev += cpair.first * std::abs(cpair.second-mean_non_lr);
	ui->lblAverageDisparity->setText("Average: " + QString::number(disp_average) + ", L/R passed: " + QString::number(disp_lr_norm*100, 'f', 2) + "%, Non-LR-Average: " + QString::number(mean_non_lr) + " , stddev: " + QString::number(stddev) );
}

void RegionWidget::mutualDisparity(DisparityRegion& baseRegion, RegionContainer& base, RegionContainer& match, QTreeWidget *tree, int dispMin)
{
	std::vector<DisparityRegion>& other_regions = match.regions;
	cv::Mat disp = getDisparityBySegments(base);
	cv::Mat occmap = occlusionStat<short>(disp, 1.0);
	//cv::imshow("occ_test", getValueScaledImage<unsigned char, unsigned char>(occmap));
	intervals::substractRegionValue<unsigned char>(occmap, baseRegion.warped_interval, 1);

	tree->clear();
	QStringList header;
	header << "Disp" << "Other Disp" << "DispDiff" << "disp_neigh_color" << "Costs" << "own_occ" << /*"E_base" <<*/ "E" << "E_own" << "E_Other";
	tree->setColumnCount(header.size());
	tree->setHeaderLabels(header);

	int pot_trunc = 10;

	disparity_hypothesis_vector dhv(base.regions, match.regions);
	std::vector<float> optimization_vector;
	int dispMax = dispMin + baseRegion.other_regions.size()-1;
	dhv(occmap, baseRegion, pot_trunc, dispMin, dispMin, dispMax, m_config->optimizer.base_eval, optimization_vector);

	int i = 0;
	for(auto it = baseRegion.other_regions.begin(); it != baseRegion.other_regions.end(); ++it)
	{
		disparity_hypothesis hyp(optimization_vector, i);
		short currentDisp = i + dispMin;
		//disparity_hypothesis hyp(occmap, baseRegion, currentDisp, base.regions, other_regions, pot_trunc, dispMin);

		float avg_disp = getOtherRegionsAverage(other_regions, *it, [](const DisparityRegion& cregion){return (float)cregion.disparity;});
		//float stddev = getOtherRegionsAverage(other_regions, *it, [](const DisparityRegion& cregion){return cregion.stats.stddev;});
		float disp_pot = getOtherRegionsAverage(other_regions, *it, [&](const DisparityRegion& cregion){return (float)std::min(std::abs(currentDisp+cregion.disparity), 10);});
		float e_other = getOtherRegionsAverage(other_regions, *it, [&](const DisparityRegion& cregion){return cregion.optimization_energy(-currentDisp-m_matchDispMin);});

		//ratings
		//float stddev_dev = baseRegion.stats.stddev-stddev;
		//float disp_dev = std::abs(currentDisp+avg_disp);

		//float e_base = m_config->base_eval(hyp, current);
		float e_base = baseRegion.optimization_energy(i);
		//float rating = (disp_pot/5.0f -2.0f) * baseRegion.stats.stddev/stddev + e_base;
		float rating = m_config->optimizer.prop_eval(baseRegion, base, match, currentDisp);
		//output
		QStringList item;
		item << QString::number(i+dispMin);
		item << QString::number(avg_disp);
		item << QString::number(disp_pot);
		item << QString::number(hyp.neighbor_color_pot);
		item << QString::number(hyp.costs);
		item << QString::number(hyp.occ_avg);

		if(!it->empty())
			item << QString::number(rating);
		else
			item << QString::number(30.0f);
		item << QString::number(e_base);
		item << QString::number(e_other);
		tree->addTopLevelItem(new QTreeWidgetItem(item));
		i++;
	}
}

void RegionWidget::neighborTree(std::vector<DisparityRegion>& regionsBase, int index, int /*dispMin*/)
{
	ui->treeNeighbors->clear();
	QStringList headers2;
	headers2 << "Nr" << "Borderlength" << "Disparity" << "color_diff" << "entropy_diff" << "Acc(base)" << "Acc(match)" << "Acc(base)2" << "Acc(match)2";;
	ui->treeNeighbors->setHeaderLabels(headers2);
	/*cv::Mat& base_costs = regionsBase[index].disparity_costs;
	int base_disparity_idx = regionsBase[index].disparity-dispMin;
	stat_t* base_stat = &(regionsBase[index].stats);*/
	for(const std::pair<std::size_t, std::size_t>& cpair : regionsBase[index].neighbors)
	{
		int cidx = cpair.first;
		/*cv::Mat& match_costs = regionsBase[cidx].disparity_costs;
		int match_disparity_idx = regionsBase[cidx].disparity-dispMin;
		stat_t* match_stat = &(regionsBase[cidx].stats);*/

		/*float accost_base_diff = base_costs.at<float>(match_disparity_idx)-base_costs.at<float>(base_disparity_idx);
		float accost_match_diff = match_costs.at<float>(base_disparity_idx)-match_costs.at<float>(match_disparity_idx);

		float range_base = base_stat->max - base_stat->min;
		float range_match = match_stat->max - match_stat->min;*/

		QStringList item;
		item << QString::number(cidx);
		item << QString::number(cpair.second);
		item << QString::number(regionsBase[cidx].disparity);
		item << QString::number(cv::norm(regionsBase[index].average_color - regionsBase[cidx].average_color));
		//item << QString::number(std::abs(regionsBase[cidx].entropy - regionsBase[index].entropy));
		//item << QString::number(m_regionsLeft[cidx]);
		//item << QString::number(accost_base_diff/range_base*100, 'f', 2) + "%";
		//item << QString::number(accost_match_diff/range_match*100, 'f', 2) + "%";
		//item << QString::number(accost_base_diff/base_costs.at<float>(base_disparity_idx)*100, 'f', 2) + "%";
		//item << QString::number(accost_match_diff/match_costs.at<float>(match_disparity_idx)*100, 'f', 2) + "%";

		ui->treeNeighbors->addTopLevelItem(new QTreeWidgetItem(item));
	}
}

void optimizeWidth(QTreeWidget *widget)
{
	for(int i = 0; i < widget->columnCount(); ++i)
		widget->resizeColumnToContents(i);
}

void RegionWidget::setData(std::shared_ptr<RegionContainer>& base, std::shared_ptr<RegionContainer>& match, int index, InitialDisparityConfig *config, bool delta)
{
	m_config = config;
	m_base = base;
	m_match = match;
	m_index = index;

	int dispMin = m_base->task.dispMin;
	m_baseDispMin = m_base->task.dispMin;
	m_matchDispMin = m_match->task.dispMin;

	std::vector<DisparityRegion>& regionsBase = m_base->regions;
	std::vector<DisparityRegion>& regionsMatch = m_match->regions;

	stat_t* pixelAnalysis = &(regionsBase[index].stats);
	QString info = QString("min: %1, max: %2, mean: %3, stddev: %4, range: %5, confidence: %6,\n maxima: %7, badmaxima: %8, disparity: %9, confidence2: %10, occ_val: %11, dilate: %12,\n confidence_range: %13, minima_variance: %14,base_disp: %15, confidence3: %16").arg(pixelAnalysis->min).arg(pixelAnalysis->max).arg(pixelAnalysis->mean).arg(pixelAnalysis->stddev).arg(pixelAnalysis->max-pixelAnalysis->min).arg(pixelAnalysis->confidence).arg(pixelAnalysis->minima.size()).arg(pixelAnalysis->bad_minima.size()).arg(pixelAnalysis->disparity_idx).arg(pixelAnalysis->confidence2).arg(pixelAnalysis->occ_val).arg(regionsBase[index].dilation).arg(pixelAnalysis->confidence_range).arg(pixelAnalysis->confidence_variance).arg(regionsBase[index].base_disparity).arg(regionsBase[index].confidence3);
	ui->stat->setText(info);

	QString maximatest;
	for(int idx : pixelAnalysis->minima)
		maximatest += QString::number(idx+dispMin)+", ";
	ui->stat2->setText(maximatest);
	if(!delta)
		ui->plot->setValues(regionsBase[index].disparity_costs.ptr<float>(0), &(regionsBase[index].stats), regionsBase[index].disparity_costs.size[0], regionsBase[index].disparity_offset);
	else
	{
		auto range = getSubrange(regionsBase[index].base_disparity, config->region_refinement_delta, m_base->task );
		int rsize = range.second - range.first + 1;
		ui->plot->setValues(regionsBase[index].disparity_costs.ptr<float>(0), &(regionsBase[index].stats), rsize, regionsBase[index].disparity_offset);
	}

	ui->lblIndex->setText("Index: " + QString::number(index));
	ui->lblDisparity->setText("Disparity: " + QString::number(regionsBase[index].disparity));
	//ui->lblEntropy->setText("Entropy: " + QString::number(regionsBase[index].entropy));
	ui->lblPixelcount->setText("Pixelcount: " + QString::number(regionsBase[index].m_size));


	warpTree(index, regionsBase[index], regionsMatch, ui->treeConnected, dispMin, regionsBase[index].disparity);
	mutualDisparity(regionsBase[index], *m_base, *m_match, ui->twMutualDisparity, dispMin);
	neighborTree(regionsBase, index, dispMin);

	//fill occlusion tree
	ui->twOcc->clear();
	/*for(unsigned int i = 0; i < regionsBase[index].occlusion.size(); ++i)
	{
		QStringList item;
		item << QString::number(i);
		item << QString::number(regionsBase[index].occlusion[i]);
		ui->twOcc->addTopLevelItem(new QTreeWidgetItem(item));
	}*/

	//resize columns
	optimizeWidth(ui->treeNeighbors);
	optimizeWidth(ui->treeConnected);
	optimizeWidth(ui->twOcc);
	optimizeWidth(ui->twMutualDisparity);

	showResultHistory(regionsBase[index]);
}

RegionWidget::~RegionWidget()
{
	delete ui;
}

void RegionWidget::on_twMutualDisparity_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	warpTree(m_index, m_base->regions[m_index], m_match->regions, ui->treeConnected, m_baseDispMin, item->text(0).toInt());
}

void RegionWidget::on_treeConnected_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	emit matchRegionSelected(item->text(0).toInt());
}

void RegionWidget::on_treeNeighbors_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	emit baseRegionSelected(item->text(0).toInt());
}

void RegionWidget::showResultHistory(DisparityRegion &region)
{
	ui->twResulthistory->clear();
	QStringList header;
	header << "costs" << "disp" << "start" << "end" << "base";
	ui->twResulthistory->setHeaderLabels(header);

	for(const EstimationStep& step : region.results)
	{
		QStringList item;
		item << QString::number(step.costs) << QString::number(step.disparity) << QString::number(step.searchrange_start) << QString::number(step.searchrange_end) << QString::number(step.base_disparity);
		ui->twResulthistory->addTopLevelItem(new QTreeWidgetItem(item));
	}

	optimizeWidth(ui->twResulthistory);
}


