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
#include "disparity_region.h"
#include "disparity_region_algorithms.h"
#include <segmentation/intervals.h>

#include "region_optimizer.h"
#include "configrun.h"
#include <segmentation/intervals_algorithms.h>
#include "initial_disparity.h"
#include "disparity_range.h"

#include <opencv2/highgui/highgui.hpp>

#include "manual_region_optimizer.h"

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

void RegionWidget::warpTree(int index, disparity_region& baseRegion, std::vector<disparity_region>& other_regions , QTreeWidget *tree, int dispMin, int currentDisparity)
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
	for(const corresponding_region& cregion : baseRegion.corresponding_regions[currentDisparity-dispMin])
	{
		disparity_region& matchRegion = other_regions[cregion.index];
		double mutual_percent = cregion.percent;
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
		matchItem << QString::number(matchRegion.get_corresponding_region(index, -currentDisparity- m_match->task.dispMin).percent*100, 'f', 2) + "%";
		//matchItem << QString::number(matchRegion.stats.stddev/baseRegion.stats.stddev);
		matchItem << "-";
		if(matchRegion.optimization_energy.data)
			matchItem << QString::number(matchRegion.optimization_energy(-currentDisparity-m_match->task.dispMin));
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

void RegionWidget::mutualDisparity(disparity_region& baseRegion, region_container& base, region_container& match, QTreeWidget *tree, int dispMin)
{
	std::vector<disparity_region>& other_regions = match.regions;

	tree->clear();
	QStringList header;
	header << "Disp" << "Other Disp" << "lr_pot" << "neigh_color_pot" << "costs" << "own_occ" << "warp_costs" << /*"E_base" <<*/ "E" << "E_own" << "E_Other";
	tree->setColumnCount(header.size());
	tree->setHeaderLabels(header);

	int pot_trunc = 15;

	manual_optimizer_feature_calculator dhv(base, match);
	int dispMax = dispMin + baseRegion.corresponding_regions.size()-1;
	disparity_range range(dispMin, dispMax);
	dhv.update(pot_trunc, baseRegion, range);
	stat_t cstat;
	generate_stats(baseRegion, cstat);

	int i = 0;
	for(auto it = baseRegion.corresponding_regions.begin(); it != baseRegion.corresponding_regions.end(); ++it)
	{
		disparity_hypothesis hyp = dhv.get_disparity_hypothesis(i);
		short currentDisp = i + dispMin;

		float avg_disp = corresponding_regions_average(other_regions, *it, [](const disparity_region& cregion){return (float)cregion.disparity;});
		float e_other = baseRegion.optimization_energy.data ? corresponding_regions_average(other_regions, *it, [&](const disparity_region& cregion){return cregion.optimization_energy(-currentDisp-m_match->task.dispMin);}) : 0;

		//ratings
		//float stddev_dev = baseRegion.stats.stddev-stddev;
		//float disp_dev = std::abs(currentDisp+avg_disp);

		//float e_base = m_config->base_eval(hyp, current);
		float e_base = baseRegion.optimization_energy.data ? baseRegion.optimization_energy(i) : 0;
		//float rating = (disp_pot/5.0f -2.0f) * baseRegion.stats.stddev/stddev + e_base;
		float rating = baseRegion.optimization_energy.data ? m_config->optimizer.prop_eval(baseRegion, base, match, currentDisp, cstat) : 0;
		//output
		QStringList item;
		item << QString::number(i+dispMin);
		item << QString::number(avg_disp);
		item << QString::number(hyp.lr_pot);
		item << QString::number(hyp.neighbor_color_pot);
		item << QString::number(hyp.costs);
		item << QString::number(hyp.occ_avg);
		item << QString::number(hyp.warp_costs);

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

void RegionWidget::neighborTree(std::vector<disparity_region>& regionsBase, int index, int /*dispMin*/)
{
	ui->treeNeighbors->clear();
	QStringList headers2;
	headers2 << "Nr" << "Borderlength" << "Disparity" << "color_diff";
	ui->treeNeighbors->setHeaderLabels(headers2);
	for(const auto& cpair : regionsBase[index].neighbors)
	{
		int cidx = cpair.first;
		QStringList item;
		item << QString::number(cidx);
		item << QString::number(cpair.second);
		item << QString::number(regionsBase[cidx].disparity);
		item << QString::number(cv::norm(regionsBase[index].average_color - regionsBase[cidx].average_color));

		ui->treeNeighbors->addTopLevelItem(new QTreeWidgetItem(item));
	}
}

void optimizeWidth(QTreeWidget *widget)
{
	for(int i = 0; i < widget->columnCount(); ++i)
		widget->resizeColumnToContents(i);
}

void RegionWidget::setData(std::shared_ptr<region_container>& base, std::shared_ptr<region_container>& match, int index, initial_disparity_config *config, bool delta)
{
	m_config = config;
	m_base = base;
	m_match = match;
	m_index = index;

	int dispMin = m_base->task.dispMin;

	std::vector<disparity_region>& regionsBase = m_base->regions;
	std::vector<disparity_region>& regionsMatch = m_match->regions;

	stat_t cstat;
	generate_stats(regionsBase[index], cstat);
	QString info = QString("min: %1, max: %2, mean: %3, stddev: %4, range: %5,\n disparity: %6, confidence2: %7, \n confidence_range: %8, minima_variance: %9,base_disp: %10").arg(cstat.min).arg(cstat.max).arg(cstat.mean).arg(cstat.stddev).arg(cstat.max-cstat.min).arg(cstat.disparity_idx).arg(cstat.confidence2).arg(cstat.confidence_range).arg(cstat.confidence_variance).arg(regionsBase[index].base_disparity);
	ui->stat->setText(info);

	if(!delta)
		ui->plot->setValues(regionsBase[index].disparity_costs.ptr<float>(0), cstat, regionsBase[index].disparity_costs.size[0], regionsBase[index].disparity_offset);
	else
	{
		disparity_range drange = task_subrange(m_base->task, regionsBase[index].base_disparity, config->region_refinement_delta);
		ui->plot->setValues(regionsBase[index].disparity_costs.ptr<float>(0), cstat, drange.size(), regionsBase[index].disparity_offset);
	}

	ui->lblIndex->setText("Index: " + QString::number(index));
	ui->lblDisparity->setText("Disparity: " + QString::number(regionsBase[index].disparity));
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
	warpTree(m_index, m_base->regions[m_index], m_match->regions, ui->treeConnected, m_base->task.dispMin, item->text(0).toInt());
}

void RegionWidget::on_treeConnected_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	emit matchRegionSelected(item->text(0).toInt());
}

void RegionWidget::on_treeNeighbors_itemDoubleClicked(QTreeWidgetItem *item, int /*column*/)
{
	emit baseRegionSelected(item->text(0).toInt());
}

void RegionWidget::showResultHistory(disparity_region& /*region*/)
{
	ui->twResulthistory->clear();
	QStringList header;
	header << "costs" << "disp" << "start" << "end" << "base";
	ui->twResulthistory->setHeaderLabels(header);

	/*for(const EstimationStep& step : region.results)
	{
		QStringList item;
		item << QString::number(step.costs) << QString::number(step.disparity) << QString::number(step.searchrange_start) << QString::number(step.searchrange_end) << QString::number(step.base_disparity);
		ui->twResulthistory->addTopLevelItem(new QTreeWidgetItem(item));
	}*/

	optimizeWidth(ui->twResulthistory);
}


