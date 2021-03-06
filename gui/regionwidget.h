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

#ifndef REGIONWIDGET_H
#define REGIONWIDGET_H

#include <QWidget>
#include <QTreeWidgetItem>
#include <memory>

class disparity_region;
class region_container;
class initial_disparity_config;

namespace Ui {
class RegionWidget;
}

class RegionWidget : public QWidget
{
	Q_OBJECT
	
public:
	explicit RegionWidget(QWidget *parent = 0);
	void showResultHistory(disparity_region& region);
	void warpTree(int index, disparity_region& baseRegion, std::vector<disparity_region>& other_regions , QTreeWidget *tree, int dispMin, int currentDisparity);
	void mutualDisparity(disparity_region& baseRegion, region_container &base, region_container& match, QTreeWidget *tree, int dispMin);
	void optimizationViewer(disparity_region& baseRegion, region_container &base, region_container& match, QTreeWidget *tree, int dispMin);
	void neighborTree(std::vector<disparity_region>& regionsBase, int index, int dispMin);
	void setData(std::shared_ptr<region_container>& base, std::shared_ptr<region_container>& match, int index, initial_disparity_config *config, bool delta);
	void setInverted(bool inverted);
	~RegionWidget();

signals:
	void matchRegionSelected(int index);
	void baseRegionSelected(int index);

private slots:
	void on_twMutualDisparity_itemDoubleClicked(QTreeWidgetItem *item, int column);

	void on_treeConnected_itemDoubleClicked(QTreeWidgetItem *item, int column);

	void on_treeNeighbors_itemDoubleClicked(QTreeWidgetItem *item, int column);

private:
	std::shared_ptr<region_container> m_base, m_match;
	int m_index;
	initial_disparity_config *m_config;
	Ui::RegionWidget *ui;
};

#endif // REGIONWIDGET_H
