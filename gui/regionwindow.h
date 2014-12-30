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

#ifndef REGIONWINDOW_H
#define REGIONWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <QTreeWidgetItem>
#include <memory>

class region_container;
class disparity_region;
class debug_store;
class initial_disparity_config;
class disparity_hypothesis_weight_vector;
class manual_region_optimizer;

namespace Ui {
class RegionWindow;
}

class RegionWindow : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit RegionWindow(QWidget *parent = 0);
	~RegionWindow();
	void setData(std::shared_ptr<region_container>& left, std::shared_ptr<region_container>& right);
	void setStore(debug_store* store, initial_disparity_config* config);

private slots:
	void on_treeSegments_itemDoubleClicked(QTreeWidgetItem *item, int column);
	void selectPointOnLeftDisparity(int x, int y);
	void selectPointOnRightDisparity(int x, int y);
	void selectLeftRegion(int index);
	void selectRightRegion(int index);

	void on_hsZoom_valueChanged(int value);

	void on_cbTask_currentIndexChanged(int index);

	void on_pbOptimize_clicked();

	void on_pbRefreshBase_clicked();

	void on_pbResetOptimization_clicked();

private:
	disparity_hypothesis_weight_vector get_weight_vector() const;
	void refreshImages(std::vector<disparity_region> markLeft, bool markLeftOnRight, std::vector<disparity_region> markRight, bool markRightOnLeft);
	void fillTree(int index, disparity_region& baseRegion, std::vector<disparity_region>& other_regions , QTreeWidget *tree);
	Ui::RegionWindow *ui;
	std::shared_ptr<region_container> m_left, m_right;
	debug_store *m_store;
	initial_disparity_config *m_config;

	manual_region_optimizer *optimizer;
};

#endif // REGIONWINDOW_H
