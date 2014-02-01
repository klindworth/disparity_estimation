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

#ifndef COSTMAPVIEWER_H
#define COSTMAPVIEWER_H

#include <QWidget>
#include <opencv2/core/core.hpp>

#include "costmap_utils.h"

namespace Ui {
class CostmapViewer;
}

class CostmapViewer : public QWidget
{
	Q_OBJECT
	
public:
	explicit CostmapViewer(QWidget *parent = 0);
	void setCostMap(cv::Mat& cost_map, cv::Mat offset = cv::Mat());
	void setWindowMap(cv::Mat& cost_map);
	void setOriginals(cv::Mat& left, cv::Mat &right);
	void setWindowSize(int windowsize);
	void setInverted(bool inverted);
	~CostmapViewer();

private slots:
	void subwindowClicked(int x, int y, cv::Mat window);
	void rowSelected(int row, int col);
	void zoomClicked(int x, int y);
	void datapointSelected(int point);
	void setActivePixel(int x, int y);
	void refreshImage();

private:
	void resetSelection();

	int m_windowsize;
	int m_offsetx, m_offsety;
	DataStore2D<stat_t> m_analysis;
	cv::Mat m_costmap;
	cv::Mat m_windowMap;
	cv::Mat m_offset;
	cv::Mat m_disparity;
	int m_subsampling;
	bool m_inverted, m_bWindowMap;
	Ui::CostmapViewer *ui;
};

#endif // COSTMAPVIEWER_H
