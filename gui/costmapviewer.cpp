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

#include "costmapviewer.h"
#include "ui_costmapviewer.h"
#include "genericfunctions.h"
#include "costmap_utils.h"
#include "disparity_utils.h"

#include <QTableWidgetItem>
#include <QString>

#include <iostream>
#include "intervals.h"

CostmapViewer::CostmapViewer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::CostmapViewer)
{
	ui->setupUi(this);
	m_windowsize = 11;
	m_inverted = false;
	m_bWindowMap = false;
	//ui->disparityImage->setWindowSize(m_windowsize); TODO
	//connect(ui->disparityImage, SIGNAL(subwindowClicked(int,int,cv::Mat)), this, SLOT(subwindowClicked(int,int,cv::Mat)));
	ui->disparityZoom->setScaling(10.0f);
	connect(ui->table, SIGNAL(cellClicked(int,int)), this, SLOT(rowSelected(int,int)));
	connect(ui->stattable, SIGNAL(cellClicked(int,int)), this, SLOT(rowSelected(int,int)));
	connect(ui->disparityZoom, SIGNAL(mouseClicked(int,int)), this, SLOT(zoomClicked(int,int)));
	connect(ui->plot, SIGNAL(datapointSelected(int)), this, SLOT(datapointSelected(int)));
	connect(ui->cbWarp, SIGNAL(toggled(bool)), this, SLOT(refreshImage()));

	ui->disparityImage->setScaling(1.0f);
	ui->originalLeft->setScaling(0.5f);
	ui->originalRight->setScaling(0.5f);
	ui->disparityRangeViewer->setScaling(5.0f);
	ui->windowOriginal->setScaling(5.0f);

	/*ui->disparityImage->setScaling(0.25f);
	ui->originalLeft->setScaling(0.125f);
	ui->originalRight->setScaling(0.125f);
	ui->disparityRangeViewer->setScaling(5.0f);
	ui->windowOriginal->setScaling(5.0f);*/

	m_subsampling = 4;
}

void CostmapViewer::setWindowSize(int windowsize)
{
	m_windowsize = windowsize;
	//ui->disparityImage->setWindowSize(m_windowsize); TODO
}

void CostmapViewer::setOriginals(cv::Mat &left, cv::Mat& right)
{
	assert(left.dims == 2 && right.dims == 2);

	ui->originalLeft->setCVMat(left);
	ui->originalRight->setCVMat(right);
}

DataStore2D<stat_t> analyzeCostmap(const cv::Mat& src)
{
	assert(src.dims == 3);

	//cv::Mat derivedCost = deriveCostmap(src);

	DataStore2D<stat_t> stat_store(src.size[0], src.size[1]);

	float *derivedCost = new float[src.size[2]-1];

	for(int i = 0; i < src.size[0]; ++i)
	{
		for(int j = 0; j < src.size[1]; ++j)
		{
			stat_store(i,j) = stat_t();
			derivePartialCostmap(src.ptr<float>(i,j,0), derivedCost, src.size[2]);
			analyzeDisparityRange(stat_store(i,j), src.ptr<float>(i,j,0), derivedCost, src.size[2]);
		}
	}

	delete[] derivedCost;

	return stat_store;
}

void CostmapViewer::setCostMap(cv::Mat& cost_map, cv::Mat offset)
{
	assert(cost_map.dims == 3);

	resetSelection();

	m_offset = offset;
	m_costmap = cost_map;
	m_analysis = analyzeCostmap(cost_map);
	m_bWindowMap = false;

	int dispMin = 0;

	if(offset.size[0] > 0)
		dispMin = - m_costmap.size[2]/2+1;
	else
	{
		if(!m_inverted)
			dispMin = 0;
		else
			dispMin = -m_costmap.size[2]+1;
	}

	//calculate disparity
	m_disparity = createDisparity(m_costmap, dispMin, m_subsampling);

	double min,max;
	cv::minMaxIdx(m_disparity, &min, &max);
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;

	if(offset.size[0] > 0)
		m_disparity += m_offset*m_subsampling;

	cv::minMaxIdx(m_disparity, &min, &max);
	std::cout << "new" << std::endl;
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;
	std::cout << "offsets" << std::endl;
	cv::minMaxIdx(m_offset, &min, &max);
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;

	//show disparity image
	refreshImage();
}

void CostmapViewer::resetSelection()
{
	ui->stattable->clear();
	ui->table->clear();
	ui->plot->clear();

	/*ui->disparityImage->markArea(-1,-1,0,0);
	ui->originalLeft->markArea(-1,-1,0,0);
	ui->originalRight->markArea(-1,-1,0,0);*/
}

void CostmapViewer::setWindowMap(cv::Mat& cost_map)
{
	assert(cost_map.dims == 2);

	/*ui->stattable->clear();
	ui->table->clear();
	ui->plot->clear();*/

	m_windowMap = cost_map;

	m_bWindowMap = true;
}

void CostmapViewer::subwindowClicked(int x, int y, cv::Mat window)
{
	int window_x = window.cols;
	int window_y = window.rows;

	if(m_bWindowMap)
	{
		window_x = m_windowMap.at<cv::Vec2b>(y,x)[1];
		window_y = m_windowMap.at<cv::Vec2b>(y,x)[0];
	}

	//ui->disparityImage->markArea(x-window_x/2-1,y-window_y/2-1, window_x+2, window_y+2);
	//ui->originalLeft->markArea(x-window_x/2-1,y-window_y/2-1, window_x+2, window_y+2);
	ui->disparityZoom->setCVMat(window);
	ui->table->clear();
	ui->table->setRowCount(window.rows * window.cols);
	ui->table->setColumnCount(m_costmap.size[2]);
	ui->stattable->clear();
	//ui->stattable->setRowCount(window.rows * window.cols);
	//ui->stattable->setColumnCount(m_analysis.size[2]);
	m_offsetx = x-window_x/2;
	m_offsety = y-window_y/2;

	for(int i = 0; i < window.rows; ++i)
	{
		for(int j = 0; j < window.cols; ++j)
		{
			for(int k = 0; k < m_costmap.size[2]; ++k)
			{
				QTableWidgetItem *item = new QTableWidgetItem(QString::number(m_costmap.at<float>(i+m_offsety, j+m_offsetx, k)));
				ui->table->setItem(i*window.rows+j,k, item);
			}
			/*for(int k = 0; k < m_analysis.size[2]; ++k)
				{
					QTableWidgetItem *item = new QTableWidgetItem(QString::number(m_analysis.at<float>(i+m_offsety, j+m_offsetx, k)));
					ui->stattable->setItem(i*window.rows+j,k, item);
				}*/
		}
	}
	zoomClicked(window_x/2+1, window_y/2+1);
	setActivePixel(x,y);
}

template<typename T>
T clamp(const T& value, const T& min, const T& max)
{
	return std::min(std::max(value, min), max);
}

void CostmapViewer::setActivePixel(int x, int y)
{
	int window_x = m_windowsize;
	int window_y = m_windowsize;

	if(m_bWindowMap)
	{
		window_y = m_windowMap.at<cv::Vec2b>(y,x)[0];
		window_x = m_windowMap.at<cv::Vec2b>(y,x)[1];
	}

	int rangeStart;
	int rangeSize;
	int rangeEnd;

	int offset = 0;
	if(m_offset.size[0] > 0)
	{
		if(!m_inverted)
			offset = m_offset.at<short>(y,x) - m_costmap.size[2]/2;
		else
			offset = m_offset.at<short>(y,x) + m_costmap.size[2]/2;
	}

	if(m_inverted)
	{
		rangeStart = clamp(x - m_costmap.size[2] - window_x/2 + offset, 0, m_costmap.size[1]-1);
		rangeEnd = clamp(x+window_x/2 + offset, 0, m_costmap.size[1]-1);
	}
	else
	{
		rangeStart = clamp(x - window_x/2 + offset, 0, m_costmap.size[1]-1) ;
		rangeEnd = clamp(x+m_costmap.size[2] + window_x/2 + offset, 0, m_costmap.size[1]-1);
	}
	rangeSize = rangeEnd - rangeStart;
	std::cout << "rangesize: " << rangeSize - window_x << std::endl;
	/*cv::Mat disparityRange = cv::Mat(ui->originalRight->image(), cv::Range(y-window_y/2, y+window_y/2+1), cv::Range(rangeStart, rangeEnd+1));
	ui->originalRight->markArea(rangeStart,y-window_y/2-1, rangeSize, window_y+2);
	ui->disparityRangeViewer->setCVMat(disparityRange);
	cv::Mat zoomOriginal = cv::Mat(ui->originalLeft->image(), cv::Range(y-window_y/2, y+window_y/2+1), cv::Range(x-window_x/2, x+window_x/2+1));
	ui->windowOriginal->setCVMat(zoomOriginal);*/ //TODO

	/*cv::Scalar mean, stddev;
	cv::meanStdDev(zoomOriginal, mean, stddev);

	ui->patch_info->setText("stddev: " + QString::number(stddev[0]) + ", window_x: " + QString::number(window_x) + ", window_y: " + QString::number(window_y));*/

	//mark choosen disparity
	int mindisp = m_analysis(y,x).disparity_idx; //sollte an Raendern nicht funktionieren
	//ui->disparityRangeViewer->markArea(mindisp,0, window_x, window_y);
}

void CostmapViewer::zoomClicked(int x, int y)
{
	ui->table->setCurrentCell(y*m_windowsize + x,0);
	ui->stattable->setCurrentCell(y*m_windowsize + x,0);
	rowSelected(y*m_windowsize + x, 0);
}

void CostmapViewer::datapointSelected(int point)
{
	std::cout << ui->table->currentRow() << std::endl;
	ui->table->setCurrentCell(ui->table->currentRow(), point);
}

void CostmapViewer::rowSelected(int row, int col)
{
	stat_t *pixelAnalysis = m_analysis.ptr(m_offsety + row / m_windowsize, m_offsetx + row % m_windowsize);
	float *values = m_costmap.ptr<float>(m_offsety + row / m_windowsize, m_offsetx + row % m_windowsize);

	ui->plot->setValues(values, pixelAnalysis, m_costmap.size[2], 0); //FIXME

	QString info = QString("min: %1, max: %2, mean: %3, stddev: %4, range: %5, confidence: %6, maxima: %7, badmaxima: %8, disparity: %9").arg(pixelAnalysis->min).arg(pixelAnalysis->max).arg(pixelAnalysis->mean).arg(pixelAnalysis->stddev).arg(pixelAnalysis->max-pixelAnalysis->min).arg(pixelAnalysis->confidence).arg(pixelAnalysis->minima.size()).arg(pixelAnalysis->bad_minima.size()).arg(pixelAnalysis->disparity_idx);

	ui->rowinfo->setText(info);

	ui->plot->setMarker(col);
}

void CostmapViewer::setInverted(bool inverted)
{
	m_inverted = inverted;
	ui->plot->setInverted(inverted);
}

void CostmapViewer::refreshImage()
{
	resetSelection();

	if(ui->cbWarp->isChecked())
	{
		cv::Mat warped = warpDisparity<short>(m_disparity, 1.0f/m_subsampling);

		cv::Mat disparity_image = createDisparityImage(warped);
		ui->disparityImage->setCVMat(disparity_image);
	}
	else
		ui->disparityImage->setCVMat(createDisparityImage(m_disparity));
}

CostmapViewer::~CostmapViewer()
{
	delete ui;
}

