#include "detailviewer.h"
#include "ui_detailviewer.h"

#include <QLabel>
#include <cvsimpleviewer.h>

#include <iostream>

#include <opencv2/core/core.hpp>

DetailViewer::DetailViewer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::DetailViewer)
{
	ui->setupUi(this);

	QVBoxLayout *layout = new QVBoxLayout(ui->scrollArea);
	ui->scrollArea->setLayout(layout);
	m_widgetcontainer = new QWidget();
	layout->addWidget(m_widgetcontainer);
	m_layout = new QVBoxLayout(m_widgetcontainer);
	m_layout->setSpacing(6);
	m_layout->setContentsMargins(11, 11, 11, 11);
	m_widgetcontainer->setLayout(m_layout);
	ui->scrollArea->setWidget(m_widgetcontainer);

	m_scaling = 1.0f;
}

void DetailViewer::setMatList(const std::vector<std::pair<QString, cv::Mat>>& list)
{
	for(QWidget *cwidget : m_widgets)
	{
		m_layout->removeWidget(cwidget);
		delete cwidget;
	}
	m_widgets.clear();
	m_viewer.clear();

	for(auto& item : list)
	{
		QLabel *label = new QLabel(item.first);
		m_layout->addWidget(label);
		m_widgets.push_back(label);

		if(!item.second.data)
			std::cout << "invalid data" << std::endl;

		CVSimpleViewer *viewer = new CVSimpleViewer(this);
		m_layout->addWidget(viewer);
		viewer->setCVMat(item.second, true);
		viewer->setScaling(m_scaling);
		m_widgets.push_back(viewer);
		m_viewer.push_back(viewer);
	}
	m_layout->update();
}

DetailViewer::~DetailViewer()
{
	delete ui;
}

void DetailViewer::on_hsZoom_valueChanged(int value)
{
	m_scaling = (float)value/8;
	for(SimpleImageViewer *cviewer : m_viewer)
		cviewer->setScaling(m_scaling);
	m_layout->update();
}
