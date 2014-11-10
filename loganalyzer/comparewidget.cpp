#include "comparewidget.h"

#include <QLabel>
#include <QGridLayout>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTableWidget>
#include <QTextEdit>

#include <cvsimpleviewer.h>

#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

CompareWidget::CompareWidget(QWidget *parent) :
	QWidget(parent)
{
	gridLayout = new QGridLayout(this);
	gridLayout->setSpacing(6);
	gridLayout->setContentsMargins(11, 11, 11, 11);
	gridLayout->setObjectName(QString::fromUtf8("gridLayout"));

	m_scaling = 1.0f;
}

void CompareWidget::reset(const std::vector<QString> &header)
{
	for(QWidget* cwidget : widgets)
	{
		gridLayout->removeWidget(cwidget);
		cwidget->deleteLater();
	}
	widgets.clear();

	for(CVSimpleViewer* cviewer : viewers)
	{
		gridLayout->removeWidget(cviewer);
		delete cviewer;
	}
	viewers.clear();

	for(std::size_t i = 0; i < header.size(); ++i)
	{
		QLabel *label = new QLabel("<b>" + header[i] + "</b>", this);
		gridLayout->addWidget(label, 0, i*2+1, 1, 2);
		widgets.push_back(label);
	}
}

void CompareWidget::insertNotesRow(const std::vector<QString> &notes)
{
	int row = gridLayout->rowCount();
	for(std::size_t i = 0; i < notes.size(); ++i)
	{
		QTextEdit *label = new QTextEdit(this);
		label->setPlainText(notes[i]);
		label->setReadOnly(true);
		gridLayout->addWidget(label, row, i*2+1, 1, 2);
		widgets.push_back(label);
	}
}

int CompareWidget::addRow(const QString& name, const std::vector<cv::Mat> &images, int offset)
{
	QLabel *header = new QLabel(name, this);
	gridLayout->addWidget(header, gridLayout->rowCount(), 0, 1, 2);
	widgets.push_back(header);

	int row = gridLayout->rowCount();

	for(std::size_t i = 0; i < images.size(); ++i)
	{
		CVSimpleViewer *viewer = new CVSimpleViewer(this);
		viewer->setScaling(m_scaling);
		//if(!images[i].isEmpty())
		viewer->setCVMat(images[i], true);

		gridLayout->addWidget(viewer, row, i + offset);
		widgets.push_back(viewer);
		viewers.push_back(viewer);
	}

	return row;
}

int CompareWidget::addRow(const QString& name,const std::vector<std::vector<int> >& hist, const std::vector<cv::Mat> &images)
{
	if(hist.size() > 0)
	{
		int row = addRow(name, images, 1);

		QTableWidget *table = new QTableWidget(hist[0].size()+1, hist.size(), this);
		QStringList vertHeaders;
		vertHeaders << "mean";
		for(std::size_t i = 0; i < hist[0].size(); ++i)
			vertHeaders << QString::number(i);
		table->setVerticalHeaderLabels(vertHeaders);
		table->setMinimumWidth(325);

		gridLayout->addWidget(table, row, 0);
		bool bcumulative = false;
		//hist
		for(std::size_t i = 0; i < hist.size(); ++i)
		{
			int sum = std::accumulate(hist[i].begin(), hist[i].end(), 0);

			float mean = 0.0f;
			float cumulative = 0.0f;
			for(std::size_t j = 0; j < hist[i].size(); ++j)
			{
				float percent = (float)hist[i][j]/sum;
				cumulative += percent;

				QTableWidgetItem *item = new QTableWidgetItem();
				if(bcumulative)
					item->setText(QString::number(cumulative*100, 'g', 3) + "%");
				else
					item->setText(QString::number(percent*100, 'g', 3) + "%");

				mean += j*percent;
				table->setItem(j+1, i, item);
			}
			QTableWidgetItem *item = new QTableWidgetItem(QString::number(mean, 'g', 5));
			table->setItem(0, i, item);

			widgets.push_back(table);
		}

		gridLayout->update();
		//
		return row;
	}
	return 0;
}

void CompareWidget::setScaling(float scaling)
{
	m_scaling = scaling;

	for(CVSimpleViewer *cviewer : viewers)
		cviewer->setScaling(scaling);
}

CompareWidget::~CompareWidget()
{
}
