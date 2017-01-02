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

#include "miniplot.h"

#include <QPainter>
#include <QMouseEvent>
#include <limits>
#include <cmath>
#include <iostream>

MiniPlot::MiniPlot(QWidget *parent) :
	QWidget(parent)
{
	m_ready = false;
	m_values = nullptr;
	m_perdatapoint = 0;
	m_len = 0;
	m_margin = 3;
	m_invert = false;
	setMinimumHeight(150);
	m_marker = -1;
	setMouseTracking(true);
}

void MiniPlot::setValues(float *values, const stat_t& analysis, int len, int offset)
{
	m_offset = offset;
	m_analysis = analysis;
	m_values = values;
	m_len = len;
	update();
}

void MiniPlot::setMarker(int x)
{
	m_marker = x;
	update();
}

void MiniPlot::setInverted(bool inverted)
{
	m_invert = inverted;
	update();
}

void MiniPlot::clear()
{
	m_len = 0;
	m_ready = false;
}

void MiniPlot::paintEvent(QPaintEvent *)
{
	if(m_len > 0)
	{
		m_ready = true;
		QPainter painter(this);

		QPalette pal = QPalette();
		QColor penColor(pal.color(QPalette::WindowText));
		QPen pen;
		pen.setColor(penColor);
		painter.setPen(pen);

		float min = m_analysis.min;
		float max = m_analysis.max;

		//calculate dimensions
		m_perdatapoint = (this->width() - 2*m_margin) / (m_len-1);
		double perval = (this->height() - 2*m_margin)/(double)(max - min);

		//draw axis
		QPoint last(m_margin,m_margin);
		painter.drawLine(last, QPoint(m_margin, this->height()-m_margin));
		painter.drawLine(QPoint(this->width()-m_margin, this->height()-m_margin), QPoint(m_margin, this->height()-m_margin));

		int offset = this->height() - m_margin;

		//draw data
		if(!m_invert)
		{
			for(int i = 0; i < m_len; ++i)
			{
				int val = perval*(m_values[i]-min);
				QPoint current(m_perdatapoint*i+m_margin, offset-val);
				painter.drawLine(last, current);
				last = current;
			}
		}
		else
		{
			for(int i = m_len-1; i >= 0; --i)
			{
				int val = perval*(m_values[i]-min);
				QPoint current(m_perdatapoint*(m_len-1-i)+m_margin, offset-val);
				painter.drawLine(last, current);
				last = current;
			}
		}

		//draw marker
		if(m_marker >= 0 && m_marker < m_len)
		{
			QPen pen;
			pen.setColor(penColor);
			pen.setWidth(4);
			painter.setPen(pen);
			int val = perval*(m_values[m_marker]-min);
			int xpos;
			if(!m_invert)
				xpos = m_marker;
			else
				xpos = m_len - 1 - m_marker;
			QPoint current(m_perdatapoint*xpos+m_margin, offset-val);
			painter.drawPoint(current);
		}

		//draw mean/std
		pen.setStyle(Qt::DashLine);
		pen.setColor(penColor);
		painter.setPen(pen);

		int meanpos = offset-(m_analysis.mean-min)*perval;
		painter.drawLine(QPoint(m_margin, meanpos), QPoint(this->width()-2*m_margin, meanpos));
		int stddevDelta = m_analysis.stddev*perval;
		pen.setStyle(Qt::DotLine);
		painter.setPen(pen);
		painter.drawLine(QPoint(m_margin, meanpos-stddevDelta), QPoint(this->width()-2*m_margin, meanpos-stddevDelta));
		painter.drawLine(QPoint(m_margin, meanpos+stddevDelta), QPoint(this->width()-2*m_margin, meanpos+stddevDelta));
	}
}

int MiniPlot::getDisparity(int x)
{
	if(m_ready)
		return getValueIndex(x) + m_offset;
	return 0;
}

int MiniPlot::getValueIndex(int x)
{
	if(m_ready)
	{
		if(!m_invert)
			return ((x - m_margin+m_perdatapoint/2)/m_perdatapoint);
		else
			return(m_len - 1- (x - m_margin+m_perdatapoint/2)/m_perdatapoint);
	}
	return -1;
}

void MiniPlot::mouseMoveEvent(QMouseEvent *ev)
{
	int x = ev->x();
	int idx = getValueIndex(x);
	if(idx >= 0 && idx < m_len)
	{
		setToolTip("disparity: " + QString::number(getDisparity(x)) + ", value: " + QString::number(m_values[idx]));
		emit datapointHovered(idx);
	}
}

void MiniPlot::mouseReleaseEvent(QMouseEvent *ev)
{
	int idx = getValueIndex(ev->x());
	if(idx >= 0 && idx < m_len)
		emit datapointSelected(idx);
}
