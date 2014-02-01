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

#ifndef MINIPLOT_H
#define MINIPLOT_H

#include <QWidget>

#include "costmap_utils.h"

class MiniPlot : public QWidget
{
	Q_OBJECT
public:
	explicit MiniPlot(QWidget *parent = 0);
	void setValues(float *values, stat_t *analysis, int len, int offset);
	void setInverted(bool inverted);
	void setMarker(int x);
	void clear();

protected:
	void mouseReleaseEvent(QMouseEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev);
	void paintEvent(QPaintEvent *);

private:
	int getValueIndex(int x);
	int getDisparity(int x);

	float *m_values;
	stat_t *m_analysis;
	int m_len;
	bool m_ready;
	int m_perdatapoint;
	int m_margin;
	int m_offset;
	bool m_invert;
	int m_marker;
	
signals:
	void datapointSelected(int point);
	void datapointHovered(int point);
public slots:
	
};

#endif // MINIPLOT_H
