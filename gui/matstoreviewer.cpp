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

#include "matstoreviewer.h"
#include "ui_matstoreviewer.h"

#include <iostream>

MatStoreViewer::MatStoreViewer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::MatStoreViewer)
{
	ui->setupUi(this);
	connect(ui->matchooser, SIGNAL(currentIndexChanged(int)), this, SLOT(showMat(int)));
	connect(ui->cbTaskList, SIGNAL(currentIndexChanged(int)), this, SLOT(showTask(int)));
}

void MatStoreViewer::refreshList(DebugMatStore &store)
{
	m_store = &store;
	for(auto it = store.tasks.begin(); it != store.tasks.end(); ++it)
	{
		ui->cbTaskList->addItem(it->name.c_str());
	}
}

void MatStoreViewer::refreshList(TaskStore &store)
{
	m_taskStore = &store;
	ui->matchooser->clear();
	for(std::vector<struct viewerMat>::iterator it = store.costmaps.begin(); it != store.costmaps.end(); ++it)
	{
		ui->matchooser->addItem(it->name.c_str());
	}
}

void MatStoreViewer::showMat(int idx)
{
	if(idx >= 0)
	{
		viewerMat *mat = &(m_taskStore->costmaps.at(idx));
		ui->costmapviewer->setInverted(mat->forward);
		ui->costmapviewer->setCostMap(mat->cost_map, mat->offset);
		ui->costmapviewer->setWindowSize(mat->windowsize);
		ui->costmapviewer->setOriginals(mat->left, mat->right);

		if(mat->windows.size[0] > 0)
			ui->costmapviewer->setWindowMap(mat->windows);
	}
}

void MatStoreViewer::showTask(int idx)
{
	refreshList(m_store->tasks.at(idx));
}

void MatStoreViewer::on_pbClone_clicked()
{
	MatStoreViewer *nviewer = new MatStoreViewer();
	nviewer->refreshList(*m_store);
	nviewer->show();
}

MatStoreViewer::~MatStoreViewer()
{
	delete ui;
}

