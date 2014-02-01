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

#include "imagestore.h"
#include "ui_imagestore.h"

#include "debugmatstore.h"

ImageStore::ImageStore(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::ImageStore)
{
	ui->setupUi(this);

	connect(ui->imagelist,  SIGNAL(currentIndexChanged(int)), this, SLOT(showMat(int)));
	connect(ui->cbTasklist, SIGNAL(currentIndexChanged(int)), this, SLOT(showTask(int)));
}

ImageStore::~ImageStore()
{
	delete ui;
}

void ImageStore::showMat(int idx)
{
	if(idx >= 0)
		ui->imgviewer->setCVMat(m_taskStore->simpleMatrices.at(idx).first);
}

void ImageStore::showTask(int idx)
{
	refreshList(m_store->tasks.at(idx));
}

void ImageStore::refreshList(DebugMatStore &store)
{
	m_store = &store;
	for(auto it = store.tasks.begin(); it != store.tasks.end(); ++it)
	{
		qDebug(it->name.c_str());
		ui->cbTasklist->addItem(it->name.c_str());
	}
}

void ImageStore::refreshList(TaskStore &store)
{
	qDebug("settask");
	m_taskStore = &store;

	ui->imagelist->clear();
	for(std::vector<std::pair<cv::Mat, std::string> >::iterator it = store.simpleMatrices.begin(); it != store.simpleMatrices.end(); ++it)
	{
		ui->imagelist->addItem(it->second.c_str());
	}
}

void ImageStore::on_pbClone_clicked()
{
	ImageStore *store = new ImageStore();
	store->refreshList(*m_store);
	store->show();
}

