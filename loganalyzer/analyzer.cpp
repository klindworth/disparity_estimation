#include "analyzer.h"
#include "ui_analyzer.h"

#include "comparewidget.h"

#include <QFileDialog>
#include <QDir>
#include <QTreeWidgetItem>
#include <QSettings>
#include <QTextStream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <memory>
#include <iostream>

#include "disparity_toolkit/genericfunctions.h"
#include "disparity_toolkit/taskanalysis.h"
#include "disparity_toolkit/disparity_utils.h"
#include "disparity_toolkit/stereotask.h"
#include "detailviewer.h"

#include <cvio/hdf5internals.h>

/*#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>*/

Analyzer::Analyzer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::Analyzer)
{
	ui->setupUi(this);

	QStringList header;
	header << "Name" << "Task" << "Config" << "window" << "quantizer" << "Segmentation" << "runtime" << "optimization" << "dilate" << "dilate_step" << "refinement" << "damping" << "color_var" << "spatial var" << "size" << "compact" << "deltaDisp";

	ui->twOverview->setHeaderLabels(header);
	ui->twOverview->setColumnCount(header.size());

	ui->pbSave->setVisible(false);
	ui->teNotes->setVisible(false);

	QDir dir;
	QSettings settings;
	ui->lePath->setText(settings.value("path", dir.absolutePath()).toString());

	m_basePath = QDir(ui->lePath->text());
	m_resultsPath = m_basePath;
	m_resultsPath.cd("results");

	on_pbRefresh_clicked();
}

Analyzer::~Analyzer()
{
	delete ui;
}

void optimizeWidth(QTreeWidget *widget)
{
	for(int i = 0; i < widget->columnCount(); ++i)
		widget->resizeColumnToContents(i);
}

void Analyzer::on_toolButton_clicked()
{
	QString path = QFileDialog::getExistingDirectory(this, "Choose dir", ui->lePath->text());
	if(!path.isEmpty())
	{
		QSettings settings;
		settings.setValue("path", path);

		ui->lePath->setText(path);
		on_pbRefresh_clicked();
	}
}

template<typename T, typename storage_type>
QString read_number(const std::string& elem, const storage_type& fs)
{
	T temp;
	fs[elem] >> temp;
	return QString::number(temp);
}

template<typename T, typename storage_type>
void read_numbers(const storage_type& fs, QStringList& list, std::initializer_list<std::string> elems)
{
	for(const std::string& elem : elems)
		list << read_number<T>(elem, fs);
}

template<typename storage_type>
QString read_string(const std::string& elem, const storage_type& fs)
{
	std::string temp;
	fs[elem] >> temp;
	return QString(temp.c_str());
}

void addHist(QStringList& item, std::vector<int>& hist, bool bcumulative)
{
	int sum = std::accumulate(hist.begin(), hist.end(), 0);

	float mean = 0.0f;
	float cumulative = 0.0f;
	for(std::size_t j = 0; j < hist.size(); ++j)
	{
		float percent = (float)hist[j]/sum;
		cumulative += percent;

		if(bcumulative)
			item << QString::number(cumulative*100, 'g', 3) + "%";
		else
			item << QString::number(percent*100, 'g', 3) + "%";

		mean += j*percent;
	}
	item << QString::number(mean, 'g', 5);
}

void addSubItem(QStringList& subitem, const cv::FileNode& node)
{
	subitem << read_string("taskname", node);
	subitem << "" << "" << "";
	subitem << read_number<int>("total_runtime", node);

	for(int j = 0; j < 4; ++j)
		subitem << "";
}

void Analyzer::on_pbRefresh_clicked()
{
	m_basePath = QDir(ui->lePath->text());
	m_resultsPath = m_basePath;
	m_resultsPath.cd("results");

	ui->twOverview->clear();

	QStringList filter;
	filter << "*.yml";

	QStringList files = m_resultsPath.entryList(filter);

	for(int i = 0; i < files.size(); ++i)
	{
		//qDebug(dir.absoluteFilePath(files.at(i)).toAscii());
		try{
			QStringList item;
			QString filenameYML = files.at(i);
			item << filenameYML.left(filenameYML.size() - 4);

			cv::FileStorage fs(m_resultsPath.absoluteFilePath(files.at(i)).toStdString(), cv::FileStorage::READ);
			if(!fs.isOpened())
				qDebug("not");
			//item << readString("left", fs) + ", " + readString("right", fs);
			item << read_string("testset", fs);
			item << read_string("configname", fs);
			QString segm = read_string("segmentation", fs);
			if(!segm.isEmpty())
			{
				QString subitem = read_number<int>("min_windowsize", fs) + "-" + read_number<int>("max_windowsize", fs);
				item << subitem;
			}
			else
				item << read_number<int>("windowsize", fs);
			item << read_number<int>("quantizer", fs);
			item << segm;
			//item << readInt("total_runtime", fs);
			item << "";
			read_numbers<int>(fs, item, {"optimization_rounds", "dilate", "dilate_step", "refinement", "enable_damping"});
			item << read_number<double>("color_var", fs);
			item << read_number<int>("spatial_var", fs);
			item << read_number<int>("superpixel_size" ,fs);
			item << read_number<double>("superpixel_compactness", fs);
			item << read_number<int>("deltaDisp", fs);

			int total_runtime = 0;
			QTreeWidgetItem* parent = new QTreeWidgetItem(item);

			cv::FileNode node = fs["analysis"];
			for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
			{
				QStringList subitem_left;
				QStringList subitem_right;

				addSubItem(subitem_left, *it);
				addSubItem(subitem_right, *it);

				int runtime;
				(*it)["total_runtime"] >> runtime;
				total_runtime += runtime;

				subitem_left[1] = "forward";
				subitem_right[1] = "backward";
				new QTreeWidgetItem(parent, subitem_left);
				new QTreeWidgetItem(parent, subitem_right);
			}
			parent->setText(6, QString::number(total_runtime));

			ui->twOverview->addTopLevelItem(parent);
			//parent->setExpanded(true);
		}
		catch(cv::Exception ex)
		{
			qDebug("exception caught:");
			qDebug(ex.msg.c_str());
		}
	}
	optimizeWidth(ui->twOverview);
}

stereo_task load_task(cv::FileNodeIterator it, std::string name, std::string base_folder)
{
	/*std::string leftGround, rightGround, left, right, leftOcc, rightOcc;
	int dispRange, groundSubsampling;
	(*it)["left"] >> left;
	(*it)["right"] >> right;
	(*it)["occLeft"] >> leftOcc;
	(*it)["occRight"] >> rightOcc;
	(*it)["groundLeft"] >> leftGround;
	(*it)["groundRight"] >> rightGround;
	(*it)["dispRange"] >> dispRange;
	(*it)["groundTruthSubsampling"] >> groundSubsampling;

	std::vector<std::string*> pathToTransform {&leftGround, &rightGround, &left, &right, &leftOcc, &rightOcc};
	for(std::string* cpath : pathToTransform)
		*cpath = base_folder + "/" + *cpath;

	return stereo_task(name, left, right, leftGround, rightGround, leftOcc, rightOcc, groundSubsampling, dispRange);*/

	return stereo_task::load_from_filestorage(*it, base_folder);
}

disparity_map load_disparity(cvio::hdf5file& hfile, const std::string& path, int default_subsampling)
{
	cvio::hdf5dataset dataset_right(hfile, path);
	int subsampling = dataset_right.get_attribute("sampling", default_subsampling);
	std::cout << "subsampling_right: " << subsampling << std::endl;

	return disparity_map(hfile.load(path), subsampling);
}

void Analyzer::setSubTask(const QString& base, const QString& name)
{
	m_currentFilename = base + ".yml";

	std::cout << m_currentFilename.toStdString() << std::endl;

	cv::FileStorage fs(m_resultsPath.absoluteFilePath(m_currentFilename).toStdString(), cv::FileStorage::READ);

	QString prefix = base+ "_" + name;

	ui->compare->reset(std::vector<QString>{base});

	int windowsize, subsampling;
	fs["windowsize"] >> windowsize;
	fs["subsampling"] >> subsampling;
	std::cout << "subsampling: " << subsampling << std::endl;
	if(subsampling == 0)
		subsampling = 1;
	cv::FileNode node = fs["analysis"];
	for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
	{
		QString cname = read_string("taskname", *it);
		if(cname == name)
		{
			std::cout << "found" << std::endl;

			std::string result_path = m_resultsPath.absolutePath().toStdString();
			std::string abs_prefix = result_path + "/" + prefix.toStdString();

			stereo_task task = load_task(it, name.toStdString(), m_basePath.absolutePath().toStdString());

			std::cout << abs_prefix << std::endl;
			cvio::hdf5file hfile(abs_prefix + ".hdf5");

			disparity_map disp_left  = load_disparity(hfile, "/results/left_disparity",  subsampling);
			disparity_map disp_right = load_disparity(hfile, "/results/right_disparity", subsampling);

			task_analysis analysis(task, disp_left, disp_right, windowsize/2);

			cv::Mat warpedLeft   = disparity::warp_image<cv::Vec3b>(task.left,  disp_left,  cv::Vec3b(0,0,0), 1.0f/disp_left.sampling);
			cv::Mat warpedRight  = disparity::warp_image<cv::Vec3b>(task.right, disp_right, cv::Vec3b(0,0,0), 1.0f/disp_right.sampling);

			cv::Mat disp_left_img  = disparity::create_image(disp_left);
			cv::Mat disp_right_img = disparity::create_image(disp_right);
			cv::Mat disp_left_img_color, disp_right_img_color;
			cv::cvtColor(disp_left_img, disp_left_img_color, CV_GRAY2BGR);
			cv::cvtColor(disp_right_img, disp_right_img_color, CV_GRAY2BGR);

			//PCL
			/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc (new pcl::PointCloud<pcl::PointXYZRGB>());
			pc->height = disp_left.rows;
			pc->width = disp_left.cols;
			pc->is_dense = true;
	std::stringstream stream = generate_latex(rows, setnames);

			int total_points = task.left.total();
			pc->reserve(total_points);

			for(int y = 0; y < disp_left.rows; ++y)
			{
				for(int x = 0; x < disp_left.cols; ++x)
				{
					cv::Vec3b ccolor = task.left.at<cv::Vec3b>(y,x);
					short disp = disp_left.at<short>(y,x);

					pcl::PointXYZRGB point(ccolor[2], ccolor[1], ccolor[0]);
					point.x = x;
					point.y = y;
					point.z = (task.dispRange-std::abs(disp))*3;
					pc->push_back(point);
				}
			}

			pcl::visualization::CloudViewer viewerpc("Cloud Viewer");
			viewerpc.showCloud(pc);

			while (!viewerpc.wasStopped ())
				;
			*/
			//PCL-END

			std::vector<std::pair<QString, cv::Mat>> images;
			images.push_back(std::make_pair("left", task.left));
			images.push_back(std::make_pair("right", task.right));
			images.push_back(std::make_pair("left disp", disp_left_img));
			images.push_back(std::make_pair("right disp", disp_right_img));
			images.push_back(std::make_pair("left error", analysis.diff_mat_left));
			images.push_back(std::make_pair("right error", analysis.diff_mat_right));
			images.push_back(std::make_pair("left warped to right", warpedLeft));
			images.push_back(std::make_pair("right warped to left", warpedRight));

			bool mix = false;
			cv::Mat mixedLeft, mixedRight;
			if(mix)
			{
				mixedLeft  = 0.5*task.left  + 0.5*warpedRight;
				mixedRight = 0.5*task.right + 0.5*warpedLeft;
			}
			else
			{
				mixedLeft  = task.left  + warpedRight;
				mixedRight = task.right + warpedLeft;
			}
			images.push_back(std::make_pair("left mix", mixedLeft));
			images.push_back(std::make_pair("right mix", mixedRight));

			cv::Mat img_disp_left_overlay  = 0.7*disp_left_img_color  + 0.3*task.left;
			cv::Mat img_disp_right_overlay = 0.7*disp_right_img_color + 0.3*task.right;
			images.push_back(std::make_pair("left disp image overlay", img_disp_left_overlay));
			images.push_back(std::make_pair("right disp image overlay", img_disp_right_overlay));

			std::cout << "images added " << images.size() << std::endl;

			DetailViewer *viewer = new DetailViewer();
			viewer->setMatList(images);
			viewer->show();

			//mainwindow
			std::vector<cv::Mat> imagesLeft {disp_left_img, analysis.diff_mat_left};
			std::vector<cv::Mat> imagesRight {disp_right_img, analysis.diff_mat_right};
			ui->compare->addRow(name + " (forward)", imagesLeft);
			ui->compare->addRow(name + " (backward)", imagesRight);
		}
	}
}

void Analyzer::setTask(const QString& base)
{
	ui->compare->reset(std::vector<QString>{base});

	m_currentFilename = base + ".yml";
	m_notesFilename = base + ".txt";

	if(m_resultsPath.exists(m_notesFilename))
	{
		QFile file(m_resultsPath.absoluteFilePath(m_notesFilename));
		file.open(QFile::ReadOnly);
		QTextStream noteStream(&file);
		ui->teNotes->setText(noteStream.readAll());
	}
	else
		ui->teNotes->setText("");
	ui->pbSave->setVisible(true);
	ui->teNotes->setVisible(true);
}

class CompareRow
{
public:
	CompareRow(const QString& pname) : name(pname) {}
	QString name;
	std::vector<std::vector<int>> hist;
	std::vector<cv::Mat> images;
};

CompareRow* getElement(std::vector<std::unique_ptr<CompareRow>>&rows, const QString& name)
{
	auto it = std::find_if(rows.begin(), rows.end(), [&](const std::unique_ptr<CompareRow>& row){return name == row->name;});

	if(it == rows.end())
	{
		rows.push_back(std::unique_ptr<CompareRow>(new CompareRow(name)));
		return rows.back().get();
	}
	else
		return it->get();
}

void Analyzer::on_pbSave_clicked()
{
	if(!m_notesFilename.isEmpty())
	{
		QFile file(m_resultsPath.absoluteFilePath(m_notesFilename));
		file.open(QFile::WriteOnly | QFile::Truncate);
		QTextStream stream(&file);
		stream << ui->teNotes->toPlainText();
	}
}

void Analyzer::on_cbCumulative_clicked()
{
	on_pbRefresh_clicked();
}

std::string generate_latex(const std::vector<std::unique_ptr<CompareRow>>& rows, const std::vector<QString>& setnames)
{
	assert(rows.size() > 0);
	assert(rows[0].hist.size() > 0);
	std::stringstream stream;
	std::size_t trunc = 6;
	int cols = std::min(rows[0]->hist[0].size(), trunc);
	std::string header = "|l||";
	for(int i = 0; i < cols + 2; ++i)
		header += "l|";
	stream << "\\begin{tabular}{" << header << "}\\hline\n";
	stream << "~ & ";
	for(int i = 0; i < cols-1; ++i)
		stream << i << " & ";
	stream << "$\\geq$ " << (cols-1) << " & $\\mu$ & ~ \\\\ \\hline \n";

	for(const std::unique_ptr<CompareRow>& crow : rows)
	{
		stream << "\\hline\\multicolumn{" << cols+3 << "}{|l|}{" << crow->name.toStdString() << "}\\\\ \\hline \n";
		std::vector<std::stringstream> rowstream(crow->hist.size());
		std::vector<float> rowmeans(crow->hist.size());
		for(std::size_t i = 0; i < crow->hist.size(); ++i)
		{
			int sum = std::accumulate(crow->hist[i].begin(), crow->hist[i].end(), 0);
			int sum_trunc = std::accumulate(crow->hist[i].begin() + trunc, crow->hist[i].end(), 0);

			std::string name = setnames[i].toStdString();
			while(name.find('_') != std::string::npos)
				name.replace(name.find('_'), 1, "-");
			rowstream[i] << name;

			rowstream[i] << std::setprecision(3);
			float mean = 0.0f;
			for(int j = 0; j < cols; ++j)
			{
				float cval = (j == cols-1) ? (float)sum_trunc/sum :(float)crow->hist[i][j]/sum;
				rowstream[i] << " & " << cval*100 << "\\%";
				mean += cval*j;
			}
			rowmeans[i] = mean;
			rowstream[i] << std::setprecision(5);
			rowstream[i] << " & " << mean;
			//rowstream[i] << "\\\\ \\hline \n";
		}

		float min_mean = *(std::min_element(rowmeans.begin(), rowmeans.end()));
		for(std::size_t i = 0; i < crow->hist.size(); ++i)
		{
			if(rowmeans[i] == min_mean)
				rowstream[i] << " & - ";
			else
				rowstream[i] << " & +" << ((rowmeans[i]/min_mean)-1)*100 << " \\%";
			rowstream[i] << "\\\\ \\hline \n";
		}

		for(std::stringstream& cstream : rowstream)
			stream << cstream.str();
	}
	stream << "\\end{tabular}";

	return stream.str();
}

void Analyzer::setTasks(QList<QTreeWidgetItem*> items)
{
	std::vector<std::unique_ptr<CompareRow>> rows;
	std::vector<QString> setnames;
	std::vector<QString> notes;

	for(int k = 0; k < items.size(); ++k)
	{
		QTreeWidgetItem *item = items.at(k);

		if(!item->parent())
		{
			QString base = item->text(0);
			m_currentFilename = base + ".yml";
			QString notesFilename = base + ".txt";

			if(m_resultsPath.exists(notesFilename))
			{
				QFile file(m_resultsPath.absoluteFilePath(notesFilename));
				file.open(QFile::ReadOnly);
				QTextStream noteStream(&file);
				notes.push_back(noteStream.readAll());
			}
			else
				notes.push_back(QString());

			setnames.push_back(base);

			cv::FileStorage fs(m_resultsPath.absoluteFilePath(m_currentFilename).toStdString(), cv::FileStorage::READ);

			int windowsize, subsampling;
			fs["windowsize"] >> windowsize;
			fs["subsampling"] >> subsampling;
			if(subsampling == 0)
				subsampling = 1;
			std::cout << "subsampling: " << subsampling << std::endl;
			cv::FileNode node = fs["analysis"];
			for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
			{
				QString name = read_string("taskname", *it);
				QString prefix = base+ "_" + name;

				std::string base_folder = m_basePath.absolutePath().toStdString();
				std::string result_path = m_resultsPath.absolutePath().toStdString();
				std::string abs_prefix = result_path + "/" + prefix.toStdString();

				stereo_task task = load_task(it, name.toStdString(), base_folder);

				std::cout << abs_prefix << std::endl;

				cvio::hdf5file hfile(abs_prefix + ".hdf5");

				disparity_map disp_left  = load_disparity(hfile, "/results/left_disparity",  subsampling);
				disparity_map disp_right = load_disparity(hfile, "/results/right_disparity", subsampling);

				task_analysis analysis(task, disp_left, disp_right, windowsize/2);


				CompareRow* crow_left  = getElement(rows, name +" (left)");
				crow_left->images.push_back(disparity::create_image(disp_left));
				crow_left->images.push_back(analysis.diff_mat_left);
				crow_left->hist.push_back(analysis.error_hist_left);

				CompareRow* crow_right = getElement(rows, name +" (right)");
				crow_right->images.push_back(disparity::create_image(disp_right));
				crow_right->images.push_back(analysis.diff_mat_right);
				crow_right->hist.push_back(analysis.error_hist_right);
			}
		}
	}

	ui->compare->reset(setnames);
	ui->compare->insertNotesRow(notes);
	for(std::unique_ptr<CompareRow>& crow : rows)
	{
		ui->compare->addRow(crow->name, crow->hist, crow->images);
	}

	std::cout << generate_latex(rows, setnames) << std::endl;
}

void Analyzer::on_twOverview_itemSelectionChanged()
{
	on_pbSave_clicked();
	m_notesFilename = "";
	ui->pbSave->setVisible(false);
	ui->teNotes->setVisible(false);

	if(ui->twOverview->selectedItems().size() == 1)
	{
		QTreeWidgetItem *item = ui->twOverview->selectedItems().at(0);
		if(item->parent())
		{
			//Subtask-Mode
			QString base = item->parent()->text(0);
			setSubTask(base, item->text(0));
		}
		else
		{
			//TaskMode
			setTask(item->text(0));
			setTasks(ui->twOverview->selectedItems());
		}
	}
	else if(ui->twOverview->selectedItems().size() >= 1)
	{
		if(ui->twOverview->selectedItems().size() <= 6)
			setTasks(ui->twOverview->selectedItems());
		else
			std::cout << "to much tasks" << std::endl;
	}
}

void Analyzer::on_horizontalSlider_valueChanged(int value)
{
	ui->compare->setScaling(value/8.0f);
}
