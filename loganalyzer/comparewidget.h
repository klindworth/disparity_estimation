#ifndef COMPAREWIDGET_H
#define COMPAREWIDGET_H

#include <QWidget>
#include <vector>


class QGridLayout;
class CVSimpleViewer;

namespace cv {
class Mat;
}

class CompareWidget : public QWidget
{
	Q_OBJECT
	
public:
	explicit CompareWidget(QWidget *parent = 0);
	~CompareWidget();
	int addRow(const QString &name, const std::vector<cv::Mat>& images, int offset = 0);
	int addRow(const QString& name, const std::vector<std::vector<int> > &hist, const std::vector<cv::Mat> &images);
	void reset(const std::vector<QString>& header);
	void insertNotesRow(const std::vector<QString> &notes);
	void setScaling(float scaling);
	
private:
	QGridLayout *gridLayout;
	std::vector<QWidget*> widgets;
	std::vector<CVSimpleViewer*> viewers;
	float m_scaling;
};

#endif // COMPAREWIDGET_H
