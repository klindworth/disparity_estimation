#ifndef DETAILVIEWER_H
#define DETAILVIEWER_H

#include <QWidget>
#include <QVBoxLayout>

namespace Ui {
class DetailViewer;
}

namespace cv {
	class Mat;
}

class CVSimpleViewer;

class DetailViewer : public QWidget
{
	Q_OBJECT
	
public:
	explicit DetailViewer(QWidget *parent = 0);
	void setMatList(const std::vector<std::pair<QString, cv::Mat>>& list);
	~DetailViewer();
	
private slots:
	void on_hsZoom_valueChanged(int value);

private:
	float m_scaling;
	std::vector<QWidget*> m_widgets;
	QWidget *m_widgetcontainer;
	std::vector<CVSimpleViewer*> m_viewer;
	QVBoxLayout* m_layout;
	Ui::DetailViewer *ui;
};

#endif // DETAILVIEWER_H
