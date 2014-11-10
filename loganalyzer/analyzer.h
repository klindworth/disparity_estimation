#ifndef ANALYZER_H
#define ANALYZER_H

#include <QWidget>
#include <QDir>

class QTreeWidgetItem;

namespace Ui {
class Analyzer;
}

class Analyzer : public QWidget
{
	Q_OBJECT
	
public:
	explicit Analyzer(QWidget *parent = 0);
	void setSubTask(const QString &base, const QString &name);
	void setTask(const QString& base);
	void setTasks(QList<QTreeWidgetItem*> items);
	~Analyzer();
	
private slots:
	void on_toolButton_clicked();

	void on_pbRefresh_clicked();

	void on_pbSave_clicked();

	void on_cbCumulative_clicked();

	void on_twOverview_itemSelectionChanged();

	void on_horizontalSlider_valueChanged(int value);

private:
	QString m_currentFilename, m_notesFilename;
	QString m_text;
	QDir m_basePath, m_resultsPath;
	Ui::Analyzer *ui;
};

#endif // ANALYZER_H
