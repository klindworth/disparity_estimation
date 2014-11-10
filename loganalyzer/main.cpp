#include "analyzer.h"
#include <QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("Kai Klindworth");
	QCoreApplication::setOrganizationDomain("kk-surf.de");
	QCoreApplication::setApplicationName("LogAnalyzer");
	Analyzer w;
	w.show();
	
	return a.exec();
}
