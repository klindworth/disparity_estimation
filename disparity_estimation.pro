QT       += core gui

TARGET = disparity_estimation
TEMPLATE = app


SOURCES += main.cpp \
    genericfunctions.cpp \
    slidinghimigradient.cpp \
    debugmatstore.cpp \
    costmap_utils.cpp \
    region.cpp \
    gui/matstoreviewer.cpp \
    gui/costmapviewer.cpp \
    gui/miniplot.cpp \
    initial_disparity.cpp \
    stereotask.cpp \
    gui/regionwidget.cpp \
    gui/imagestore.cpp \
    window_size.cpp \
    disparity_utils.cpp \
    region_optimizer.cpp \
    configrun.cpp \
    taskanalysis.cpp \
    gui/regionwindow.cpp \
    segmentation.cpp \
    slidingGradient.cpp \
    region_descriptor.cpp \
    refinement.cpp

HEADERS  += \
    genericfunctions.h \
    fast_array.h \
    slidingGradient.h \
    slidinghimigradient.h \
    debugmatstore.h \
    slidingEntropy.h \
    costmap_creators.h \
    it_metrics.h \
    costmap_utils.h \
    region.h \
    gui/matstoreviewer.h \
    gui/costmapviewer.h \
    gui/miniplot.h \
    sparse_counter.h \
    initial_disparity.h \
    stereotask.h \
    gui/regionwidget.h \
    gui/imagestore.h \
    window_size.h \
    disparity_utils.h \
    intervals.h \
    region_optimizer.h \
    configrun.h \
    taskanalysis.h \
    intervals_algorithms.h \
    misc.h \
    gui/regionwindow.h \
    segmentation.h \
    region_descriptor.h \
    region_metrics.h \
    refinement.h

FORMS    += \
    gui/matstoreviewer.ui \
    gui/costmapviewer.ui \
    gui/regionwidget.ui \
    gui/imagestore.ui \
    gui/regionwindow.ui

INCLUDEPATH += gui contourRelaxation
LIBS += -lcvwidgets

unix {
	CONFIG += link_pkgconfig
	PKGCONFIG += opencv
}

QMAKE_CXXFLAGS += -std=c++0x -march=native -Wextra

QMAKE_CXXFLAGS_DEBUG += -msse3 -O0 -fsanitize=address -fno-omit-frame-pointer #-fopenmp
QMAKE_CXXFLAGS_RELEASE += -O3 -DNDEBUG -msse3 -fopenmp
QMAKE_LFLAGS_RELEASE += -fopenmp
QMAKE_LFLAGS_DEBUG += -fsanitize=address #-fopenmp
