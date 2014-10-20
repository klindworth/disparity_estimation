cmake_minimum_required(VERSION 2.8)

project(disparity_estimation)

find_package(Qt4 REQUIRED QtCore QtGui)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
include(${QT_USE_FILE})


set(SOURCES main.cpp
    genericfunctions.cpp
    slidinghimigradient.cpp
    debugmatstore.cpp
    costmap_utils.cpp
    region.cpp
    initial_disparity.cpp
    stereotask.cpp
    window_size.cpp
    disparity_utils.cpp
    region_optimizer.cpp
    configrun.cpp
    taskanalysis.cpp
    slidingGradient.cpp

    refinement.cpp
    
    
    gui/matstoreviewer.cpp
    gui/costmapviewer.cpp
    gui/miniplot.cpp    
    gui/regionwidget.cpp
    gui/imagestore.cpp    
    gui/regionwindow.cpp   

    segmentation/region_descriptor.cpp
    segmentation/segmentation.cpp
    segmentation/segmentation_cr.cpp
)


set(HEADERS
    genericfunctions.h
    fast_array.h
    slidingGradient.h
    slidinghimigradient.h
    debugmatstore.h
    slidingEntropy.h
    costmap_creators.h
    it_metrics.h
    costmap_utils.h
    region.h

    sparse_counter.h
    initial_disparity.h
    stereotask.h

    window_size.h
    disparity_utils.h

    region_optimizer.h
    configrun.h
    taskanalysis.h
    
    misc.h

    region_metrics.h
    refinement.h

    segmentation/region_descriptor.h
    segmentation/region_descriptor_algorithms.h
    segmentation/intervals_algorithms.h
    segmentation/intervals.h
    segmentation/segmentation.h
    segmentation/segmentation_cr.h
)

set(QTHEADERS
    gui/matstoreviewer.h
    gui/costmapviewer.h
    gui/miniplot.h
    gui/regionwindow.h
    gui/regionwidget.h
    gui/imagestore.h
)

set(FORMS
    gui/matstoreviewer.ui
    gui/costmapviewer.ui
    gui/regionwidget.ui
    gui/imagestore.ui
    gui/regionwindow.ui
)

qt4_wrap_cpp(HEADERS_MOC ${QTHEADERS})
qt4_wrap_ui(FORMS_HEADERS ${FORMS})
qt4_add_resources(RESOURCES_RCC ${RESOURCES})

include_directories(. gui segmentation ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${QT_DEFINITIONS})

add_executable(${PROJECT_NAME}
    ${QTHEADERS}
    ${HEADERS}
    ${SOURCES}
    ${HEADERS_MOC}
    ${FORMS_HEADERS}
    ${RESOURCES_RCC}
)

target_link_libraries(${PROJECT_NAME}
    ${QT_LIBRARIES} ${OpenCV_LIBS} cvwidgets
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -DNDEBUG -Og")
