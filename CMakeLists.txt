cmake_minimum_required(VERSION 3.0)

project(disparity_estimation)

find_package(Qt5Widgets)
find_package(OpenCV REQUIRED core imgproc highgui)
find_package(Boost REQUIRED filesystem system)
find_package(OpenMP)
#include(${QT_USE_FILE})


set(SOURCES main.cpp
    genericfunctions.cpp
    debugmatstore.cpp
    costmap_utils.cpp
    disparity_region.cpp
    initial_disparity.cpp
    stereotask.cpp
    window_size.cpp
    disparity_utils.cpp
    optimizer/region_optimizer.cpp
    configrun.cpp
    taskanalysis.cpp
    metrics/pixelwise/gradient_disparitywise.cpp
    refinement.cpp
    metrics/pixelwise/sncc_disparitywise_calculator.cpp
    optimizer/manual_region_optimizer.cpp
    optimizer/ml_region_optimizer.cpp

    gui/miniplot.cpp    
    gui/regionwidget.cpp
    gui/imagestore.cpp    
    gui/regionwindow.cpp   
)

set(HEADERS
    genericfunctions.h
    fast_array.h
    metrics/pixelwise/gradient_disparitywise.h
    debugmatstore.h
    metrics/sliding_entropy.h
    costmap_creators.h
    metrics/it_metrics.h
    costmap_utils.h
    disparity_range.h
    disparity_region.h
    disparity_region_algorithms.h
    disparitywise_calculator.h
    metrics/pixelwise/sncc_disparitywise_calculator.h
    metrics/pixelwise/converter_to_region.h
    metrics/pixelwise/sad_disparitywise.h
    metrics/entropy.h

    sparse_counter.h
    initial_disparity.h
    stereotask.h

    window_size.h
    disparity_utils.h

    optimizer/region_optimizer.h
    optimizer/manual_region_optimizer.h
    optimizer/ml_region_optimizer.h
    configrun.h
    taskanalysis.h

    region_metrics.h
    refinement.h

    neural_network/network.h
    neural_network/blas_wrapper.h
    neural_network/layer.h
)

set(QTHEADERS
    gui/miniplot.h
    gui/regionwindow.h
    gui/regionwidget.h
    gui/imagestore.h
)

set(FORMS
    gui/regionwidget.ui
    gui/imagestore.ui
    gui/regionwindow.ui
)

qt5_wrap_cpp(HEADERS_MOC ${QTHEADERS})
qt5_wrap_ui(FORMS_HEADERS ${FORMS})

include_directories(. gui neural_network loganalyzer optimizer metrics/pixelwise metrics ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${QT_DEFINITIONS})

add_executable(${PROJECT_NAME}
    ${QTHEADERS}
    ${HEADERS}
    ${SOURCES}
    ${HEADERS_MOC}
    ${FORMS_HEADERS}
)

target_link_libraries(${PROJECT_NAME}
    Qt5::Widgets ${OpenCV_LIBS} ${Boost_LIBRARIES} cvwidgets cvsegmentation_lib openblas
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -O0 -fsanitize=address -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -Wnoexcept -std=c++11 -DNDEBUG -O3")

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}_tests neural_network/tests.cpp neural_network/network.h disparity_utils_tests.cpp)
target_link_libraries(${PROJECT_NAME}_tests ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} openblas)

add_subdirectory(segmentation)

#analyzer

set(SOURCES_ANALYZER
    loganalyzer/main.cpp
    loganalyzer/analyzer.cpp
    loganalyzer/comparewidget.cpp
    loganalyzer/detailviewer.cpp

    genericfunctions.cpp
    disparity_utils.cpp
    taskanalysis.cpp
    stereotask.cpp
)

set(QTHEADERS_ANALYZER
    loganalyzer/analyzer.h
    loganalyzer/comparewidget.h
    loganalyzer/detailviewer.h
)

set(FORMS_ANALYZER
    loganalyzer/analyzer.ui
    loganalyzer/detailviewer.ui
)

qt5_wrap_ui(FORMS_ANALYZER_HEADERS ${FORMS_ANALYZER})

add_executable(${PROJECT_NAME}_loganalyzer ${SOURCES_ANALYZER} ${FORMS_ANALYZER_HEADERS})

set_target_properties(${PROJECT_NAME}_loganalyzer PROPERTIES AUTOMOC TRUE)
#set_target_properties(${PROJECT_NAME}_loganalyzer PROPERTIES AUTOUIC TRUE)

target_link_libraries(${PROJECT_NAME}_loganalyzer
    Qt5::Widgets ${OpenCV_LIBS} ${Boost_LIBRARIES} cvwidgets cvsegmentation_lib
)
