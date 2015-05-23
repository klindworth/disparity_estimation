cmake_minimum_required(VERSION 3.0)

project(disparity_estimation)

find_package(Qt5Widgets)
find_package(OpenCV REQUIRED core imgproc highgui)
find_package(Boost REQUIRED filesystem system)
find_package(HDF5 REQUIRED)
find_package(OpenMP)
find_package(Doxygen)
#include(${QT_USE_FILE})


set(SOURCES
    main.cpp
    debugmatstore.cpp
    costmap_utils.cpp
    disparity_region.cpp
    initial_disparity.cpp
    window_size.cpp
    configrun.cpp
    refinement.cpp
    region_ground_truth.cpp

    disparity_toolkit/taskanalysis.cpp
    disparity_toolkit/genericfunctions.cpp
    disparity_toolkit/stereotask.cpp
    disparity_toolkit/disparity_utils.cpp
    disparity_toolkit/task_collection.cpp

    metrics/pixelwise/gradient_disparitywise.cpp
    metrics/pixelwise/sncc_disparitywise_calculator.cpp

    optimizer/region_optimizer.cpp
    optimizer/manual_region_optimizer.cpp
    optimizer/ml_region_optimizer.cpp
    optimizer/ml_region_optimizer_disp.cpp

    gui/miniplot.cpp    
    gui/regionwidget.cpp
    gui/imagestore.cpp    
    gui/regionwindow.cpp   
)

set(HEADERS
    fast_array.h
    debugmatstore.h
    costmap_creators.h
    costmap_utils.h
    configrun.h

    region_ground_truth.h
    window_size.h
    region_metrics.h
    refinement.h
    sparse_counter.h
    initial_disparity.h

    disparity_region.h
    disparity_region_algorithms.h
    disparitywise_calculator.h

    metrics/pixelwise/sncc_disparitywise_calculator.h
    metrics/pixelwise/converter_to_region.h
    metrics/pixelwise/sad_disparitywise.h
    metrics/pixelwise/gradient_disparitywise.h
    metrics/entropy.h
    metrics/it_metrics.h
    metrics/sliding_entropy.h

    disparity_toolkit/stereotask.h
    disparity_toolkit/disparity_utils.h
    disparity_toolkit/disparity_map.h
    disparity_toolkit/disparity_range.h
    disparity_toolkit/taskanalysis.h
    disparity_toolkit/genericfunctions.h
    disparity_toolkit/task_collection.h

    optimizer/region_optimizer.h
    optimizer/manual_region_optimizer.h
    optimizer/ml_region_optimizer.h
    optimizer/ml_region_optimizer_disp.h
    optimizer/ml_region_optimizer_algorithms.h

    neural_network/network.h
    neural_network/blas_wrapper.h
    neural_network/layer.h
    neural_network/data_normalizer.h
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
    Qt5::Widgets ${OpenCV_LIBS} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} cvio cvsegmentation_lib openblas
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -Wall -Wextra -std=c++11 -O0 -fsanitize=address -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -Wall -Wextra -Wnoexcept -std=c++11 -DNDEBUG -O3")

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}_tests neural_network/tests.cpp neural_network/network.h disparity_toolkit/disparity_utils_tests.cpp disparity_toolkit/disparity_range_tests.cpp)
target_link_libraries(${PROJECT_NAME}_tests ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} openblas)

add_subdirectory(segmentation)

#analyzer

set(SOURCES_ANALYZER
    loganalyzer/main.cpp
    loganalyzer/analyzer.cpp
    loganalyzer/comparewidget.cpp
    loganalyzer/detailviewer.cpp

    disparity_toolkit/genericfunctions.cpp
    disparity_toolkit/disparity_utils.cpp
    disparity_toolkit/taskanalysis.cpp
    disparity_toolkit/stereotask.cpp
    disparity_toolkit/task_collection.cpp
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
    Qt5::Widgets ${OpenCV_LIBS} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} cvio cvsegmentation_lib
)

if(DOXYGEN_FOUND)

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
	COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating API documentation with Doxygen"
	VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
