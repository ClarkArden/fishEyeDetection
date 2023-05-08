cmake_minimum_required(VERSION 3.6) 
PROJECT (fishEyeDect)
find_package(OpenCV REQUIRED )
# set( NAME_SRC
#     src/*.cc
# )

# set( NAME_HEADERS       
#      include/*.h
#      include/*.hpp
# )

file( GLOB NAME_SRC
    ${PROJECT_SOURCE_DIR}/src/*.cc;  
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

file( GLOB NAME_HEADERS
    ${PROJECT_SOURCE_DIR}/include/*.hpp;
    ${PROJECT_SOURCE_DIR}/include/*.h

) 

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( fishEyeDect ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( fishEyeDect ${OpenCV_LIBS} )