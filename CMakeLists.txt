cmake_minimum_required(VERSION 3.10)
project(avthp)
add_subdirectory(libjpeg-turbo)
find_path(INCLUDE_PATH libavcodec/avcodec.h)
include_directories(${INCLUDE_PATH})
add_executable(avthp dsp.c main.c)
find_library(AVCODEC_LIBRARY avcodec)
find_library(AVFORMAT_LIBRARY avformat)
find_library(AVUTIL_LIBRARY avutil)
find_library(SWSCALE_LIBRARY swscale)
find_library(SWRESAMPLE_LIBRARY swresample)
if(NOT WIN32)
  find_library(M_LIBRARY m)
endif()
target_link_libraries(avthp libjpeg-turbo
                      ${AVFORMAT_LIBRARY}
                      ${AVCODEC_LIBRARY}
                      ${SWSCALE_LIBRARY}
                      ${SWRESAMPLE_LIBRARY}
                      ${AVUTIL_LIBRARY}
                      ${M_LIBRARY})