set(MODULE_NAME apps)
file(GLOB HEADER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB SOURCE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${MODULE_NAME} ${HEADER_LIST} ${SOURCE_LIST})
target_include_directories(${MODULE_NAME} PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include" 
                                            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${MODULE_NAME} PUBLIC cuda_bundle gst_bundle jsonlib_bundle libsig_bundle
nvdslib_bundle nvds_common nvds_infer opencv_bundle polyclipping json-glib-1.0
spdlog_bundle yaml-cpp config logger request poco_libs common manager processor)
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20)
