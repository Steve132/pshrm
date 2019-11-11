

add_executable(clvalidate 
	clvalidate.cpp)
set_property(TARGET clvalidate PROPERTY CXX_STANDARD 11)
target_include_directories(clvalidate PRIVATE deps)
target_link_libraries(clvalidate OpenCL)

add_library(simpleimage
	SimpleImage.hpp
	SimpleImage.cpp
)
set_property(TARGET simpleimage PROPERTY CXX_STANDARD 11)
target_include_directories(simpleimage PRIVATE deps)
target_compile_definitions(simpleimage PRIVATE -DSIMPLE_IMAGE_CIMG_SUPPORT)

add_library(pshrm
	pshrm.hpp
	pshrm.cpp
	hdr_view.cpp
	hdr_view.hpp
)
target_include_directories(pshrm PRIVATE deps)

set_property(TARGET pshrm PROPERTY CXX_STANDARD 11)
target_link_libraries(pshrm simpleimage X11 pthread)


add_custom_command(OUTPUT raytrace.cl.c COMMAND
	clvalidate -i "${CMAKE_CURRENT_SOURCE_DIR}/raytrace.cl" -oc raytrace.cl.c
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/raytrace.cl"
)



#add_custom_command(OUTPUT test.c 
#	COMMAND echo "/home/steven/master_root/projects/programming/graphics/pshrm/src/raytrace.cl:4:1: error: could not convert ‘0’ from ‘int’ to ‘SimpleImage<float>" && exit 1
#	VERBATIM
#	)

add_executable(pshrm_test
	pshrm_test.cpp
	raytrace.cl.c
	raytrace.cl
)
set_property(TARGET pshrm_test PROPERTY CXX_STANDARD 11)
target_link_libraries(pshrm_test pshrm simpleimage OpenCL)
target_include_directories(pshrm_test PRIVATE deps)

