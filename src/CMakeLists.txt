
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

add_executable(pshrm_test
	pshrm_test.cpp
)
set_property(TARGET pshrm_test PROPERTY CXX_STANDARD 11)
target_link_libraries(pshrm_test pshrm simpleimage)
