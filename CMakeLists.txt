cmake_minimum_required(VERSION 3.18)

PROJECT(instant-ngp
        VERSION 1.0
        DESCRIPTION "Instant neural graphics primitives"
        LANGUAGES C CXX CUDA
        )
set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

if (NOT NGP_DEPLOY)
    set(NGP_VERSION "${NGP_VERSION}dev")
endif ()

set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)

option(NGP_BUILD_WITH_GUI "Build with GUI support (requires GLFW and GLEW)?" ON)
option(NGP_BUILD_WITH_VULKAN "Build with Vulkan to enable DLSS support?" ON)
option(NGP_BUILD_WITH_OPTIX "Build with OptiX to enable hardware ray tracing?" ON)

option(USE_TORCH "Use Torch C++" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instant-ngp/dependencies/glfw/CMakeLists.txt")
    message(FATAL_ERROR
            "Some instant-ngp dependencies are missing. "
            "If you forgot the \"--recursive\" flag when cloning this project, "
            "this can be fixed by calling \"git submodule update --init --recursive\"."
            )
endif ()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else ()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fopenmp")
endif ()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################

################
# tiny-cuda-nn #
################

set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(instant-ngp/dependencies/tiny-cuda-nn)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

if (NGP_BUILD_WITH_GUI)
    find_package(Vulkan)
    if (Vulkan_FOUND AND NGP_BUILD_WITH_VULKAN)
        set(NGP_VULKAN ON)
        list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
        include_directories("${Vulkan_INCLUDE_DIRS}")
        set(NGP_LIBRARIES ${NGP_LIBRARIES} ${Vulkan_LIBRARIES})

        set(GUI_SOURCES
                ${GUI_SOURCES}
                instant-ngp/src/dlss.cu
                )

        list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
        list(APPEND NGP_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
        set(NGP_LIBRARIES ${NGP_LIBRARIES} ${Vulkan_LIBRARIES})

        list(APPEND GUI_SOURCES instant-ngp/src/dlss.cu)
        include_directories("instant-ngp/dependencies/dlss/include")

        # DLSS depends on vulkan, so appears here
        list(APPEND NGP_INCLUDE_DIRECTORIES "instant-ngp/dependencies/dlss/include")

        if (MSVC)
            list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/instant-ngp/dependencies/dlss/lib/Windows_x86_64/x86_64")
            if (CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d_dbg)
            else ()
                set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d)
            endif ()
        else ()
            list(APPEND NGP_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/instant-ngp/dependencies/dlss/lib/Linux_x86_64")
            set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx)
        endif ()
    else ()
        set(NGP_VULKAN OFF)
        if (NGP_BUILD_WITH_VULKAN)
            message(WARNING
                    "Vulkan was not found. Neural graphics primitives will still compile "
                    "and run correctly, but DLSS will not be supported."
                    )
        endif ()
    endif ()

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
    set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
    set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
    set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

    add_subdirectory(instant-ngp/dependencies/glfw)

    set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

    mark_as_advanced(
            GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
            GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
            GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
            BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
    )

    include_directories("instant-ngp/dependencies/glfw/include")
    include_directories("instant-ngp/dependencies/imgui")

    list(APPEND NGP_INCLUDE_DIRECTORIES "instant-ngp/dependencies/glfw/include" "instant-ngp/dependencies/imgui")

    if (MSVC)
        include_directories("instant-ngp/dependencies/gl3w")
        link_directories("instant-ngp/dependencies")
        list(APPEND NGP_INCLUDE_DIRECTORIES "instant-ngp/dependencies/gl3w")
        list(APPEND GUI_SOURCES "instant-ngp/dependencies/gl3w/GL/gl3w.c")
        set(NGP_LIBRARIES ${NGP_LIBRARIES} opengl32 $<TARGET_OBJECTS:glfw_objects>)
    else ()
        find_package(GLEW REQUIRED)
        list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
        set(NGP_LIBRARIES ${NGP_LIBRARIES} GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects> ${CMAKE_DL_LIBS})
    endif ()

    set(GUI_SOURCES
            ${GUI_SOURCES}
            instant-ngp/dependencies/imguizmo/ImGuizmo.cpp
            instant-ngp/dependencies/imgui/imgui.cpp
            instant-ngp/dependencies/imgui/backends/imgui_impl_glfw.cpp
            instant-ngp/dependencies/imgui/backends/imgui_impl_opengl3.cpp
            instant-ngp/dependencies/imgui/imgui_draw.cpp
            instant-ngp/dependencies/imgui/imgui_tables.cpp
            instant-ngp/dependencies/imgui/imgui_widgets.cpp
            )

    list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif (NGP_BUILD_WITH_GUI)

list(APPEND NGP_INCLUDE_DIRECTORIES
        "instant-ngp/dependencies"
        "instant-ngp/dependencies/eigen"
        "instant-ngp/dependencies/filesystem"
        "instant-ngp/dependencies/nanovdb"
        "instant-ngp/dependencies/tinylogger"
        )

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(OptiX)
# set OptiX_INSTALL_DIR via your environment if it's not found automatically

if ((OptiX_FOUND OR OptiX_INCLUDE) AND NGP_BUILD_WITH_OPTIX)
    set(NGP_OPTIX ON)
    include_directories("${OptiX_INCLUDE}")
    list(APPEND NGP_DEFINITIONS -DNGP_OPTIX)
else ()
    set(NGP_OPTIX OFF)
    if (NGP_BUILD_WITH_OPTIX)
        message(WARNING
                "OptiX was not found. Neural graphics primitives will still compile "
                "and run correctly. However, SDF training in 'raystab' and 'pathescape' "
                "modes will be significantly slower."
                )
    endif ()
endif ()

include_directories("instant-ngp/dependencies")
include_directories("instant-ngp/dependencies/tiny-cuda-nn/include")
include_directories("instant-ngp/dependencies/tiny-cuda-nn/dependencies")
include_directories("instant-ngp/dependencies/tinylogger")
include_directories("instant-ngp/dependencies/eigen")
include_directories("instant-ngp/dependencies/filesystem")
include_directories("instant-ngp/dependencies/nanovdb")

##########
# Python #
##########
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

##############
# flann #
##############
# add_library(flann STATIC IMPORTED)
# set_property(TARGET flann PROPERTY IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/dependencies/flann/build/lib/libflann_cuda_s.a")

##############
# liblz4 #
##############
# add_library(liblz4 STATIC IMPORTED)
# set_property(TARGET liblz4 PROPERTY IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/liblz4.a")

##############
# torch #
##############

if (USE_TORCH)
    add_definitions(-DUSE_TORCH)
    list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch_cuda/libtorch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(${TORCH_INCLUDE_DIRS})
endif ()

###############################################################################
# Program
###############################################################################

list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")
list(APPEND NGP_INCLUDE_DIRECTORIES "instant-ngp/include")
include_directories("instant-ngp/include")

set(NGP_SOURCES
        ${GUI_SOURCES}
        instant-ngp/src/camera_path.cu
        instant-ngp/src/common_device.cu
        instant-ngp/src/marching_cubes.cu
        instant-ngp/src/nerf_loader.cu
        instant-ngp/src/render_buffer.cu
        instant-ngp/src/testbed.cu
        instant-ngp/src/testbed_image.cu
        instant-ngp/src/testbed_nerf.cu
        instant-ngp/src/testbed_sdf.cu
        instant-ngp/src/testbed_volume.cu
        instant-ngp/src/thread_pool.cpp
        instant-ngp/src/tinyexr_wrapper.cu
        instant-ngp/src/tinyobj_loader_wrapper.cpp
        instant-ngp/src/triangle_bvh.cu
        )

include_directories("include")

set(RTA_SRC
        src/core.cu
        src/flame.cu
        src/tiny_mesh.cu
        src/recorder.cu
        src/masking.cpp
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

if (NGP_OPTIX)
    add_library(optix_program OBJECT
            instant-ngp/src/optix/pathescape.cu
            instant-ngp/src/optix/raystab.cu
            instant-ngp/src/optix/raytrace.cu
            )

    set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
    target_compile_definitions(optix_program PUBLIC ${NGP_DEFINITIONS} -DTCNN_MIN_GPU_ARCH=0)
    target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

    # OptiX programs will be compiled as PTX and packaged
    # as headers to be included from the binary dir.
    list(APPEND NGP_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
    set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

    find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
    if (NOT bin_to_c)
        message(FATAL_ERROR
                "bin2c not found:\n"
                "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
                "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
                )
    endif ()

    add_custom_command(
            OUTPUT "${OPTIX_PTX_HEADER}"
            COMMAND ${CMAKE_COMMAND}
            "-DBIN_TO_C_COMMAND=${bin_to_c}"
            "-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
            "-DOUTPUT=${OPTIX_PTX_HEADER}"
            -P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
            VERBATIM
            DEPENDS optix_program
            COMMENT "Converting PTX files to a C header"
    )

    list(APPEND NGP_SOURCES ${OPTIX_PTX_HEADER})
endif ()

add_library(ngp STATIC ${NGP_SOURCES})
set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(ngp PUBLIC ${NGP_DEFINITIONS})
target_compile_options(ngp PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(ngp PUBLIC ${NGP_INCLUDE_DIRECTORIES})
target_link_directories(ngp PUBLIC ${NGP_LINK_DIRECTORIES})
target_link_libraries(ngp PUBLIC ${NGP_LIBRARIES} tiny-cuda-nn)

add_executable(rta src/main.cu ${RTA_SRC})
target_link_libraries(rta PUBLIC ngp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
target_compile_options(rta PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(rta PUBLIC ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})

# Copy DLSS shared libraries
if (NGP_VULKAN)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(NGX_BUILD_DIR "dev")
    else ()
        set(NGX_BUILD_DIR "rel")
    endif ()

    if (MSVC)
        add_custom_command(TARGET rta POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/instant-ngp/dependencies/dlss/lib/Windows_x86_64/${NGX_BUILD_DIR}/nvngx_dlss.dll" $<TARGET_FILE_DIR:rta>
                COMMAND_EXPAND_LISTS
                )
    else ()
        file(GLOB DLSS_SOS "${CMAKE_CURRENT_SOURCE_DIR}/instant-ngp/dependencies/dlss/lib/Linux_x86_64/${NGX_BUILD_DIR}/libnvidia-ngx-dlss.so.*")
        add_custom_command(TARGET rta POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy ${DLSS_SOS} $<TARGET_FILE_DIR:rta>
                COMMAND_EXPAND_LISTS
                )
    endif ()
endif ()


if (Python_FOUND)
    add_library(pyngp SHARED instant-ngp/src/python_api.cu)
    set_target_properties(pyngp PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
    target_link_libraries(pyngp PRIVATE ngp PUBLIC ${PYTHON_LIBRARIES} pybind11::module)
    target_compile_definitions(pyngp PUBLIC -DNGP_PYTHON)
    pybind11_extension(pyngp)

    if (MSVC)
        # Copy dlls needed at runtime
        file(GLOB CUDA_DLLS "${CUDA_COMPILER_BIN}/cublas*.dll" "${CUDA_COMPILER_BIN}/cudart64*.dll" "${CUDA_COMPILER_BIN}/cusolver64*.dll")
        if (CUDA_DLLS)
            add_custom_command(TARGET pyngp POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_DLLS} $<TARGET_FILE_DIR:pyngp>
                    COMMAND_EXPAND_LISTS
                    )
        endif ()
    endif ()
endif ()
