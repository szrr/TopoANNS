cmake_minimum_required(VERSION 3.27)

project(TopoANNS VERSION 0.1.0 LANGUAGES CXX CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.3)
    message(FATAL_ERROR
        "TopoANNS requires CUDA 12.3 or newer; found "
        "${CMAKE_CUDA_COMPILER_VERSION}. Set CMAKE_CUDA_COMPILER to a "
        "supported nvcc executable.")
endif()

option(TOPOANNS_BUILD_TOOLS "Build TopoANNS command-line tools" ON)
option(TOPOANNS_BUILD_TESTS "Build TopoANNS tests" ON)
option(TOPOANNS_BUILD_DISKANN_TOOLS
       "Build index-conversion tools that depend on an external DiskANN checkout"
       OFF)

set(TOPOANNS_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures used for TopoANNS targets")
set(TOPOANNS_BAM_LIBRARY "" CACHE FILEPATH
    "Optional path to a prebuilt libnvm library; the bundled BaM sources are used when empty")
set(TOPOANNS_DISKANN_ROOT "" CACHE PATH
    "DiskANN source tree, required only when TOPOANNS_BUILD_DISKANN_TOOLS=ON")
set(TOPOANNS_DISKANN_LIBRARY "" CACHE FILEPATH
    "Path to the DiskANN library, required only when TOPOANNS_BUILD_DISKANN_TOOLS=ON")
set(TOPOANNS_DISKANN_EXTRA_LIBRARIES
    "tcmalloc;mkl_intel_ilp64;mkl_intel_thread;mkl_core;iomp5;pthread;m;dl"
    CACHE STRING "Additional libraries required by the external DiskANN build")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "${TOPOANNS_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(TOPOANNS_BAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bam")
set(TOPOANNS_BAM_INCLUDE_DIR "${TOPOANNS_BAM_ROOT}/include")
set(TOPOANNS_FREESTANDING_INCLUDE_DIR
    "${TOPOANNS_BAM_INCLUDE_DIR}/freestanding/include")

if(NOT EXISTS "${TOPOANNS_FREESTANDING_INCLUDE_DIR}/simt/atomic" OR
   NOT EXISTS "${TOPOANNS_BAM_INCLUDE_DIR}/freestanding/libcxx/include/__config")
    message(FATAL_ERROR
        "The freestanding submodule is missing. Run "
        "'git submodule update --init --recursive' and configure again.")
endif()

if(TOPOANNS_BAM_LIBRARY)
    if(NOT EXISTS "${TOPOANNS_BAM_LIBRARY}")
        message(FATAL_ERROR
            "TOPOANNS_BAM_LIBRARY does not exist: ${TOPOANNS_BAM_LIBRARY}")
    endif()
    add_library(topoanns_bam SHARED IMPORTED GLOBAL)
    set_target_properties(topoanns_bam PROPERTIES
        IMPORTED_LOCATION "${TOPOANNS_BAM_LIBRARY}")
else()
    add_library(topoanns_bam SHARED
        third_party/bam/src/admin.cpp
        third_party/bam/src/ctrl.cpp
        third_party/bam/src/dma.cpp
        third_party/bam/src/error.cpp
        third_party/bam/src/mutex.cpp
        third_party/bam/src/queue.cpp
        third_party/bam/src/rpc.cpp
        third_party/bam/src/linux/device.cpp
        third_party/bam/src/linux/dma.cpp
    )
    target_include_directories(topoanns_bam
        PUBLIC
            "${TOPOANNS_BAM_INCLUDE_DIR}"
        PRIVATE
            "${TOPOANNS_BAM_ROOT}/src"
            "${TOPOANNS_FREESTANDING_INCLUDE_DIR}"
    )
    target_compile_definitions(topoanns_bam PRIVATE _CUDA __CUDA__)
    target_link_libraries(topoanns_bam PRIVATE CUDA::cudart Threads::Threads)
    set_target_properties(topoanns_bam PROPERTIES
        CXX_EXTENSIONS ON
        OUTPUT_NAME nvm)
endif()

add_library(topoanns_core STATIC
    src/io/bam_vector_provider.cu
    src/io/bvecs_io.cpp
    src/io/diskann_disk_index.cpp
    src/io/hpq_index.cpp
    src/io/pq_index.cpp
    src/io/topology_layout.cpp
    src/io/vector_page_layout.cpp
    src/io/vector_page_provider.cpp
    src/builder/vector_store_builder.cpp
    src/search/entry_provider.cpp
    src/search/pq_distance_oracle.cpp
    src/search/pq_query_tables.cu
    src/search/rerank_exact.cu
    src/search/search_resources.cpp
    src/search/search_stop_condition.cpp
    src/search/topoanns_search.cpp
    src/search/topology_search.cpp
    src/search/topology_search_kernel.cu
    src/providers/pinned_topology.cu
)

target_include_directories(topoanns_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${TOPOANNS_BAM_INCLUDE_DIR}"
        "${TOPOANNS_FREESTANDING_INCLUDE_DIR}"
        "${TOPOANNS_BAM_ROOT}/graph_kernel_operation"
)
target_link_libraries(topoanns_core
    PUBLIC
        CUDA::cudart
        topoanns_bam
        Threads::Threads
        ${CMAKE_DL_LIBS}
        rt
)
target_compile_options(topoanns_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(topoanns_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

function(topoanns_add_core_tool target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE topoanns_core)
endfunction()

if(TOPOANNS_BUILD_TOOLS)
    topoanns_add_core_tool(topoanns_build_sift_subset
        tools/topoanns_build_sift_subset.cpp)
    topoanns_add_core_tool(topoanns_build_float_subset
        tools/topoanns_build_float_subset.cpp)
    topoanns_add_core_tool(topoanns_bvecs_to_fbin
        tools/topoanns_bvecs_to_fbin.cpp)
    topoanns_add_core_tool(topoanns_bam_4k_trace_bench
        tools/topoanns_bam_4k_trace_bench.cpp)
    topoanns_add_core_tool(topoanns_bam_write_vectors
        tools/topoanns_bam_write_vectors.cpp)
    topoanns_add_core_tool(topoanns_bam_write_topology
        tools/topoanns_bam_write_topology.cpp)
    topoanns_add_core_tool(topoanns_bam_write_combined_nodes
        tools/topoanns_bam_write_combined_nodes.cpp)
    topoanns_add_core_tool(topoanns_bam_verify_combined_nodes
        tools/topoanns_bam_verify_combined_nodes.cpp)
    topoanns_add_core_tool(topoanns_debug_topology_merge_diff
        tools/topoanns_debug_topology_merge_diff.cpp)

    add_executable(topoanns_cuda_device_map tools/topoanns_cuda_device_map.cpp)
    target_link_libraries(topoanns_cuda_device_map PRIVATE CUDA::cudart)

    find_package(BLAS REQUIRED)
    find_package(OpenMP REQUIRED)

    add_library(topoanns_rvq_support STATIC
        src/search/rvq_entry_provider.cpp
        src/search/rvq_device_entries.cu
        src/legacy_rvq/gas_rvq_wrapper.cu
        src/legacy_rvq/gas_legacy/anns/functions/distance_kernel.cu
        src/legacy_rvq/gas_legacy/anns/functions/selectMin1.cu
    )
    target_include_directories(topoanns_rvq_support
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/legacy_rvq/gas_legacy/anns"
    )
    target_link_libraries(topoanns_rvq_support
        PUBLIC
            topoanns_core
            CUDA::cudart
            CUDA::cublas
            BLAS::BLAS
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(topoanns_rvq_support PRIVATE
        ANNS_SEARCH_DIM=128
        ANNS_RVQ_DISABLE_SEARCH_LOG
        USE_L2_DIST_
        _CUDA
        __CUDA__
    )
    target_compile_options(topoanns_rvq_support PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--compiler-options=-fopenmp>
    )
    set_target_properties(topoanns_rvq_support PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    function(topoanns_add_rvq_tool target)
        add_executable(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE topoanns_rvq_support)
        set_target_properties(${target} PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endfunction()

    topoanns_add_rvq_tool(topoanns_rvq_tool tools/topoanns_rvq_tool.cpp)
    target_include_directories(topoanns_rvq_tool PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/legacy_rvq/gas_legacy/anns")
    topoanns_add_rvq_tool(topoanns_eval_sift tools/topoanns_eval_sift.cpp)
    topoanns_add_rvq_tool(topoanns_overlap_eval
        tools/topoanns_overlap_eval.cpp)
    topoanns_add_rvq_tool(topoanns_overlap_pipeline_eval
        tools/topoanns_overlap_pipeline_eval.cpp
        src/search/rerank_reuse_profiler.cpp)
    topoanns_add_rvq_tool(topoanns_pq2_rank_distribution
        tools/topoanns_pq2_rank_distribution.cpp)
    topoanns_add_rvq_tool(topoanns_pq34_bound_analysis
        tools/topoanns_pq34_bound_analysis.cpp)
    topoanns_add_rvq_tool(topoanns_learned_stop_dataset
        tools/topoanns_learned_stop_dataset.cpp)
    topoanns_add_rvq_tool(topoanns_learned_rerank_stop_dataset
        tools/topoanns_learned_rerank_stop_dataset.cpp)

    if(TOPOANNS_BUILD_DISKANN_TOOLS)
        if(NOT TOPOANNS_DISKANN_ROOT OR
           NOT EXISTS "${TOPOANNS_DISKANN_ROOT}/include")
            message(FATAL_ERROR
                "Set TOPOANNS_DISKANN_ROOT to a DiskANN source checkout.")
        endif()
        if(NOT TOPOANNS_DISKANN_LIBRARY OR
           NOT EXISTS "${TOPOANNS_DISKANN_LIBRARY}")
            message(FATAL_ERROR
                "Set TOPOANNS_DISKANN_LIBRARY to the built DiskANN library.")
        endif()

        function(topoanns_add_diskann_tool target source)
            add_executable(${target} ${source})
            target_include_directories(${target} PRIVATE
                "${TOPOANNS_DISKANN_ROOT}/include")
            target_link_libraries(${target} PRIVATE
                "${TOPOANNS_DISKANN_LIBRARY}"
                ${TOPOANNS_DISKANN_EXTRA_LIBRARIES})
        endfunction()

        topoanns_add_diskann_tool(topoanns_generate_pq_from_pivots
            tools/topoanns_generate_pq_from_pivots.cpp)
        topoanns_add_diskann_tool(topoanns_build_residual_pq
            tools/topoanns_build_residual_pq.cpp)
        topoanns_add_diskann_tool(topoanns_build_hpq
            tools/topoanns_build_hpq.cpp)
        target_link_libraries(topoanns_build_hpq PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(TOPOANNS_BUILD_TESTS)
    enable_testing()

    function(topoanns_add_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE topoanns_core)
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    topoanns_add_test(test_topology_layout tests/test_topology_layout.cpp)
    topoanns_add_test(test_vector_page_layout tests/test_vector_page_layout.cpp)
    topoanns_add_test(test_bvecs_io tests/test_bvecs_io.cpp)
    topoanns_add_test(test_diskann_disk_index tests/test_diskann_disk_index.cpp)
    topoanns_add_test(test_pinned_topology_uva tests/test_pinned_topology_uva.cu)
    set_target_properties(test_pinned_topology_uva PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    topoanns_add_test(test_vector_store_builder
        tests/test_vector_store_builder.cpp)
    topoanns_add_test(test_search_stop_condition
        tests/test_search_stop_condition.cpp)
    topoanns_add_test(test_search_resources tests/test_search_resources.cpp)
    topoanns_add_test(test_pq2_residual_refine
        tests/test_pq2_residual_refine.cpp)
    topoanns_add_test(test_pq2_bound_filter tests/test_pq2_bound_filter.cpp)
    topoanns_add_test(test_entry_provider tests/test_entry_provider.cpp)
    topoanns_add_test(test_pq_loader tests/test_pq_loader.cpp)
    topoanns_add_test(test_pq_query_tables tests/test_pq_query_tables.cpp)
    topoanns_add_test(test_pq_distance_oracle
        tests/test_pq_distance_oracle.cpp)
    topoanns_add_test(test_topology_search_gpu
        tests/test_topology_search_gpu.cpp)
    topoanns_add_test(test_topology_search_batch_gpu
        tests/test_topology_search_batch_gpu.cpp)
    topoanns_add_test(test_rerank_exact tests/test_rerank_exact.cpp)
    topoanns_add_test(test_topoanns_search_batch
        tests/test_topoanns_search_batch.cpp)
    topoanns_add_test(test_topoanns_search_device_entries
        tests/test_topoanns_search_device_entries.cpp)
    topoanns_add_test(test_topoanns_search_fused
        tests/test_topoanns_search_fused.cpp)
    topoanns_add_test(test_topoanns_search_fused_sweep
        tests/test_topoanns_search_fused_sweep.cpp)
endif()
