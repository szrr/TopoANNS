#pragma once
#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.cuh"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <byteswap.h>

using error = std::runtime_error;
using std::string;

const char* const ctrls_paths[] = {"/dev/libnvm0"};

__global__
void sequential_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, //unsigned long long* req_count,
                                uint32_t num_ctrls, uint64_t reqs_per_thread, uint32_t access_type, uint64_t s_offset, uint64_t o_offset){
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // uint32_t bid = blockIdx.x;
    // uint32_t smid = get_smid();

    uint32_t ctrl = (tid/32) % (num_ctrls);
    uint32_t queue = (tid/32) % (ctrls[ctrl]->n_qps);
    uint64_t itr=0;

    // printf("[sequential_access_kernel] 1\n");

    for (;tid < pc->n_pages; tid = tid+n_reqs){
            uint64_t start_block = (o_offset+s_offset + tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log ;
            uint64_t pc_idx = (tid);
            //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
            //start_block = tid;
            uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
            itr = itr+1;
            // printf("[sequential_access_kernel] 2\n");
                if (access_type == READ) {
                    read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, pc_idx);
               }
                else {
                    // printf("[sequential_access_kernel] 3\n");
                    write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, pc_idx);
                    // printf("[sequential_access_kernel] 4\n");
                }
    }
}

__global__
void verify_kernel(uint64_t* orig_h, uint64_t* nvme_h, uint64_t n_elems,uint32_t n_reqs){
        uint64_t tid = blockIdx.x*blockDim.x + threadIdx.x;

        for (;tid < n_elems; tid = tid+n_reqs){
           uint64_t orig_val = orig_h[tid];
           uint64_t nvme_val = nvme_h[tid];
           if(orig_val != nvme_val)
              printf("MISMATCH: at %llu\torig_val:%llu\tnvme_val:%llu\tn_reqs:%lu\tn_elms:%llu\n",tid, (unsigned long long)orig_val, (unsigned long long)nvme_h, n_reqs, n_elems);
        }
        __syncthreads();//really not needed.
}


void writeSSD(int* d_graph, float* d_data, size_t graph_size, size_t data_size,
              page_cache_t& h_pc, page_cache_d_t* d_pc, Settings& settings, int degree_of_graph, int dim_of_data) {
    // 更新 settings.numPages
    settings.numPages = ceil((float)settings.maxPageCacheSize / settings.pageSize);

    // 计算总元素数
    size_t total_graph_elements = graph_size * degree_of_graph;
    size_t total_data_elements = data_size * dim_of_data;

    // 设置最大每次处理的元素数，防止一次处理过多数据
    const size_t MAX_COPY_ELEMENTS = settings.pageSize * 1024 / sizeof(float);  // 可根据需要调整

    // 写入 d_graph 到 SSD
    size_t graph_offset_elements = 0;
    size_t remaining_graph_elements = total_graph_elements;
    printf("开始写入 d_graph 到 SSD...\n");
    while (remaining_graph_elements > 0) {
        size_t elements_per_page = settings.pageSize / sizeof(int);
        size_t max_elements = elements_per_page * settings.numPages;
        size_t copy_elements = std::min(max_elements, remaining_graph_elements);
        // 指针偏移
        int* src_ptr = d_graph + graph_offset_elements;

        // 复制数据
        size_t cpysize = copy_elements * sizeof(int);
        cuda_err_chk(cudaMemcpy(h_pc.pdt.base_addr, src_ptr, cpysize, cudaMemcpyDeviceToDevice));
        uint64_t b_size = settings.blkSize; // 64
        uint64_t g_size = (settings.numThreads + b_size - 1) / b_size;
        printf("b_size = %d\n", (int)b_size);
        printf("g_size = %d\n", (int)g_size);
        // 启动写入内核
        sequential_access_kernel<<<g_size, b_size>>>(
            h_pc.pdt.d_ctrls, d_pc, settings.pageSize, settings.numThreads,
            settings.n_ctrls, settings.numReqs, WRITE, graph_offset_elements * sizeof(int), 0
        );
        cuda_err_chk(cudaDeviceSynchronize());
        // 更新偏移量和剩余元素数量
        graph_offset_elements += copy_elements;
        remaining_graph_elements -= copy_elements;

        // 打印调试信息
        printf("已写入 d_graph 元素数：%zu，剩余元素数：%zu\n", graph_offset_elements, remaining_graph_elements);
    }

    // 写入 d_data 到 SSD
    uint64_t d_data_offset = ((total_graph_elements * sizeof(int) + settings.pageSize - 1) / settings.pageSize) * settings.pageSize;
    size_t data_offset_elements = 0;
    size_t remaining_data_elements = total_data_elements;
    printf("remaining_data_elements = %ld\n", remaining_data_elements);

    printf("开始写入 d_data 到 SSD...\n");
    while (remaining_data_elements > 0) {
        // printf("remaining_data_elements = %llu\n", remaining_data_elements);
        size_t elements_per_page = settings.pageSize / sizeof(float);
        // printf("settings.pageSize = %llu\n", settings.pageSize);
        size_t max_elements = elements_per_page * settings.numPages;
        // printf("settings.numPages = %llu\n", settings.numPages);

        size_t copy_elements = max_elements < remaining_data_elements ? max_elements: remaining_data_elements;
        std::cout << "#" << max_elements << ", " << remaining_data_elements << ", " << copy_elements << "\n";
        // printf("max_elements = %llu\n", max_elements);
        // printf("copy_elements = %llu\n", copy_elements);
        // printf("remaining_data_elements = %llu\n", remaining_data_elements);

        // 指针偏移
        float* src_ptr = d_data + data_offset_elements;

        // 复制数据
        size_t cpysize = copy_elements * sizeof(float);
        cuda_err_chk(cudaMemcpy(h_pc.pdt.base_addr, src_ptr, cpysize, cudaMemcpyDeviceToDevice));

        uint64_t b_size = settings.blkSize; // 64
        uint64_t g_size = (settings.numThreads + b_size - 1) / b_size;
        printf("b_size = %d\n", (int)b_size);
        printf("g_size = %d\n", (int)g_size);

        // 启动写入内核
        sequential_access_kernel<<<g_size, b_size>>>(
            h_pc.pdt.d_ctrls, d_pc, settings.pageSize, settings.numThreads,
            settings.n_ctrls, settings.numReqs, WRITE, d_data_offset + data_offset_elements * sizeof(float), 0
        );
        cuda_err_chk(cudaDeviceSynchronize());

        // 更新偏移量和剩余元素数量
        data_offset_elements += copy_elements;
        remaining_data_elements -= copy_elements;

        // 打印调试信息
        printf("已写入 d_data 元素数：%zu，剩余元素数：%zu\n", data_offset_elements, remaining_data_elements);
    }

    printf("数据写入 SSD 完成。\n");
}
template<typename T>
__global__ void read_kernel(T* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        printf("idx: %d, value: %d\n", idx, data[idx]);
    }
}
template<typename T>
void writeArray2Disk(T* d_data, size_t num_elements, size_t offset, page_cache_t& h_pc, page_cache_d_t* d_pc, Settings& settings) {
    // Calculate the number of pages that can fit in max cache size
    settings.numPages = settings.maxPageCacheSize / settings.pageSize;

    // Define max elements to process per iteration to avoid exceeding GPU cache size
    const size_t MAX_COPY_ELEMENTS = settings.numPages * settings.pageSize / sizeof(T);

    // Track the remaining elements and current offset
    size_t offset_elements = 0;
    size_t remaining_elements = num_elements;

    printf("There are %lu elements, starting write data to SSD...\n", num_elements);

    while (remaining_elements > 0) {
        // Determine the number of elements to copy in this iteration
        size_t copy_elements = std::min(MAX_COPY_ELEMENTS, remaining_elements); // copy element num at a time
        size_t cpy_size = copy_elements * sizeof(T); // copy bytes
        T* src_ptr = d_data + offset_elements; // Adjust pointer for the current chunk
        size_t current_offset = offset + offset_elements * sizeof(T); // Calculate current byte offset
        // read_kernel<<<1,128>>>(src_ptr, 8);
        // cuda_err_chk(cudaDeviceSynchronize());
        // Step 1: Copy data from the device memory to the GPU's BAM cache
        cuda_err_chk(cudaMemcpy(h_pc.pdt.base_addr, src_ptr, cpy_size, cudaMemcpyDeviceToDevice));

        // Define kernel execution configuration
        uint64_t b_size = settings.blkSize; // 64
        uint64_t g_size = (settings.numThreads + b_size - 1) / b_size; //1

        // Step 2: Write the data chunk to SSD
        sequential_access_kernel<<<g_size, b_size>>>(
            h_pc.pdt.d_ctrls, d_pc, settings.pageSize, settings.numThreads,
            settings.n_ctrls, settings.numReqs, WRITE, current_offset, 0
        );
        cuda_err_chk(cudaDeviceSynchronize());

        // Update offset and remaining element count for the next iteration
        offset_elements += copy_elements;
        remaining_elements -= copy_elements;

        // Print progress for debugging
        printf("Wrote %zu elements, %zu elements remaining.\n", offset_elements, remaining_elements);
    }

    printf("Data write to SSD completed.\n");
}


template<typename T>
void writeArray2Disk(string data_path, size_t num_elements, size_t offset, page_cache_t& h_pc, page_cache_d_t* d_pc, Settings& settings) {
    // Calculate the number of pages that can fit in max cache size
    settings.numPages = settings.maxPageCacheSize / settings.pageSize;
    // settings.numPages = ceil((float)settings.maxPageCacheSize / settings.pageSize);

    // Define max elements to process per iteration to avoid exceeding GPU cache size
    const size_t MAX_COPY_ELEMENTS = settings.numPages * settings.pageSize / sizeof(T);

    // Track the remaining elements and current offset
    size_t offset_elements = 0;
    size_t remaining_elements = num_elements;
    std::ifstream in(data_path, std::ios::binary);
    in.seekg(4096, std::ios::beg);

    printf("Starting data write to SSD...\n");
    printf("Wrote %zu elements.\n", num_elements);

    while (remaining_elements > 0 ) {

        // Determine the number of elements to copy in this iteration
        size_t copy_elements = std::min(MAX_COPY_ELEMENTS, remaining_elements); // copy element num at a time
        size_t cpy_size = copy_elements * sizeof(T); // copy bytes
        T* h_data = new T[copy_elements];
        in.read((char*)h_data, cpy_size);
        // if(remaining_elements == num_elements){
        //     for(int i = 0; i < 8; i++){
        //         std::cout<<h_data[i]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        // T* src_ptr; // Adjust pointer for the current chunk
        // cuda_err_chk(cudaMalloc((void**)&src_ptr, cpy_size));
        // cuda_err_chk(cudaMemcpy(src_ptr, h_data, cpy_size, cudaMemcpyHostToDevice));
        size_t current_offset = offset + offset_elements * sizeof(T); // Calculate current byte offset

        // Step 1: Copy data from the device memory to the GPU's BAM cache
        cuda_err_chk(cudaMemcpy(h_pc.pdt.base_addr, h_data, cpy_size, cudaMemcpyHostToDevice));

        // Define kernel execution configuration
        uint64_t b_size = settings.blkSize; // 64
        uint64_t g_size = (settings.numThreads + b_size - 1) / b_size; //1

        // Step 2: Write the data chunk to SSD
        sequential_access_kernel<<<g_size, b_size>>>(
            h_pc.pdt.d_ctrls, d_pc, settings.pageSize, settings.numThreads,
            settings.n_ctrls, settings.numReqs, WRITE, current_offset, 0
        );
        cuda_err_chk(cudaDeviceSynchronize());

        // Update offset and remaining element count for the next iteration
        offset_elements += copy_elements;
        remaining_elements -= copy_elements;

        // Print progress for debugging
        delete[] h_data;
        // cudaFree(src_ptr);
        printf("Wrote %zu elements, %zu elements remaining.\n", offset_elements, remaining_elements);
    }
    in.close();
    printf("Data write to SSD completed.\n");
}
