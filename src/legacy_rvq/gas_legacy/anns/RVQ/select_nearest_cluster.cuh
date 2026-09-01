#pragma once

#include<cuda_runtime.h>
#include "../graph/graph_kernel_operation/structure_on_device.cuh"
#include "../graph/graph_kernel_operation/warpselect/WarpSelect.cuh"
 #define num_of_top_coarse_cell 32

template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void selectNearestCluster(float* d_query, float* d_coarse_codebook, float* d_fine_codebook, int* d_result,
                                     int total_num_of_points, int dim, int num_of_results, int num_of_coarse_cell, int num_of_fine_cell){
    #define DIM 100
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;
    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* dis_of_coarse = shared_memory_space_s;
    int* crt_result = d_result + b_id * num_of_results;

    WarpSelect<float, int, false, Comparator<float>, NumWarpQ, NumThreadQ, WARP_SIZE>heap(MAX, total_num_of_points, num_of_top_coarse_cell);
    //读取query
#if DIM > 0
    float q1 = 0;
    if (t_id < DIM) {
        q1 = d_query[b_id * DIM + t_id];
    }
#endif
#if DIM > 32
    float q2 = 0;
    if (t_id + 32 < DIM) {
        q2 = d_query[b_id * DIM + t_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (t_id + 64 < DIM) {
        q3 = d_query[b_id * DIM + t_id + 64];
       }
#endif
#if DIM > 96
    float q4 = 0;
    if (t_id + 96 < DIM) {
        q4 = d_query[b_id * DIM + t_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (t_id + 128 < DIM) {
        q5 = d_query[b_id * DIM + t_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (t_id + 160 < DIM) {
        q6 = d_query[b_id * DIM + t_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (t_id + 192 < DIM) {
        q7 = d_query[b_id * DIM + t_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (t_id + 224 < DIM) {
        q8 = d_query[b_id * DIM + t_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (t_id + 256 < DIM) {
        q9 = d_query[b_id * DIM + t_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (t_id + 288 < DIM) {
        q10 = d_query[b_id * DIM + t_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (t_id + 320 < DIM) {
        q11 = d_query[b_id * DIM + t_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (t_id + 352 < DIM) {
        q12 = d_query[b_id * DIM + t_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (t_id + 384 < DIM) {
        q13 = d_query[b_id * DIM + t_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (t_id + 416 < DIM) {
        q14 = d_query[b_id * DIM + t_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (t_id + 448 < DIM) {
        q15 = d_query[b_id * DIM + t_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (t_id + 480 < DIM) {
        q16 = d_query[b_id * DIM + t_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (t_id + 512 < DIM) {
        q17 = d_query[b_id * DIM + t_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (t_id + 544 < DIM) {
        q18 = d_query[b_id * DIM + t_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (t_id + 576 < DIM) {
        q19 = d_query[b_id * DIM + t_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (t_id + 608 < DIM) {
        q20 = d_query[b_id * DIM + t_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (t_id + 640 < DIM) {
        q21 = d_query[b_id * DIM + t_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (t_id + 672 < DIM) {
        q22 = d_query[b_id * DIM + t_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (t_id + 704 < DIM) {
        q23 = d_query[b_id * DIM + t_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (t_id + 736 < DIM) {
        q24 = d_query[b_id * DIM + t_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (t_id + 768 < DIM) {
        q25 = d_query[b_id * DIM + t_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (t_id + 800 < DIM) {
        q26 = d_query[b_id * DIM + t_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (t_id + 832 < DIM) {
        q27 = d_query[b_id * DIM + t_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (t_id + 864 < DIM) {
        q28 = d_query[b_id * DIM + t_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (t_id + 896 < DIM) {
        q29 = d_query[b_id * DIM + t_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (t_id + 224 < DIM) {
        q30 = d_query[b_id * DIM + t_id + 928];
    }
#endif

//计算query与第一层cell距离
    for(int i = 0; i < num_of_coarse_cell; i++){
#if DIM > 0
    float p1 = 0;
    if (t_id < DIM) {
        p1 = d_coarse_codebook[i * DIM + t_id];
    }
#endif
#if DIM > 32
    float p2 = 0;
    if (t_id + 32 < DIM) {
        p2 = d_coarse_codebook[i * DIM + t_id + 32];
    }
#endif
#if DIM > 64
    float p3 = 0;
    if (t_id + 64 < DIM) {
        p3 = d_coarse_codebook[i * DIM + t_id + 64];
       }
#endif
#if DIM > 96
    float p4 = 0;
    if (t_id + 96 < DIM) {
        p4 = d_coarse_codebook[i * DIM + t_id + 96];
    }
#endif
#if DIM > 128
    float p5 = 0;
    if (t_id + 128 < DIM) {
        p5 = d_coarse_codebook[i * DIM + t_id + 128];
    }
#endif
#if DIM > 160
    float p6 = 0;
    if (t_id + 160 < DIM) {
        p6 = d_coarse_codebook[i * DIM + t_id + 160];
    }
#endif
#if DIM > 192
    float p7 = 0;
    if (t_id + 192 < DIM) {
        p7 = d_coarse_codebook[i * DIM + t_id + 192];
    }
#endif
#if DIM > 224
    float p8 = 0;
    if (t_id + 224 < DIM) {
        p8 = d_coarse_codebook[i * DIM + t_id + 224];
    }
#endif
#if DIM > 256
    float p9 = 0;
    if (t_id + 256 < DIM) {
        p9 = d_coarse_codebook[i * DIM + t_id + 256];
    }
#endif
#if DIM > 288
    float p10 = 0;
    if (t_id + 288 < DIM) {
        p10 = d_coarse_codebook[i * DIM + t_id + 288];
    }
#endif
#if DIM > 320
    float p11 = 0;
    if (t_id + 320 < DIM) {
        p11 = d_coarse_codebook[i * DIM + t_id + 320];
    }
#endif
#if DIM > 352
    float p12 = 0;
    if (t_id + 352 < DIM) {
        p12 = d_coarse_codebook[i * DIM + t_id + 352];
    }
#endif
#if DIM > 384
    float p13 = 0;
    if (t_id + 384 < DIM) {
        p13 = d_coarse_codebook[i * DIM + t_id + 384];
    }
#endif
#if DIM > 416
    float p14 = 0;
    if (t_id + 416 < DIM) {
        p14 = d_coarse_codebook[i * DIM + t_id + 416];
    }
#endif
#if DIM > 448
    float p15 = 0;
    if (t_id + 448 < DIM) {
        p15 = d_coarse_codebook[i * DIM + t_id + 448];
    }
#endif
#if DIM > 480
    float p16 = 0;
    if (t_id + 480 < DIM) {
        p16 = d_coarse_codebook[i * DIM + t_id + 480];
    }
#endif
#if DIM > 512
    float p17 = 0;
    if (t_id + 512 < DIM) {
        p17 = d_coarse_codebook[i * DIM + t_id + 512];
    }
#endif
#if DIM > 544
    float p18 = 0;
    if (t_id + 544 < DIM) {
        p18 = d_coarse_codebook[i * DIM + t_id + 544];
    }
#endif
#if DIM > 576
    float p19 = 0;
    if (t_id + 576 < DIM) {
        p19 = d_coarse_codebook[i * DIM + t_id + 576];
    }
#endif
#if DIM > 608
    float p20 = 0;
    if (t_id + 608 < DIM) {
        p20 = d_coarse_codebook[i * DIM + t_id + 608];
    }
#endif
#if DIM > 640
    float p21 = 0;
    if (t_id + 640 < DIM) {
        p21 = d_coarse_codebook[i * DIM + t_id + 640];
    }
#endif
#if DIM > 672
    float p22 = 0;
    if (t_id + 672 < DIM) {
        p22 = d_coarse_codebook[i * DIM + t_id + 672];
    }
#endif
#if DIM > 704
    float p23 = 0;
    if (t_id + 704 < DIM) {
        p23 = d_coarse_codebook[i * DIM + t_id + 704];
    }
#endif
#if DIM > 736
    float p24 = 0;
    if (t_id + 736 < DIM) {
        p24 = d_coarse_codebook[i * DIM + t_id + 736];
    }
#endif
#if DIM > 768
    float p25 = 0;
    if (t_id + 768 < DIM) {
        p25 = d_coarse_codebook[i * DIM + t_id + 768];
    }
#endif
#if DIM > 800
    float p26 = 0;
    if (t_id + 800 < DIM) {
        p26 = d_coarse_codebook[i * DIM + t_id + 800];
    }
#endif
#if DIM > 832
    float p27 = 0;
    if (t_id + 832 < DIM) {
        p27 = d_coarse_codebook[i * DIM + t_id + 832];
    }
#endif
#if DIM > 864
    float p28 = 0;
    if (t_id + 864 < DIM) {
        p28 = d_coarse_codebook[i * DIM + t_id + 864];
    }
#endif
#if DIM > 896
    float p29 = 0;
    if (t_id + 896 < DIM) {
        p29 = d_coarse_codebook[i * DIM + t_id + 896];
    }
#endif
#if DIM > 928
    float p30 = 0;
    if (t_id + 224 < DIM) {
        p30 = d_coarse_codebook[i * DIM + t_id + 928];
    }
#endif


#if USE_L2_DIST_
    #if DIM > 0
        float delta1 = (p1 - q1) * (p1 - q1);
    #endif
    #if DIM > 32
        float delta2 = (p2 - q2) * (p2 - q2);
    #endif
    #if DIM > 64
        float delta3 = (p3 - q3) * (p3 - q3);
    #endif
    #if DIM > 96
        float delta4 = (p4 - q4) * (p4 - q4);
    #endif
    #if DIM > 128
        float delta5 = (p5 - q5) * (p5 - q5);
    #endif
    #if DIM > 160
        float delta6 = (p6 - q6) * (p6 - q6);
    #endif
    #if DIM > 192
        float delta7 = (p7 - q7) * (p7 - q7);
    #endif
    #if DIM > 224
        float delta8 = (p8 - q8) * (p8 - q8);
    #endif
    #if DIM > 256
        float delta9 = (p9 - q9) * (p9 - q9);
    #endif
    #if DIM > 288
        float delta10 = (p10 - q10) * (p10 - q10);
    #endif
    #if DIM > 320
        float delta11 = (p11 - q11) * (p11 - q11);
    #endif
    #if DIM > 352
        float delta12 = (p12 - q12) * (p12 - q12);
    #endif
    #if DIM > 384
        float delta13 = (p13 - q13) * (p13 - q13);
    #endif
    #if DIM > 416
        float delta14 = (p14 - q14) * (p14 - q14);
    #endif
    #if DIM > 448
        float delta15 = (p15 - q15) * (p15 - q15);
    #endif
    #if DIM > 480
        float delta16 = (p16 - q16) * (p16 - q16);
    #endif
    #if DIM > 512
        float delta17 = (p17 - q17) * (p17 - q17);
    #endif
    #if DIM > 544
        float delta18 = (p18 - q18) * (p18 - q18);
    #endif
    #if DIM > 576
        float delta19 = (p19 - q19) * (p19 - q19);
    #endif
    #if DIM > 608
        float delta20 = (p20 - q20) * (p20 - q20);
    #endif
    #if DIM > 640
        float delta21 = (p21 - q21) * (p21 - q21);
    #endif
    #if DIM > 672
        float delta22 = (p22 - q22) * (p22 - q22);
    #endif
    #if DIM > 704
        float delta23 = (p23 - q23) * (p23 - q23);
    #endif
    #if DIM > 736
        float delta24 = (p24 - q24) * (p24 - q24);
    #endif
    #if DIM > 768
        float delta25 = (p25 - q25) * (p25 - q25);
    #endif
    #if DIM > 800
        float delta26 = (p26 - q26) * (p26 - q26);
    #endif
    #if DIM > 832
        float delta27 = (p27 - q27) * (p27 - q27);
    #endif
    #if DIM > 864
        float delta28 = (p28 - q28) * (p28 - q28);
    #endif
    #if DIM > 896
        float delta29 = (p29 - q29) * (p29 - q29);
    #endif
    #if DIM > 928
        float delta30 = (p30 - q30) * (p30 - q30);
    #endif
#endif


#if USE_L2_DIST_
        float dist = 0;
    #if DIM > 0
        dist += delta1;
    #endif
    #if DIM > 32
        dist += delta2;
    #endif
    #if DIM > 64
        dist += delta3;
    #endif
    #if DIM > 96
        dist += delta4;
    #endif
    #if DIM > 128
        dist += delta5;
    #endif
    #if DIM > 160
        dist += delta6;
    #endif
    #if DIM > 192
        dist += delta7;
    #endif
    #if DIM > 224
        dist += delta8;
    #endif
    #if DIM > 256
        dist += delta9;
    #endif
    #if DIM > 288
        dist += delta10;
    #endif
    #if DIM > 320
        dist += delta11;
    #endif
    #if DIM > 352
        dist += delta12;
    #endif
    #if DIM > 384
        dist += delta13;
    #endif
    #if DIM > 416
        dist += delta14;
    #endif
    #if DIM > 448
        dist += delta15;
    #endif
    #if DIM > 480
        dist += delta16;
    #endif
    #if DIM > 512
        dist += delta17;
    #endif
    #if DIM > 544
        dist += delta18;
    #endif
    #if DIM > 576
        dist += delta19;
    #endif
    #if DIM > 608
        dist += delta20;
    #endif
    #if DIM > 640
        dist += delta21;
    #endif
    #if DIM > 672
        dist += delta22;
    #endif
    #if DIM > 704
        dist += delta23;
    #endif
    #if DIM > 736
        dist += delta24;
    #endif
    #if DIM > 768
        dist += delta25;
    #endif
    #if DIM > 800
        dist += delta26;
    #endif
    #if DIM > 832
        dist += delta27;
    #endif
    #if DIM > 864
        dist += delta28;
    #endif
    #if DIM > 896
        dist += delta29;
    #endif
    #if DIM > 928
        dist += delta30;
    #endif
#endif


#if USE_L2_DIST_
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
    dist = sqrt(dist);
#endif

        dist = __shfl_sync(0xFFFFFFFF, dist, 0);
        if(t_id == i % size_of_warp){
            heap.addThreadQ(dist, i);
        }
        if(i % size_of_warp * NumThreadQ == 0){
            heap.reduce();
        }
    }
    heap.checkThreadQ();
//取最近的min(num_of_top_coarse_cell, num_of_coarse_cell)个第一层聚类中心
    int num_of_select_coarse = min(num_of_top_coarse_cell, num_of_coarse_cell);
    for(int i = 0; i < (num_of_select_coarse + size_of_warp - 1) / size_of_warp; i++){
        int unrollt_id = t_id + size_of_warp * i;
        if(unrollt_id < num_of_select_coarse){
            dis_of_coarse[unrollt_id].first = heap.warpK[i];
            dis_of_coarse[unrollt_id].second = heap.warpV[i];
        }
    }
    heap.reset();
//计算选取的第一层聚类中心与第二层cell的距离
    for(int i = 0; i < num_of_select_coarse; i++){
        int coarse_id = dis_of_coarse[i].second;
    //计算残差
    #if DIM > 0
        float p1 = 0;
        if (t_id < DIM) {
            p1 = q1 - d_coarse_codebook[coarse_id * DIM + t_id];
        }
    #endif
    #if DIM > 32
        float p2 = 0;
        if (t_id + 32 < DIM) {
            p2 = q2 - d_coarse_codebook[coarse_id * DIM + t_id + 32];
        }
    #endif
    #if DIM > 64
        float p3 = 0;
        if (t_id + 64 < DIM) {
            p3 = q3 - d_coarse_codebook[coarse_id * DIM + t_id + 64];
        }
    #endif
    #if DIM > 96
        float p4 = 0;
        if (t_id + 96 < DIM) {
            p4 = q4 - d_coarse_codebook[coarse_id * DIM + t_id + 96];
        }
    #endif
    #if DIM > 128
        float p5 = 0;
        if (t_id + 128 < DIM) {
            p5 = q5 - d_coarse_codebook[coarse_id * DIM + t_id + 128];
        }
    #endif
    #if DIM > 160
        float p6 = 0;
        if (t_id + 160 < DIM) {
            p6 = q6 - d_coarse_codebook[coarse_id * DIM + t_id + 160];
        }
    #endif
    #if DIM > 192
        float p7 = 0;
        if (t_id + 192 < DIM) {
            p7 = q7 - d_coarse_codebook[coarse_id * DIM + t_id + 192];
        }
    #endif
    #if DIM > 224
        float p8 = 0;
        if (t_id + 224 < DIM) {
            p8 = q8 - d_coarse_codebook[coarse_id * DIM + t_id + 224];
        }
    #endif
    #if DIM > 256
        float p9 = 0;
        if (t_id + 256 < DIM) {
            p9 = q9 - d_coarse_codebook[coarse_id * DIM + t_id + 256];
        }
    #endif
    #if DIM > 288
        float p10 = 0;
        if (t_id + 288 < DIM) {
            p10 = q10 - d_coarse_codebook[coarse_id * DIM + t_id + 288];
        }
    #endif
    #if DIM > 320
        float p11 = 0;
        if (t_id + 320 < DIM) {
            p11 = q11 - d_coarse_codebook[coarse_id * DIM + t_id + 320];
        }
    #endif
    #if DIM > 352
        float p12 = 0;
        if (t_id + 352 < DIM) {
            p12 = q12 - d_coarse_codebook[coarse_id * DIM + t_id + 352];
        }
    #endif
    #if DIM > 384
        float p13 = 0;
        if (t_id + 384 < DIM) {
            p13 = q13 - d_coarse_codebook[coarse_id * DIM + t_id + 384];
        }
    #endif
    #if DIM > 416
        float p14 = 0;
        if (t_id + 416 < DIM) {
            p14 = q14 - d_coarse_codebook[coarse_id * DIM + t_id + 416];
        }
    #endif
    #if DIM > 448
        float p15 = 0;
        if (t_id + 448 < DIM) {
            p15 = q15 - d_coarse_codebook[coarse_id * DIM + t_id + 448];
        }
    #endif
    #if DIM > 480
        float p16 = 0;
        if (t_id + 480 < DIM) {
            p16 = q16 - d_coarse_codebook[coarse_id * DIM + t_id + 480];
        }
    #endif
    #if DIM > 512
        float p17 = 0;
        if (t_id + 512 < DIM) {
            p17 = q17 - d_coarse_codebook[coarse_id * DIM + t_id + 512];
        }
    #endif
    #if DIM > 544
        float p18 = 0;
        if (t_id + 544 < DIM) {
            p18 = q18 - d_coarse_codebook[coarse_id * DIM + t_id + 544];
        }
    #endif
    #if DIM > 576
        float p19 = 0;
        if (t_id + 576 < DIM) {
            p19 = q19 - d_coarse_codebook[coarse_id * DIM + t_id + 576];
        }
    #endif
    #if DIM > 608
        float p20 = 0;
        if (t_id + 608 < DIM) {
            p20 = q20 - d_coarse_codebook[coarse_id * DIM + t_id + 608];
        }
    #endif
    #if DIM > 640
        float p21 = 0;
        if (t_id + 640 < DIM) {
            p21 = q21 - d_coarse_codebook[coarse_id * DIM + t_id + 640];
        }
    #endif
    #if DIM > 672
        float p22 = 0;
        if (t_id + 672 < DIM) {
            p22 = q22 - d_coarse_codebook[coarse_id * DIM + t_id + 672];
        }
    #endif
    #if DIM > 704
        float p23 = 0;
        if (t_id + 704 < DIM) {
            p23 = q23 - d_coarse_codebook[coarse_id * DIM + t_id + 704];
        }
    #endif
    #if DIM > 736
        float p24 = 0;
        if (t_id + 736 < DIM) {
            p24 = q24 - d_coarse_codebook[coarse_id * DIM + t_id + 736];
        }
    #endif
    #if DIM > 768
        float p25 = 0;
        if (t_id + 768 < DIM) {
            p25 = q25 - d_coarse_codebook[coarse_id * DIM + t_id + 768];
        }
    #endif
    #if DIM > 800
        float p26 = 0;
        if (t_id + 800 < DIM) {
            p26 = q26 - d_coarse_codebook[coarse_id * DIM + t_id + 800];
        }
    #endif
    #if DIM > 832
        float p27 = 0;
        if (t_id + 832 < DIM) {
            p27 = q27 - d_coarse_codebook[coarse_id * DIM + t_id + 832];
        }
    #endif
    #if DIM > 864
        float p28 = 0;
        if (t_id + 864 < DIM) {
            p28 = q28 - d_coarse_codebook[coarse_id * DIM + t_id + 864];
        }
    #endif
    #if DIM > 896
        float p29 = 0;
        if (t_id + 896 < DIM) {
            p29 = q29 - d_coarse_codebook[coarse_id * DIM + t_id + 896];
        }
    #endif
    #if DIM > 928
        float p30 = 0;
        if (t_id + 224 < DIM) {
            p30 = q30 - d_coarse_codebook[coarse_id * DIM + t_id + 928];
        }
    #endif
        //计算与二层cell距离
        for(int l = 0; l < num_of_fine_cell; l++){
        #if DIM > 0
            float r1 = 0;
            if (t_id < DIM) {
                r1 = d_fine_codebook[l * DIM + t_id];
            }
        #endif
        #if DIM > 32
            float r2 = 0;
            if (t_id + 32 < DIM) {
                r2 = d_fine_codebook[l * DIM + t_id + 32];
            }
        #endif
        #if DIM > 64
            float r3 = 0;
            if (t_id + 64 < DIM) {
                r3 = d_fine_codebook[l * DIM + t_id + 64];
            }
        #endif
        #if DIM > 96
            float r4 = 0;
            if (t_id + 96 < DIM) {
                r4 = d_fine_codebook[l * DIM + t_id + 96];
            }
        #endif
        #if DIM > 128
            float r5 = 0;
            if (t_id + 128 < DIM) {
                r5 = d_fine_codebook[l * DIM + t_id + 128];
            }
        #endif
        #if DIM > 160
            float r6 = 0;
            if (t_id + 160 < DIM) {
                r6 = d_fine_codebook[l * DIM + t_id + 160];
            }
        #endif
        #if DIM > 192
            float r7 = 0;
            if (t_id + 192 < DIM) {
                r7 = d_fine_codebook[l * DIM + t_id + 192];
            }
        #endif
        #if DIM > 224
            float r8 = 0;
            if (t_id + 224 < DIM) {
                r8 = d_fine_codebook[l * DIM + t_id + 224];
            }
        #endif
        #if DIM > 256
            float r9 = 0;
            if (t_id + 256 < DIM) {
                r9 = d_fine_codebook[l * DIM + t_id + 256];
            }
        #endif
        #if DIM > 288
            float r10 = 0;
            if (t_id + 288 < DIM) {
                r10 = d_fine_codebook[l * DIM + t_id + 288];
            }
        #endif
        #if DIM > 320
            float r11 = 0;
            if (t_id + 320 < DIM) {
                r11 = d_fine_codebook[l * DIM + t_id + 320];
            }
        #endif
        #if DIM > 352
            float r12 = 0;
            if (t_id + 352 < DIM) {
                r12 = d_fine_codebook[l * DIM + t_id + 352];
            }
        #endif
        #if DIM > 384
            float r13 = 0;
            if (t_id + 384 < DIM) {
                r13 = d_fine_codebook[l * DIM + t_id + 384];
            }
        #endif
        #if DIM > 416
            float r14 = 0;
            if (t_id + 416 < DIM) {
                r14 = d_fine_codebook[l * DIM + t_id + 416];
            }
        #endif
        #if DIM > 448
            float r15 = 0;
            if (t_id + 448 < DIM) {
                r15 = d_fine_codebook[l * DIM + t_id + 448];
            }
        #endif
        #if DIM > 480
            float r16 = 0;
            if (t_id + 480 < DIM) {
                r16 = d_fine_codebook[l * DIM + t_id + 480];
            }
        #endif
        #if DIM > 512
            float r17 = 0;
            if (t_id + 512 < DIM) {
                r17 = d_fine_codebook[l * DIM + t_id + 512];
            }
        #endif
        #if DIM > 544
            float r18 = 0;
            if (t_id + 544 < DIM) {
                r18 = d_fine_codebook[l * DIM + t_id + 544];
            }
        #endif
        #if DIM > 576
            float r19 = 0;
            if (t_id + 576 < DIM) {
                r19 = d_fine_codebook[l * DIM + t_id + 576];
            }
        #endif
        #if DIM > 608
            float r20 = 0;
            if (t_id + 608 < DIM) {
                r20 = d_fine_codebook[l * DIM + t_id + 608];
            }
        #endif
        #if DIM > 640
            float r21 = 0;
            if (t_id + 640 < DIM) {
                r21 = d_fine_codebook[l * DIM + t_id + 640];
            }
        #endif
        #if DIM > 672
            float r22 = 0;
            if (t_id + 672 < DIM) {
                r22 = d_fine_codebook[l * DIM + t_id + 672];
            }
        #endif
        #if DIM > 704
            float r23 = 0;
            if (t_id + 704 < DIM) {
                r23 = d_fine_codebook[l * DIM + t_id + 704];
            }
        #endif
        #if DIM > 736
            float r24 = 0;
            if (t_id + 736 < DIM) {
                r24 = d_fine_codebook[l * DIM + t_id + 736];
            }
        #endif
        #if DIM > 768
            float r25 = 0;
            if (t_id + 768 < DIM) {
                r25 = d_fine_codebook[l * DIM + t_id + 768];
            }
        #endif
        #if DIM > 800
            float r26 = 0;
            if (t_id + 800 < DIM) {
                r26 = d_fine_codebook[l * DIM + t_id + 800];
            }
        #endif
        #if DIM > 832
            float r27 = 0;
            if (t_id + 832 < DIM) {
                r27 = d_fine_codebook[l * DIM + t_id + 832];
            }
        #endif
        #if DIM > 864
            float r28 = 0;
            if (t_id + 864 < DIM) {
                r28 = d_fine_codebook[l * DIM + t_id + 864];
            }
        #endif
        #if DIM > 896
            float r29 = 0;
            if (t_id + 896 < DIM) {
                r29 = d_fine_codebook[l * DIM + t_id + 896];
            }
        #endif
        #if DIM > 928
            float r30 = 0;
            if (t_id + 224 < DIM) {
                r30 = d_fine_codebook[l * DIM + t_id + 928];
            }
        #endif
        #if USE_L2_DIST_
            #if DIM > 0
                float delta1 = (p1 - r1) * (p1 - r1);
            #endif
            #if DIM > 32
                float delta2 = (p2 - r2) * (p2 - r2);
            #endif
            #if DIM > 64
                float delta3 = (p3 - r3) * (p3 - r3);
            #endif
            #if DIM > 96
                float delta4 = (p4 - r4) * (p4 - r4);
            #endif
            #if DIM > 128
                float delta5 = (p5 - r5) * (p5 - r5);
            #endif
            #if DIM > 160
                float delta6 = (p6 - r6) * (p6 - r6);
            #endif
            #if DIM > 192
                float delta7 = (p7 - r7) * (p7 - r7);
            #endif
            #if DIM > 224
                float delta8 = (p8 - r8) * (p8 - r8);
            #endif
            #if DIM > 256
                float delta9 = (p9 - r9) * (p9 - r9);
            #endif
            #if DIM > 288
                float delta10 = (p10 - r10) * (p10 - r10);
            #endif
            #if DIM > 320
                float delta11 = (p11 - r11) * (p11 - r11);
            #endif
            #if DIM > 352
                float delta12 = (p12 - r12) * (p12 - r12);
            #endif
            #if DIM > 384
                float delta13 = (p13 - r13) * (p13 - r13);
            #endif
            #if DIM > 416
                float delta14 = (p14 - r14) * (p14 - r14);
            #endif
            #if DIM > 448
                float delta15 = (p15 - r15) * (p15 - r15);
            #endif
            #if DIM > 480
                float delta16 = (p16 - r16) * (p16 - r16);
            #endif
            #if DIM > 512
                float delta17 = (p17 - r17) * (p17 - r17);
            #endif
            #if DIM > 544
                float delta18 = (p18 - r18) * (p18 - r18);
            #endif
            #if DIM > 576
                float delta19 = (p19 - r19) * (p19 - r19);
            #endif
            #if DIM > 608
                float delta20 = (p20 - r20) * (p20 - r20);
            #endif
            #if DIM > 640
                float delta21 = (p21 - r21) * (p21 - r21);
            #endif
            #if DIM > 672
                float delta22 = (p22 - r22) * (p22 - r22);
            #endif
            #if DIM > 704
                float delta23 = (p23 - r23) * (p23 - r23);
            #endif
            #if DIM > 736
                float delta24 = (p24 - r24) * (p24 - r24);
            #endif
            #if DIM > 768
                float delta25 = (p25 - r25) * (p25 - r25);
            #endif
            #if DIM > 800
                float delta26 = (p26 - r26) * (p26 - r26);
            #endif
            #if DIM > 832
                float delta27 = (p27 - r27) * (p27 - r27);
            #endif
            #if DIM > 864
                float delta28 = (p28 - r28) * (p28 - r28);
            #endif
            #if DIM > 896
                float delta29 = (p29 - r29) * (p29 - r29);
            #endif
            #if DIM > 928
                float delta30 = (p30 - r30) * (p30 - r30);
            #endif
        #endif

        #if USE_L2_DIST_
                float dist = 0;
            #if DIM > 0
                dist += delta1;
            #endif
            #if DIM > 32
                dist += delta2;
            #endif
            #if DIM > 64
                dist += delta3;
            #endif
            #if DIM > 96
                dist += delta4;
            #endif
            #if DIM > 128
                dist += delta5;
            #endif
            #if DIM > 160
                dist += delta6;
            #endif
            #if DIM > 192
                dist += delta7;
            #endif
            #if DIM > 224
                dist += delta8;
            #endif
            #if DIM > 256
                dist += delta9;
            #endif
            #if DIM > 288
                dist += delta10;
            #endif
            #if DIM > 320
                dist += delta11;
            #endif
            #if DIM > 352
                dist += delta12;
            #endif
            #if DIM > 384
                dist += delta13;
            #endif
            #if DIM > 416
                dist += delta14;
            #endif
            #if DIM > 448
                dist += delta15;
            #endif
            #if DIM > 480
                dist += delta16;
            #endif
            #if DIM > 512
                dist += delta17;
            #endif
            #if DIM > 544
                dist += delta18;
            #endif
            #if DIM > 576
                dist += delta19;
            #endif
            #if DIM > 608
                dist += delta20;
            #endif
            #if DIM > 640
                dist += delta21;
            #endif
            #if DIM > 672
                dist += delta22;
            #endif
            #if DIM > 704
                dist += delta23;
            #endif
            #if DIM > 736
                dist += delta24;
            #endif
            #if DIM > 768
                dist += delta25;
            #endif
            #if DIM > 800
                dist += delta26;
            #endif
            #if DIM > 832
                dist += delta27;
            #endif
            #if DIM > 864
                dist += delta28;
            #endif
            #if DIM > 896
                dist += delta29;
            #endif
            #if DIM > 928
                dist += delta30;
            #endif
        #endif


        #if USE_L2_DIST_
            dist += __shfl_down_sync(FULL_MASK, dist, 16);
            dist += __shfl_down_sync(FULL_MASK, dist, 8);
            dist += __shfl_down_sync(FULL_MASK, dist, 4);
            dist += __shfl_down_sync(FULL_MASK, dist, 2);
            dist += __shfl_down_sync(FULL_MASK, dist, 1);
        #endif

        #if USE_L2_DIST_
            dist = sqrt(dist);
        #endif

            dist = __shfl_sync(0xFFFFFFFF, dist, 0);
            if(t_id == l % size_of_warp){
                heap.addThreadQ(dist + dis_of_coarse[i].first, coarse_id * num_of_fine_cell + l);
            }
            if(l % size_of_warp * NumThreadQ == 0){
                heap.reduce();
            }
        }
        heap.checkThreadQ();
    }
//写出结果
    for(int i = 0; i < (num_of_results + size_of_warp - 1) / size_of_warp; i++){
        int unrollt_id = t_id + i * size_of_warp;
        if(unrollt_id < num_of_results){
            crt_result[unrollt_id] = heap.warpV[i];
        }
    }
}