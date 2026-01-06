/**
 * @file kernel.cu
 * @brief 实现将在GPU上执行的CUDA Kernel函数。
 *
 * 此文件包含了仿真核心计算逻辑的实现，包括：
 * 1. initCurandStateKernel: 用于初始化CURAND状态。
 * 2. rayTracingKernel: 主光线追踪Kernel，每个线程处理一个微面元。
 * 3. 大量的 __device__ 辅助函数，用于数学计算、随机采样和几何操作。
 */

#include "kernel.cuh"
#include "intersections.cuh" // 引入几何求交函数
#include "sampling.cuh"      // 引入随机采样函数
#include <device_launch_parameters.h>  // 这个头文件定义了 blockIdx, threadIdx 等
#include "cuda_runtime.h"

 // ==================================================================================
 // Kernel 实现: rayTracingKernel
 // ==================================================================================
__global__ void rayTracingKernel(
    const Heliostat* d_heliostats,
    int num_heliostats,
    const Receiver* d_receiver,
    const AccelerationGrid* d_grid,
    float3 sun_direction,
    AtomicCounters* d_counters,
    int rays_per_microfacet,
    int microfacets_per_heliostat_x,
    float sun_cone_half_angle_rad,
    float normal_perturbation_sigma_rad,
    MicrofacetShadingData* d_shading_data
) {
    // --- 1. 线程身份识别 ---
    // 计算当前线程负责的全局微面元索引
    unsigned long long global_microfacet_idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // 计算每个定日镜上的微面元总数
    const int microfacets_per_heliostat = microfacets_per_heliostat_x * microfacets_per_heliostat_x; // 假设x和y方向数量相同
    if (global_microfacet_idx >= (unsigned long long)num_heliostats * microfacets_per_heliostat) {
        return; // 超出范围的线程直接退出
    }

    // 计算所属的定日镜ID和在该定日镜上的局部ID
    int heliostat_id = global_microfacet_idx / microfacets_per_heliostat;
    int local_microfacet_idx = global_microfacet_idx % microfacets_per_heliostat;

    // 从局部ID计算出在定日镜表面上的二维索引(i, j)
    int j = local_microfacet_idx / microfacets_per_heliostat_x;
    int i = local_microfacet_idx % microfacets_per_heliostat_x;

    // --- 2. 加载数据到快速内存 (寄存器) ---
    const Heliostat h = d_heliostats[heliostat_id];
    curandState local_rand_state;
    // 使用一个固定的种子(也可以从CPU传入)和全局唯一的线程ID来初始化。
    // 这样可以保证每次运行的结果可复现，同时每个线程的随机序列都不同。
    curand_init(12345ULL, global_microfacet_idx, 0, &local_rand_state);


    // --- 3. 微面元建模 ---
    // a. 计算微面元在定日镜局部坐标系下的中心位置
    float microfacet_size = h.width / microfacets_per_heliostat_x;
    float local_x = (i + 0.5f) * microfacet_size - h.width * 0.5f;
    float local_y = (j + 0.5f) * microfacet_size - h.height * 0.5f;

    // b. 将局部位置转换到世界坐标系
    //    [修正] 使用与 intersectRayPlane 中完全一致的、基于物理约束的局部坐标系。
    //    这确保了微面元的位置计算和边界检测使用相同的坐标系基准。
    float3 local_x_axis = make_float3(h.ideal_normal.y, -h.ideal_normal.x, 0.0f);
    if (length_sq(local_x_axis) < 1e-6f) { // 处理法线垂直于地面的特殊情况
        local_x_axis = make_float3(1.0f, 0.0f, 0.0f);
    }
    local_x_axis = normalize(local_x_axis);
    // 局部Y轴通过叉乘得到，以确保坐标系是正交的。
    float3 local_y_axis = normalize(cross(h.ideal_normal, local_x_axis));

    float3 microfacet_world_pos = h.center + local_x * local_x_axis + local_y * local_y_axis;

    // c. 对理想法线进行随机扰动，模拟镜面不平整
    float3 perturbed_normal = sampleNormalPerturbation(h.ideal_normal, &local_rand_state, normal_perturbation_sigma_rad);

    // [新增] 初始化当前微面元的局部计数器
    unsigned int local_shadow_losses = 0;
    unsigned int local_block_losses = 0;

    // --- 4. 光锥追踪循环 ---
    for (int k = 0; k < rays_per_microfacet; ++k) {
        // a. 在太阳光锥内采样一条入射光线
        float3 incident_dir = sampleSunConeBuie(sun_direction, &local_rand_state, sun_cone_half_angle_rad);

        // b. 计算反射光线方向 (根据扰动后的真实法线)
        float3 reflect_dir = reflect(-incident_dir, perturbed_normal);

        // --- c. 阴影测试 (向太阳方向追) ---
        Ray shadow_ray;
        shadow_ray.origin = microfacet_world_pos;
        shadow_ray.direction = -incident_dir;
        shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * 1e-4f;

        if (traverseGridAndIntersect(shadow_ray, heliostat_id, *d_grid, d_heliostats)) {
            atomicAdd(&d_counters->shadow_rays, 1);
            local_shadow_losses++;
            continue; // 被阴影遮挡，追踪下一条光线
        }

        // --- d. 遮挡测试 (向吸收塔方向追) ---
        Ray block_ray;
        block_ray.origin = microfacet_world_pos;
        block_ray.direction = reflect_dir;
        block_ray.origin = block_ray.origin + block_ray.direction * 1e-3f;

        if (traverseGridAndIntersect(block_ray, heliostat_id, *d_grid, d_heliostats)) {
            atomicAdd(&d_counters->blocked_rays, 1);
            local_block_losses++;
            continue; // 被其他镜子遮挡，追踪下一条光线
        }

        // --- e. 截断测试 (与吸收塔求交) ---
        float t_intersection;
        if (intersectRayCylinder(block_ray, *d_receiver, t_intersection)) {
            if (t_intersection > 0) {
                atomicAdd(&d_counters->hit_rays, 1);
            }
        }
    } // [修正] for 循环在这里结束

    // =========================================================================
    // *** [修正] 将最终结果写入输出缓冲区 ***
    // =========================================================================
    // 在循环结束后，将这个微面元累积的总损失 *一次性* 写入全局数组。
    // 这避免了在循环内对全局内存的重复写入，显著提高了性能。
    d_shading_data[global_microfacet_idx].shadow_ray_losses = local_shadow_losses;
    d_shading_data[global_microfacet_idx].block_ray_losses = local_block_losses;
}