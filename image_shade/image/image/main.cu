#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <fstream> // [新增] 用于文件操作

#include <cuda_runtime.h>

#include "vector_math.cuh"
#include "simulation.cuh"
#include "kernel.cuh"

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

namespace SimConfig {
    const std::string HELIOSTAT_DATA_FILE = "data.csv";
    const float LATITUDE = 39.4f;
    const float LONGITUDE = 98.5f;
    const float ALTITUDE_KM = 3.0f;
    const float HELIOSTAT_WIDTH = 6.0f;
    const float HELIOSTAT_HEIGHT = 6.0f;
    const float RECEIVER_RADIUS = 3.5f;
    const float RECEIVER_HEIGHT = 8.0f;
    const float3 AIM_POINT = make_float3(0.0f, 0.0f, 80.0f);
    const float3 RECEIVER_GEOMETRY_BASE = make_float3(0.0f, 0.0f, 76.0f);
    const float MIRROR_REFLECTIVITY = 0.92f;
    const float SUN_CONE_HALF_ANGLE_MRAD = 4.65f; // 可以根据需要改回10.0f以使用Buie模型
    const float NORMAL_PERTURBATION_SIGMA_MRAD = 0.0f; // 您设定的测试值
    const int RAYS_PER_MICROFACET = 512;
    const float MICROFACET_SIZE = 0.01f;
}

float calculateDNI(float altitude_km, float sun_altitude_rad) {
    if (sun_altitude_rad <= 0) return 0.0f;
    const float G0 = 1.366f;
    float a = 0.4237f - 0.00821f * powf(6.0f - altitude_km, 2);
    float b = 0.5055f + 0.00595f * powf(6.5f - altitude_km, 2);
    float c = 0.2711f + 0.01858f * powf(2.5f - altitude_km, 2);
    float sin_alpha_s = sinf(sun_altitude_rad);
    if (sin_alpha_s < 1e-6) return 0.0f;
    return G0 * (a + b * expf(-c / sin_alpha_s));
}

// [新增] 辅助函数，用于将角度转换为弧度
static inline float degreesToRadians(float degrees) {
    return degrees * 3.1415926535f / 180.0f;
}


int main(int argc, char** argv) {
    std::cout << "======================================================" << std::endl;
    std::cout << " Tower Solar Power Plant Simulation - 2023-A-Q1" << std::endl;
    std::cout << "======================================================" << std::endl;

    // ...
        // =========================================================================
        // *** [可视化修改 1/3] 添加控制变量和数据缓冲区 ***
        // =========================================================================
    const bool GENERATE_VISUALIZATION_DATA = true;
    const int TARGET_HELIOSTAT_ID = 1000;
    const int VIS_TARGET_MONTH = 1;
    const float VIS_TARGET_HOUR = 9.0f;

    MicrofacetShadingData* d_shading_data = nullptr;
    // =========================================================================

    // --- 1. & 2. 初始化与GPU资源分配 ---
    std::cout << "\n[Phase 1&2] Initializing CPU/GPU resources..." << std::endl;
    std::vector<Heliostat> h_heliostats;
    loadHeliostatData(SimConfig::HELIOSTAT_DATA_FILE, h_heliostats, SimConfig::HELIOSTAT_WIDTH, SimConfig::HELIOSTAT_HEIGHT);
    const int num_heliostats = h_heliostats.size();
    if (num_heliostats == 0) { std::cerr << "Error: No data loaded." << std::endl; return -1; }

    AccelerationGrid h_grid;
    initializeAccelerationGrid(h_heliostats, h_grid, make_float3(10.f, 10.f, 10.f)); // 您可以调整格子大小
    std::cout << "-> Acceleration grid structure initialized." << std::endl;

    Receiver h_receiver;
    h_receiver.center = SimConfig::RECEIVER_GEOMETRY_BASE;
    h_receiver.radius = SimConfig::RECEIVER_RADIUS;
    h_receiver.height = SimConfig::RECEIVER_HEIGHT;

    Heliostat* d_heliostats;
    Receiver* d_receiver;
    AccelerationGrid* d_grid_struct;
    AtomicCounters* d_counters;
    CHECK_CUDA_ERROR(cudaMalloc(&d_heliostats, num_heliostats * sizeof(Heliostat)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_receiver, sizeof(Receiver)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grid_struct, sizeof(AccelerationGrid)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_counters, sizeof(AtomicCounters)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_starts, h_grid.cpu_cell_starts.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_entries, h_grid.cpu_cell_entries.size() * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_receiver, &h_receiver, sizeof(Receiver), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_starts, h_grid.cpu_cell_starts.data(), h_grid.cpu_cell_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_entries, h_grid.cpu_cell_entries.data(), h_grid.cpu_cell_entries.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grid_struct, &h_grid, sizeof(AccelerationGrid), cudaMemcpyHostToDevice));

    const int microfacets_per_heliostat_x = static_cast<int>(SimConfig::HELIOSTAT_WIDTH / SimConfig::MICROFACET_SIZE);
    const int num_microfacets_per_helio = microfacets_per_heliostat_x * microfacets_per_heliostat_x;
    const unsigned long long total_microfacets = (unsigned long long)num_heliostats * num_microfacets_per_helio;
    const unsigned long long total_rays_to_cast_per_timepoint = total_microfacets * SimConfig::RAYS_PER_MICROFACET;

    // =========================================================================
 // *** [可视化修改 2/3] 分配GPU内存 (修正版) ***
 // =========================================================================
 // 我们在 main 函数开头已经声明过 d_shading_data = nullptr;
 // 这里我们只在需要时为它分配内存，不再重新声明。
    if (GENERATE_VISUALIZATION_DATA) {
        // 之前我们为 total_microfacets 分配了空间，但为了节省内存，
        // 我们可以只为单个目标定日镜分配空间，但这需要修改Kernel。
        // 让我们坚持之前的方案：为所有微面元分配空间。
        CHECK_CUDA_ERROR(cudaMalloc(&d_shading_data, total_microfacets * sizeof(MicrofacetShadingData)));
    }
    // =========================================================================

    std::cout << "-> Initialization complete. Simulation scale:" << std::endl;
    // ...


    // --- 3. 仿真循环与数据累加 ---
    std::cout << "\n[Phase 3/5] Starting simulation loop..." << std::endl;

    std::ofstream ray_log_file("ray_stats.log"); // [新增] 创建日志文件

    unsigned long long grand_total_rays_cast = 0;
    unsigned long long grand_total_shadow_blocked = 0;
    unsigned long long grand_total_hit = 0;

    std::map<int, std::vector<double>> monthly_cosine_eff, monthly_sb_eff, monthly_trunc_eff, monthly_optical_eff, monthly_power_per_area;

    const int months[] = { 1 };
    const float hours[] = { 9.0f };

    int total_timepoints_to_run = sizeof(months) / sizeof(int) * sizeof(hours) / sizeof(int);
    int current_timepoint = 0;

    for (int month : months) {
        for (float hour : hours) {
            current_timepoint++;
            // [优化] 更新控制台进度条
            std::cout << "\r-> Simulating... [" << std::setw(3) << current_timepoint << "/" << total_timepoints_to_run << "] "
                << "Month: " << std::setw(2) << month << ", Hour: " << std::fixed << std::setprecision(2) << hour << std::flush;
            float3 sun_direction = calculateSunPosition(month, 21, hour, SimConfig::LATITUDE, SimConfig::LONGITUDE);
            float sun_altitude_rad = asinf(sun_direction.z);
            float current_dni = calculateDNI(SimConfig::ALTITUDE_KM, sun_altitude_rad);

            for (int i = 0; i < num_heliostats; ++i) {
                float3 to_receiver = normalize(SimConfig::AIM_POINT - h_heliostats[i].center);
                h_heliostats[i].ideal_normal = normalize(sun_direction + to_receiver);
            }

            // [重构] 2. 根据当前姿态，动态填充加速网格
            populateAccelerationGrid(h_heliostats, h_grid);

            // [重构] 3. 将更新后的网格数据上传到GPU
            // 我们需要检查GPU上是否已有足够空间，不够则重新分配
            // 一个简单的方法是每次都重新分配，虽然效率略低，但逻辑最清晰
            if (h_grid.d_cell_entries) CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_entries));
            CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_entries, h_grid.cpu_cell_entries.size() * sizeof(int)));

            // 拷贝数据
            CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_starts, h_grid.cpu_cell_starts.data(), h_grid.cpu_cell_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_entries, h_grid.cpu_cell_entries.data(), h_grid.cpu_cell_entries.size() * sizeof(int), cudaMemcpyHostToDevice));

            // 更新GPU端的Grid结构体，确保指针正确
            CHECK_CUDA_ERROR(cudaMemcpy(d_grid_struct, &h_grid, sizeof(AccelerationGrid), cudaMemcpyHostToDevice));

            CHECK_CUDA_ERROR(cudaMemcpy(d_heliostats, h_heliostats.data(), num_heliostats * sizeof(Heliostat), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemset(d_counters, 0, sizeof(AtomicCounters)));

            const int threads_per_block = 256;
            const int blocks_per_grid = (total_microfacets + threads_per_block - 1) / threads_per_block;

            rayTracingKernel << <blocks_per_grid, threads_per_block >> > (
                d_heliostats, num_heliostats, d_receiver, d_grid_struct, sun_direction, d_counters,
                SimConfig::RAYS_PER_MICROFACET, microfacets_per_heliostat_x,
                SimConfig::SUN_CONE_HALF_ANGLE_MRAD / 1000.0f,
                SimConfig::NORMAL_PERTURBATION_SIGMA_MRAD / 1000.0f,
                d_shading_data // *** [可视化修改] 添加新参数 ***
                );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            AtomicCounters h_counters;
            CHECK_CUDA_ERROR(cudaMemcpy(&h_counters, d_counters, sizeof(AtomicCounters), cudaMemcpyDeviceToHost));

            // [新增] 将当前时间点的光线计数累加到全局总和中
            grand_total_rays_cast += total_rays_to_cast_per_timepoint;
            grand_total_shadow_blocked += h_counters.shadow_rays + h_counters.blocked_rays;
            grand_total_hit += h_counters.hit_rays;

            // [新增] 将详细光线统计写入日志文件
            ray_log_file << "Month: " << month << ", Hour: " << hour << "\n"
                << "  Total Rays: " << total_rays_to_cast_per_timepoint << "\n"
                << "  Shadow/Block: " << h_counters.shadow_rays + h_counters.blocked_rays << "\n"
                << "  Hit: " << h_counters.hit_rays << "\n\n";

            // --- 计算各项效率 ---
            unsigned long long valid_rays_for_trunc_test = total_rays_to_cast_per_timepoint - h_counters.shadow_rays - h_counters.blocked_rays;

            double cosine_eff = 0;
            for (const auto& h : h_heliostats) cosine_eff += dot(sun_direction, h.ideal_normal);
            cosine_eff = (num_heliostats > 0) ? cosine_eff / num_heliostats : 0.0;

            double sb_eff = (total_rays_to_cast_per_timepoint > 0) ? (double)valid_rays_for_trunc_test / total_rays_to_cast_per_timepoint : 0.0;
            double trunc_eff = (valid_rays_for_trunc_test > 0) ? (double)h_counters.hit_rays / valid_rays_for_trunc_test : 0.0;

            double total_atm_eff = 0.0;
            if (num_heliostats > 0) for (const auto& h : h_heliostats) {
                float d_HR = length(h.center - SimConfig::AIM_POINT);
                total_atm_eff += (d_HR <= 1000.0f) ? (0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR * d_HR) : (0.99321 - 0.0001176 * 1000.0 + 1.97e-8 * 1000.0 * 1000.0);
            }
            double avg_atm_eff = (num_heliostats > 0) ? total_atm_eff / num_heliostats : 0.0;

            double optical_eff = cosine_eff * sb_eff * trunc_eff * avg_atm_eff * SimConfig::MIRROR_REFLECTIVITY;
            double current_power_per_area = current_dni * optical_eff;

            // --- 存储月度数据 ---
            monthly_cosine_eff[month].push_back(cosine_eff);
            monthly_sb_eff[month].push_back(sb_eff);
            monthly_trunc_eff[month].push_back(trunc_eff);
            monthly_optical_eff[month].push_back(optical_eff);
            monthly_power_per_area[month].push_back(current_power_per_area);
        }
    }

    // 在 for (int month : months) 循环结束之后

    // [新增] 在关闭日志文件前，写入全局统计总结
    ray_log_file << "========================================\n"
        << "           Grand Total Summary          \n"
        << "========================================\n"
        << "Total Rays Cast Across All Time Points : " << grand_total_rays_cast << "\n"
        << "Total Shadowed/Blocked Rays          : " << grand_total_shadow_blocked << "\n"
        << "Total Hit Receiver Rays              : " << grand_total_hit << "\n";
    ray_log_file.close();
    std::cout << "\nSimulation loop finished." << std::endl;

    // =========================================================================
   // *** [可视化修改 3/3] 数据取回与文件写入 ***
   // =========================================================================
    if (GENERATE_VISUALIZATION_DATA) {
        std::cout << "\n-> Generating visualization data for Heliostat ID: " << TARGET_HELIOSTAT_ID << "..." << std::endl;

        // 1. 从GPU取回目标定日镜的数据
        std::vector<MicrofacetShadingData> h_shading_data(num_microfacets_per_helio);
        unsigned long long offset = (unsigned long long)TARGET_HELIOSTAT_ID * num_microfacets_per_helio;

        CHECK_CUDA_ERROR(cudaMemcpy(h_shading_data.data(), d_shading_data + offset, num_microfacets_per_helio * sizeof(MicrofacetShadingData), cudaMemcpyDeviceToHost));

        // 2. 将数据写入CSV文件
        std::string viz_filename = "heliostat_" + std::to_string(TARGET_HELIOSTAT_ID) + "_shading_map.csv";
        std::ofstream viz_file(viz_filename);

        viz_file << "X_Index,Y_Index,Shading_Efficiency\n";
        for (int j = 0; j < microfacets_per_heliostat_x; ++j) {
            for (int i = 0; i < microfacets_per_heliostat_x; ++i) {
                int index = j * microfacets_per_heliostat_x + i;
                unsigned int total_losses = h_shading_data[index].shadow_ray_losses + h_shading_data[index].block_ray_losses;

                double efficiency = 1.0 - (double)total_losses / SimConfig::RAYS_PER_MICROFACET;

                viz_file << i << "," << j << "," << std::fixed << std::setprecision(4) << efficiency << "\n";
            }
        }
        viz_file.close();
        std::cout << "-> Shading map saved to " << viz_filename << std::endl;
    }
    // =========================================================================

    // --- 4. 结果输出到CSV文件 ---
    std::cout << "\n[Phase 4/5] Writing results to results.csv..." << std::endl;
    std::ofstream csv_file("results.csv");
    csv_file << std::fixed << std::setprecision(4);

    // 表1
    csv_file << "Table 1: Monthly Averages\n";
    csv_file << "Date,Avg. Optical Eff.,Avg. Cosine Eff.,Avg. S/B Eff.,Avg. Truncation Eff.,Power per Area (kW/m^2)\n";

    double year_total_cosine = 0, year_total_sb = 0, year_total_trunc = 0, year_total_optical = 0, year_total_power = 0;

    for (int month : months) {
        double month_sum_cos = 0, month_sum_sb = 0, month_sum_trunc = 0, month_sum_optical = 0, month_sum_power = 0;
        int count = monthly_cosine_eff[month].size();

        for (double val : monthly_cosine_eff[month]) month_sum_cos += val;
        for (double val : monthly_sb_eff[month]) month_sum_sb += val;
        for (double val : monthly_trunc_eff[month]) month_sum_trunc += val;
        for (double val : monthly_optical_eff[month]) month_sum_optical += val;
        for (double val : monthly_power_per_area[month]) month_sum_power += val;

        double avg_opt = (count > 0) ? month_sum_optical / count : 0.0;
        double avg_cos = (count > 0) ? month_sum_cos / count : 0.0;
        double avg_sb = (count > 0) ? month_sum_sb / count : 0.0;
        double avg_trunc = (count > 0) ? month_sum_trunc / count : 0.0;
        double avg_power = (count > 0) ? month_sum_power / count : 0.0;

        csv_file << month << "/21," << avg_opt << "," << avg_cos << "," << avg_sb << "," << avg_trunc << "," << avg_power << "\n";

        year_total_optical += month_sum_optical;
        year_total_cosine += month_sum_cos;
        year_total_sb += month_sum_sb;
        year_total_trunc += month_sum_trunc;
        year_total_power += month_sum_power;
    }

    // 表2
    double annual_avg_optical = (total_timepoints_to_run > 0) ? year_total_optical / total_timepoints_to_run : 0.0;
    double annual_avg_cosine = (total_timepoints_to_run > 0) ? year_total_cosine / total_timepoints_to_run : 0.0;
    double annual_avg_sb = (total_timepoints_to_run > 0) ? year_total_sb / total_timepoints_to_run : 0.0;
    double annual_avg_trunc = (total_timepoints_to_run > 0) ? year_total_trunc / total_timepoints_to_run : 0.0;
    double annual_avg_power_per_area = (total_timepoints_to_run > 0) ? year_total_power / total_timepoints_to_run : 0.0;
    double total_mirror_area = num_heliostats * SimConfig::HELIOSTAT_WIDTH * SimConfig::HELIOSTAT_HEIGHT;
    double annual_avg_total_power_MW = annual_avg_power_per_area * total_mirror_area / 1000.0;

    csv_file << "\n\nTable 2: Annual Averages\n";
    csv_file << "Metric,Value\n";
    csv_file << "Year Avg Optical Efficiency," << annual_avg_optical << "\n";
    csv_file << "Year Avg Cosine Efficiency," << annual_avg_cosine << "\n";
    csv_file << "Year Avg Shadow/Block Efficiency," << annual_avg_sb << "\n";
    csv_file << "Year Avg Truncation Efficiency," << annual_avg_trunc << "\n";
    csv_file << "Year Avg Heat Power (MW)," << annual_avg_total_power_MW << "\n";
    csv_file << "Year Avg Power per Area (kW/m^2)," << annual_avg_power_per_area << "\n";

    csv_file.close();
    std::cout << "-> Results successfully written to results.csv." << std::endl;
    std::cout << "-> Detailed ray statistics saved to ray_stats.log." << std::endl;


    // --- 5. 释放资源 ---
    std::cout << "\n[Phase 5/5] Cleaning up..." << std::endl;
    // *** [可视化修改] 释放缓冲区内存 ***
    if (d_shading_data) CHECK_CUDA_ERROR(cudaFree(d_shading_data));
    CHECK_CUDA_ERROR(cudaFree(d_heliostats));
    CHECK_CUDA_ERROR(cudaFree(d_receiver));
    CHECK_CUDA_ERROR(cudaFree(d_grid_struct));
    CHECK_CUDA_ERROR(cudaFree(d_counters));
    CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_starts));
    CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_entries));
    std::cout << "\nSimulation finished successfully." << std::endl;

    return 0;
}