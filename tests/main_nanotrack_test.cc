/**
 * @file main_nanotrack_test.cc
 * @brief NanoTrack单目标跟踪测试程序
 *
 * 测试流水线: 文件源 -> YOLO26预处理 -> YOLO检测 -> NanoTrack跟踪 -> OSD -> fakesink
 * 无GUI显示
 *
 * 测试视频路径: /mnt/nfs/datasets/video/uav.mp4
 * NanoTrack配置: assets/configs/nanotrack.json
 */

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include "nodes/vp_file_src_node.h"
#include "nodes/infer/vp_yolo26_preprocess_node.h"
#include "nodes/infer/vp_rk_first_yolo26.h"
#include "nodes/track/vp_nanotrack_node.h"
#include "nodes/osd/vp_osd_node.h"
#include "nodes/vp_fakesink_des_node.h"
#include "vp_utils/logger/vp_logger.h"

static std::atomic<bool> g_should_exit{false};

static void handle_exit_signal(int signal_number) {
    (void)signal_number;
    g_should_exit.store(true);
}

int main(int argc, char** argv) {
    // Register signal handlers
    std::signal(SIGINT, handle_exit_signal);
    std::signal(SIGTERM, handle_exit_signal);

    // Initialize logger first
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // 默认配置
    std::string video_path = "/mnt/nfs/datasets/video/uav4.mp4";
    std::string yolo_config = "assets/configs/yolo26.json";
    std::string nanotrack_config = "assets/configs/nanotrack.json";

    // 解析命令行参数
    if (argc > 1 && argv[1] != nullptr) {
        video_path = argv[1];
    }
    if (argc > 2 && argv[2] != nullptr) {
        yolo_config = argv[2];
    }
    if (argc > 3 && argv[3] != nullptr) {
        nanotrack_config = argv[3];
    }

    // 检查文件是否存在
    std::ifstream ftest(video_path);
    if (!ftest.good()) {
        std::cerr << "Error: Video file not found: " << video_path << std::endl;
        return 1;
    }

    std::ifstream fconfig(yolo_config);
    if (!fconfig.good()) {
        std::cerr << "Error: YOLO config file not found: " << yolo_config << std::endl;
        return 1;
    }

    std::ifstream fnanotrack(nanotrack_config);
    if (!fnanotrack.good()) {
        std::cerr << "Error: NanoTrack config file not found: " << nanotrack_config << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "NanoTrack Test Configuration:" << std::endl;
    std::cout << "  Video path: " << video_path << std::endl;
    std::cout << "  YOLO config: " << yolo_config << std::endl;
    std::cout << "  NanoTrack config: " << nanotrack_config << std::endl;
    std::cout << "========================================" << std::endl;

    // 创建流水线节点
    // 1. 文件源节点 - 读取视频帧 (node_name, channel_index, file_path, resize_ratio, cycle, gst_decoder_name)
    // 使用 mppvideodec 硬件解码器替代 avdec_h264 软解码器
    auto file_src = std::make_shared<vp_nodes::vp_file_src_node>("file_src", 0, video_path, 1.0, false, "mppvideodec");

    // 2. YOLO26预处理节点 (node_name, json_path)
    auto yolo_preprocess = std::make_shared<vp_nodes::vp_yolo26_preprocess_node>("yolo_preprocess", yolo_config);

    // 3. YOLO检测节点 (node_name, json_path)
    auto yolo_detect = std::make_shared<vp_nodes::vp_rk_first_yolo26>("yolo_detect", yolo_config);

    // 4. NanoTrack跟踪节点 (node_name, json_path)
    auto nanotrack = std::make_shared<vp_nodes::vp_nanotrack_node>("nanotrack", nanotrack_config);

    // 5. OSD节点 (node_name)
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");

    // 6. fakesink节点 (node_name, channel_index, osd, gst_encoder_name)
    auto fakesink = std::make_shared<vp_nodes::vp_fakesink_des_node>("fakesink", 0, true);

    // 构建流水线: file_src -> yolo_preprocess -> yolo_detect -> nanotrack -> osd -> fakesink
    yolo_preprocess->attach_to({file_src});
    yolo_detect->attach_to({yolo_preprocess});
    nanotrack->attach_to({yolo_detect});
    osd->attach_to({nanotrack});
    fakesink->attach_to({osd});

    // 启动流水线
    std::cout << "Starting NanoTrack test pipeline..." << std::endl;
    file_src->start();

    // 等待退出信号 (Ctrl+C) 或超时
    std::cout << "Pipeline running. Press Ctrl+C to exit..." << std::endl;

    // 运行30秒后自动退出
    int elapsed = 0;
    const int timeout_seconds = 30;
    while (!g_should_exit.load() && elapsed < timeout_seconds) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        elapsed++;
    }

    if (elapsed >= timeout_seconds) {
        std::cout << "Timeout reached, exiting..." << std::endl;
    }

    // 释放资源
    std::cout << "Stopping pipeline..." << std::endl;
    file_src->detach_recursively();

    return 0;
}
