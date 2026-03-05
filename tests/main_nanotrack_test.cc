/**
 * @file main_nanotrack_test.cc
 * @brief Test program for NanoTrack single object tracking node.
 *
 * This test demonstrates the NanoTrack tracker functionality using the
 * VideoPipe pipeline framework:
 * 1. File source -> YOLO26 preprocess -> YOLO detection -> NanoTrack -> OSD -> fakesink
 * 2. No GUI display, uses vp_analysis_board for monitoring
 *
 * Usage: ./main_nanotrack_test <video_path> [config_path]
 */

#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>

#include "vp_file_src_node.h"
#include "vp_yolo26_preprocess_node.h"
#include "vp_rk_first_yolo_node.h"
#include "vp_nanotrack_node.h"
#include "vp_osd_node.h"
#include "vp_fakesink_des_node.h"
#include "vp_utils/vp_analysis_board/vp_analysis_board.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [nanotrack_config_path]" << std::endl;
        std::cerr << "  nanotrack_config_path defaults to assets/configs/nanotrack.json" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];
    std::string nanotrack_config = (argc > 2) ? argv[2] : "assets/configs/nanotrack.json";

    // Log configuration
    spdlog::info("NanoTrack Test Configuration:");
    spdlog::info("  Video path: {}", video_path);
    spdlog::info("  NanoTrack config: {}", nanotrack_config);

    // Create pipeline nodes
    // 1. File source node - reads video frames
    auto file_src = std::make_shared<vp_nodes::vp_file_src_node>("file_src", video_path, 1.0, false);

    // 2. YOLO26 preprocess node - prepares frames for YOLO inference
    auto yolo_preprocess = std::make_shared<vp_nodes::vp_yolo26_preprocess_node>("yolo_preprocess");

    // 3. YOLO detection node - detects objects using RKNN
    std::string yolo_config = "assets/configs/uav.json";
    auto yolo_detect = std::make_shared<vp_nodes::vp_rk_first_yolo_node>("yolo_detect", yolo_config);

    // 4. NanoTrack node - single object tracking
    auto nanotrack = std::make_shared<vp_nodes::vp_nanotrack_node>("nanotrack", nanotrack_config);

    // 5. OSD node - draw tracking results
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");

    // 6. Fake sink node - terminates pipeline without display
    auto fakesink = std::make_shared<vp_nodes::vp_fakesink_des_node>("fakesink");

    // Build pipeline: file_src -> preprocess -> yolo -> nanotrack -> osd -> fakesink
    file_src->attach_to({yolo_preprocess});
    yolo_preprocess->attach_to({yolo_detect});
    yolo_detect->attach_to({nanotrack});
    nanotrack->attach_to({osd});
    osd->attach_to({fakesink});

    // Start pipeline
    spdlog::info("Starting NanoTrack test pipeline...");
    auto start_time = std::chrono::steady_clock::now();

    // Use analysis board to monitor pipeline (optional, for debugging)
    vp_utils::vp_analysis_board board;
    board.consume(file_src);

    // Wait for pipeline to complete
    file_src->wait();
    fakesink->wait();

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    spdlog::info("NanoTrack test completed in {} ms", duration.count());

    return 0;
}
