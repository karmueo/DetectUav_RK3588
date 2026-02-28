/*
 * DetectUav_RK3588 简化示例：MP4 读取 + MPP 硬解 + NV12 直显 SDL + 数据流窗口
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <string>
#include <thread>

#include "nodes/vp_mpp_sdl_src_node.h"
#include "nodes/vp_nv12_sdl_des_node.h"
#include "vp_utils/analysis_board/vp_analysis_board.h"
#include "vp_utils/logger/vp_logger.h"
#include "vp_utils/vp_utils.h"

// 进程退出标志。
static std::atomic<bool> g_should_exit{false};

/**
 * @brief 处理退出信号，仅设置退出标志位。
 *
 * @param signal_number 信号编号。
 */
static void handle_exit_signal(int signal_number) {
    (void)signal_number;
    g_should_exit.store(true);
}

/**
 * @brief 解析命令行参数。
 *
 * 参数约定：
 * 1) argv[1]: 输入 MP4 路径（可选）
 * 2) argv[2]: SDL 视频驱动（可选，如 x11/wayland/kmsdrm）
 * 3) argv[3]: SDL 渲染驱动（可选，如 opengl/opengles2）
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @param file_path 输出输入文件路径。
 * @param sdl_video_driver 输出 SDL 视频驱动。
 * @param sdl_render_driver 输出 SDL 渲染驱动。
 */
static void parse_args(int argc,
                       char** argv,
                       std::string& file_path,
                       std::string& sdl_video_driver,
                       std::string& sdl_render_driver) {
    if (argc > 1 && argv[1] != nullptr) {
        file_path = argv[1];
    }
    if (argc > 2 && argv[2] != nullptr) {
        sdl_video_driver = argv[2];
    }
    if (argc > 3 && argv[3] != nullptr) {
        sdl_render_driver = argv[3];
    }
}

/**
 * @brief 示例程序主入口。
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @return int 进程退出码。
 */
int main(int argc, char** argv) {
    // 注册退出信号处理。
    std::signal(SIGINT, handle_exit_signal);
    std::signal(SIGTERM, handle_exit_signal);

    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // 输入 MP4 路径。
    std::string file_path = "assets/videos/person.mp4";
    // SDL 视频驱动。
    std::string sdl_video_driver = "";
    // SDL 渲染驱动。
    std::string sdl_render_driver = "";
    parse_args(argc, argv, file_path, sdl_video_driver, sdl_render_driver);

    VP_INFO(vp_utils::string_format("[simple_main] file=%s sdl_video_driver=%s sdl_render_driver=%s",
                                    file_path.c_str(),
                                    sdl_video_driver.empty() ? "auto" : sdl_video_driver.c_str(),
                                    sdl_render_driver.empty() ? "auto" : sdl_render_driver.c_str()));

    // 重置 NV12 SDL 显示节点退出请求标志。
    vp_nodes::vp_nv12_sdl_des_reset_exit_flag();

    // MPP 文件源节点（纯硬解码并向下游下发 NV12 数据）。
    auto src_0 = std::make_shared<vp_nodes::vp_mpp_sdl_src_node>(
        "file_src_0",      // node_name：源节点名称。
        0,                 // channel_index：通道索引。
        file_path,         // file_path：输入 MP4 文件路径。
        true,              // cycle：是否循环播放。
        false              // pace_by_src_fps：是否按源帧率节奏限速。
    );
    // NV12 SDL 显示终端节点（直接显示解码后的 NV12 画面）。
    auto nv12_des_0 = std::make_shared<vp_nodes::vp_nv12_sdl_des_node>(
        "nv12_des_0",      // node_name：终端节点名称。
        0,                 // channel_index：通道索引。
        sdl_video_driver,  // sdl_video_driver：SDL 视频驱动。
        sdl_render_driver  // sdl_render_driver：SDL 渲染驱动。
    );
    nv12_des_0->attach_to({src_0});

    src_0->start();

    // 数据流分析看板（非阻塞窗口）。
    vp_utils::vp_analysis_board board({src_0});
    board.display(1, false);

    while (!g_should_exit.load()) {
        if (vp_nodes::vp_nv12_sdl_des_should_exit()) {
            VP_INFO("[simple_main] nv12 sdl exit requested, exiting...");
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 拆链触发线程有序退出。
    src_0->detach_recursively();

    return 0;
}
