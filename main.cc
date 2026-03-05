/*
 * DetectUav_RK3588 主程序（MPP 硬解码 + YOLO26 推理 + OSD 显示 + NV12 SDL 预览）
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

#include "nodes/infer/vp_rk_first_yolo26.h"
#include "nodes/infer/vp_yolo26_preprocess_node.h"
#include "nodes/track/vp_nanotrack_node.h"
#include "nodes/osd/vp_osd_node.h"
#include "nodes/vp_bgr_to_nv12_node.h"
#include "nodes/vp_mpp_sdl_src_node.h"
#include "nodes/vp_nv12_sdl_des_node.h"
#include "vp_utils/analysis_board/vp_analysis_board.h"

// 进程退出标志。
static std::atomic<bool> g_should_exit{false};

/**
 * @brief 终端原始模式守卫（用于捕获 ESC，无需回车）。
 */
class TerminalRawModeGuard {
public:
    /**
     * @brief 构造并尝试切换终端到非规范模式。
     */
    TerminalRawModeGuard() {
        // 标准输入是否为 TTY。
        const bool is_tty = (isatty(STDIN_FILENO) == 1);
        if (!is_tty) {
            return;
        }

        // 旧终端配置读取结果。
        const int get_ret = tcgetattr(STDIN_FILENO, &old_termios_);
        if (get_ret != 0) {
            return;
        }

        // 新终端配置（基于旧配置拷贝）。
        struct termios new_termios = old_termios_;
        new_termios.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        new_termios.c_cc[VMIN] = 0;
        new_termios.c_cc[VTIME] = 0;

        // 切换终端配置结果。
        const int set_ret = tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
        if (set_ret == 0) {
            enabled_ = true;
        }
    }

    /**
     * @brief 析构时恢复终端配置。
     */
    ~TerminalRawModeGuard() {
        if (enabled_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
        }
    }

private:
    // 原终端配置。
    struct termios old_termios_ {};
    // 是否已成功启用原始模式。
    bool enabled_ = false;
};

/**
 * @brief 非阻塞检测终端是否按下 ESC。
 *
 * @return true 检测到 ESC。
 * @return false 未检测到 ESC。
 */
static bool check_terminal_escape_pressed() {
    // 读集合。
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(STDIN_FILENO, &read_fds);

    // 超时配置（立即返回）。
    struct timeval timeout {};
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    // select 结果。
    const int ready = select(STDIN_FILENO + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &read_fds)) {
        return false;
    }

    // 读取到的字符。
    unsigned char ch = 0;
    const ssize_t read_size = read(STDIN_FILENO, &ch, 1);
    if (read_size == 1 && ch == 27U) {
        return true;
    }
    return false;
}

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
 * @brief 解析可选命令行参数。
 *
 * 参数约定：
 * 1) argv[1]: 输入视频路径（可选）
 * 2) argv[2]: SDL 视频驱动（可选，如 x11/wayland/kmsdrm）
 * 3) argv[3]: SDL 渲染驱动（可选，如 opengl/opengles2）
 * 4) argv[4]: YOLO26 配置路径（可选）
 * 5) argv[5]: NanoTrack 配置路径（可选）
 * 6) argv[6]: 屏幕 sink（可选，如 ximagesink/waylandsink/kmssink）
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @param file_path 输出视频路径。
 * @param sdl_video_driver 输出 SDL 视频驱动。
 * @param sdl_render_driver 输出 SDL 渲染驱动。
 * @param yolo26_config_path 输出 YOLO26 配置路径。
 * @param nanotrack_config_path 输出 NanoTrack 配置路径。
 * @param screen_sink 输出屏幕 sink 名称。
 */
static void parse_args(int argc,
                       char** argv,
                       std::string& file_path,
                       std::string& sdl_video_driver,
                       std::string& sdl_render_driver,
                       std::string& yolo26_config_path,
                       std::string& nanotrack_config_path,
                       std::string& screen_sink) {
    if (argc > 1 && argv[1] != nullptr) {
        file_path = argv[1];
    }
    if (argc > 2 && argv[2] != nullptr) {
        sdl_video_driver = argv[2];
    }
    if (argc > 3 && argv[3] != nullptr) {
        sdl_render_driver = argv[3];
    }
    if (argc > 4 && argv[4] != nullptr) {
        yolo26_config_path = argv[4];
    }
    if (argc > 5 && argv[5] != nullptr) {
        nanotrack_config_path = argv[5];
    }
    if (argc > 6 && argv[6] != nullptr) {
        screen_sink = argv[6];
    }
}

/**
 * @brief 主程序入口，构建线性视频管线。
 *
 * `src -> yolo26_preprocess -> yolo26 -> nanotrack -> osd -> bgr_to_nv12 -> nv12_sdl_des`。
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @return int 进程退出码。
 */
int main(int argc, char** argv) {
    // 注册退出信号处理，支持 Ctrl+C 优雅退出。
    std::signal(SIGINT, handle_exit_signal);
    std::signal(SIGTERM, handle_exit_signal);

    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // 终端 ESC 监听守卫（自动恢复终端配置）。
    TerminalRawModeGuard terminal_raw_mode_guard;
    // 重置 NV12 SDL 窗口退出请求标志。
    vp_nodes::vp_nv12_sdl_des_reset_exit_flag();

    // 默认输入视频路径。
    std::string file_path = "/mnt/nfs/datasets/video/uav4.mp4";
    // SDL 视频驱动（默认自动选择）。
    std::string sdl_video_driver = "";
    // SDL 渲染驱动（默认自动选择）。
    std::string sdl_render_driver = "";
    // YOLO26 配置路径。
    std::string yolo26_config_path = "assets/configs/yolo26.json";
    // NanoTrack 配置路径。
    std::string nanotrack_config_path = "assets/configs/nanotrack.json";
    // 屏幕显示 sink（保留参数兼容，当前流程未使用该参数）。
    std::string screen_sink = "autovideosink";
    parse_args(argc, argv, file_path, sdl_video_driver, sdl_render_driver, yolo26_config_path, nanotrack_config_path, screen_sink);

    VP_INFO(vp_utils::string_format("[main] file=%s sdl_video_driver=%s sdl_render_driver=%s yolo26_cfg=%s nanotrack_cfg=%s sink=%s",
                                    file_path.c_str(),
                                    sdl_video_driver.empty() ? "auto" : sdl_video_driver.c_str(),
                                    sdl_render_driver.empty() ? "auto" : sdl_render_driver.c_str(),
                                    yolo26_config_path.c_str(),
                                    nanotrack_config_path.c_str(),
                                    screen_sink.c_str()));

    // MPP 文件源节点（纯硬解码并向下游下发 NV12 数据）。
    auto src_0 = std::make_shared<vp_nodes::vp_mpp_sdl_src_node>(
        "file_src_0",     // node_name：源节点名称。
        0,                // channel_index：通道索引。
        file_path,        // file_path：输入视频路径。
        true,             // cycle：是否循环播放。
        true             // pace_by_src_fps：是否按源帧率限速。
    );

    // YOLO26 预处理节点（直接处理 NV12，内部产出 BGR 与模型输入 RGB）。
    auto yolo26_pre_0 = std::make_shared<vp_nodes::vp_yolo26_preprocess_node>("yolo26_pre_0", yolo26_config_path);
    // YOLO26 检测节点。
    auto yolo26_0 = std::make_shared<vp_nodes::vp_rk_first_yolo26>("yolo26_0", yolo26_config_path);
    // NanoTrack 单目标跟踪节点。
    auto nanotrack_0 = std::make_shared<vp_nodes::vp_nanotrack_node>("nanotrack_0", nanotrack_config_path);
    // OSD 绘制节点。
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    // BGR 转 NV12 适配节点（把 OSD 结果转为 NV12 供 SDL 显示）。
    auto bgr_to_nv12_0 = std::make_shared<vp_nodes::vp_bgr_to_nv12_node>("bgr_to_nv12_0");
    // NV12 SDL 直显终端节点（显示解码原始画面）。
    auto nv12_des_0 = std::make_shared<vp_nodes::vp_nv12_sdl_des_node>(
        "nv12_des_0",      // node_name：终端节点名称。
        0,                  // channel_index：通道索引。
        sdl_video_driver,   // sdl_video_driver：SDL 视频驱动。
        sdl_render_driver   // sdl_render_driver：SDL 渲染驱动。
    );

    // 业务主链路（保持检测/OSD处理结构，输出改为 SDL NV12 显示）。
    yolo26_pre_0->attach_to({src_0});
    yolo26_0->attach_to({yolo26_pre_0});
    nanotrack_0->attach_to({yolo26_0});
    osd_0->attach_to({nanotrack_0});
    bgr_to_nv12_0->attach_to({osd_0});
    nv12_des_0->attach_to({bgr_to_nv12_0});

    src_0->start();

    // 数据流分析看板（非阻塞窗口）。
    vp_utils::vp_analysis_board board({src_0});
    board.display(1, false);

    while (!g_should_exit.load()) {
        if (vp_nodes::vp_nv12_sdl_des_should_exit()) {
            VP_INFO("[main] nv12 sdl exit requested, exiting...");
            g_should_exit.store(true);
            break;
        }
        if (check_terminal_escape_pressed()) {
            VP_INFO("[main] ESC detected from terminal, exiting...");
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 退出前拆链，触发各节点线程有序停止。
    src_0->detach_recursively();

    return 0;
}
