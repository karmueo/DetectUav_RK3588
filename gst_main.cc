/*
 * DetectUav_RK3588 GStreamer 显示主程序（MPP 硬解码 + YOLO26 推理 + OSD 显示 + GStreamer sink 预览）
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
#include "nodes/osd/vp_osd_node.h"
#include "nodes/vp_mpp_sdl_src_node.h"
#include "nodes/vp_nv12_to_bgr_node.h"
#include "nodes/vp_screen_des_node.h"
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
 * 2) argv[2]: YOLO26 配置路径（可选）
 * 3) argv[3]: 屏幕 sink（可选，如 ximagesink/waylandsink/kmssink/autovideosink）
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @param file_path 输出视频路径。
 * @param yolo26_config_path 输出 YOLO26 配置路径。
 * @param screen_sink 输出屏幕 sink 名称。
 */
static void parse_args(int argc,
                       char** argv,
                       std::string& file_path,
                       std::string& yolo26_config_path,
                       std::string& screen_sink) {
    if (argc > 1 && argv[1] != nullptr) {
        file_path = argv[1];
    }
    if (argc > 2 && argv[2] != nullptr) {
        yolo26_config_path = argv[2];
    }
    if (argc > 3 && argv[3] != nullptr) {
        screen_sink = argv[3];
    }
}

/**
 * @brief 主程序入口，构建线性视频管线。
 *
 * `src -> nv12_to_bgr -> yolo26 -> osd -> screen_des(gstreamer sink)`。
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
    // 重置 screen_des 退出请求标志。
    vp_nodes::vp_screen_des_reset_exit_flag();

    // 默认输入视频路径。
    std::string file_path = "/mnt/nfs/datasets/video/uav.mp4";
    // YOLO26 配置路径。
    std::string yolo26_config_path = "assets/configs/yolo26.json";
    // 屏幕显示 sink 名称。
    std::string screen_sink = "autovideosink";
    parse_args(argc, argv, file_path, yolo26_config_path, screen_sink);

    VP_INFO(vp_utils::string_format("[gst_main] file=%s yolo26_cfg=%s sink=%s",
                                    file_path.c_str(),
                                    yolo26_config_path.c_str(),
                                    screen_sink.c_str()));

    // MPP 文件源节点（纯硬解码并向下游下发 NV12 数据）。
    auto src_0 = std::make_shared<vp_nodes::vp_mpp_sdl_src_node>(
        "file_src_0",     // node_name：源节点名称。
        0,                // channel_index：通道索引。
        file_path,        // file_path：输入视频路径。
        true,             // cycle：是否循环播放。
        false             // pace_by_src_fps：是否按源帧率限速。
    );

    // NV12 转 BGR 适配节点（供推理/OSD/显示链路使用）。
    auto nv12_to_bgr_0 = std::make_shared<vp_nodes::vp_nv12_to_bgr_node>("nv12_to_bgr_0");
    // YOLO26 检测节点。
    auto yolo26_0 = std::make_shared<vp_nodes::vp_rk_first_yolo26>("yolo26_0", yolo26_config_path);
    // OSD 绘制节点。
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    // GStreamer 屏显终端节点。
    auto gst_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>(
        "gst_des_0",      // node_name：终端节点名称。
        0,                 // channel_index：通道索引。
        true,              // osd：优先显示 OSD 帧。
        vp_objects::vp_size{}, // display_w_h：输出分辨率（空表示跟随源）。
        false,             // fast_mode：是否启用快速模式。
        screen_sink        // video_sink：GStreamer sink。
    );

    // 业务主链路（前序节点保持一致，终端切换为 GStreamer sink 显示）。
    nv12_to_bgr_0->attach_to({src_0});
    yolo26_0->attach_to({nv12_to_bgr_0});
    osd_0->attach_to({yolo26_0});
    gst_des_0->attach_to({osd_0});

    src_0->start();

    // 数据流分析看板（非阻塞窗口）。
    vp_utils::vp_analysis_board board({src_0});
    board.display(1, false);

    while (!g_should_exit.load()) {
        if (vp_nodes::vp_screen_des_should_exit()) {
            VP_INFO("[gst_main] screen des exit requested, exiting...");
            g_should_exit.store(true);
            break;
        }
        if (check_terminal_escape_pressed()) {
            VP_INFO("[gst_main] ESC detected from terminal, exiting...");
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 退出前拆链，触发各节点线程有序停止。
    src_0->detach_recursively();

    return 0;
}
