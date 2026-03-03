/*
 * RTSP 推流主程序（MPP 硬解码 + YOLO26 推理 + OSD 显示 + RTSP H.264 推流）
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
#include "nodes/osd/vp_osd_node.h"
#include "nodes/vp_mpp_sdl_src_node.h"
#include "nodes/vp_rtsp_des_node.h"
#include "vp_utils/analysis_board/vp_analysis_board.h"

static std::atomic<bool> g_should_exit{false};

class TerminalRawModeGuard {
public:
    TerminalRawModeGuard() {
        const bool is_tty = (isatty(STDIN_FILENO) == 1);
        if (!is_tty) {
            return;
        }

        const int get_ret = tcgetattr(STDIN_FILENO, &old_termios_);
        if (get_ret != 0) {
            return;
        }

        struct termios new_termios = old_termios_;
        new_termios.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        new_termios.c_cc[VMIN] = 0;
        new_termios.c_cc[VTIME] = 0;

        const int set_ret = tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
        if (set_ret == 0) {
            enabled_ = true;
        }
    }

    ~TerminalRawModeGuard() {
        if (enabled_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
        }
    }

private:
    struct termios old_termios_ {};
    bool enabled_ = false;
};

static bool check_terminal_escape_pressed() {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(STDIN_FILENO, &read_fds);

    struct timeval timeout {};
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    const int ready = select(STDIN_FILENO + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &read_fds)) {
        return false;
    }

    unsigned char ch = 0;
    const ssize_t read_size = read(STDIN_FILENO, &ch, 1);
    return (read_size == 1 && ch == 27U);
}

static void handle_exit_signal(int signal_number) {
    (void)signal_number;
    g_should_exit.store(true);
}

static bool parse_bool_arg(const std::string& value, bool default_value) {
    if (value.empty()) {
        return default_value;
    }
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "OFF") {
        return false;
    }
    return default_value;
}

static void parse_args(int argc,
                       char** argv,
                       std::string& file_path,
                       std::string& yolo26_config_path,
                       std::string& rtsp_url,
                       bool& enable_rtsp_push,
                       std::string& rtsp_encoder) {
    if (argc > 1 && argv[1] != nullptr) {
        file_path = argv[1];
    }
    if (argc > 2 && argv[2] != nullptr) {
        yolo26_config_path = argv[2];
    }
    if (argc > 3 && argv[3] != nullptr) {
        rtsp_url = argv[3];
    }
    if (argc > 4 && argv[4] != nullptr) {
        enable_rtsp_push = parse_bool_arg(argv[4], enable_rtsp_push);
    }
    if (argc > 5 && argv[5] != nullptr) {
        rtsp_encoder = argv[5];
    }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_exit_signal);
    std::signal(SIGTERM, handle_exit_signal);

    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    TerminalRawModeGuard terminal_raw_mode_guard;
    std::string file_path = "/mnt/nfs/datasets/video/uav.mp4";
    std::string yolo26_config_path = "assets/configs/yolo26.json";
    std::string rtsp_url = "rtsp://127.0.0.1:8554/live/uav";
    bool enable_rtsp_push = true;
    std::string rtsp_encoder = "mpph264enc";

    parse_args(argc,
               argv,
               file_path,
               yolo26_config_path,
               rtsp_url,
               enable_rtsp_push,
               rtsp_encoder);

    VP_INFO(vp_utils::string_format("[rtsp_main] file=%s yolo26_cfg=%s rtsp_url=%s rtsp_enable=%d rtsp_encoder=%s",
                                    file_path.c_str(),
                                    yolo26_config_path.c_str(),
                                    rtsp_url.c_str(),
                                    enable_rtsp_push ? 1 : 0,
                                    rtsp_encoder.c_str()));

    auto src_0 = std::make_shared<vp_nodes::vp_mpp_sdl_src_node>(
        "file_src_0",
        0,
        file_path,
        true,
        true);

    auto yolo26_pre_0 = std::make_shared<vp_nodes::vp_yolo26_preprocess_node>("yolo26_pre_0", yolo26_config_path);
    auto yolo26_0 = std::make_shared<vp_nodes::vp_rk_first_yolo26>("yolo26_0", yolo26_config_path);
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");

    yolo26_pre_0->attach_to({src_0});
    yolo26_0->attach_to({yolo26_pre_0});
    osd_0->attach_to({yolo26_0});

    std::shared_ptr<vp_nodes::vp_rtsp_des_node> rtsp_des_0 = nullptr;
    if (enable_rtsp_push) {
        rtsp_des_0 = std::make_shared<vp_nodes::vp_rtsp_des_node>(
            "rtsp_des_0",
            0,
            rtsp_url,
            vp_objects::vp_size{},
            true,
            rtsp_encoder,
            false);
        rtsp_des_0->attach_to({osd_0});
    }

    src_0->start();

    vp_utils::vp_analysis_board board({src_0});
    board.display(1, false);

    while (!g_should_exit.load()) {
        if (check_terminal_escape_pressed()) {
            VP_INFO("[rtsp_main] ESC detected from terminal, exiting...");
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    src_0->detach_recursively();
    return 0;
}
