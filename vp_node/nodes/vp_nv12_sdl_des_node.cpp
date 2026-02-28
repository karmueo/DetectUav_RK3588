#include "vp_nv12_sdl_des_node.h"

#include <cstring>
#include <mutex>

#include "vp_utils/logger/vp_logger.h"
#include "vp_utils/vp_utils.h"

namespace vp_nodes {

// NV12 SDL 终端节点的全局退出请求标志。
static std::atomic<bool> g_nv12_sdl_des_exit_requested{false};

bool vp_nv12_sdl_des_should_exit() {
    return g_nv12_sdl_des_exit_requested.load();
}

void vp_nv12_sdl_des_reset_exit_flag() {
    g_nv12_sdl_des_exit_requested.store(false);
}

vp_nv12_sdl_des_node::vp_nv12_sdl_des_node(std::string node_name,
                                           int channel_index,
                                           std::string sdl_video_driver,
                                           std::string sdl_render_driver,
                                           bool fullscreen)
    : vp_des_node(std::move(node_name), channel_index),
      sdl_video_driver(std::move(sdl_video_driver)),
      sdl_render_driver(std::move(sdl_render_driver)),
      fullscreen(fullscreen) {
    this->initialized();
}

vp_nv12_sdl_des_node::~vp_nv12_sdl_des_node() {
    deinitialized();
    release_sdl();
}

bool vp_nv12_sdl_des_node::init_sdl(int frame_width, int frame_height) {
    if (!sdl_video_driver.empty()) {
        // 设置 SDL 视频驱动返回值。
        const int set_video_env_ret = setenv("SDL_VIDEODRIVER", sdl_video_driver.c_str(), 1);
        if (set_video_env_ret != 0) {
            VP_WARN(vp_utils::string_format("[%s] set SDL_VIDEODRIVER failed", node_name.c_str()));
        }
    }
    if (!sdl_render_driver.empty()) {
        // 设置 SDL 渲染驱动 hint 返回值。
        const int set_render_env_ret = setenv("SDL_HINT_RENDER_DRIVER", sdl_render_driver.c_str(), 1);
        if (set_render_env_ret != 0) {
            VP_WARN(vp_utils::string_format("[%s] set SDL_HINT_RENDER_DRIVER failed", node_name.c_str()));
        }
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    SDL_SetHint(SDL_HINT_RENDER_BATCHING, "1");
#ifdef SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR
    SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "1");
#endif
#ifdef SDL_HINT_VIDEO_X11_FORCE_EGL
    SDL_SetHint(SDL_HINT_VIDEO_X11_FORCE_EGL, "1");
#endif

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_Init failed: %s", node_name.c_str(), SDL_GetError()));
        return false;
    }

    // SDL 窗口标志。
    Uint32 window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    if (fullscreen) {
        window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }

    sdl_window = SDL_CreateWindow("DetectUav_RK3588 NV12 SDL",
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  frame_width,
                                  frame_height,
                                  window_flags);
    if (sdl_window == nullptr) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_CreateWindow failed: %s", node_name.c_str(), SDL_GetError()));
        return false;
    }

    // SDL 渲染器创建标志。
    const Uint32 renderer_flags = SDL_RENDERER_ACCELERATED;
    // 指定渲染器索引，默认自动。
    int render_index = -1;
    if (!sdl_render_driver.empty()) {
        const int driver_count = SDL_GetNumRenderDrivers();
        for (int i = 0; i < driver_count; ++i) {
            // 渲染器信息。
            SDL_RendererInfo renderer_info;
            if (SDL_GetRenderDriverInfo(i, &renderer_info) == 0 &&
                renderer_info.name != nullptr &&
                !strcmp(renderer_info.name, sdl_render_driver.c_str())) {
                render_index = i;
                break;
            }
        }
        if (render_index < 0) {
            VP_WARN(vp_utils::string_format("[%s] render driver %s not found, fallback auto",
                                            node_name.c_str(), sdl_render_driver.c_str()));
        }
    }
    sdl_renderer = SDL_CreateRenderer(sdl_window, render_index, renderer_flags);
    if (sdl_renderer == nullptr) {
        VP_WARN(vp_utils::string_format("[%s] SDL accelerated renderer failed: %s, fallback",
                                        node_name.c_str(), SDL_GetError()));
        sdl_renderer = SDL_CreateRenderer(sdl_window, -1, 0);
        if (sdl_renderer == nullptr) {
            VP_ERROR(vp_utils::string_format("[%s] SDL_CreateRenderer failed: %s", node_name.c_str(), SDL_GetError()));
            return false;
        }
    }

    // 当前渲染器信息。
    SDL_RendererInfo active_renderer_info;
    if (SDL_GetRendererInfo(sdl_renderer, &active_renderer_info) == 0) {
        VP_INFO(vp_utils::string_format("[%s] SDL video_driver=%s render_driver=%s flags=0x%x",
                                        node_name.c_str(),
                                        SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "unknown",
                                        active_renderer_info.name ? active_renderer_info.name : "unknown",
                                        active_renderer_info.flags));
    }

    texture_width = frame_width;
    texture_height = frame_height;
    sdl_texture = SDL_CreateTexture(sdl_renderer,
                                    SDL_PIXELFORMAT_NV12,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    texture_width,
                                    texture_height);
    if (sdl_texture == nullptr) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_CreateTexture NV12 failed: %s", node_name.c_str(), SDL_GetError()));
        return false;
    }

    sdl_inited = true;
    VP_INFO(vp_utils::string_format("[%s] sdl ready w=%d h=%d", node_name.c_str(), texture_width, texture_height));
    return true;
}

bool vp_nv12_sdl_des_node::ensure_texture(int frame_width, int frame_height) {
    if (sdl_texture != nullptr && texture_width == frame_width && texture_height == frame_height) {
        return true;
    }

    if (sdl_texture != nullptr) {
        SDL_DestroyTexture(sdl_texture);
        sdl_texture = nullptr;
    }

    texture_width = frame_width;
    texture_height = frame_height;
    sdl_texture = SDL_CreateTexture(sdl_renderer,
                                    SDL_PIXELFORMAT_NV12,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    texture_width,
                                    texture_height);
    if (sdl_texture == nullptr) {
        VP_ERROR(vp_utils::string_format("[%s] recreate SDL texture failed: %s", node_name.c_str(), SDL_GetError()));
        return false;
    }

    return true;
}

void vp_nv12_sdl_des_node::pump_sdl_events() {
    // SDL 事件对象。
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            g_nv12_sdl_des_exit_requested.store(true);
        }
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
            g_nv12_sdl_des_exit_requested.store(true);
        }
    }
}

void vp_nv12_sdl_des_node::meta_flow(std::shared_ptr<vp_objects::vp_meta> meta) {
    if (meta == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(this->in_queue_lock);
    const bool queue_full = this->in_queue.size() >= this->max_in_queue_size;
    const bool is_frame_meta = meta->meta_type == vp_objects::vp_meta_type::FRAME;
    if (queue_full && is_frame_meta) {
        this->dropped_frames++;
        // 每秒打印一次丢帧统计，避免日志刷屏。
        const auto now_tp = std::chrono::steady_clock::now();
        if (this->dropped_log_tp.time_since_epoch().count() == 0) {
            this->dropped_log_tp = now_tp;
            this->dropped_frames_last_log = this->dropped_frames;
        } else {
            const auto delta_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - this->dropped_log_tp).count();
            if (delta_ms >= 1000) {
                const uint64_t dropped_delta = this->dropped_frames - this->dropped_frames_last_log;
                VP_WARN(vp_utils::string_format("[%s] drop_frame backlog=%zu dropped=%llu(+%llu/s)",
                                                this->node_name.c_str(),
                                                this->in_queue.size(),
                                                static_cast<unsigned long long>(this->dropped_frames),
                                                static_cast<unsigned long long>(dropped_delta)));
                this->dropped_log_tp = now_tp;
                this->dropped_frames_last_log = this->dropped_frames;
            }
        }
        return;
    }

    this->in_queue.push(meta);
    invoke_meta_arriving_hooker(this->node_name, this->in_queue.size(), meta);
    this->in_queue_semaphore.signal();
}

void vp_nv12_sdl_des_node::release_sdl() {
    if (sdl_texture != nullptr) {
        SDL_DestroyTexture(sdl_texture);
        sdl_texture = nullptr;
    }
    if (sdl_renderer != nullptr) {
        SDL_DestroyRenderer(sdl_renderer);
        sdl_renderer = nullptr;
    }
    if (sdl_window != nullptr) {
        SDL_DestroyWindow(sdl_window);
        sdl_window = nullptr;
    }
    if (sdl_inited) {
        SDL_Quit();
        sdl_inited = false;
    }
}

std::shared_ptr<vp_objects::vp_meta>
vp_nv12_sdl_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
    // 输入 NV12 帧。
    cv::Mat& frame = meta->frame;

    // NV12 宽度。
    const int frame_width = frame.cols;
    // NV12 总行数（Y + UV）。
    const int frame_rows = frame.rows;
    // NV12 高度。
    const int frame_height = frame_rows * 2 / 3;

    // 基础格式校验。
    const bool valid_type = (frame.type() == CV_8UC1);
    const bool valid_size = (frame_rows > 0 && frame_width > 0 && frame_rows == frame_height * 3 / 2);
    if (!valid_type || !valid_size) {
        VP_WARN(vp_utils::string_format("[%s] invalid nv12 frame: type=%d size=%dx%d",
                                        node_name.c_str(), frame.type(), frame.cols, frame.rows));
        return vp_des_node::handle_frame_meta(meta);
    }

    if (!sdl_inited) {
        if (!init_sdl(frame_width, frame_height)) {
            g_nv12_sdl_des_exit_requested.store(true);
            return vp_des_node::handle_frame_meta(meta);
        }
    }

    pump_sdl_events();
    if (g_nv12_sdl_des_exit_requested.load()) {
        return vp_des_node::handle_frame_meta(meta);
    }

    if (!ensure_texture(frame_width, frame_height)) {
        g_nv12_sdl_des_exit_requested.store(true);
        return vp_des_node::handle_frame_meta(meta);
    }

    // NV12 的 Y 面起始地址。
    const uint8_t* y_plane = frame.ptr<uint8_t>(0);
    // NV12 的 UV 面起始地址。
    const uint8_t* uv_plane = frame.ptr<uint8_t>(frame_height);

    // 直接按 NV12 两平面更新纹理，减少 lock/unlock 额外开销。
    const int update_ret = SDL_UpdateNVTexture(sdl_texture,
                                               nullptr,
                                               y_plane,
                                               frame_width,
                                               uv_plane,
                                               frame_width);
    if (update_ret != 0) {
        VP_WARN(vp_utils::string_format("[%s] SDL_UpdateNVTexture failed: %s", node_name.c_str(), SDL_GetError()));
        return vp_des_node::handle_frame_meta(meta);
    }

    if (SDL_RenderCopy(sdl_renderer, sdl_texture, nullptr, nullptr) != 0) {
        VP_WARN(vp_utils::string_format("[%s] SDL_RenderCopy failed: %s", node_name.c_str(), SDL_GetError()));
    } else {
        SDL_RenderPresent(sdl_renderer);
    }

    return vp_des_node::handle_frame_meta(meta);
}

std::shared_ptr<vp_objects::vp_meta>
vp_nv12_sdl_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
    return vp_des_node::handle_control_meta(meta);
}

} // namespace vp_nodes
