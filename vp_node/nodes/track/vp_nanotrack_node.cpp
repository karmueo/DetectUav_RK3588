/**
 * @file vp_nanotrack_node.cpp
 * @brief NanoTrack单目标跟踪节点实现
 * @details 基于NanoTrack的单目标跟踪节点实现
 */

#include "vp_nanotrack_node.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"
#include "objects/vp_frame_meta.h"

using json = nlohmann::json;

namespace vp_nodes {

vp_nanotrack_node::vp_nanotrack_node(std::string node_name, std::string config_path)
    : vp_track_node(node_name),
      config_path_(config_path) {
    // 加载配置文件
    load_config(config_path);

    // 创建NanoTrack跟踪器
    try {
        tracker_ = std::make_unique<NanoTrack>(backbone_path_, backbone_search_path_, head_path_, npu_core_);
        tracker_->set_params(window_influence_, penalty_k_, lr_);
        spdlog::info("[vp_nanotrack_node] NanoTrack tracker created successfully");
    } catch (const std::exception& e) {
        spdlog::error("[vp_nanotrack_node] Failed to create NanoTrack tracker: {}", e.what());
        throw;
    }
}

vp_nanotrack_node::~vp_nanotrack_node() {
    spdlog::info("[vp_nanotrack_node] Destroyed");
}

void vp_nanotrack_node::load_config(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json config;
    ifs >> config;

    backbone_path_ = config.value("backbone_path", "");
    backbone_search_path_ = config.value("backbone_search_path", "");
    head_path_ = config.value("head_path", "");

    conf_threshold_ = config.value("conf_threshold", 0.5f);
    track_conf_threshold_ = config.value("track_conf_threshold", 0.3f);
    enable_iou_check_ = config.value("enable_iou_check", true);
    iou_threshold_ = config.value("iou_threshold", 0.3f);

    window_influence_ = config.value("window_influence", 0.455f);
    penalty_k_ = config.value("penalty_k", 0.138f);
    lr_ = config.value("lr", 0.348f);

    npu_core_ = config.value("npu_core", 3);

    spdlog::info("[vp_nanotrack_node] Config loaded:");
    spdlog::info("  backbone_path: {}", backbone_path_);
    spdlog::info("  backbone_search_path: {}", backbone_search_path_);
    spdlog::info("  head_path: {}", head_path_);
    spdlog::info("  conf_threshold: {}", conf_threshold_);
    spdlog::info("  track_conf_threshold: {}", track_conf_threshold_);
    spdlog::info("  enable_iou_check: {}", enable_iou_check_);
    spdlog::info("  iou_threshold: {}", iou_threshold_);
    spdlog::info("  npu_core: {}", npu_core_);
}

std::shared_ptr<vp_objects::vp_meta> vp_nanotrack_node::handle_frame_meta(
    std::shared_ptr<vp_objects::vp_frame_meta> meta) {
    // 获取当前帧
    const cv::Mat& frame = meta->frame;

    // 获取检测目标
    const auto& targets = meta->targets;

    // 如果跟踪未初始化，尝试选择目标进行初始化
    if (!tracking_initialized_) {
        if (!targets.empty()) {
            // 过滤低置信度目标
            std::vector<std::shared_ptr<vp_objects::vp_frame_target>> valid_targets;
            for (const auto& t : targets) {
                if (t->primary_score >= conf_threshold_) {
                    valid_targets.push_back(t);
                }
            }

            if (!valid_targets.empty()) {
                // 选择一个目标进行跟踪
                auto selected = select_target(valid_targets);
                if (selected) {
                    // 初始化跟踪器
                    std::array<float, 4> bbox = {
                        static_cast<float>(selected->x),
                        static_cast<float>(selected->y),
                        static_cast<float>(selected->width),
                        static_cast<float>(selected->height)
                    };

                    try {
                        tracker_->init(frame, bbox);
                        tracking_initialized_ = true;
                        last_target_rect_ = selected->get_rect();
                        selected->track_id = fixed_track_id_;
                        spdlog::info("[vp_nanotrack_node] Tracking initialized with target id: {}, bbox: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]",
                                    fixed_track_id_, bbox[0], bbox[1], bbox[2], bbox[3]);
                    } catch (const std::exception& e) {
                        spdlog::error("[vp_nanotrack_node] Failed to init tracker: {}", e.what());
                    }
                }
            }
        }
    } else {
        // 跟踪已初始化，执行跟踪
        try {
            TrackOutput output = tracker_->track(frame);

            // 检查置信度
            if (output.score >= track_conf_threshold_) {
                // IOU检查
                if (enable_iou_check_) {
                    vp_objects::vp_rect current_rect(
                        static_cast<int>(output.bbox[0]),
                        static_cast<int>(output.bbox[1]),
                        static_cast<int>(output.bbox[2]),
                        static_cast<int>(output.bbox[3])
                    );
                    float iou = calculate_iou(last_target_rect_, current_rect);

                    if (iou < iou_threshold_) {
                        spdlog::warn("[vp_nanotrack_node] IOU check failed: {:.3f} < {:.3f}, resetting tracking", iou, iou_threshold_);
                        tracking_initialized_ = false;
                        return meta;
                    }
                    last_target_rect_ = current_rect;
                }

                // 更新目标位置
                for (auto& t : targets) {
                    if (t->track_id == fixed_track_id_) {
                        t->x = static_cast<int>(output.bbox[0]);
                        t->y = static_cast<int>(output.bbox[1]);
                        t->width = static_cast<int>(output.bbox[2]);
                        t->height = static_cast<int>(output.bbox[3]);
                        t->primary_score = output.score;

                        // 记录跟踪历史
                        t->tracks.push_back(t->get_rect());
                        break;
                    }
                }
            } else {
                spdlog::warn("[vp_nanotrack_node] Tracking score too low: {:.3f} < {:.3f}", output.score, track_conf_threshold_);
            }
        } catch (const std::exception& e) {
            spdlog::error("[vp_nanotrack_node] Tracking failed: {}", e.what());
            tracking_initialized_ = false;
        }
    }

    return meta;
}

void vp_nanotrack_node::track(int channel_index,
                               const std::vector<vp_objects::vp_rect>& target_rects,
                               const std::vector<std::vector<float>>& target_embeddings,
                               std::vector<int>& track_ids) {
    // 单目标跟踪不需要实现多目标跟踪接口
    // 实际跟踪逻辑在handle_frame_meta中实现
    track_ids.clear();
}

std::shared_ptr<vp_objects::vp_frame_target> vp_nanotrack_node::select_target(
    const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) {
    if (targets.empty()) {
        return nullptr;
    }

    // 选择置信度最高的目标
    auto max_score_target = targets[0];
    float max_score = targets[0]->primary_score;

    for (const auto& t : targets) {
        if (t->primary_score > max_score) {
            max_score = t->primary_score;
            max_score_target = t;
        }
    }

    return max_score_target;
}

float vp_nanotrack_node::calculate_iou(const vp_objects::vp_rect& rect1,
                                        const vp_objects::vp_rect& rect2) {
    float x1 = static_cast<float>(rect1.x);
    float y1 = static_cast<float>(rect1.y);
    float w1 = static_cast<float>(rect1.width);
    float h1 = static_cast<float>(rect1.height);

    float x2 = static_cast<float>(rect2.x);
    float y2 = static_cast<float>(rect2.y);
    float w2 = static_cast<float>(rect2.width);
    float h2 = static_cast<float>(rect2.height);

    float endx = std::max(x1 + w1, x2 + w2);
    float startx = std::min(x1, x2);
    float width = w1 + w2 - (endx - startx);

    float endy = std::max(y1 + h1, y2 + h2);
    float starty = std::min(y1, y2);
    float height = h1 + h2 - (endy - starty);

    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    float area = width * height;
    float area1 = w1 * h1;
    float area2 = w2 * h2;

    return area / (area1 + area2 - area);
}

void vp_nanotrack_node::reset_tracking() {
    tracking_initialized_ = false;
    spdlog::info("[vp_nanotrack_node] Tracking reset");
}

void vp_nanotrack_node::init_target(const cv::Mat& frame_bgr, const std::array<float, 4>& bbox) {
    try {
        tracker_->init(frame_bgr, bbox);
        tracking_initialized_ = true;
        last_target_rect_ = vp_objects::vp_rect(
            static_cast<int>(bbox[0]),
            static_cast<int>(bbox[1]),
            static_cast<int>(bbox[2]),
            static_cast<int>(bbox[3])
        );
        spdlog::info("[vp_nanotrack_node] Target manually initialized: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]",
                    bbox[0], bbox[1], bbox[2], bbox[3]);
    } catch (const std::exception& e) {
        spdlog::error("[vp_nanotrack_node] Failed to init target: {}", e.what());
    }
}

}  // namespace vp_nodes
