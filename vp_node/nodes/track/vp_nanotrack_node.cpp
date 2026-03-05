/**
 * @file vp_nanotrack_node.cpp
 * @brief NanoTrack single object tracking node implementation.
 *
 * Implements the vp_nanotrack_node which integrates NanoTrack tracker
 * into the VideoPipe pipeline for single object tracking.
 *
 * State machine implementation:
 * - SEARCHING: Looking for valid detection to initialize tracking
 * - TRACKING: Actively tracking with exit condition monitoring
 *
 * Target selection strategy:
 * - Select target closest to frame center
 * - Confidence must be >= selection_conf_threshold
 * - Optional class ID filtering via target_class_id
 *
 * Exit conditions:
 * - track_score < exit_conf_threshold
 * - IoU(track_bbox, detect_bbox) < iou_threshold
 * - Consecutive frames without detection >= max_no_detection_frames
 *
 * @creation 2026-03-05
 * @modified 2026-03-05
 */

#include "vp_nanotrack_node.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "nanotrack.h"
#include "nlohmann/json.hpp"
#include "vp_utils/vp_utils.h"

namespace vp_nodes {

namespace {
using json = nlohmann::json;
}

vp_nanotrack_node::vp_nanotrack_node(std::string node_name,
                                      std::string json_path,
                                      float selection_conf_threshold,
                                      float exit_conf_threshold)
    : vp_node(std::move(node_name)),
      json_path_(std::move(json_path)) {

    // Load configuration and initialize tracker
    NanoTrackConfig nanotrack_config;
    if (NanoTrack::load_config(json_path_, nanotrack_config) != 0) {
        VP_WARN(vp_utils::string_format("[%s] Failed to load config from %s, using defaults",
                node_name.c_str(), json_path_.c_str()));
        throw std::runtime_error("Failed to load NanoTrack config: " + json_path_);
    }

    // Load tracking strategy configuration from JSON
    std::ifstream config_file(json_path_);
    if (config_file.is_open()) {
        try {
            json cfg = json::parse(config_file);
            config_.target_class_id = cfg.value("target_class_id", -1);
            config_.selection_conf_threshold = cfg.value("selection_conf_threshold", 0.5f);
            config_.exit_conf_threshold = cfg.value("exit_conf_threshold", 0.3f);
            config_.iou_threshold = cfg.value("iou_threshold", 0.3f);
            config_.max_no_detection_frames = cfg.value("max_no_detection_frames", 30);
        } catch (const json::exception& e) {
            VP_WARN(vp_utils::string_format("[%s] Failed to parse tracking config: %s, using defaults",
                    node_name.c_str(), e.what()));
        }
    }

    // Override with constructor parameters if provided (for backward compatibility)
    if (selection_conf_threshold > 0) {
        config_.selection_conf_threshold = selection_conf_threshold;
    }
    if (exit_conf_threshold > 0) {
        config_.exit_conf_threshold = exit_conf_threshold;
    }

    // Apply exit confidence threshold to NanoTrack internal config
    if (config_.exit_conf_threshold > 0) {
        nanotrack_config.track_conf_threshold = config_.exit_conf_threshold;
    }

    tracker_ = std::make_shared<NanoTrack>(nanotrack_config);
    VP_INFO(vp_utils::string_format("[%s] NanoTrack node initialized "
            "(state=SEARCHING, target_class=%d, selection_conf=%.2f, exit_conf=%.2f, iou_threshold=%.2f, max_no_detect=%d)",
            node_name.c_str(), config_.target_class_id, config_.selection_conf_threshold,
            config_.exit_conf_threshold, config_.iou_threshold, config_.max_no_detection_frames));

    this->initialized();
}

vp_nanotrack_node::~vp_nanotrack_node() {
    deinitialized();
    VP_INFO(vp_utils::string_format("[%s] NanoTrack node destroyed", node_name.c_str()));
}

std::shared_ptr<vp_objects::vp_meta> vp_nanotrack_node::handle_frame_meta(
    std::shared_ptr<vp_objects::vp_frame_meta> meta) {

    if (meta == nullptr || meta->frame.empty()) {
        return meta;
    }

    const int frame_width = meta->frame.cols;
    const int frame_height = meta->frame.rows;

    // State machine: SEARCHING -> TRACKING -> SEARCHING (on exit)
    switch (state_) {
        case TrackState::SEARCHING: {
            // Check if we have detection results
            if (!meta->targets.empty()) {
                std::array<float, 4> best_bbox;
                if (select_best_target(meta->targets, frame_width, frame_height, best_bbox)) {
                    // Initialize tracker with selected target
                    tracker_->init(meta->frame, best_bbox);
                    state_ = TrackState::TRACKING;
                    no_detection_frames_ = 0;

                    // Set tracking state in frame meta
                    meta->single_track_active = true;
                    meta->single_track_bbox = best_bbox;
                    meta->single_track_score = 1.0f;  // Initial score is 1.0
                    meta->single_track_init_frame = meta->frame_index;

                    VP_INFO(vp_utils::string_format("[%s] TRACKING started: bbox=[%.1f, %.1f, %.1f, %.1f]",
                            node_name.c_str(), best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3]));
                }
            }
            break;
        }

        case TrackState::TRACKING: {
            TrackOutput output = tracker_->track(meta->frame);

            // Clamp bounding box to frame boundaries
            auto clamp_bbox = [frame_width, frame_height](std::array<float, 4>& bbox) {
                bbox[0] = std::max(0.0f, std::min(bbox[0], static_cast<float>(frame_width - 1)));
                bbox[1] = std::max(0.0f, std::min(bbox[1], static_cast<float>(frame_height - 1)));
                bbox[2] = std::max(1.0f, std::min(bbox[2], static_cast<float>(frame_width) - bbox[0]));
                bbox[3] = std::max(1.0f, std::min(bbox[3], static_cast<float>(frame_height) - bbox[1]));
            };

            clamp_bbox(output.bbox);

            // Update frame meta with tracking result
            meta->single_track_active = true;
            meta->single_track_bbox = output.bbox;
            meta->single_track_score = output.score;

            // Check for tracking exit conditions
            if (should_exit_tracking(output, meta->targets)) {
                meta->single_track_exit = true;
                reset_to_searching();
            }
            break;
        }
    }

    // Periodic debug logging
    ++frame_counter_;
    if (frame_counter_ % static_cast<uint64_t>(debug_log_interval_) == 0) {
        VP_INFO(vp_utils::string_format("[%s] Frame %llu, state=%s, score=%.3f",
                node_name.c_str(),
                static_cast<unsigned long long>(frame_counter_),
                state_ == TrackState::TRACKING ? "TRACKING" : "SEARCHING",
                meta->single_track_score));
    }

    return meta;
}

std::shared_ptr<vp_objects::vp_meta> vp_nanotrack_node::handle_control_meta(
    std::shared_ptr<vp_objects::vp_control_meta> meta) {
    return meta;
}

bool vp_nanotrack_node::select_best_target(
    const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets,
    int frame_width,
    int frame_height,
    std::array<float, 4>& best_bbox) {

    if (targets.empty()) {
        return false;
    }

    const float frame_center_x = static_cast<float>(frame_width) / 2.0f;
    const float frame_center_y = static_cast<float>(frame_height) / 2.0f;

    // Filter targets by class ID and confidence threshold
    std::vector<const std::shared_ptr<vp_objects::vp_frame_target>*> valid_targets;
    for (const auto& target : targets) {
        // AC-01: Filter by target_class_id if specified (AC-04: -1 means no filter)
        if (config_.target_class_id >= 0 && target->primary_class_id != config_.target_class_id) {
            continue;
        }
        // AC-02: Filter by selection confidence threshold
        if (target->primary_score >= config_.selection_conf_threshold) {
            valid_targets.push_back(&target);
        }
    }

    if (valid_targets.empty()) {
        VP_DEBUG(vp_utils::string_format("[%s] No targets with class_id=%d, confidence >= %.2f",
                node_name.c_str(), config_.target_class_id, config_.selection_conf_threshold));
        return false;
    }

    // AC-03: Find target closest to frame center
    const std::shared_ptr<vp_objects::vp_frame_target>* best_target = nullptr;
    float min_distance = std::numeric_limits<float>::max();

    for (const auto& target_ptr : valid_targets) {
        const auto& target = *target_ptr;
        float distance = distance_to_center(target, frame_center_x, frame_center_y);

        if (distance < min_distance) {
            min_distance = distance;
            best_target = target_ptr;
        }
    }

    if (best_target == nullptr) {
        return false;
    }

    // Convert vp_frame_target to bbox format [x, y, w, h]
    const auto& selected = *best_target;
    best_bbox[0] = static_cast<float>(selected->x);
    best_bbox[1] = static_cast<float>(selected->y);
    best_bbox[2] = static_cast<float>(selected->width);
    best_bbox[3] = static_cast<float>(selected->height);

    VP_DEBUG(vp_utils::string_format("[%s] Selected target class=%d at center (%.1f, %.1f) "
            "with confidence %.3f, distance to frame center: %.1f",
            node_name.c_str(),
            selected->primary_class_id,
            selected->x + selected->width / 2.0f,
            selected->y + selected->height / 2.0f,
            selected->primary_score,
            min_distance));

    return true;
}

float vp_nanotrack_node::distance_to_center(
    const std::shared_ptr<vp_objects::vp_frame_target>& target,
    float frame_center_x,
    float frame_center_y) {

    // Calculate target center
    float target_center_x = static_cast<float>(target->x) + static_cast<float>(target->width) / 2.0f;
    float target_center_y = static_cast<float>(target->y) + static_cast<float>(target->height) / 2.0f;

    // Euclidean distance
    float dx = target_center_x - frame_center_x;
    float dy = target_center_y - frame_center_y;
    return std::sqrt(dx * dx + dy * dy);
}

bool vp_nanotrack_node::should_exit_tracking(
    const TrackOutput& output,
    const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) {

    // AC-05: Check confidence exit condition (AC-09: threshold <= 0 disables this check)
    if (config_.exit_conf_threshold > 0 && output.score < config_.exit_conf_threshold) {
        VP_WARN(vp_utils::string_format("[%s] TRACKING exit: score %.3f < threshold %.3f",
                node_name.c_str(), output.score, config_.exit_conf_threshold));
        return true;
    }

    // AC-08: Check consecutive frames without detection (AC-10: threshold <= 0 disables this check)
    if (config_.max_no_detection_frames > 0) {
        if (targets.empty()) {
            ++no_detection_frames_;
            if (no_detection_frames_ >= config_.max_no_detection_frames) {
                VP_WARN(vp_utils::string_format("[%s] TRACKING exit: %d consecutive frames without detection",
                        node_name.c_str(), no_detection_frames_));
                return true;
            }
        } else {
            // Reset counter when detections are present
            no_detection_frames_ = 0;
        }
    }

    // AC-06: Check IoU exit condition (AC-07: no detection -> skip IoU check, AC-09: threshold <= 0 disables)
    if (config_.iou_threshold > 0 && !targets.empty()) {
        bool has_valid_iou = false;
        for (const auto& target : targets) {
            // Optional: also filter by class ID for IoU check
            if (config_.target_class_id >= 0 && target->primary_class_id != config_.target_class_id) {
                continue;
            }
            float iou = calculate_iou(output.bbox, target);
            if (iou >= config_.iou_threshold) {
                has_valid_iou = true;
                break;
            }
        }
        if (!has_valid_iou) {
            VP_WARN(vp_utils::string_format("[%s] TRACKING exit: no detection with IoU >= %.3f",
                    node_name.c_str(), config_.iou_threshold));
            return true;
        }
    }

    return false;
}

float vp_nanotrack_node::calculate_iou(
    const std::array<float, 4>& bbox,
    const std::shared_ptr<vp_objects::vp_frame_target>& target) {

    // bbox format: [x, y, w, h]
    // vp_frame_target format: x, y, width, height

    // Calculate intersection coordinates
    float x1 = std::max(bbox[0], static_cast<float>(target->x));
    float y1 = std::max(bbox[1], static_cast<float>(target->y));
    float x2 = std::min(bbox[0] + bbox[2], static_cast<float>(target->x + target->width));
    float y2 = std::min(bbox[1] + bbox[3], static_cast<float>(target->y + target->height));

    // Calculate intersection area
    float intersection_width = std::max(0.0f, x2 - x1);
    float intersection_height = std::max(0.0f, y2 - y1);
    float intersection_area = intersection_width * intersection_height;

    // Calculate union area
    float bbox_area = bbox[2] * bbox[3];
    float target_area = static_cast<float>(target->width) * static_cast<float>(target->height);
    float union_area = bbox_area + target_area - intersection_area;

    // Avoid division by zero
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection_area / union_area;
}

void vp_nanotrack_node::reset_to_searching() {
    tracker_->reset();
    state_ = TrackState::SEARCHING;
    no_detection_frames_ = 0;
    VP_INFO(vp_utils::string_format("[%s] State -> SEARCHING", node_name.c_str()));
}

}  // namespace vp_nodes
