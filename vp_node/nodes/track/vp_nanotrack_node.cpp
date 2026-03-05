/**
 * @file vp_nanotrack_node.cpp
 * @brief NanoTrack single object tracking node implementation.
 *
 * Implements the vp_nanotrack_node which integrates NanoTrack tracker
 * into the VideoPipe pipeline for single object tracking.
 *
 * Target selection strategy:
 * - Select target closest to frame center
 * - Confidence must be >= selection_conf_threshold
 * - Exit tracking if confidence < exit_conf_threshold
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
      json_path_(std::move(json_path)),
      selection_conf_threshold_(selection_conf_threshold),
      exit_conf_threshold_(exit_conf_threshold) {

    // Load configuration and initialize tracker
    NanoTrackConfig config;
    if (NanoTrack::load_config(json_path_, config) != 0) {
        VP_WARN(vp_utils::string_format("[%s] Failed to load config from %s, using defaults",
                node_name.c_str(), json_path_.c_str()));
        throw std::runtime_error("Failed to load NanoTrack config: " + json_path_);
    }

    // Override exit confidence threshold if specified
    if (exit_conf_threshold > 0) {
        config.track_conf_threshold = exit_conf_threshold;
    }

    tracker_ = std::make_shared<NanoTrack>(config);
    VP_INFO(vp_utils::string_format("[%s] NanoTrack node initialized (selection_conf=%.2f, exit_conf=%.2f)",
            node_name.c_str(), selection_conf_threshold_, exit_conf_threshold_));

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

    // Case 1: Tracker not initialized - look for detection results to initialize
    if (!tracker_->is_initialized()) {
        // Check if we have detection results
        if (!meta->targets.empty()) {
            std::array<float, 4> best_bbox;
            if (select_best_target(meta->targets, frame_width, frame_height, best_bbox)) {
                // Initialize tracker with selected target
                tracker_->init(meta->frame, best_bbox);

                // Set tracking state in frame meta
                meta->single_track_active = true;
                meta->single_track_bbox = best_bbox;
                meta->single_track_score = 1.0f;  // Initial score is 1.0
                meta->single_track_init_frame = meta->frame_index;

                VP_INFO(vp_utils::string_format("[%s] Tracker initialized with bbox: [%.1f, %.1f, %.1f, %.1f]",
                        node_name.c_str(), best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3]));
            }
        }
    }
    // Case 2: Tracker is initialized - perform tracking
    else {
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

        // Check for tracking exit condition
        if (output.score < exit_conf_threshold_) {
            meta->single_track_exit = true;
            tracker_->reset();

            VP_WARN(vp_utils::string_format("[%s] Tracking exited: score %.3f < threshold %.3f",
                    node_name.c_str(), output.score, exit_conf_threshold_));
        }
    }

    // Periodic debug logging
    ++frame_counter_;
    if (frame_counter_ % static_cast<uint64_t>(debug_log_interval_) == 0) {
        VP_INFO(vp_utils::string_format("[%s] Frame %llu, tracking=%s, score=%.3f",
                node_name.c_str(),
                static_cast<unsigned long long>(frame_counter_),
                tracker_->is_initialized() ? "YES" : "NO",
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

    // Filter targets by confidence threshold
    std::vector<const std::shared_ptr<vp_objects::vp_frame_target>*> valid_targets;
    for (const auto& target : targets) {
        if (target->primary_score >= selection_conf_threshold_) {
            valid_targets.push_back(&target);
        }
    }

    if (valid_targets.empty()) {
        VP_DEBUG(vp_utils::string_format("[%s] No targets with confidence >= %.2f",
                node_name.c_str(), selection_conf_threshold_));
        return false;
    }

    // Find target closest to frame center
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

    VP_DEBUG(vp_utils::string_format("[%s] Selected target at center (%.1f, %.1f) with confidence %.3f, distance to frame center: %.1f",
            node_name.c_str(),
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

}  // namespace vp_nodes
