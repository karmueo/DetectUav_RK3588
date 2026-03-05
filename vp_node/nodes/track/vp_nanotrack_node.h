/**
 * @file vp_nanotrack_node.h
 * @brief NanoTrack single object tracking node for VideoPipe pipeline.
 *
 * This node integrates the NanoTrack tracker into the VideoPipe framework, providing:
 * - Automatic target selection from detection results
 * - Target initialization based on detection or manual input
 * - Continuous tracking with confidence monitoring
 * - Automatic exit on low confidence or tracking failure
 *
 * State machine:
 * - SEARCHING: Looking for valid detection to initialize tracking
 * - TRACKING: Actively tracking the selected target
 *
 * Exit conditions (configurable):
 * - track_score < exit_conf_threshold
 * - IoU(track_bbox, detect_bbox) < iou_threshold
 * - Consecutive frames without detection >= max_no_detection_frames
 *
 * @creation 2026-03-05
 * @modified 2026-03-05
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include "nodes/base/vp_node.h"
#include "objects/vp_frame_meta.h"

// Forward declaration
class NanoTrack;
struct NanoTrackConfig;
struct TrackOutput;

namespace vp_nodes {

/**
 * @brief Tracking state enumeration.
 */
enum class TrackState {
    SEARCHING,  ///< Looking for valid target to initialize tracking
    TRACKING    ///< Actively tracking the selected target
};

/**
 * @brief Tracking configuration parameters.
 */
struct TrackConfig {
    int target_class_id = -1;               ///< Target class ID filter (-1 = no filter)
    float selection_conf_threshold = 0.5f;  ///< Minimum confidence for target selection
    float exit_conf_threshold = 0.3f;       ///< Minimum confidence before tracking exit
    float iou_threshold = 0.3f;             ///< Minimum IoU threshold for tracking validation
    int max_no_detection_frames = 30;       ///< Max consecutive frames without detection before exit
};

/**
 * @brief Single object tracking node using NanoTrack algorithm.
 *
 * This node performs single object tracking using the NanoTrack algorithm.
 * Workflow:
 * 1. Wait for detection results from upstream nodes
 * 2. Select target closest to frame center with confidence above threshold
 * 3. Initialize tracker with selected target
 * 4. Track target in subsequent frames
 * 5. Exit tracking if confidence drops below threshold or tracking failure
 *
 * Target selection strategy:
 * - Select the target closest to frame center
 * - Targets must have confidence >= selection_conf_threshold
 * - Optional class ID filtering via target_class_id
 */
class vp_nanotrack_node : public vp_node {
public:
    /**
     * @brief Construct NanoTrack tracking node.
     * @param node_name Node name for identification.
     * @param json_path Path to NanoTrack configuration JSON file.
     * @param selection_conf_threshold Minimum confidence for target selection (default: 0.5).
     * @param exit_conf_threshold Minimum confidence before tracking exit (default: 0.3).
     */
    vp_nanotrack_node(std::string node_name,
                     std::string json_path,
                     float selection_conf_threshold = 0.5f,
                     float exit_conf_threshold = 0.3f);

    /**
     * @brief Destructor.
     */
    ~vp_nanotrack_node() override;

protected:
    /**
     * @brief Handle frame metadata.
     * @param meta Frame metadata pointer.
     * @return Processed metadata.
     */
    virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) override;

    /**
     * @brief Handle control metadata.
     * @param meta Control metadata pointer.
     * @return Processed metadata.
     */
    virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) override;

private:
    /**
     * @brief Select best target from detection results.
     *
     * Strategy: Select target closest to frame center with confidence >= threshold.
     *
     * @param targets Detected targets.
     * @param frame_width Frame width.
     * @param frame_height Frame height.
     * @param best_bbox Output bounding box [x, y, w, h].
     * @return true if valid target found, false otherwise.
     */
    bool select_best_target(const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets,
                            int frame_width,
                            int frame_height,
                            std::array<float, 4>& best_bbox);

    /**
     * @brief Calculate distance from target center to frame center.
     * @param target Target to evaluate.
     * @param frame_center_x Frame center X coordinate.
     * @param frame_center_y Frame center Y coordinate.
     * @return Euclidean distance.
     */
    float distance_to_center(const std::shared_ptr<vp_objects::vp_frame_target>& target,
                             float frame_center_x,
                             float frame_center_y);

    /**
     * @brief Check if tracking should exit based on tracking output and detection results.
     *
     * Exit conditions:
     * 1. track_score < exit_conf_threshold (if threshold > 0)
     * 2. No detection with IoU >= iou_threshold (if threshold > 0 and detections exist)
     * 3. Consecutive frames without detection >= max_no_detection_frames (if threshold > 0)
     *
     * @param output Current tracking output.
     * @param targets Detection results from current frame.
     * @return true if tracking should exit, false otherwise.
     */
    bool should_exit_tracking(const TrackOutput& output,
                              const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets);

    /**
     * @brief Calculate IoU between tracking bbox and detection target.
     * @param bbox Tracking bounding box [x, y, w, h].
     * @param target Detection target.
     * @return IoU value in range [0, 1].
     */
    float calculate_iou(const std::array<float, 4>& bbox,
                        const std::shared_ptr<vp_objects::vp_frame_target>& target);

    /**
     * @brief Reset tracker and switch to SEARCHING state.
     */
    void reset_to_searching();

    /// Current tracking state.
    TrackState state_ = TrackState::SEARCHING;

    /// Tracking configuration.
    TrackConfig config_;

    /// NanoTrack tracker instance.
    std::shared_ptr<NanoTrack> tracker_;

    /// Configuration path.
    std::string json_path_;

    /// Counter for consecutive frames without detection.
    int no_detection_frames_ = 0;

    /// Frame counter for logging.
    uint64_t frame_counter_ = 0;

    /// Debug log interval.
    int debug_log_interval_ = 300;
};

}  // namespace vp_nodes
