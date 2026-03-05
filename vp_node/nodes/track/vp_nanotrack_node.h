/**
 * @file vp_nanotrack_node.h
 * @brief NanoTrack single object tracking node for VideoPipe pipeline.
 *
 * This node integrates the NanoTrack tracker into the VideoPipe framework, providing:
 * - Automatic target selection from detection results
 * - Target initialization based on detection or * - Continuous tracking with confidence monitoring
 * - Automatic exit on low confidence
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

namespace vp_nodes {

/**
 * @brief Single object tracking node using NanoTrack algorithm.
 *
 * This node performs single object tracking using the NanoTrack algorithm.
 * It workflow:
 * 1. Wait for detection results from upstream nodes
 * 2. Select target closest to frame center with confidence above threshold
 * 3. Initialize tracker with selected target
 * 4. Track target in subsequent frames
 * 5. Exit tracking if confidence drops below threshold
 *
 * Target selection strategy:
 * - Select the target with highest confidence among targets near frame center
 * - Targets must have confidence >= selection_conf_threshold
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
     * @return true if valid target found,     */
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

    /// NanoTrack tracker instance.
    std::shared_ptr<NanoTrack> tracker_;

    /// Configuration path.
    std::string json_path_;

    /// Minimum confidence for target selection.
    float selection_conf_threshold_;

    /// Minimum confidence before tracking exit.
    float exit_conf_threshold_;

    /// Frame counter for logging.
    uint64_t frame_counter_ = 0;

    /// Debug log interval.
    int debug_log_interval_ = 300;
};

}  // namespace vp_nodes
