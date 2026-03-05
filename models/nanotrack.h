/**
 * @file nanotrack.h
 * @brief NanoTrack single object tracker using RKNN inference with RGA preprocessing.
 *
 * This module implements a lightweight single object tracking algorithm based on
 * the NanoTrack architecture. It uses three RKNN models (backbone, backbone_search, head)
 * and leverages RGA for hardware-accelerated image preprocessing.
 *
 * @creation 2026-03-05
 * @modified 2026-03-05
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

#include "config.h"
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

/**
 * @brief NanoTrack tracking parameters.
 *
 * These parameters control the tracking behavior including window influence,
 * penalty calculation, and learning rate.
 */
struct NanoTrackParams {
    float WINDOW_INFLUENCE = 0.455f;   ///< Cosine window influence weight.
    float PENALTY_K = 0.138f;          ///< Scale/ratio penalty coefficient.
    float LR = 0.348f;                 ///< Learning rate for bbox update.
    int EXEMPLAR_SIZE = 127;           ///< Template crop size (z-axis).
    int INSTANCE_SIZE = 255;           ///< Search region crop size (x-axis).
    float CONTEXT_AMOUNT = 0.5f;       ///< Context padding ratio.
    int OUTPUT_SIZE = 15;              ///< Score map output size.
    int STRIDE = 16;                   ///< Feature map stride.
    int CLS_OUT_CHANNELS = 2;          ///< Classification output channels.
};

/**
 * @brief Tracking output structure containing bbox and confidence score.
 */
struct TrackOutput {
    std::array<float, 4> bbox;  ///< Bounding box [x, y, width, height].
    float score = 0.0f;         ///< Confidence score.
};

/**
 * @brief Configuration for NanoTrack model loading.
 */
struct NanoTrackConfig : public Config {
    std::string backbone_path;         ///< Path to backbone template model.
    std::string backbone_search_path;  ///< Path to backbone search model.
    std::string head_path;             ///< Path to head model.
    int npu_core = 3;                  ///< NPU core assignment (0-2 specific, 3=auto).
    float track_conf_threshold = 0.3f; ///< Tracking confidence threshold for exit.
};

/**
 * @brief Internal tensor wrapper for RKNN input/output.
 */
struct NanoTensor {
    std::vector<float> data;           ///< Tensor data buffer.
    std::array<int, 4> dims{1, 1, 1, 1}; ///< Tensor dimensions [N, C, H, W].
    rknn_tensor_format fmt = RKNN_TENSOR_NHWC; ///< Tensor format.
};

/**
 * @brief RKNN model wrapper for NanoTrack.
 *
 * Encapsulates RKNN context, model loading, and inference execution.
 */
class NanoRknnModel {
public:
    NanoRknnModel() = default;
    ~NanoRknnModel() { release(); }

    /**
     * @brief Load RKNN model from file.
     * @param model_path Path to .rknn model file.
     * @param npu_core NPU core assignment.
     * @throws std::runtime_error if loading fails.
     */
    void load(const std::string& model_path, int npu_core);

    /**
     * @brief Release RKNN resources.
     */
    void release();

    /**
     * @brief Get input/output number info.
     */
    const rknn_input_output_num& io_num() const { return io_num_; }

    /**
     * @brief Get input tensor attributes.
     */
    const std::vector<rknn_tensor_attr>& input_attrs() const { return input_attrs_; }

    /**
     * @brief Get output tensor attributes.
     */
    const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }

    /**
     * @brief Run inference with given inputs.
     * @param inputs Vector of RKNN input tensors.
     * @return Vector of output tensors with float data.
     * @throws std::runtime_error if inference fails.
     */
    std::vector<NanoTensor> run(const std::vector<rknn_input>& inputs);

private:
    /**
     * @brief Set NPU core mask for inference.
     * @param npu_core Core index (0-2) or 3 for auto.
     */
    void set_core(int npu_core);

    /**
     * @brief Query input/output tensor attributes.
     */
    void query_io_attrs();

    rknn_context ctx_ = 0;                       ///< RKNN context.
    std::vector<uint8_t> model_data_;            ///< Model binary data.
    rknn_input_output_num io_num_{0, 0};         ///< Input/output counts.
    std::vector<rknn_tensor_attr> input_attrs_;  ///< Input tensor attributes.
    std::vector<rknn_tensor_attr> output_attrs_; ///< Output tensor attributes.
};

/**
 * @brief NanoTrack single object tracker using RKNN and RGA.
 *
 * Implements the NanoTrack algorithm for real-time single object tracking.
 * Uses three separate RKNN models and RGA for hardware-accelerated preprocessing.
 *
 * Usage:
 * @code
 * NanoTrackConfig config;
 * config.backbone_path = "backbone.rknn";
 * config.backbone_search_path = "backbone_search.rknn";
 * config.head_path = "head.rknn";
 * NanoTrack tracker(config);
 *
 * // Initialize with first frame and target bbox
 * tracker.init(frame, {x, y, w, h});
 *
 * // Track in subsequent frames
 * TrackOutput out = tracker.track(frame);
 * @endcode
 */
class NanoTrack {
public:
    /**
     * @brief Construct NanoTrack with configuration.
     * @param config Model and tracking configuration.
     * @throws std::runtime_error if model loading fails.
     */
    explicit NanoTrack(const NanoTrackConfig& config);

    /**
     * @brief Destructor releasing RKNN resources.
     */
    ~NanoTrack();

    /**
     * @brief Load configuration from JSON file.
     * @param json_path Path to JSON configuration file.
     * @param config Output configuration structure.
     * @return 0 on success, -1 on failure.
     */
    static int load_config(const std::string& json_path, NanoTrackConfig& config);

    /**
     * @brief Initialize tracker with first frame and target bbox.
     * @param frame_bgr Input frame in BGR format.
     * @param bbox_xywh Initial bounding box [x, y, width, height].
     */
    void init(const cv::Mat& frame_bgr, const std::array<float, 4>& bbox_xywh);

    /**
     * @brief Track target in current frame.
     * @param frame_bgr Input frame in BGR format.
     * @return Tracking output with bbox and confidence score.
     * @throws std::runtime_error if tracker not initialized.
     */
    TrackOutput track(const cv::Mat& frame_bgr);

    /**
     * @brief Check if tracker is initialized.
     * @return true if initialized, false otherwise.
     */
    bool is_initialized() const { return initialized_; }

    /**
     * @brief Reset tracker to uninitialized state.
     */
    void reset();

    /**
     * @brief Get tracking confidence threshold.
     * @return Confidence threshold value.
     */
    float get_track_conf_threshold() const { return track_conf_threshold_; }

private:
    /**
     * @brief Convert tensor from NHWC to NCHW format.
     * @param in Input tensor in NHWC format.
     * @return Tensor in NCHW format.
     */
    static NanoTensor to_nchw(const NanoTensor& in);

    /**
     * @brief Run backbone inference on crop.
     * @param input_bgr Input BGR image crop.
     * @param is_template true for template backbone, false for search backbone.
     * @return Feature tensor in NCHW format.
     */
    NanoTensor run_backbone(const cv::Mat& input_bgr, bool is_template);

    /**
     * @brief Run head inference on feature pair.
     * @param z_nchw Template feature tensor.
     * @param x_nchw Search feature tensor.
     * @return Pair of classification and localization tensors.
     */
    std::pair<NanoTensor, NanoTensor> run_head(const NanoTensor& z_nchw, const NanoTensor& x_nchw);

    /**
     * @brief Generate anchor point grid.
     * @param stride Feature stride.
     * @param size Score map size.
     * @return Vector of anchor points.
     */
    static std::vector<cv::Point2f> generate_points(int stride, int size);

    /**
     * @brief Initialize cosine window for score penalization.
     */
    void init_window();

    /**
     * @brief Convert classification output to confidence scores.
     * @param cls_nchw Classification tensor.
     * @return Vector of confidence scores.
     */
    std::vector<float> convert_score(const NanoTensor& cls_nchw) const;

    /**
     * @brief Convert localization output to bounding boxes.
     * @param loc_nchw Localization tensor.
     * @param points Anchor points.
     * @return Array of bbox coordinate vectors [cx, cy, w, h].
     */
    std::array<std::vector<float>, 4> convert_bbox(const NanoTensor& loc_nchw,
                                                    const std::vector<cv::Point2f>& points) const;

    /**
     * @brief Clip bounding box coordinates to frame boundaries.
     */
    static void bbox_clip(float& cx, float& cy, float& w, float& h, int frame_h, int frame_w);

    /**
     * @brief Extract crop region with padding and resize using RGA.
     * @param im Input image.
     * @param pos Center position of crop.
     * @param model_sz Target model input size.
     * @param original_sz Original crop size before resize.
     * @param avg_chans Channel mean values for padding.
     * @return Cropped and resized patch.
     */
    cv::Mat get_subwindow(const cv::Mat& im, const cv::Point2f& pos, int model_sz,
                          int original_sz, const cv::Scalar& avg_chans);

    NanoTrackParams params_;          ///< Tracking parameters.
    NanoRknnModel backbone_;          ///< Template backbone model.
    NanoRknnModel backbone_search_;   ///< Search backbone model.
    NanoRknnModel head_;              ///< Head model.

    int score_size_ = 15;             ///< Score map size.
    std::vector<float> window_;       ///< Cosine window for penalization.
    std::vector<cv::Point2f> points_; ///< Anchor points.

    bool initialized_ = false;        ///< Initialization flag.
    cv::Point2f center_pos_;          ///< Current target center position.
    cv::Size2f target_size_;          ///< Current target size.
    cv::Scalar channel_average_;      ///< Channel mean for padding.

    NanoTensor z_feat_nchw_;          ///< Cached template feature.
    std::vector<float> z_feat_nhwc_buf_; ///< Buffer for NHWC conversion.
    std::vector<float> x_feat_nhwc_buf_; ///< Buffer for NHWC conversion.

    float track_conf_threshold_;      ///< Confidence threshold for tracking exit.

    // RGA cache buffers
    int cache_src_width_ = 0;         ///< Cached source width.
    int cache_src_height_ = 0;        ///< Cached source height.
    int cache_dst_width_ = 0;         ///< Cached destination width.
    int cache_dst_height_ = 0;        ///< Cached destination height.
    std::vector<uint8_t> cache_rgb_resize_data_; ///< Cached RGB resize buffer.
};
