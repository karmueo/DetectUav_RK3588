/**
 * @file nanotrack.cpp
 * @brief NanoTrack single object tracker implementation using RKNN and RGA.
 *
 * Implements the NanoTrack tracking algorithm with:
 * - RKNN model loading and inference
 * - RGA-based image preprocessing (crop, resize, color conversion)
 * - Template-feature based tracking with cosine window penalization
 *
 * @creation 2026-03-05
 * @modified 2026-03-05
 */

#include "nanotrack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>

#include "RgaUtils.h"
#include "im2d.h"
#include "im2d_common.h"

using json = nlohmann::json;

// ============================================================================
// NanoRknnModel Implementation
// ============================================================================

void NanoRknnModel::load(const std::string& model_path, int npu_core) {
    release();

    std::ifstream ifs(model_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open model: " + model_path);
    }

    ifs.seekg(0, std::ios::end);
    size_t model_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    model_data_.resize(model_size);
    ifs.read(reinterpret_cast<char*>(model_data_.data()), static_cast<std::streamsize>(model_size));

    int ret = rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
    }

    set_core(npu_core);
    query_io_attrs();

    spdlog::info("NanoRknnModel loaded: {} ({} bytes)", model_path, model_size);
}

void NanoRknnModel::release() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    model_data_.clear();
    input_attrs_.clear();
    output_attrs_.clear();
    io_num_ = {0, 0};
}

void NanoRknnModel::set_core(int npu_core) {
    rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
    if (npu_core == 0) mask = RKNN_NPU_CORE_0;
    else if (npu_core == 1) mask = RKNN_NPU_CORE_1;
    else if (npu_core == 2) mask = RKNN_NPU_CORE_2;
    else mask = RKNN_NPU_CORE_AUTO;

    int ret = rknn_set_core_mask(ctx_, mask);
    if (ret != RKNN_SUCC) {
        spdlog::warn("rknn_set_core_mask failed: {}", ret);
    }
}

void NanoRknnModel::query_io_attrs() {
    int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_query in/out num failed: " + std::to_string(ret));
    }

    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_query input attr failed: " + std::to_string(ret));
        }
    }

    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_query output attr failed: " + std::to_string(ret));
        }
    }
}

std::vector<NanoTensor> NanoRknnModel::run(const std::vector<rknn_input>& inputs) {
    if (ctx_ == 0) {
        throw std::runtime_error("Model not initialized");
    }
    if (inputs.size() != static_cast<size_t>(io_num_.n_input)) {
        throw std::runtime_error("Input number mismatch");
    }

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, const_cast<rknn_input*>(inputs.data()));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed: " + std::to_string(ret));
    }

    std::vector<rknn_output> outputs(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));
    }

    std::vector<NanoTensor> result;
    result.reserve(io_num_.n_output);

    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        const auto& attr = output_attrs_[i];
        int n = 1, c = 1, h = 1, w = 1;

        if (attr.n_dims >= 4) {
            if (attr.fmt == RKNN_TENSOR_NHWC) {
                n = attr.dims[0];
                h = attr.dims[1];
                w = attr.dims[2];
                c = attr.dims[3];
            } else {
                n = attr.dims[0];
                c = attr.dims[1];
                h = attr.dims[2];
                w = attr.dims[3];
            }
        } else {
            int elems = 1;
            for (uint32_t d = 0; d < attr.n_dims; ++d) {
                elems *= attr.dims[d];
            }
            n = 1;
            c = elems;
            h = 1;
            w = 1;
        }

        size_t elems = static_cast<size_t>(n) * c * h * w;
        NanoTensor t;
        t.data.resize(elems);
        t.dims = {n, c, h, w};
        t.fmt = attr.fmt;
        std::memcpy(t.data.data(), outputs[i].buf, elems * sizeof(float));
        result.push_back(std::move(t));
    }

    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    return result;
}

// ============================================================================
// NanoTrack Implementation
// ============================================================================

NanoTrack::NanoTrack(const NanoTrackConfig& config)
    : params_() {

    spdlog::info("Loading NanoTrack models...");

    backbone_.load(config.backbone_path, config.npu_core);
    spdlog::info("Backbone loaded: {}", config.backbone_path);

    backbone_search_.load(config.backbone_search_path, config.npu_core);
    spdlog::info("Backbone_search loaded: {}", config.backbone_search_path);

    head_.load(config.head_path, config.npu_core);
    spdlog::info("Head loaded: {}", config.head_path);

    score_size_ = params_.OUTPUT_SIZE;
    init_window();
    points_ = generate_points(params_.STRIDE, score_size_);

    spdlog::info("NanoTrack initialized (score_size={})", score_size_);
}

NanoTrack::~NanoTrack() {
    spdlog::info("Freeing NanoTrack models...");
}

int NanoTrack::load_config(const std::string& json_path, NanoTrackConfig& config) {
    try {
        std::ifstream f(json_path);
        if (!f.is_open()) {
            spdlog::error("Failed to open config file: {}", json_path);
            return -1;
        }

        json j_conf;
        f >> j_conf;

        config.backbone_path = j_conf.value("backbone_path", "");
        config.backbone_search_path = j_conf.value("backbone_search_path", "");
        config.head_path = j_conf.value("head_path", "");
        config.npu_core = j_conf.value("npu_core", 3);

        if (config.backbone_path.empty() ||
            config.backbone_search_path.empty() ||
            config.head_path.empty()) {
            spdlog::error("Missing required model paths in config");
            return -1;
        }

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse config: {}", e.what());
        return -1;
    }
}

void NanoTrack::init(const cv::Mat& frame_bgr, const std::array<float, 4>& bbox_xywh) {
    center_pos_ = cv::Point2f(
        bbox_xywh[0] + (bbox_xywh[2] - 1.0f) / 2.0f,
        bbox_xywh[1] + (bbox_xywh[3] - 1.0f) / 2.0f
    );
    target_size_ = cv::Size2f(bbox_xywh[2], bbox_xywh[3]);

    channel_average_ = cv::mean(frame_bgr);

    float w_z = target_size_.width + params_.CONTEXT_AMOUNT * (target_size_.width + target_size_.height);
    float h_z = target_size_.height + params_.CONTEXT_AMOUNT * (target_size_.width + target_size_.height);
    int s_z = static_cast<int>(std::round(std::sqrt(w_z * h_z)));

    cv::Mat z_crop = get_subwindow(frame_bgr, center_pos_, params_.EXEMPLAR_SIZE, s_z, channel_average_);
    z_feat_nchw_ = run_backbone(z_crop, true);
    initialized_ = true;

    spdlog::debug("NanoTrack initialized with bbox: [{}, {}, {}, {}]",
                  bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]);
}

TrackOutput NanoTrack::track(const cv::Mat& frame_bgr) {
    if (!initialized_) {
        throw std::runtime_error("Tracker not initialized");
    }

    float w_z = target_size_.width + params_.CONTEXT_AMOUNT * (target_size_.width + target_size_.height);
    float h_z = target_size_.height + params_.CONTEXT_AMOUNT * (target_size_.width + target_size_.height);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = static_cast<float>(params_.EXEMPLAR_SIZE) / s_z;
    float s_x = s_z * (static_cast<float>(params_.INSTANCE_SIZE) / static_cast<float>(params_.EXEMPLAR_SIZE));

    cv::Mat x_crop = get_subwindow(frame_bgr, center_pos_, params_.INSTANCE_SIZE,
                                   static_cast<int>(std::round(s_x)), channel_average_);

    NanoTensor x_feat_nchw = run_backbone(x_crop, false);
    auto head_out = run_head(z_feat_nchw_, x_feat_nchw);

    std::vector<float> scores = convert_score(head_out.first);
    std::array<std::vector<float>, 4> pred_bbox = convert_bbox(head_out.second, points_);

    // Calculate penalty and pscore
    auto change = [](float r) { return std::max(r, 1.0f / r); };
    auto sz = [](float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    };

    int n = static_cast<int>(scores.size());
    std::vector<float> penalty(n, 0.0f);
    std::vector<float> pscore(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float s_c = change(sz(pred_bbox[2][i], pred_bbox[3][i]) /
                          sz(target_size_.width * scale_z, target_size_.height * scale_z));
        float r_c = change((target_size_.width / target_size_.height) /
                          (pred_bbox[2][i] / pred_bbox[3][i]));
        penalty[i] = std::exp(-(r_c * s_c - 1.0f) * params_.PENALTY_K);
        pscore[i] = penalty[i] * scores[i];
        pscore[i] = pscore[i] * (1.0f - params_.WINDOW_INFLUENCE) +
                    window_[i] * params_.WINDOW_INFLUENCE;
    }

    // Find best index
    int best_idx = 0;
    float best_pscore = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (pscore[i] > best_pscore) {
            best_pscore = pscore[i];
            best_idx = i;
        }
    }

    std::array<float, 4> bbox{
        pred_bbox[0][best_idx] / scale_z,
        pred_bbox[1][best_idx] / scale_z,
        pred_bbox[2][best_idx] / scale_z,
        pred_bbox[3][best_idx] / scale_z,
    };

    float lr = penalty[best_idx] * scores[best_idx] * params_.LR;

    float cx = bbox[0] + center_pos_.x;
    float cy = bbox[1] + center_pos_.y;
    float width = target_size_.width * (1.0f - lr) + bbox[2] * lr;
    float height = target_size_.height * (1.0f - lr) + bbox[3] * lr;

    bbox_clip(cx, cy, width, height, frame_bgr.rows, frame_bgr.cols);
    center_pos_ = cv::Point2f(cx, cy);
    target_size_ = cv::Size2f(width, height);

    TrackOutput out;
    out.bbox = {cx - width / 2.0f, cy - height / 2.0f, width, height};
    out.score = scores[best_idx];
    return out;
}

void NanoTrack::reset() {
    initialized_ = false;
    center_pos_ = cv::Point2f();
    target_size_ = cv::Size2f();
    z_feat_nchw_ = NanoTensor();
}

NanoTensor NanoTrack::to_nchw(const NanoTensor& in) {
    if (in.fmt == RKNN_TENSOR_NCHW) {
        return in;
    }

    int n = in.dims[0];
    int c = in.dims[1];
    int h = in.dims[2];
    int w = in.dims[3];

    std::vector<float> out(static_cast<size_t>(n) * c * h * w, 0.0f);
    for (int ni = 0; ni < n; ++ni) {
        for (int yi = 0; yi < h; ++yi) {
            for (int xi = 0; xi < w; ++xi) {
                for (int ci = 0; ci < c; ++ci) {
                    size_t src_idx = static_cast<size_t>(ni) * h * w * c + yi * w * c + xi * c + ci;
                    size_t dst_idx = static_cast<size_t>(ni) * c * h * w + ci * h * w + yi * w + xi;
                    out[dst_idx] = in.data[src_idx];
                }
            }
        }
    }

    NanoTensor t;
    t.data = std::move(out);
    t.dims = {n, c, h, w};
    t.fmt = RKNN_TENSOR_NCHW;
    return t;
}

NanoTensor NanoTrack::run_backbone(const cv::Mat& input_bgr, bool is_template) {
    if (!input_bgr.isContinuous()) {
        throw std::runtime_error("backbone input must be continuous");
    }

    rknn_input in;
    std::memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.size = static_cast<uint32_t>(input_bgr.total() * input_bgr.elemSize());
    in.fmt = RKNN_TENSOR_NHWC;
    in.pass_through = 0;
    in.buf = const_cast<unsigned char*>(input_bgr.data);

    std::vector<rknn_input> inputs{in};
    auto outs = is_template ? backbone_.run(inputs) : backbone_search_.run(inputs);
    if (outs.empty()) {
        throw std::runtime_error("backbone output empty");
    }
    return to_nchw(outs[0]);
}

std::pair<NanoTensor, NanoTensor> NanoTrack::run_head(const NanoTensor& z_nchw, const NanoTensor& x_nchw) {
    int zn = z_nchw.dims[0], zc = z_nchw.dims[1], zh = z_nchw.dims[2], zw = z_nchw.dims[3];
    int xn = x_nchw.dims[0], xc = x_nchw.dims[1], xh = x_nchw.dims[2], xw = x_nchw.dims[3];

    // NCHW -> NHWC conversion for head input
    auto nchw_to_nhwc = [](const std::vector<float>& data, int n, int c, int h, int w) {
        std::vector<float> out(static_cast<size_t>(n) * h * w * c, 0.0f);
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int yi = 0; yi < h; ++yi) {
                    for (int xi = 0; xi < w; ++xi) {
                        size_t src_idx = static_cast<size_t>(ni) * c * h * w + ci * h * w + yi * w + xi;
                        size_t dst_idx = static_cast<size_t>(ni) * h * w * c + yi * w * c + xi * c + ci;
                        out[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        return out;
    };

    z_feat_nhwc_buf_ = nchw_to_nhwc(z_nchw.data, zn, zc, zh, zw);
    x_feat_nhwc_buf_ = nchw_to_nhwc(x_nchw.data, xn, xc, xh, xw);

    rknn_input in0;
    std::memset(&in0, 0, sizeof(in0));
    in0.index = 0;
    in0.type = RKNN_TENSOR_FLOAT32;
    in0.size = static_cast<uint32_t>(z_feat_nhwc_buf_.size() * sizeof(float));
    in0.fmt = RKNN_TENSOR_NHWC;
    in0.pass_through = 0;
    in0.buf = z_feat_nhwc_buf_.data();

    rknn_input in1;
    std::memset(&in1, 0, sizeof(in1));
    in1.index = 1;
    in1.type = RKNN_TENSOR_FLOAT32;
    in1.size = static_cast<uint32_t>(x_feat_nhwc_buf_.size() * sizeof(float));
    in1.fmt = RKNN_TENSOR_NHWC;
    in1.pass_through = 0;
    in1.buf = x_feat_nhwc_buf_.data();

    std::vector<rknn_input> inputs{in0, in1};
    auto outs = head_.run(inputs);
    if (outs.size() < 2) {
        throw std::runtime_error("head output number < 2");
    }

    NanoTensor cls = to_nchw(outs[0]);
    NanoTensor loc = to_nchw(outs[1]);
    return {std::move(cls), std::move(loc)};
}

std::vector<cv::Point2f> NanoTrack::generate_points(int stride, int size) {
    std::vector<cv::Point2f> points(static_cast<size_t>(size) * size);
    int ori = -(size / 2) * stride;
    size_t idx = 0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            points[idx++] = cv::Point2f(static_cast<float>(ori + stride * x),
                                        static_cast<float>(ori + stride * y));
        }
    }
    return points;
}

void NanoTrack::init_window() {
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> hanning(score_size_);
    for (int i = 0; i < score_size_; ++i) {
        hanning[i] = 0.5f - 0.5f * std::cos(2.0f * kPi * i / static_cast<float>(score_size_ - 1));
    }

    window_.resize(static_cast<size_t>(score_size_) * score_size_);
    size_t idx = 0;
    for (int y = 0; y < score_size_; ++y) {
        for (int x = 0; x < score_size_; ++x) {
            window_[idx++] = hanning[y] * hanning[x];
        }
    }
}

std::vector<float> NanoTrack::convert_score(const NanoTensor& cls_nchw) const {
    int n = cls_nchw.dims[0];
    int c = cls_nchw.dims[1];
    int h = cls_nchw.dims[2];
    int w = cls_nchw.dims[3];
    if (n != 1) {
        throw std::runtime_error("cls batch must be 1");
    }

    int hw = h * w;
    std::vector<float> score(hw, 0.0f);

    if (params_.CLS_OUT_CHANNELS == 1 || c == 1) {
        for (int i = 0; i < hw; ++i) {
            float v = cls_nchw.data[i];
            score[i] = 1.0f / (1.0f + std::exp(-v));
        }
        return score;
    }

    for (int i = 0; i < hw; ++i) {
        float v0 = cls_nchw.data[0 * hw + i];
        float v1 = cls_nchw.data[1 * hw + i];
        float vmax = std::max(v0, v1);
        float e0 = std::exp(v0 - vmax);
        float e1 = std::exp(v1 - vmax);
        score[i] = e1 / (e0 + e1);
    }
    return score;
}

std::array<std::vector<float>, 4> NanoTrack::convert_bbox(const NanoTensor& loc_nchw,
                                                          const std::vector<cv::Point2f>& points) const {
    int n = loc_nchw.dims[0];
    int c = loc_nchw.dims[1];
    int h = loc_nchw.dims[2];
    int w = loc_nchw.dims[3];
    if (n != 1 || c != 4) {
        throw std::runtime_error("loc shape must be [1,4,H,W]");
    }

    int hw = h * w;
    if (static_cast<int>(points.size()) != hw) {
        throw std::runtime_error("points size mismatch with loc");
    }

    std::array<std::vector<float>, 4> out;
    for (auto& v : out) {
        v.resize(hw, 0.0f);
    }

    const float* d0 = loc_nchw.data.data() + 0 * hw;
    const float* d1 = loc_nchw.data.data() + 1 * hw;
    const float* d2 = loc_nchw.data.data() + 2 * hw;
    const float* d3 = loc_nchw.data.data() + 3 * hw;

    for (int i = 0; i < hw; ++i) {
        float x1 = points[i].x - d0[i];
        float y1 = points[i].y - d1[i];
        float x2 = points[i].x + d2[i];
        float y2 = points[i].y + d3[i];

        out[0][i] = (x1 + x2) * 0.5f;
        out[1][i] = (y1 + y2) * 0.5f;
        out[2][i] = x2 - x1;
        out[3][i] = y2 - y1;
    }

    return out;
}

void NanoTrack::bbox_clip(float& cx, float& cy, float& w, float& h, int frame_h, int frame_w) {
    cx = std::max(0.0f, std::min(cx, static_cast<float>(frame_w)));
    cy = std::max(0.0f, std::min(cy, static_cast<float>(frame_h)));
    w = std::max(10.0f, std::min(w, static_cast<float>(frame_w)));
    h = std::max(10.0f, std::min(h, static_cast<float>(frame_h)));
}

cv::Mat NanoTrack::get_subwindow(const cv::Mat& im, const cv::Point2f& pos, int model_sz,
                                  int original_sz, const cv::Scalar& avg_chans) {
    const float sz = static_cast<float>(original_sz);
    const float c = (sz + 1.0f) / 2.0f;

    float context_xmin_f = std::floor(pos.x - c + 0.5f);
    float context_xmax_f = context_xmin_f + sz - 1.0f;
    float context_ymin_f = std::floor(pos.y - c + 0.5f);
    float context_ymax_f = context_ymin_f + sz - 1.0f;

    int left_pad = std::max(0, static_cast<int>(-context_xmin_f));
    int top_pad = std::max(0, static_cast<int>(-context_ymin_f));
    int right_pad = std::max(0, static_cast<int>(context_xmax_f - static_cast<float>(im.cols) + 1.0f));
    int bottom_pad = std::max(0, static_cast<int>(context_ymax_f - static_cast<float>(im.rows) + 1.0f));

    int context_xmin = static_cast<int>(context_xmin_f) + left_pad;
    int context_ymin = static_cast<int>(context_ymin_f) + top_pad;

    cv::Mat src;
    if (left_pad || right_pad || top_pad || bottom_pad) {
        cv::copyMakeBorder(im, src, top_pad, bottom_pad, left_pad, right_pad,
                          cv::BORDER_CONSTANT, avg_chans);
    } else {
        src = im;
    }

    int roi_sz = original_sz;
    roi_sz = std::max(1, roi_sz);
    context_xmin = std::max(0, std::min(context_xmin, src.cols - roi_sz));
    context_ymin = std::max(0, std::min(context_ymin, src.rows - roi_sz));

    cv::Rect roi(context_xmin, context_ymin, roi_sz, roi_sz);
    cv::Mat patch = src(roi).clone();

    // Use OpenCV resize (RGA for full-frame preprocessing is handled elsewhere)
    // For crop regions, OpenCV resize is acceptable as it's on small patches
    if (model_sz != original_sz) {
        cv::resize(patch, patch, cv::Size(model_sz, model_sz));
    }
    return patch;
}
