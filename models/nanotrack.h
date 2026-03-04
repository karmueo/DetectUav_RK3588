/**
 * @file nanotrack.h
 * @brief NanoTrack单目标跟踪模型封装
 * @details 基于RKNN的NanoTrack跟踪器实现，提供单目标跟踪功能
 */

#ifndef _NANOTRACK_H_
#define _NANOTRACK_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

/**
 * @brief NanoTrack跟踪参数
 */
struct TrackParams {
    float window_influence = 0.455f;
    float penalty_k = 0.138f;
    float lr = 0.348f;
    int exemplar_size = 127;
    int instance_size = 255;
    float context_amount = 0.5f;
    int output_size = 15;
    int stride = 16;
    int cls_out_channels = 2;
};

/**
 * @brief NanoTrack跟踪输出结果
 */
struct TrackOutput {
    std::array<float, 4> bbox;  // x, y, w, h
    float score = 0.0f;
};

/**
 * @brief Tensor结构封装
 */
struct Tensor {
    std::vector<float> data;
    std::array<int, 4> dims{1, 1, 1, 1};
    rknn_tensor_format fmt = RKNN_TENSOR_NHWC;
};

/**
 * @brief RKNN模型封装类
 */
class RknnModel {
public:
    RknnModel() = default;
    ~RknnModel() { release(); }

    /**
     * @brief 加载RKNN模型
     * @param model_path 模型文件路径
     * @param npu_core NPU核心编号 (0-3)
     */
    void load(const std::string& model_path, int npu_core);

    /**
     * @brief 释放模型资源
     */
    void release();

    /**
     * @brief 运行推理
     * @param inputs 输入tensor列表
     * @return 输出tensor列表
     */
    std::vector<Tensor> run(const std::vector<rknn_input>& inputs);

    const rknn_input_output_num& io_num() const { return io_num_; }
    const std::vector<rknn_tensor_attr>& input_attrs() const { return input_attrs_; }
    const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }

private:
    void set_core(int npu_core);
    void query_io_attrs();

    rknn_context ctx_ = 0;
    std::vector<uint8_t> model_data_;
    rknn_input_output_num io_num_{0, 0};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
};

/**
 * @brief NanoTrack单目标跟踪器
 */
class NanoTrack {
public:
    /**
     * @brief 构造函数
     * @param backbone_path 主干网络模型路径
     * @param backbone_search_path 搜索区域主干网络模型路径
     * @param head_path 头部网络模型路径
     * @param npu_core NPU核心编号
     */
    NanoTrack(const std::string& backbone_path,
              const std::string& backbone_search_path,
              const std::string& head_path,
              int npu_core = 3);

    ~NanoTrack();

    /**
     * @brief 初始化跟踪目标
     * @param frame_bgr 输入图像 (BGR格式)
     * @param bbox_xywh 目标框 (x, y, w, h)
     */
    void init(const cv::Mat& frame_bgr, const std::array<float, 4>& bbox_xywh);

    /**
     * @brief 跟踪单帧
     * @param frame_bgr 输入图像 (BGR格式)
     * @return 跟踪结果
     */
    TrackOutput track(const cv::Mat& frame_bgr);

    /**
     * @brief 设置跟踪参数
     * @param window_influence 窗口影响因子
     * @param penalty_k 惩罚系数
     * @param lr 学习率
     */
    void set_params(float window_influence, float penalty_k, float lr);

    /**
     * @brief 检查是否已初始化
     * @return 是否已初始化
     */
    bool is_initialized() const { return initialized_; }

private:
    // 内部辅助函数
    static Tensor to_nchw(const Tensor& in);
    static std::vector<float> nchw_to_nhwc(const std::vector<float>& data, int n, int c, int h, int w);

    Tensor run_backbone(const cv::Mat& input_bgr, bool is_template);
    std::pair<Tensor, Tensor> run_head(const Tensor& z_nchw, const Tensor& x_nchw);

    // 图像裁剪
    cv::Mat get_subwindow(const cv::Mat& im, const cv::Point2f& pos, int model_sz,
                          int original_sz, const cv::Scalar& avg_chans);

    // 初始化辅助
    void init_window();
    std::vector<cv::Point2f> generate_points(int stride, int size);

    // 后处理
    std::vector<float> convert_score(const Tensor& cls_nchw) const;
    std::array<std::vector<float>, 4> convert_bbox(const Tensor& loc_nchw,
                                                    const std::vector<cv::Point2f>& points) const;

    static void bbox_clip(float& cx, float& cy, float& w, float& h, int frame_h, int frame_w);

    // 模型
    TrackParams params_;
    RknnModel backbone_;
    RknnModel backbone_search_;
    RknnModel head_;

    // 跟踪状态
    int score_size_ = 15;
    std::vector<float> window_;
    std::vector<cv::Point2f> points_;

    bool initialized_ = false;
    cv::Point2f center_pos_;
    cv::Size2f target_size_;
    cv::Scalar channel_average_;

    // 特征缓存
    Tensor z_feat_nchw_;
    std::vector<float> z_feat_nhwc_buf_;
    std::vector<float> x_feat_nhwc_buf_;
};

#endif  // _NANOTRACK_H_
