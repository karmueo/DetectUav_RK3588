/**
 * @file vp_nanotrack_node.h
 * @brief NanoTrack单目标跟踪节点
 * @details 基于NanoTrack的单目标跟踪节点实现，继承自vp_track_node
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vp_track_node.h"
#include "../../../models/nanotrack.h"

namespace vp_nodes {

/**
 * @brief NanoTrack单目标跟踪节点
 * @details 实现基于RKNN的单目标跟踪，支持从检测结果中选择目标并进行跟踪
 */
class vp_nanotrack_node : public vp_track_node {
private:
    // 跟踪实现 (必须实现纯虚函数)
    virtual void track(int channel_index,
                       const std::vector<vp_objects::vp_rect>& target_rects,
                       const std::vector<std::vector<float>>& target_embeddings,
                       std::vector<int>& track_ids) override;

    /**
     * @brief 加载配置文件
     * @param config_path 配置文件路径
     */
    void load_config(const std::string& config_path);

    /**
     * @brief 从检测目标中选择要跟踪的目标
     * @param targets 检测目标列表
     * @return 选中的目标，如果无有效目标则返回nullptr
     */
    std::shared_ptr<vp_objects::vp_frame_target> select_target(
        const std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets);

    /**
     * @brief 计算两个矩形的IOU
     * @param rect1 第一个矩形
     * @param rect2 第二个矩形
     * @return IOU值
     */
    float calculate_iou(const vp_objects::vp_rect& rect1, const vp_objects::vp_rect& rect2);

    // NanoTrack跟踪器
    std::unique_ptr<NanoTrack> tracker_;

    // 配置文件路径
    std::string config_path_;

    // 跟踪器配置参数
    std::string backbone_path_;
    std::string backbone_search_path_;
    std::string head_path_;
    int npu_core_ = 3;

    // 跟踪参数
    float conf_threshold_ = 0.5f;
    float track_conf_threshold_ = 0.3f;
    bool enable_iou_check_ = true;
    float iou_threshold_ = 0.3f;
    float window_influence_ = 0.455f;
    float penalty_k_ = 0.138f;
    float lr_ = 0.348f;

    // 跟踪状态
    bool tracking_initialized_ = false;
    int fixed_track_id_ = 0;  // 单目标跟踪使用固定ID

    // 上一帧的目标位置，用于IOU检查
    vp_objects::vp_rect last_target_rect_;

protected:
    /**
     * @brief 处理帧元数据
     * @param meta 帧元数据
     * @return 处理后的元数据
     */
    virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) override final;

public:
    /**
     * @brief 构造函数
     * @param node_name 节点名称
     * @param config_path 配置文件路径
     */
    vp_nanotrack_node(std::string node_name, std::string config_path);

    /**
     * @brief 析构函数
     */
    virtual ~vp_nanotrack_node();

    /**
     * @brief 重置跟踪状态
     * @details 清除当前跟踪目标，重新开始跟踪
     */
    void reset_tracking();

    /**
     * @brief 手动初始化跟踪目标
     * @param frame_bgr 初始化帧 (BGR格式)
     * @param bbox 目标框 (x, y, w, h)
     */
    void init_target(const cv::Mat& frame_bgr, const std::array<float, 4>& bbox);
};

}  // namespace vp_nodes
