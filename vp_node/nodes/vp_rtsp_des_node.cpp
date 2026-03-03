#include "vp_rtsp_des_node.h"

#include "vp_utils/vp_utils.h"

namespace vp_nodes {

vp_rtsp_des_node::vp_rtsp_des_node(std::string node_name,
                                   int channel_index,
                                   std::string rtsp_url,
                                   vp_objects::vp_size resolution_w_h,
                                   bool osd,
                                   std::string gst_encoder_name,
                                   bool append_channel_index)
    : vp_des_node(node_name, channel_index),
      rtsp_url(std::move(rtsp_url)),
      gst_encoder_name(std::move(gst_encoder_name)),
      resolution_w_h(resolution_w_h),
      osd(osd),
      append_channel_index(append_channel_index) {
    if (this->gst_encoder_name == "mpph264enc") {
        // 兼顾 VLC 兼容性和长时间稳定性：固定码率 + 周期 IDR + 每个 IDR 携带头信息。
        this->gst_encoder_pipeline =
            "mpph264enc rc-mode=cbr bps=4000000 gop=50 header-mode=each-idr profile=baseline level=40";
    } else {
        this->gst_encoder_pipeline = this->gst_encoder_name;
    }
    if (this->append_channel_index) {
        this->rtsp_url = this->rtsp_url + "_" + std::to_string(channel_index);
    }
    this->gst_pipeline = "";
    VP_INFO(vp_utils::string_format("[%s] rtsp url=%s encoder=[%s]",
                                    this->node_name.c_str(),
                                    this->rtsp_url.c_str(),
                                    this->gst_encoder_pipeline.c_str()));
    this->initialized();
}

vp_rtsp_des_node::~vp_rtsp_des_node() {
    if (rtsp_writer.isOpened()) {
        rtsp_writer.release();
    }
    deinitialized();
}

std::shared_ptr<vp_objects::vp_meta>
vp_rtsp_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
    if (this->output_fps > 0) {
        const auto now_tp = std::chrono::steady_clock::now();
        if (this->last_pushed_frame_tp.time_since_epoch().count() != 0) {
            const auto delta_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - this->last_pushed_frame_tp).count();
            const int min_interval_ms = 1000 / this->output_fps;
            if (delta_ms < min_interval_ms) {
                return vp_des_node::handle_frame_meta(meta);
            }
        }
        this->last_pushed_frame_tp = now_tp;
    }

    cv::Mat push_frame;
    if (this->resolution_w_h.width > 0 && this->resolution_w_h.height > 0) {
        cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame,
                   push_frame,
                   cv::Size(this->resolution_w_h.width, this->resolution_w_h.height));
    } else {
        push_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
    }

    if (push_frame.empty()) {
        return vp_des_node::handle_frame_meta(meta);
    }

    if (!rtsp_writer.isOpened()) {
        const auto now_tp = std::chrono::steady_clock::now();
        if (open_last_retry_tp.time_since_epoch().count() != 0) {
            const auto retry_delta_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - open_last_retry_tp).count();
            if (retry_delta_ms < open_retry_interval_ms) {
                return vp_des_node::handle_frame_meta(meta);
            }
        }
        open_last_retry_tp = now_tp;

        const int output_fps = this->output_fps > 0 ? this->output_fps : ((meta->fps > 0) ? meta->fps : 25);
        this->gst_pipeline = vp_utils::string_format(this->gst_template,
                                                     push_frame.cols,
                                                     push_frame.rows,
                                                     output_fps,
                                                     this->gst_encoder_pipeline.c_str(),
                                                     this->rtsp_url.c_str());
        const bool opened = rtsp_writer.open(
            this->gst_pipeline,
            cv::CAP_GSTREAMER,
            0,
            output_fps,
            {push_frame.cols, push_frame.rows});
        if (!opened) {
            VP_ERROR(vp_utils::string_format("[%s] open rtsp writer failed, pipeline=[%s]",
                                             this->node_name.c_str(),
                                             this->gst_pipeline.c_str()));
            return vp_des_node::handle_frame_meta(meta);
        }
        VP_INFO(vp_utils::string_format("[%s] open rtsp writer success, pipeline=[%s]",
                                        this->node_name.c_str(),
                                        this->gst_pipeline.c_str()));

        pushed_frames = 0;
        push_start_tp = std::chrono::steady_clock::now();
        push_last_log_tp = push_start_tp;
    }

    rtsp_writer.write(push_frame);
    pushed_frames++;

    const auto now_tp = std::chrono::steady_clock::now();
    const auto log_delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - push_last_log_tp).count();
    if (log_delta_ms >= 1000) {
        const auto total_delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - push_start_tp).count();
        const double avg_fps = total_delta_ms > 0 ? (pushed_frames * 1000.0 / static_cast<double>(total_delta_ms)) : 0.0;
        VP_INFO(vp_utils::string_format("[%s] push_rtsp fps=%.2f frames=%llu url=%s",
                                        this->node_name.c_str(),
                                        avg_fps,
                                        static_cast<unsigned long long>(pushed_frames),
                                        this->rtsp_url.c_str()));
        push_last_log_tp = now_tp;
    }

    return vp_des_node::handle_frame_meta(meta);
}

std::shared_ptr<vp_objects::vp_meta>
vp_rtsp_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
    return vp_des_node::handle_control_meta(meta);
}

std::string vp_rtsp_des_node::to_string() {
    return rtsp_url;
}

} // namespace vp_nodes
