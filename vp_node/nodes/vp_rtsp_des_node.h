#pragma once

#include <chrono>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "base/vp_des_node.h"
#include "objects/vp_control_meta.h"
#include "objects/vp_frame_meta.h"

namespace vp_nodes {

// rtsp des node, push H.264 stream to RTSP server.
// example:
// rtsp://127.0.0.1:8554/live/uav
class vp_rtsp_des_node : public vp_des_node {
private:
    std::string gst_template =
        "appsrc is-live=true do-timestamp=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! "
        "videoconvert ! video/x-raw,format=NV12 ! "
        "queue leaky=downstream max-size-buffers=4 max-size-time=0 max-size-bytes=0 ! "
        "%s ! "
        "h264parse config-interval=-1 ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
        "rtspclientsink location=%s protocols=tcp";

    std::string gst_pipeline;
    cv::VideoWriter rtsp_writer;

    std::string rtsp_url;
    std::string gst_encoder_name = "mpph264enc";
    std::string gst_encoder_pipeline;
    vp_objects::vp_size resolution_w_h;
    bool osd = true;
    bool append_channel_index = true;
    int output_fps = 25;
    std::chrono::steady_clock::time_point last_pushed_frame_tp;

    uint64_t pushed_frames = 0;
    std::chrono::steady_clock::time_point push_start_tp;
    std::chrono::steady_clock::time_point push_last_log_tp;
    std::chrono::steady_clock::time_point open_last_retry_tp;
    int open_retry_interval_ms = 1000;

protected:
    virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

public:
    vp_rtsp_des_node(std::string node_name,
                     int channel_index,
                     std::string rtsp_url,
                     vp_objects::vp_size resolution_w_h = {},
                     bool osd = true,
                     std::string gst_encoder_name = "mpph264enc",
                     bool append_channel_index = true);
    ~vp_rtsp_des_node();

    virtual std::string to_string() override;
};

} // namespace vp_nodes
