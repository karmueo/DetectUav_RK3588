# 变更日志

## [未发布]
### 新增
- 在 `.process/` 下新增 RTSP 推流五阶段交付文档：
  - `stage1_locked_spec.md`
  - `stage2_execution_plan.md`
  - `stage3_implementation_decision_log.md`
  - `stage4_validation_release_evidence.md`
  - `stage5_code_panorama_delivery_archive.md`
- 新增 RTSP 输出能力实现：
  - `vp_node/nodes/vp_rtsp_des_node.h`
  - `vp_node/nodes/vp_rtsp_des_node.cpp`
  - `rtsp_main.cc`
- 在 `CMakeLists.txt` 中新增可执行目标 `detectuav_rk3588_rtsp`。

### 变更
- 为提升客户端兼容性和稳定性，更新 RTSP 推流管线：
  - 显式设置 `appsrc` 的 caps（BGR/宽/高/帧率）
  - 增加 `videoconvert` 转 `NV12`
  - 调整 `mpph264enc` 参数：`rc-mode=cbr`、`gop=50`、`header-mode=each-idr`、`profile=baseline`
  - 设置 `h264parse config-interval=-1`
  - 在 `vp_rtsp_des_node` 中增加开流重试节流与写入限频
- 将 `rtsp_main.cc` 调整为仅 RTSP 输出（移除 `nv12` 显示分支）。

### 说明
- Stage4 裁定已更新为 `GO`（构建、推流、解码与稳定性验证通过）。
