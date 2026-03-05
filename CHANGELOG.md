# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added - 2026-03-05

#### NanoTrack Single Object Tracking Node

**New Files:**
- `models/nanotrack.h` - NanoTrack tracker class header with RKNN integration
- `models/nanotrack.cpp` - NanoTrack implementation using RKNN inference and RGA preprocessing
- `vp_node/nodes/track/vp_nanotrack_node.h` - VideoPipe tracking node header
- `vp_node/nodes/track/vp_nanotrack_node.cpp` - VideoPipe tracking node implementation
- `assets/configs/nanotrack.json` - Model configuration file
- `tests/main_nanotrack_test.cc` - Test program for NanoTrack functionality

**Modified Files:**
- `vp_node/objects/vp_frame_meta.h` - Added single object tracking fields:
  - `single_track_active` - Tracking state flag
  - `single_track_score` - Confidence score
  - `single_track_bbox` - Bounding box [x, y, w, h]
  - `single_track_init_frame` - Initialization frame index
  - `single_track_exit` - Tracking exit flag
- `vp_node/nodes/osd/vp_osd_node.cpp` - Added single object tracking visualization:
  - Green bounding box with confidence score
  - "Tracking Lost" warning on exit

**Features:**
- Automatic target selection from detection results (closest to frame center)
- Configurable selection and exit confidence thresholds
- RKNN-accelerated inference on NPU
- RGA hardware-accelerated image preprocessing
- Automatic tracking exit on low confidence

**Technical Details:**
- Uses three RKNN models: backbone (template), backbone_search, and head
- Template-feature based tracking with cosine window penalization
- NCHW/NHWC tensor format conversion for RKNN compatibility

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| Unreleased | 2026-03-05 | NanoTrack single object tracking implementation |
