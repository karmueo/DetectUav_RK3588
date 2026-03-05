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

### Changed - 2026-03-05 (v7ee0ded)

#### NanoTrack Node State Machine Refactoring

**Commit:** 7ee0ded31417f6efbc8c8887fd62b2d6009f6eae

**Summary:**
Refactored the NanoTrack tracking node to use a state machine architecture with configurable tracking strategy parameters and multiple exit conditions.

**Key Changes:**

1. **State Machine Implementation**
   - Added `TrackState` enum with SEARCHING/TRACKING states
   - Clean state transitions with explicit exit handling
   - `reset_to_searching()` method for state reset

2. **Configuration Structure**
   - Added `TrackConfig` struct encapsulating all tracking parameters
   - JSON configuration support for runtime parameter tuning
   - Backward compatible with existing constructor API

3. **Exit Conditions (12 Acceptance Criteria)**
   - Confidence-based exit: `track_score < exit_conf_threshold`
   - IoU-based exit: `IoU(track_bbox, detect_bbox) < iou_threshold`
   - No-detection exit: consecutive frames without detection >= threshold
   - All thresholds can be disabled by setting <= 0

4. **Target Selection Enhancement**
   - Class ID filtering via `target_class_id` (-1 = no filter)
   - Confidence threshold filtering
   - Center-closest target selection strategy

**Files Changed:**
| File | Lines Changed |
|:---|:---|
| `vp_node/nodes/track/vp_nanotrack_node.h` | +86/-0 |
| `vp_node/nodes/track/vp_nanotrack_node.cpp` | +247/-71 |
| `assets/configs/nanotrack.json` | +8/-0 |
| `tests/main_nanotrack_test.cc` | +2/-0 |

**Documentation:**
- `docs/audit_report_7ee0ded.md` - Audit report with verification results
- `docs/code_panorama_7ee0ded.md` - Complete technical documentation

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 7ee0ded | 2026-03-05 | NanoTrack state machine refactoring with configurable exit conditions |
| Unreleased | 2026-03-05 | NanoTrack single object tracking implementation |
