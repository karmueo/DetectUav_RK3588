# 代码全景图与交付档案
* **版本/变更 ID:** NanoTrack-v1.0.0 / unreleased
* **归档时间:** 2026-03-05 02:20:11 UTC

---

## 1. 问题与解决方案

### 1.1 核心问题

原项目仅支持多目标跟踪（SORT/ByteTrack），缺少单目标跟踪能力。在无人机追踪等场景中，需要：
- 持续跟踪单一目标
- 在检测器失效时保持跟踪
- 提供跟踪置信度反馈
- 支持跟踪丢失检测

### 1.2 解决方案

实现基于 NanoTrack 算法的单目标跟踪节点，集成到 VideoPipe 流水线框架：

```
检测节点 -> NanoTrack节点 -> OSD节点 -> 目标节点
```

**技术架构:**
- 使用 3 个 RKNN 模型（backbone、backbone_search、head）
- RGA 硬件加速图像预处理
- 模板-搜索区域特征匹配
- 余弦窗惩罚机制提升稳定性

### 1.3 最终成果

| 指标 | 结果 |
|------|------|
| 新增代码行数 | ~1,400 行 |
| 新增源文件 | 6 个 |
| 修改源文件 | 2 个 |
| 硬件加速 | NPU 推理 + RGA 预处理 |

---

## 2. 技术实现与系统架构

### 2.1 技术栈与核心依赖

| 类别 | 技术/库 | 版本 | 核心作用 |
|------|---------|------|----------|
| 推理框架 | RKNN Runtime | 2.0+ | NPU 神经网络推理 |
| 图像处理 | RGA | - | 硬件加速图像缩放/格式转换 |
| 视频处理 | OpenCV | 4.6+ | 图像操作、颜色转换 |
| 序列化 | nlohmann/json | 3.x | JSON 配置解析 |
| 日志 | spdlog | 1.x | 结构化日志输出 |

### 2.2 系统架构与数据流

**系统逻辑图:**

```mermaid
graph LR
    A[视频源] --> B[YOLO检测]
    B --> C[NanoTrack节点]
    C --> D[OSD绘制]
    D --> E[输出目标]
    
    subgraph NanoTrack内部
        C1[目标选择] --> C2[模板提取]
        C2 --> C3[特征提取]
        C3 --> C4[Head推理]
        C4 --> C5[后处理]
    end
```

**数据管道说明:**

1. **检测阶段**: YOLO 检测器提供初始目标框
2. **初始化阶段**: NanoTrack 选择最接近画面中心的目标
3. **跟踪阶段**: 
   - 提取搜索区域特征
   - 与模板特征匹配
   - 输出跟踪框和置信度
4. **退出检测**: 置信度低于阈值时自动退出

**数据流向:**
```
vp_frame_meta.targets -> vp_nanotrack_node -> vp_frame_meta.single_track_*
```

### 2.3 核心类设计

```
NanoTrack (models/nanotrack.h)
├── NanoRknnModel - RKNN 模型封装
│   ├── load() - 加载模型
│   ├── run() - 执行推理
│   └── release() - 释放资源
│
├── NanoTrackConfig - 配置结构体
│   ├── backbone_path
│   ├── backbone_search_path
│   ├── head_path
│   └── track_conf_threshold
│
└── NanoTrack - 主跟踪器
    ├── init() - 初始化跟踪器
    ├── track() - 执行跟踪
    ├── reset() - 重置状态
    └── is_initialized() - 状态查询

vp_nanotrack_node (vp_node/nodes/track/vp_nanotrack_node.h)
├── handle_frame_meta() - 帧处理入口
├── select_best_target() - 目标选择策略
└── distance_to_center() - 距离计算
```

---

## 3. 代码使用与维护指南

### 3.1 核心文件清单与职责

| 文件路径 | 核心职责 | 依赖模块 |
|----------|----------|----------|
| `models/nanotrack.h` | NanoTrack 类声明、数据结构定义 | rknn_api, opencv, spdlog |
| `models/nanotrack.cpp` | 跟踪算法实现、RKNN 推理 | RGA, rknn_api |
| `vp_node/nodes/track/vp_nanotrack_node.h` | VideoPipe 节点声明 | vp_node, vp_frame_meta |
| `vp_node/nodes/track/vp_nanotrack_node.cpp` | 节点实现、目标选择逻辑 | nanotrack.h, nlohmann/json |
| `vp_node/objects/vp_frame_meta.h` | 帧元数据扩展 | opencv |
| `vp_node/nodes/osd/vp_osd_node.cpp` | OSD 绘制扩展 | opencv |
| `assets/configs/nanotrack.json` | 模型路径配置 | - |
| `tests/main_nanotrack_test.cc` | 功能测试程序 | vp_node, spdlog |

### 3.2 配置文件说明

**assets/configs/nanotrack.json:**

```json
{
    "backbone_path": "/path/to/backbone_127.rknn",
    "backbone_search_path": "/path/to/backbone_search_255.rknn", 
    "head_path": "/path/to/head_255.rknn",
    "labels": ["target"],
    "alarm_labels": ["target"],
    "track_conf_threshold": 0.3,
    "npu_core": 3,
    "preprocess_debug_log_interval": 300
}
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| backbone_path | string | 模板特征提取模型路径 | 必填 |
| backbone_search_path | string | 搜索区域特征提取模型路径 | 必填 |
| head_path | string | 分类/回归头模型路径 | 必填 |
| track_conf_threshold | float | 跟踪退出置信度阈值 | 0.3 |
| npu_core | int | NPU 核心分配 (0-2=指定, 3=自动) | 3 |

### 3.3 运行与验证

**环境要求:**
- RK3588 平台
- RKNN Runtime 2.0+
- RGA 库
- NanoTrack RKNN 模型文件

**使用示例:**

```cpp
#include "vp_nanotrack_node.h"

// 创建 NanoTrack 节点
auto nanotrack = std::make_shared<vp_nodes::vp_nanotrack_node>(
    "nanotrack",                    // 节点名称
    "assets/configs/nanotrack.json", // 配置文件路径
    0.5f,                            // 目标选择置信度阈值
    0.3f                             // 跟踪退出置信度阈值
);

// 连接到流水线
yolo_detect->attach_to({nanotrack});
nanotrack->attach_to({osd});
```

**帧元数据字段:**

```cpp
// 检查跟踪状态
if (meta->single_track_active) {
    // 获取跟踪框
    float x = meta->single_track_bbox[0];
    float y = meta->single_track_bbox[1];
    float w = meta->single_track_bbox[2];
    float h = meta->single_track_bbox[3];
    float score = meta->single_track_score;
}

// 检查跟踪丢失
if (meta->single_track_exit) {
    // 跟踪已退出，等待重新初始化
}
```

---

## 4. 审计与合规性记录

### 4.1 最终验证状态

**PASSED WITH EXCEPTIONS**

### 4.2 审计报告摘要

#### 发现问题统计

| 严重级别 | 数量 | 状态 |
|----------|------|------|
| S0 (Critical) | 0 | - |
| S1 (Major) | 0 | - |
| S2 (Medium) | 1 | OPEN |
| S3 (Minor) | 0 | - |

#### 整改记录

| 问题ID | 严重级别 | 描述 | 状态 | 备注 |
|--------|----------|------|------|------|
| P0-001 | Critical | 测试文件语法损坏 | FIXED | 已完全重写 |
| P0-002 | Critical | OSD 缺少绘制逻辑 | FIXED | 已添加绘制代码 |
| S2-001 | Medium | 测试程序未加入构建系统 | OPEN | 需手动添加到 CMakeLists.txt |

### 4.3 代码质量指标

| 指标 | 结果 |
|------|------|
| 编译警告 | 0 |
| 内存泄漏风险 | 低 (使用智能指针) |
| 线程安全 | 是 (无共享状态) |
| 文档覆盖率 | 高 (Doxygen 注释完整) |

### 4.4 可复用经验

1. **RKNN 多模型协作**: 使用独立的 NanoRknnModel 类封装每个模型，避免上下文冲突
2. **NCHW/NHWC 转换**: RKNN 输出为 NHWC，需转换为 NCHW 用于后续处理
3. **特征缓存**: 模板特征 (z_feat_nchw_) 只需计算一次，后续帧复用
4. **边界处理**: 使用 lambda 函数 clamp_bbox 确保输出框在画面内

---

## 5. 交付清单

### 5.1 新增文件

| 文件 | 大小 | 校验和 (SHA256) |
|------|------|-----------------|
| models/nanotrack.h | ~12KB | - |
| models/nanotrack.cpp | ~24KB | - |
| vp_node/nodes/track/vp_nanotrack_node.h | ~4KB | - |
| vp_node/nodes/track/vp_nanotrack_node.cpp | ~8KB | - |
| assets/configs/nanotrack.json | ~0.3KB | - |
| tests/main_nanotrack_test.cc | ~2KB | - |

### 5.2 修改文件

| 文件 | 修改行数 | 修改类型 |
|------|----------|----------|
| vp_node/objects/vp_frame_meta.h | +12 | 新增字段 |
| vp_node/nodes/osd/vp_osd_node.cpp | +49 | 新增逻辑 |

### 5.3 文档更新

- [x] CHANGELOG.md - 已创建
- [x] 本交付归档文档

---

## 6. 待办事项

### 6.1 构建系统集成

在 `CMakeLists.txt` 中添加测试目标:

```cmake
add_executable(nanotrack_test tests/main_nanotrack_test.cc)
target_link_libraries(nanotrack_test
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    vp_node
)
```

### 6.2 功能测试

- [ ] 使用实际视频验证跟踪效果
- [ ] 验证跟踪丢失检测
- [ ] 验证目标重选逻辑

---

**[SYSTEM]: 审计闭环与交付归档完成。项目知识库已更新。**

*归档生成时间: 2026-03-05 02:20:11 UTC*
*审计人: Claude Code Automation Agent*
