#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_PATH="${ROOT_DIR}/build/bin/detectuav_rk3588"
if [[ ! -x "${BIN_PATH}" ]]; then
  BIN_PATH="${ROOT_DIR}/build/detectuav_rk3588"
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[ERROR] detectuav_rk3588 not found. Build first:"
  echo "  ./build-linux.sh"
  echo "or:"
  echo "  cmake -S . -B build && cmake --build build -j4 && cmake --install build"
  exit 1
fi

VIDEO_PATH="${1:-/mnt/nfs/datasets/video/uav6.mp4}"
SDL_VIDEO_DRIVER="${2:-}"
SDL_RENDER_DRIVER="${3:-}"
YOLO26_CFG="${4:-${ROOT_DIR}/assets/configs/yolo26.json}"
SCREEN_SINK="${5:-autovideosink}"

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "[ERROR] video not found: ${VIDEO_PATH}"
  exit 1
fi

if [[ ! -f "${YOLO26_CFG}" ]]; then
  echo "[ERROR] yolo26 config not found: ${YOLO26_CFG}"
  exit 1
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/build/bin/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] bin: ${BIN_PATH}"
echo "[INFO] video: ${VIDEO_PATH}"
echo "[INFO] sdl_video_driver: ${SDL_VIDEO_DRIVER:-auto}"
echo "[INFO] sdl_render_driver: ${SDL_RENDER_DRIVER:-auto}"
echo "[INFO] yolo26 config: ${YOLO26_CFG}"
echo "[INFO] sink: ${SCREEN_SINK}"

exec "${BIN_PATH}" "${VIDEO_PATH}" "${SDL_VIDEO_DRIVER}" "${SDL_RENDER_DRIVER}" "${YOLO26_CFG}" "${SCREEN_SINK}"
