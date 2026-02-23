#!/bin/bash
# detect-hardware.sh — Detect system resources and recommend embedding model profile.
#
# Usage:
#   ./scripts/detect-hardware.sh              # show recommendation
#   ./scripts/detect-hardware.sh --json       # machine-readable output
#   ./scripts/detect-hardware.sh --apply      # write recommended .env settings
#
# Profiles:
#   minimal  — ≤2GB RAM free, 1-2 cores   → EmbeddingGemma 300M (314MB)
#   light    — 2-4GB RAM, 2-4 cores        → Qwen3-Embedding-0.6B (610MB)
#   standard — 4-8GB RAM, 4-8 cores        → Qwen3-Embedding-4B (4.0GB)   ← default
#   heavy    — 8-16GB RAM, 8+ cores        → Qwen3-Embedding-8B (8.1GB)
#   gpu      — NVIDIA GPU with ≥6GB VRAM   → Qwen3-Embedding-8B + GPU offload

set -euo pipefail

# ─── Gather hardware info ───────────────────────────────────────────────
TOTAL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)
AVAIL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}' || echo 0)
CPU_CORES=$(nproc 2>/dev/null || echo 1)
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")

# GPU detection
HAS_GPU=false
GPU_VRAM_MB=0
GPU_NAME="none"
if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -n "$GPU_INFO" ]; then
    HAS_GPU=true
    GPU_NAME=$(echo "$GPU_INFO" | head -1 | cut -d, -f1 | xargs)
    GPU_VRAM_MB=$(echo "$GPU_INFO" | head -1 | cut -d, -f2 | xargs)
  fi
fi

# Disk space (for model downloads)
DISK_AVAIL_GB=$(df -BG /opt 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo 0)

# ─── Determine profile ──────────────────────────────────────────────────
PROFILE="standard"
MODEL_NAME="Qwen3-Embedding-4B-Q8_0.gguf"
MODEL_ID="qwen/qwen3-embedding-4b"
DIMENSIONS=2560
MODEL_SIZE_MB=4096
THREADS=8
PARALLEL=2

if [ "$HAS_GPU" = true ] && [ "$GPU_VRAM_MB" -ge 6000 ]; then
  PROFILE="gpu"
  MODEL_NAME="Qwen3-Embedding-8B-Q8_0.gguf"
  MODEL_ID="qwen/qwen3-embedding-8b"
  DIMENSIONS=4096
  MODEL_SIZE_MB=8192
  THREADS=$CPU_CORES
  PARALLEL=4
elif [ "$TOTAL_RAM_MB" -ge 12000 ] && [ "$CPU_CORES" -ge 8 ]; then
  PROFILE="heavy"
  MODEL_NAME="Qwen3-Embedding-8B-Q8_0.gguf"
  MODEL_ID="qwen/qwen3-embedding-8b"
  DIMENSIONS=4096
  MODEL_SIZE_MB=8192
  THREADS=$((CPU_CORES > 16 ? 16 : CPU_CORES))
  PARALLEL=2
elif [ "$TOTAL_RAM_MB" -ge 6000 ] && [ "$CPU_CORES" -ge 4 ]; then
  PROFILE="standard"
  # defaults already set
  THREADS=$((CPU_CORES > 12 ? 12 : CPU_CORES))
elif [ "$TOTAL_RAM_MB" -ge 2500 ] && [ "$CPU_CORES" -ge 2 ]; then
  PROFILE="light"
  MODEL_NAME="Qwen3-Embedding-0.6B-Q8_0.gguf"
  MODEL_ID="qwen/qwen3-embedding-0.6b"
  DIMENSIONS=1024
  MODEL_SIZE_MB=610
  THREADS=$((CPU_CORES > 4 ? 4 : CPU_CORES))
  PARALLEL=1
else
  PROFILE="minimal"
  MODEL_NAME="hf_ggml-org_embeddinggemma-300M-Q8_0.gguf"
  MODEL_ID="google/embeddinggemma-300m"
  DIMENSIONS=768
  MODEL_SIZE_MB=314
  THREADS=$((CPU_CORES > 2 ? 2 : CPU_CORES))
  PARALLEL=1
fi

# Adjust parallel based on available RAM headroom
RAM_AFTER_MODEL=$((AVAIL_RAM_MB - MODEL_SIZE_MB))
if [ "$RAM_AFTER_MODEL" -lt 1000 ] && [ "$PARALLEL" -gt 1 ]; then
  PARALLEL=1
fi

# ─── Output ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "--json" ]; then
  cat <<JSON
{
  "profile": "$PROFILE",
  "model_name": "$MODEL_NAME",
  "model_id": "$MODEL_ID",
  "dimensions": $DIMENSIONS,
  "threads": $THREADS,
  "parallel": $PARALLEL,
  "hardware": {
    "total_ram_mb": $TOTAL_RAM_MB,
    "avail_ram_mb": $AVAIL_RAM_MB,
    "cpu_cores": $CPU_CORES,
    "cpu_model": "$CPU_MODEL",
    "has_gpu": $HAS_GPU,
    "gpu_name": "$GPU_NAME",
    "gpu_vram_mb": $GPU_VRAM_MB,
    "disk_avail_gb": $DISK_AVAIL_GB
  }
}
JSON
  exit 0
fi

if [ "${1:-}" = "--apply" ]; then
  ENV_FILE="${2:-.env}"
  if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Copy .env.example first." >&2
    exit 1
  fi
  # Update embedding-related vars
  sed -i "s|^AGENT_MEMORY_MODEL=.*|AGENT_MEMORY_MODEL=$MODEL_ID|" "$ENV_FILE"
  sed -i "s|^AGENT_MEMORY_DIMENSIONS=.*|AGENT_MEMORY_DIMENSIONS=$DIMENSIONS|" "$ENV_FILE"
  echo "✓ Updated $ENV_FILE with profile=$PROFILE (model=$MODEL_ID, dims=$DIMENSIONS)"
  echo ""
  echo "Also update your llama-server command:"
  echo "  llama-server --model $MODEL_NAME \\"
  echo "    --embedding --pooling last --host 0.0.0.0 --port 8090 \\"
  echo "    --ctx-size 8192 --batch-size 2048 --threads $THREADS --parallel $PARALLEL"
  exit 0
fi

# Default: pretty print
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            Agent Memory — Hardware Detection                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Hardware:"
echo "    CPU:      $CPU_MODEL ($CPU_CORES cores)"
echo "    RAM:      ${TOTAL_RAM_MB}MB total / ${AVAIL_RAM_MB}MB available"
echo "    GPU:      $GPU_NAME ($([[ $HAS_GPU == true ]] && echo "${GPU_VRAM_MB}MB VRAM" || echo "none"))"
echo "    Disk:     ${DISK_AVAIL_GB}GB free"
echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  Recommended Profile: $PROFILE"
echo "  ═══════════════════════════════════════════════════════════"
echo ""
echo "    Model:      $MODEL_NAME"
echo "    Model ID:   $MODEL_ID"
echo "    Dimensions: $DIMENSIONS"
echo "    Size:       ~${MODEL_SIZE_MB}MB"
echo "    Threads:    $THREADS"
echo "    Parallel:   $PARALLEL"
echo ""
echo "  llama-server command:"
echo "    llama-server --model $MODEL_NAME \\"
echo "      --embedding --pooling last --host 0.0.0.0 --port 8090 \\"
echo "      --ctx-size 8192 --batch-size 2048 --threads $THREADS --parallel $PARALLEL"
echo ""
echo "  Apply to .env:  ./scripts/detect-hardware.sh --apply"
echo "  JSON output:    ./scripts/detect-hardware.sh --json"
echo ""
echo "  ┌──────────────────────────────────────────────────────────┐"
echo "  │ All Profiles:                                            │"
echo "  │                                                          │"
echo "  │ minimal  — EmbeddingGemma 300M  (314MB, dim=768)        │"
echo "  │            ≤2GB RAM, 1-2 cores, Raspberry Pi / tiny VPS │"
echo "  │                                                          │"
echo "  │ light    — Qwen3-Embed-0.6B     (610MB, dim=1024)       │"
echo "  │            2-4GB RAM, 2-4 cores, small VPS              │"
echo "  │                                                          │"
echo "  │ standard — Qwen3-Embed-4B       (4.0GB, dim=2560)       │"
echo "  │            4-8GB RAM, 4-8 cores, mid-range server       │"
echo "  │                                                          │"
echo "  │ heavy    — Qwen3-Embed-8B       (8.1GB, dim=4096)       │"
echo "  │            12GB+ RAM, 8+ cores, dedicated server        │"
echo "  │                                                          │"
echo "  │ gpu      — Qwen3-Embed-8B + GPU (8.1GB, dim=4096)       │"
echo "  │            NVIDIA GPU ≥6GB VRAM, fastest inference       │"
echo "  └──────────────────────────────────────────────────────────┘"
