#!/usr/bin/env bash
# Bootstrap completo para uma instancia limpa com quatro RTX 4090 de 24 GiB.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="$SCRIPT_DIR/VAST_TREINAR_TUDO.sh"

fail() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }
need_command() { command -v "$1" >/dev/null 2>&1 || fail "Comando ausente: $1"; }

need_command nvidia-smi
need_command nproc
[[ -f "$TRAINER" ]] || fail "Nao encontrei $TRAINER"

if pgrep -af 'torch\.distributed\.run|llamafactory\.launcher|llamafactory-cli' >/dev/null 2>&1; then
  fail 'Ja existe um treino LlamaFactory ativo. Use uma instancia limpa ou encerre o processo anterior antes de iniciar.'
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
GPU_COUNT="$(printf '%s\n' "$GPU_NAMES" | sed '/^$/d' | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" == '4' ]] || fail "Este script exige exatamente 4 GPUs NVIDIA; detectadas: $GPU_COUNT."
if printf '%s\n' "$GPU_NAMES" | grep -Evqi 'RTX 4090'; then
  fail "Este script foi ajustado para RTX 4090. GPUs detectadas: $(printf '%s' "$GPU_NAMES" | tr '\n' ';')"
fi

CPU_COUNT="$(nproc)"
OMP_DEFAULT=$(( CPU_COUNT / GPU_COUNT ))
(( OMP_DEFAULT < 4 )) && OMP_DEFAULT=4
(( OMP_DEFAULT > 8 )) && OMP_DEFAULT=8

# Batch 6 explora a folga de VRAM observada; se houver OOM, reduza para 5 ou 4.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export CUTOFF_LEN="${CUTOFF_LEN:-512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_DEFAULT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '\nInstancia pronta: 4 RTX 4090, batch %s/GPU (global %s), cutoff %s, OMP %s.\n' \
  "$PER_DEVICE_BATCH_SIZE" "$(( GPU_COUNT * PER_DEVICE_BATCH_SIZE ))" "$CUTOFF_LEN" "$OMP_NUM_THREADS"
exec bash "$TRAINER"
