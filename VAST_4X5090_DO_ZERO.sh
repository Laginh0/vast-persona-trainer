#!/usr/bin/env bash
# Bootstrap completo para uma instancia limpa com quatro RTX 5090.
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

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" == '4' ]] || fail "Este script exige exatamente 4 GPUs NVIDIA; detectadas: $GPU_COUNT."

CPU_COUNT="$(nproc)"
OMP_DEFAULT=$(( CPU_COUNT / GPU_COUNT ))
(( OMP_DEFAULT < 4 )) && OMP_DEFAULT=4
(( OMP_DEFAULT > 8 )) && OMP_DEFAULT=8

# Com Qwen3-4B em QLoRA e cutoff 512, seis exemplos por RTX 5090 e um alvo agressivo.
# Caso ocorra CUDA out of memory, use PER_DEVICE_BATCH_SIZE=5 no comando de inicio.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export CUTOFF_LEN="${CUTOFF_LEN:-512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_DEFAULT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '\nInstancia pronta: %s GPUs, batch %s/GPU (global %s), cutoff %s, OMP %s.\n' \
  "$GPU_COUNT" "$PER_DEVICE_BATCH_SIZE" "$(( GPU_COUNT * PER_DEVICE_BATCH_SIZE ))" "$CUTOFF_LEN" "$OMP_NUM_THREADS"
exec bash "$TRAINER"
