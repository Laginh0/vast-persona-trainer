#!/usr/bin/env bash
# Reinicia o treino com parametros agressivos para duas RTX 5090 de 32 GiB.
# Nao encerra processos existentes: pare o treino anterior com Ctrl+C primeiro.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="$SCRIPT_DIR/VAST_TREINAR_TUDO.sh"

if pgrep -af 'torch\.distributed\.run|llamafactory\.launcher|llamafactory-cli' >/dev/null 2>&1; then
  printf 'ERRO: ja existe um treino LlamaFactory ativo. Pare-o normalmente com Ctrl+C antes de reiniciar.\n' >&2
  exit 1
fi

[[ -f "$TRAINER" ]] || { printf 'ERRO: nao encontrei %s\n' "$TRAINER" >&2; exit 1; }

# Batch 6 e um alvo agressivo, mas ainda deixa margem na GPU que apresentou maior uso.
# Se houver CUDA out of memory, rode de novo com: PER_DEVICE_BATCH_SIZE=5 bash VAST_REINICIAR_MAX_GPU.sh
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export CUTOFF_LEN="${CUTOFF_LEN:-512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

printf 'Reinicio de alto desempenho: 2 GPUs, batch %s/GPU, cutoff %s, OMP %s.\n' \
  "$PER_DEVICE_BATCH_SIZE" "$CUTOFF_LEN" "$OMP_NUM_THREADS"
exec bash "$TRAINER"
