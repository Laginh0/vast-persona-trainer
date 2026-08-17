#!/usr/bin/env bash
# Treinamento privado de persona no Vast.ai (Ubuntu/Debian com GPU NVIDIA).
# Nao armazena URL ou conversas no diretorio persistente da instancia.
# Importante: durante o treino, o conteudo precisa existir em RAM/VRAM. Um dono
# do host ou hipervisor malicioso pode inspecionar uma VM enquanto ela executa;
# este script reduz rastros em disco, mas nao elimina esse risco.
set -Eeuo pipefail
umask 077

# Ajustes de treino. Pode sobrescrever antes de executar, por exemplo:
# MODEL_NAME=Qwen/Qwen3-4B EPOCHS=1 ./VAST_TREINAR_TUDO.sh
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
MIN_RAM_MIB="${MIN_RAM_MIB:-6144}"

readonly SCRIPT_NAME="$(basename "$0")"
readonly VENV_DIR="$HOME/.venvs/lage-persona"
readonly RESULTS_DIR="$HOME/lage-persona-resultados"

BASE_PYTHON=""
RAM_MOUNT=""
WORK_DIR=""
INPUT_ZIP=""

fail() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }
note() { printf '\n==> %s\n' "$*"; }

cleanup() {
  unset DROPBOX_URL DOWNLOAD_URL INPUT_ZIP
  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR"
    WORK_DIR=""
  fi
  if [[ -n "$RAM_MOUNT" && -d "$RAM_MOUNT" ]]; then
    mountpoint -q "$RAM_MOUNT" && umount "$RAM_MOUNT" || true
    rmdir "$RAM_MOUNT" 2>/dev/null || true
    RAM_MOUNT=""
  fi
}
trap cleanup EXIT INT TERM

need_command() { command -v "$1" >/dev/null 2>&1 || fail "Comando ausente: $1"; }

prepare_ram() {
  need_command df
  need_command stat
  if [[ -d /dev/shm && "$(stat -f -c %T /dev/shm)" == "tmpfs" ]]; then
    local available
    available="$(df -Pm /dev/shm | awk 'NR==2 {print $4}')"
    if (( available >= MIN_RAM_MIB )); then
      WORK_DIR="$(mktemp -d /dev/shm/lage-persona.XXXXXXXX)"
      chmod 700 "$WORK_DIR"
      return
    fi
  fi

  if [[ "$(id -u)" -ne 0 ]]; then
    fail "/dev/shm nao possui ${MIN_RAM_MIB} MiB livres. Inicie a instancia com --shm-size=6g ou execute como root para montar um tmpfs temporario."
  fi

  RAM_MOUNT="$(mktemp -d "$HOME/.cache/lage-persona-tmpfs.XXXXXXXX")"
  mount -t tmpfs -o "size=${MIN_RAM_MIB}M,mode=0700" tmpfs "$RAM_MOUNT"
  WORK_DIR="$RAM_MOUNT"
}

configure_private_runtime() {
  [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]] || fail 'Diretorio temporario privado nao foi criado.'
  mkdir -p "$WORK_DIR/tmp" "$WORK_DIR/hf-datasets" "$WORK_DIR/torchinductor"
  chmod 700 "$WORK_DIR/tmp" "$WORK_DIR/hf-datasets" "$WORK_DIR/torchinductor"

  # Impede que temporarios do Python, cache do dataset e artefatos de compilacao
  # acabem em /tmp ou no disco persistente da instancia.
  export TMPDIR="$WORK_DIR/tmp" TEMP="$WORK_DIR/tmp" TMP="$WORK_DIR/tmp"
  export HF_DATASETS_CACHE="$WORK_DIR/hf-datasets"
  export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torchinductor"
  export PYTHONDONTWRITEBYTECODE=1

  # O script nunca grava comandos ou core dumps que possam conter referencias aos dados.
  export HISTFILE=/dev/null HISTSIZE=0
  set +o history 2>/dev/null || true
  ulimit -c 0 2>/dev/null || true
}

install_system_packages() {
  if command -v python >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 && command -v 7z >/dev/null 2>&1; then
    return
  fi
  if [[ "$(id -u)" -eq 0 ]]; then APT=(apt-get); else APT=(sudo apt-get); fi
  note 'Instalando dependencias do sistema'
  "${APT[@]}" update
  "${APT[@]}" install -y python3 python3-venv python3-pip curl ca-certificates p7zip-full util-linux
}

select_base_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    BASE_PYTHON="$PYTHON_BIN"
  elif command -v python >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
  else
    fail 'Python nao foi encontrado mesmo apos instalar as dependencias.'
  fi
  "$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10); print(sys.executable)'
}

install_python_environment() {
  note 'Preparando ambiente Python e LlamaFactory'
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  if ! "$VENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    note 'A imagem nao trouxe PyTorch CUDA funcional; instalando wheel CUDA 12.8'
    "$VENV_DIR/bin/python" -m pip install \
      torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  if ! "$VENV_DIR/bin/python" -c 'import llamafactory, bitsandbytes' >/dev/null 2>&1; then
    "$VENV_DIR/bin/python" -m pip install llamafactory==0.9.5 bitsandbytes
  fi
}

download_and_extract() {
  read -r -p 'Cole o link HTTPS do Dropbox para o arquivo .zip: ' DROPBOX_URL
  [[ "$DROPBOX_URL" == https://* ]] || fail 'O link precisa começar com https://.'

  if [[ "$DROPBOX_URL" == *dropbox.com* && "$DROPBOX_URL" != *'dl=1'* ]]; then
    if [[ "$DROPBOX_URL" == *\?* ]]; then DOWNLOAD_URL="${DROPBOX_URL}&dl=1"; else DOWNLOAD_URL="${DROPBOX_URL}?dl=1"; fi
  else
    DOWNLOAD_URL="$DROPBOX_URL"
  fi

  INPUT_ZIP="$WORK_DIR/conversas.$$.zip"
  note 'Baixando o ZIP diretamente para a RAM'
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 --output "$INPUT_ZIP" "$DOWNLOAD_URL"
  [[ -s "$INPUT_ZIP" ]] || fail 'O download retornou um arquivo vazio.'

  note 'Extraindo o ZIP apenas na RAM'
  7z x -bd "$INPUT_ZIP" "-o$WORK_DIR/raw" || fail 'Nao foi possivel abrir o ZIP. Confira o link.'
  find "$WORK_DIR/raw" -type f -iname '*.txt' -print -quit | grep -q . || fail 'Nenhum arquivo TXT foi encontrado dentro do ZIP.'
  # Depois de extrair na RAM, nao ha motivo para conservar a copia baixada.
  rm -f -- "$INPUT_ZIP"
  INPUT_ZIP=""
  unset DROPBOX_URL DOWNLOAD_URL
}

prepare_dataset() {
  note 'Preparando todos os turnos de conversa apenas na RAM'
  "$VENV_DIR/bin/python" - "$WORK_DIR/raw" "$WORK_DIR/preparado" <<'PY'
import hashlib, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

raw_dir, output_dir = map(lambda item: Path(item).resolve(), sys.argv[1:3])
output_dir.mkdir(parents=True, exist_ok=True)
patterns = (
    re.compile(r"^\[\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]+):\s?(.*)$"),
    re.compile(r"^(?:\[)?\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\])?\s+-\s+([^:]+):\s?(.*)$"),
)
email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
phone = re.compile(r"(?<!\w)(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-\s]?\d{4}(?!\w)")
cpf = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
card = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
media = re.compile(r"^(?:<\s*media omitted\s*>|<\s*m[ií]dia omitida\s*>|imagem omitida|video omitido|audio omitido)$", re.I)

def norm(value):
    value = unicodedata.normalize("NFKD", value)
    return re.sub(r"\s+", " ", "".join(char for char in value if not unicodedata.combining(char))).strip().casefold()

def redact(text):
    return phone.sub("[TELEFONE]", card.sub("[NUMERO_REMOVIDO]", cpf.sub("[CPF]", email.sub("[EMAIL]", text))))

def messages(path):
    sender, body = None, []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            match = next((pattern.match(line) for pattern in patterns if pattern.match(line)), None)
            if match:
                if sender is not None: yield sender, "\n".join(body).strip()
                sender, body = match.group(1).strip(), [match.group(2)]
            elif sender is not None:
                body.append(line)
        if sender is not None: yield sender, "\n".join(body).strip()

def limit(items, maximum_messages, maximum_chars, latest):
    source = items[-maximum_messages:] if latest else items[:maximum_messages]
    limited, result, count = len(source) != len(items), [], 0
    iterator = reversed(source) if latest else iter(source)
    for direction, text in iterator:
        if result and count + len(text) > maximum_chars:
            limited = True; break
        result.append((direction, text)); count += len(text)
    if latest: result.reverse()
    return result, limited

owner, stats, seen = "eu", Counter(), set()
with (output_dir / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as corpus, (output_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n") as dataset:
    for chat in sorted(raw_dir.rglob("*.txt")):
        stats["files"] += 1; history = []; run = []; context = []
        def commit():
            global run, context
            if not run: return
            stats["outgoing_turns"] += 1
            if any(not outgoing for outgoing, _ in context):
                selected_context, context_cut = limit(context, 10, 1200, True)
                selected_output, output_cut = limit(run, 10, 900, False)
                prompt = "\n".join(("Eu: " if outgoing else "Outra pessoa: ") + text for outgoing, text in selected_context)
                answer = "\n".join(text for _, text in selected_output)
                fingerprint = hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()
                if prompt and answer and fingerprint not in seen:
                    seen.add(fingerprint)
                    dataset.write(json.dumps({"instruction":"Responda em portugues de forma natural, seguindo o estilo de escrita demonstrado pelo usuario. Quando adequado, uma resposta pode conter varias mensagens curtas.", "input":prompt, "output":answer}, ensure_ascii=False) + "\n")
                    stats["pairs"] += 1; stats["response_messages_in_pairs"] += len(selected_output)
                    stats["context_limited_pairs"] += int(context_cut); stats["response_limited_pairs"] += int(output_cut)
            else: stats["outgoing_turns_without_received_context"] += 1
            history.extend(run); del history[:-10]; run, context = [], []
        for sender, body in messages(chat):
            text = redact(body).strip()
            if not text or media.match(text): stats["non_text_or_media"] += 1; continue
            outgoing = norm(sender) == owner; stats["messages_total"] += 1
            if outgoing:
                stats["own_messages"] += 1; corpus.write(json.dumps({"text":text}, ensure_ascii=False) + "\n")
                if not run: context = list(history)
                run.append((True, text))
            else:
                stats["received_messages"] += 1; commit(); history.append((False, text)); del history[:-10]
        commit()
info = {"lage_persona":{"file_name":"train.jsonl","formatting":"alpaca","columns":{"prompt":"instruction","query":"input","response":"output"}}}
(output_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY
}

train_model() {
  local gpu_count batch_size accumulation launcher_path
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  (( gpu_count >= 1 )) || fail 'Nenhuma GPU NVIDIA foi detectada.'
  "$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= $gpu_count; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
  launcher_path="$("$VENV_DIR/bin/python" -c 'import os, llamafactory.launcher; print(os.path.abspath(llamafactory.launcher.__file__))')"
  [[ -f "$launcher_path" ]] || fail 'O modulo LlamaFactory nao esta disponivel no ambiente de treino.'

  if (( gpu_count >= 2 )); then batch_size=4; accumulation=1; else batch_size=2; accumulation=4; fi
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
  export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  export HF_HOME="$HOME/.cache/huggingface-public-models"
  export WANDB_DISABLED=true WANDB_MODE=disabled DO_NOT_TRACK=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY

  note "Treinando $MODEL_NAME com $gpu_count GPU(s)"
  # Usa o Python do venv em todos os ranks; o torchrun da imagem base pode nao
  # enxergar os pacotes instalados em $VENV_DIR.
  "$VENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$gpu_count" "$launcher_path" \
    --model_name_or_path "$MODEL_NAME" --trust_remote_code true \
    --stage sft --do_train true --finetuning_type lora --lora_target all --lora_rank 16 --lora_alpha 32 --quantization_bit 4 \
    --dataset_dir "$WORK_DIR/preparado" --dataset lage_persona --template qwen3 --cutoff_len "$CUTOFF_LEN" \
    --per_device_train_batch_size "$batch_size" --gradient_accumulation_steps "$accumulation" \
    --learning_rate 0.00015 --num_train_epochs "$EPOCHS" --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --bf16 true --logging_steps 25 --save_strategy epoch --save_total_limit 1 --report_to none \
    --output_dir "$WORK_DIR/adapter"
}

encrypt_result() {
  local result
  [[ -d "$WORK_DIR/adapter" ]] || fail 'O adaptador final nao foi encontrado.'
  mkdir -p "$RESULTS_DIR"; chmod 700 "$RESULTS_DIR"
  result="$RESULTS_DIR/lage-lora-$(date -u +%Y%m%dT%H%M%SZ).7z"
  note 'Digite uma senha para criptografar o adaptador LoRA final'
  7z a -t7z "$result" "$WORK_DIR/adapter/*" '-mhe=on' '-p' '-mx=5'
  chmod 600 "$result"
  note 'Removendo conversas, dataset, cache e adaptador em claro da RAM'
  cleanup
  printf '\nConcluido. A instancia reteve apenas o arquivo final criptografado: %s\n' "$result"
}

main() {
  [[ "${BASH_VERSINFO[0]}" -ge 4 ]] || fail 'Use Bash 4 ou superior.'
  note "Iniciando $SCRIPT_NAME"
  install_system_packages
  select_base_python
  need_command nvidia-smi
  need_command 7z
  if swapon --noheadings --show 2>/dev/null | grep -q .; then
    note 'AVISO: swap esta ativo. O treino continuara, mas paginas com conversas em claro podem ser gravadas no disco da instancia.'
  fi
  prepare_ram
  configure_private_runtime
  install_python_environment
  download_and_extract
  prepare_dataset
  train_model
  encrypt_result
}

main "$@"
