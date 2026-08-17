#!/usr/bin/env bash
# Exporta o adaptador LoRA final sem descriptografa-lo.
# O arquivo de entrada e sempre o .7z cifrado produzido por VAST_TREINAR_TUDO.sh.
set -Eeuo pipefail
umask 077

readonly RESULTS_DIR="${RESULTS_DIR:-$HOME/lage-persona-resultados}"
ARCHIVE=""
TMP_DIR=""
TOKEN_FILE=""

fail() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }
note() { printf '\n==> %s\n' "$*"; }
need_command() { command -v "$1" >/dev/null 2>&1 || fail "Comando ausente: $1"; }

cleanup() {
  unset DROPBOX_TOKEN
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf -- "$TMP_DIR"
  fi
}
trap cleanup EXIT INT TERM

find_archive() {
  local record
  [[ -d "$RESULTS_DIR" ]] || fail "Diretorio de resultados nao encontrado: $RESULTS_DIR"
  record="$(find "$RESULTS_DIR" -maxdepth 1 -type f -name 'lage-lora-*.7z' -printf '%T@ %p\n' | sort -rn | head -n 1 || true)"
  [[ -n "$record" ]] || fail "Nenhum arquivo lage-lora-*.7z foi encontrado em $RESULTS_DIR"
  ARCHIVE="${record#* }"
  [[ -f "$ARCHIVE" ]] || fail 'O arquivo final selecionado deixou de existir.'
}

show_archive() {
  local bytes hash
  need_command stat
  need_command sha256sum
  bytes="$(stat -c '%s' -- "$ARCHIVE")"
  hash="$(sha256sum -- "$ARCHIVE" | awk '{print $1}')"
  printf '\nArquivo cifrado: %s\nTamanho: %s bytes\nSHA-256: %s\n' "$ARCHIVE" "$bytes" "$hash"
}

prepare_ram() {
  need_command mktemp
  need_command stat
  [[ -d /dev/shm && "$(stat -f -c %T /dev/shm)" == "tmpfs" ]] || fail 'Este exportador exige /dev/shm (tmpfs) para manter o token fora do disco.'
  TMP_DIR="$(mktemp -d /dev/shm/lage-export.XXXXXXXX)"
  chmod 700 "$TMP_DIR"
  TOKEN_FILE="$TMP_DIR/curl.conf"
}

upload_dropbox() {
  local folder remote_path size api_arg response
  need_command curl
  need_command stat
  prepare_ram

  read -r -s -p 'Cole um token Dropbox com permissao files.content.write (nao sera salvo): ' DROPBOX_TOKEN
  printf '\n'
  [[ -n "$DROPBOX_TOKEN" ]] || fail 'Nenhum token foi informado.'
  printf 'header = "Authorization: Bearer %s"\n' "$DROPBOX_TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  unset DROPBOX_TOKEN

  read -r -p 'Pasta de destino no Dropbox (ex.: /Apps/LagePersona): ' folder
  [[ "$folder" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail 'Use uma pasta iniciada por / e contendo apenas letras, numeros, ponto, _, - e /.'
  folder="${folder%/}"
  remote_path="$folder/$(basename "$ARCHIVE")"
  size="$(stat -c '%s' -- "$ARCHIVE")"
  (( size <= 157286400 )) || fail 'O arquivo excede 150 MiB. Baixe-o por SCP para evitar enviar uma credencial a esta instancia.'
  api_arg="{\"path\":\"$remote_path\",\"mode\":\"add\",\"autorename\":true,\"mute\":true,\"strict_conflict\":false}"

  note 'Enviando o arquivo cifrado ao Dropbox'
  response="$(curl --silent --show-error --fail-with-body --config "$TOKEN_FILE" \
    --request POST --url 'https://content.dropboxapi.com/2/files/upload' \
    --header 'Content-Type: application/octet-stream' \
    --header "Dropbox-API-Arg: $api_arg" \
    --data-binary "@$ARCHIVE")"
  [[ -n "$response" ]] || fail 'O Dropbox nao confirmou o envio.'
  printf '\nEnvio concluido para: %s\n' "$remote_path"
}

show_scp() {
  printf '\nNo seu PC, execute este comando e substitua os campos entre < > pelos dados de SSH exibidos pela Vast.ai:\n\n'
  printf 'scp -P <PORTA_SSH> <USUARIO>@<IP_DA_VAST>:%q .\n' "$ARCHIVE"
  printf '\nO arquivo ja esta cifrado; confira o SHA-256 exibido acima apos o download.\n'
}

main() {
  find_archive
  show_archive
  case "${1:-}" in
    --dropbox) upload_dropbox ;;
    --scp) show_scp ;;
    '')
      printf '\n1) Enviar o arquivo cifrado ao Dropbox\n2) Mostrar comando SCP para baixar no meu PC\n'
      read -r -p 'Escolha [1/2]: ' choice
      case "$choice" in
        1) upload_dropbox ;;
        2) show_scp ;;
        *) fail 'Escolha invalida.' ;;
      esac
      ;;
    *) fail 'Use sem argumentos, --dropbox ou --scp.' ;;
  esac
}

main "$@"
