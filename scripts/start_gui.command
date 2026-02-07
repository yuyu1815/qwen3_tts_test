#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! bash "$ROOT_DIR/scripts/start_gui.sh"; then
  echo
  echo "起動に失敗しました。まず ./scripts/setup_mac.sh を実行してください。"
  echo "Enterキーで終了します。"
  read -r
fi
