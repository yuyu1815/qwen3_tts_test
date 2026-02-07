#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT_DIR/scripts/start_gui.command" "$@"
