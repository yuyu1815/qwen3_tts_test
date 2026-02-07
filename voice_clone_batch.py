# python3 voice_clone_batch.py \
#   --ref-audio myvoice.mp3 \
#   --ref-text-file myvoice_ref.txt \
#   --text-file input.txt \
#   --out out.wav \
#   --language Japanese

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def run_ffmpeg_to_wav(in_path: Path, out_path: Path, sr: int = 16000) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。mp3/mp4 を wav に変換するため ffmpeg をインストールして PATH を通してください。")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        str(out_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 変換に失敗:\n{p.stderr}")


def load_lines_utf8(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else [""]


def concat_with_silence(wavs: list[np.ndarray], sr: int, silence_sec: float = 0.25) -> np.ndarray:
    sil = np.zeros(int(sr * silence_sec), dtype=np.float32)
    out = []
    for i, w in enumerate(wavs):
        out.append(w.astype(np.float32, copy=False))
        if i != len(wavs) - 1:
            out.append(sil)
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_runtime_params(device: str) -> tuple[torch.dtype, str]:
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype, "flash_attention_2"
    if device == "mps":
        return torch.float16, "eager"
    return torch.float32, "eager"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True, help="参照音声 (mp3/mp4/wav等) パス")
    ap.add_argument("--ref-text-file", default=None, help="参照音声の文字起こしテキスト(UTF-8) パス（推奨）")
    ap.add_argument("--x-vector-only", action="store_true", help="ref_text無し（品質低下の可能性）")
    ap.add_argument("--text-file", required=True, help="読み上げたいテキスト(UTF-8) パス")
    ap.add_argument("--out", required=True, help="出力 wav パス")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Voice Clone用 Base モデル")
    ap.add_argument("--language", default="Japanese", help="Japanese / English / auto など")
    ap.add_argument("--device", default="auto", help="auto / mps / cpu / cuda:0")
    ap.add_argument("--silence", type=float, default=0.25, help="行間無音秒")
    args = ap.parse_args()

    ref_in = Path(args.ref_audio).expanduser().resolve()
    text_file = Path(args.text_file).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not ref_in.exists():
        raise FileNotFoundError(f"ref audio not found: {ref_in}")
    if not text_file.exists():
        raise FileNotFoundError(f"text file not found: {text_file}")

    device = resolve_device(args.device)
    dtype, attn_impl = resolve_runtime_params(device)

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref_wav = td / "ref.wav"
        run_ffmpeg_to_wav(ref_in, ref_wav, sr=16000)

        ref_text = None
        if not args.x_vector_only:
            if not args.ref_text_file:
                raise RuntimeError("ref_text が必要です。--ref-text-file を指定するか、ref_text無しなら --x-vector-only を付けてください。")
            ref_text = Path(args.ref_text_file).expanduser().read_text(encoding="utf-8").strip()
            if not ref_text:
                raise RuntimeError("ref_text_file が空です。参照音声の文字起こしを入れてください。")

        lines = load_lines_utf8(text_file)

        wav_list = []
        out_sr = None
        for ln in lines:
            wavs, sr = model.generate_voice_clone(
                text=ln,
                language=args.language,
                ref_audio=str(ref_wav),
                ref_text=ref_text,
                x_vector_only_mode=bool(args.x_vector_only),
            )
            wav_list.append(wavs[0])
            out_sr = sr

        merged = concat_with_silence(wav_list, out_sr, silence_sec=args.silence)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), merged, out_sr)
        print(f"saved: {out_path} sr={out_sr} device={device} attn={attn_impl} dtype={dtype} lines={len(lines)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
