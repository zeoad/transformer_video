"""生成原创的低音量科技氛围音乐与章节转场音效。

所有声音都由 NumPy 正弦波/PAD 合成，不包含任何外部音乐素材，
因此不存在第三方音乐版权问题。输出为 48 kHz / 16-bit / stereo WAV。

命令行示例::

    python scripts/make_audio.py --duration 180 \
        --boundaries 20.2,86.8,135.4,169.9 \
        --out assets/audio/ambient.wav
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _chord_notes() -> list[list[float]]:
    """A minor / F major / C major / G major 的暗色氛围和弦。"""
    return [
        [110.00, 164.81, 261.63, 329.63],  # A2 E3 C4 E4
        [87.31, 130.81, 220.00, 261.63],   # F2 C3 A3 C4
        [130.81, 196.00, 261.63, 329.63],  # C3 G3 C4 E4
        [98.00, 146.83, 246.94, 293.66],   # G2 D3 B3 D4
    ]


def synthesize_ambient(
    duration: float,
    boundaries: list[float],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """返回形状为 ``(n_samples, 2)``、峰值不超过 0.18 的立体声信号。"""
    if duration <= 0:
        raise ValueError("duration 必须为正数")

    n_samples = int(round(duration * sample_rate))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261019)

    section = 20.0
    crossfade = 4.0
    chords = _chord_notes()
    chord_index = np.floor(t / section).astype(np.int64)
    local_time = t - chord_index * section
    blend = _smoothstep(
        (local_time - (section - crossfade)) / crossfade
    )
    current = chord_index % len(chords)
    following = (chord_index + 1) % len(chords)

    pad = np.zeros(n_samples, dtype=np.float64)
    for chord_id, chord in enumerate(chords):
        current_weight = (current == chord_id).astype(np.float64) * (1.0 - blend)
        following_weight = (following == chord_id).astype(np.float64) * blend
        weight = current_weight + following_weight
        if not np.any(weight):
            continue

        chord_signal = np.zeros(n_samples, dtype=np.float64)
        for note_index, frequency in enumerate(chord):
            phase = 0.37 * chord_id + 0.61 * note_index
            slow_lfo = 0.80 + 0.20 * np.sin(
                2.0 * math.pi * (0.031 + 0.007 * note_index) * t + phase
            )
            detuned = 0.35 * np.sin(
                2.0 * math.pi * frequency * 1.0035 * t + phase * 1.7
            )
            chord_signal += slow_lfo * (
                np.sin(2.0 * math.pi * frequency * t + phase) + detuned
            )
        chord_signal /= len(chord) * 1.35
        pad += weight * chord_signal

    # 极低音持续层：给纪录片氛围一个稳定的底座。
    pad += 0.10 * np.sin(2.0 * math.pi * 55.0 * t) * (
        0.75 + 0.25 * np.sin(2.0 * math.pi * 0.015 * t)
    )

    # 柔和、缓慢的节拍脉冲，避免机械感。
    period = 3.2
    pulse_time = np.mod(t, period)
    pulse_env = np.exp(-pulse_time * 4.5) * (pulse_time < 1.25)
    pad += 0.020 * np.sin(2.0 * math.pi * 72.0 * pulse_time) * pulse_env

    # 稀疏高频泛音，增加空间感但不抢注意力。
    shimmer_frequencies = (
        523.25,
        659.25,
        783.99,
        880.00,
        1046.50,
        1318.51,
    )
    for index, frequency in enumerate(shimmer_frequencies):
        phase = 0.4 + 0.9 * index
        tremolo = 0.55 + 0.45 * np.sin(
            2.0 * math.pi * (0.011 + 0.003 * index) * t + phase
        )
        pad += (
            0.0018
            * tremolo
            * np.sin(2.0 * math.pi * frequency * t + phase)
        )

    # 章节边界的轻微低频 swell，相当于克制的转场音效。
    for boundary in boundaries:
        if not (0.8 < boundary < duration - 0.8):
            continue
        start = boundary - 1.9
        end = boundary + 0.7
        index_start = max(0, int(start * sample_rate))
        index_end = min(n_samples, int(end * sample_rate))
        local = (
            np.arange(index_start, index_end, dtype=np.float64) / sample_rate
        ) - start
        x = np.clip(local / (end - start), 0.0, 1.0)
        envelope = np.sin(np.pi * x) ** 2
        bloom = (
            np.sin(2.0 * math.pi * 110.0 * local)
            + 0.55 * np.sin(2.0 * math.pi * 165.0 * local + 0.4)
            + 0.30 * np.sin(2.0 * math.pi * 330.0 * local + 0.9)
        )
        pad[index_start:index_end] += 0.024 * envelope * bloom

    # 去除直流，加入开头的缓慢出现和结尾的完整消失。
    pad -= float(np.mean(pad))
    fade_in = min(2.2, duration * 0.20)
    fade_out = min(3.0, duration * 0.25)
    fade_in_samples = max(1, int(fade_in * sample_rate))
    fade_out_samples = max(1, int(fade_out * sample_rate))
    pad[:fade_in_samples] *= _smoothstep(
        np.arange(fade_in_samples, dtype=np.float64) / fade_in_samples
    )
    pad[-fade_out_samples:] *= _smoothstep(
        np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float64)
    )

    peak = float(np.max(np.abs(pad)))
    if peak <= 0:
        raise RuntimeError("生成的音频为空")
    pad *= 0.18 / peak

    stereo = np.repeat(pad[:, None], 2, axis=1)
    return np.clip(stereo, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """把 [-1, 1] 浮点音频写为 16-bit PCM WAV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    channels = pcm.shape[1] if pcm.ndim == 2 else 1
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="生成原创氛围音乐")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--boundaries", type=str, default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/audio/ambient.wav"),
    )
    args = parser.parse_args()

    boundaries = [
        float(value)
        for value in args.boundaries.split(",")
        if value.strip()
    ]
    audio = synthesize_ambient(args.duration, boundaries)
    write_wav(args.out, audio)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"音频已生成: {args.out}")
    print(f"时长: {args.duration:.2f}s  采样率: {SAMPLE_RATE} Hz")
    print(f"峰值: {peak:.4f}  RMS: {rms:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
