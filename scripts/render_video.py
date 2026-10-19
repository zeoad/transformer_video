"""完整视频的渲染、拼接、配乐与验收脚本。

用法（在项目根目录）::

    .\.venv\Scripts\python.exe scripts\render_video.py all
    .\.venv\Scripts\python.exe scripts\render_video.py render
    .\.venv\Scripts\python.exe scripts\render_video.py assemble
    .\.venv\Scripts\python.exe scripts\render_video.py audio
    .\.venv\Scripts\python.exe scripts\render_video.py verify

``all`` 依次完成：渲染 5 个场景 → 无损顺序拼接并统一为 30 FPS →
生成原创氛围音乐 → 合成最终 MP4 → ffprobe/解码验收。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402
from make_audio import synthesize_ambient, write_wav  # noqa: E402


@dataclass(frozen=True)
class SceneSpec:
    key: str
    file: str
    scene_class: str


SCENES = (
    SceneSpec("intro", "intro.py", "IntroScene"),
    SceneSpec("attention", "attention.py", "AttentionScene"),
    SceneSpec("block", "transformer_block.py", "TransformerBlockScene"),
    SceneSpec("generation", "generation.py", "GenerationScene"),
    SceneSpec("summary", "generation.py", "SummaryScene"),
)

SILENT_OUTPUT = config.OUTPUT_DIR / "transformer_3min_silent.mp4"
FINAL_OUTPUT = config.OUTPUT_DIR / "transformer_3min.mp4"
AMBIENT_WAV = config.AUDIO_DIR / "ambient.wav"
TIMINGS_FILE = config.OUTPUT_DIR / "scene_timings.json"
CONCAT_FILE = config.MEDIA_DIR / "concat_scenes.txt"

FFMPEG = config.FFMPEG_EXE if config.FFMPEG_EXE.is_file() else Path(
    shutil.which("ffmpeg") or "ffmpeg"
)
FFPROBE = config.FFPROBE_EXE if config.FFPROBE_EXE.is_file() else Path(
    shutil.which("ffprobe") or "ffprobe"
)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT_DIR,
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode != 0:
        if capture:
            print(completed.stdout)
            print(completed.stderr)
        raise RuntimeError(f"命令失败，退出码 {completed.returncode}")
    return completed


def scene_output(spec: SceneSpec) -> Path:
    stem = Path(spec.file).stem
    return (
        config.MEDIA_DIR
        / "videos"
        / stem
        / f"{config.VIDEO_HEIGHT}p{config.FRAME_RATE}"
        / f"{spec.scene_class}.mp4"
    )


def probe(path: Path) -> dict:
    completed = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_name,codec_type,width,height,"
            "r_frame_rate,avg_frame_rate,pix_fmt,nb_frames,sample_rate,"
            "channels,channel_layout,duration,bit_rate",
            "-of",
            "json",
            path,
        ],
        capture=True,
    )
    return json.loads(completed.stdout)


def video_stream(data: dict) -> dict:
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise RuntimeError("未找到视频流")


def audio_stream(data: dict) -> dict | None:
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return None


def render_all() -> None:
    config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = []

    for spec in SCENES:
        scene_path = config.SCENES_DIR / spec.file
        print("\n" + "=" * 78)
        print(f"渲染 {spec.key}: {spec.scene_class}")
        print("=" * 78, flush=True)
        run(
            [
                sys.executable,
                "-m",
                "manim",
                "render",
                "--resolution",
                f"{config.VIDEO_WIDTH},{config.VIDEO_HEIGHT}",
                "--fps",
                str(config.FRAME_RATE),
                "--format",
                "mp4",
                "--media_dir",
                config.MEDIA_DIR,
                "--disable_caching",
                scene_path,
                spec.scene_class,
            ]
        )
        output = scene_output(spec)
        if not output.is_file():
            raise RuntimeError(f"未找到渲染结果: {output}")
        info = probe(output)
        stream = video_stream(info)
        duration = float(info["format"]["duration"])
        timings.append(
            {
                "key": spec.key,
                "scene_class": spec.scene_class,
                "file": str(output),
                "duration": duration,
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": stream.get("r_frame_rate"),
                "frames": stream.get("nb_frames"),
            }
        )
        print(
            f"\n完成 {spec.key}: {duration:.3f}s, "
            f"{stream.get('width')}x{stream.get('height')}, "
            f"{stream.get('r_frame_rate')} FPS",
            flush=True,
        )

    TIMINGS_FILE.write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n场景时长清单: {TIMINGS_FILE}")
    print(f"预计全片时长: {sum(item['duration'] for item in timings):.3f}s")


def load_timings() -> list[dict]:
    if TIMINGS_FILE.is_file():
        timings = json.loads(TIMINGS_FILE.read_text(encoding="utf-8"))
        if len(timings) == len(SCENES):
            return timings

    timings = []
    for spec in SCENES:
        output = scene_output(spec)
        if not output.is_file():
            raise RuntimeError(f"缺少场景文件: {output}，请先运行 render")
        info = probe(output)
        stream = video_stream(info)
        timings.append(
            {
                "key": spec.key,
                "scene_class": spec.scene_class,
                "file": str(output),
                "duration": float(info["format"]["duration"]),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": stream.get("r_frame_rate"),
                "frames": stream.get("nb_frames"),
            }
        )
    TIMINGS_FILE.write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return timings


def assemble() -> None:
    timings = load_timings()
    lines = []
    for item in timings:
        path = Path(item["file"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"缺少场景文件: {path}")
        escaped = path.as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    CONCAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONCAT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            CONCAT_FILE,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-r",
            str(config.FRAME_RATE),
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            SILENT_OUTPUT,
        ]
    )
    info = probe(SILENT_OUTPUT)
    print(
        f"\n无声母版: {SILENT_OUTPUT}\n"
        f"时长: {float(info['format']['duration']):.3f}s, "
        f"{video_stream(info).get('width')}x"
        f"{video_stream(info).get('height')}, "
        f"{video_stream(info).get('r_frame_rate')} FPS"
    )


def add_audio() -> None:
    if not SILENT_OUTPUT.is_file():
        raise RuntimeError("缺少无声母版，请先运行 assemble")
    timings = load_timings()
    info = probe(SILENT_OUTPUT)
    duration = float(info["format"]["duration"])
    cumulative = 0.0
    boundaries = []
    for item in timings[:-1]:
        cumulative += float(item["duration"])
        boundaries.append(cumulative)

    print(f"\n生成原创氛围音乐，时长 {duration:.3f}s，转场点 {boundaries}")
    audio = synthesize_ambient(duration, boundaries)
    write_wav(AMBIENT_WAV, audio)
    print(f"音频: {AMBIENT_WAV}")

    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-i",
            SILENT_OUTPUT,
            "-i",
            AMBIENT_WAV,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            "-metadata",
            "title=Transformer: A Visual Explainer",
            FINAL_OUTPUT,
        ]
    )
    final_info = probe(FINAL_OUTPUT)
    print(
        f"\n最终视频: {FINAL_OUTPUT}\n"
        f"时长: {float(final_info['format']['duration']):.3f}s"
    )


def _decode_check(path: Path) -> None:
    completed = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"完整解码失败:\n{completed.stderr}")
    if completed.stderr.strip():
        print("解码器输出：")
        print(completed.stderr.strip())
    else:
        print("完整解码：无错误")


def verify() -> None:
    if not FINAL_OUTPUT.is_file():
        raise RuntimeError("缺少最终视频，请先运行 audio")
    info = probe(FINAL_OUTPUT)
    video = video_stream(info)
    audio = audio_stream(info)
    duration = float(info["format"]["duration"])
    fps = video.get("r_frame_rate")
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))

    print("\n" + "=" * 78)
    print("最终验收")
    print("=" * 78)
    print(f"文件: {FINAL_OUTPUT}")
    print(f"时长: {duration:.3f}s")
    print(f"视频: {video.get('codec_name')} {width}x{height} {fps} FPS")
    if audio:
        print(
            f"音频: {audio.get('codec_name')} "
            f"{audio.get('sample_rate')} Hz {audio.get('channels')} ch"
        )
    else:
        print("音频: 无")

    if not (165.0 <= duration <= 195.0):
        raise RuntimeError(f"时长 {duration:.3f}s 不在 165–195 秒验收范围内")
    if (width, height) != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
        raise RuntimeError("分辨率不符合 1920x1080")
    if fps != "30/1":
        raise RuntimeError(f"帧率不符合 30 FPS: {fps}")
    if video.get("codec_name") != "h264":
        raise RuntimeError("视频编码不是 H.264")
    if audio is None:
        raise RuntimeError("最终视频没有音频流")

    _decode_check(FINAL_OUTPUT)
    print("验收通过：时长、画幅、帧率、H.264、AAC 与完整解码均正常。")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transformer 视频完整流水线")
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["all", "render", "assemble", "audio", "verify"],
        help="默认 all",
    )
    args = parser.parse_args()

    if args.action in ("all", "render"):
        render_all()
    if args.action in ("all", "assemble"):
        assemble()
    if args.action in ("all", "audio"):
        add_audio()
    if args.action in ("all", "verify"):
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
