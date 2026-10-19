"""检查 Windows 上的 Transformer 教学视频运行环境。

在项目根目录运行::

    .\.venv\Scripts\python.exe scripts\check_environment.py

只检查并报告，不安装任何软件、不修改系统设置。
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402  (必须先把项目根目录加入 sys.path)


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_utf8_console()


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str) -> None:
    _results.append((status, name, detail))
    print(f"[{status:4s}] {name}: {detail}")


def run_version(command: list[str], timeout: int = 20) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    first_line = output[0] if output else f"exit code {completed.returncode}"
    return completed.returncode == 0, first_line


def check_python() -> None:
    if sys.version_info >= (3, 11):
        report(OK, "Python", f"{sys.version.split()[0]} ({sys.executable})")
    else:
        report(FAIL, "Python", f"{sys.version.split()[0]}；需要 Python 3.11+")


def check_python_packages() -> None:
    packages = [
        ("manim", "manim"),
        ("numpy", "numpy"),
        ("pytest", "pytest"),
    ]
    for display_name, module_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            report(OK, f"Python 包 {display_name}", str(version))
        except Exception as exc:  # noqa: BLE001 - 需要把所有导入错误报告给用户
            report(FAIL, f"Python 包 {display_name}", f"无法导入：{exc}")


def check_ffmpeg() -> None:
    for name, executable in (("ffmpeg", config.FFMPEG_EXE), ("ffprobe", config.FFPROBE_EXE)):
        found = shutil.which(name)
        if found:
            ok, line = run_version([found, "-version"])
            report(OK if ok else FAIL, name, f"{found} | {line}")
        else:
            report(FAIL, name, f"未找到；期望项目内文件 {executable} 或 PATH 中的 {name}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if "libx264" in (completed.stdout or ""):
                report(OK, "H.264 编码器", "libx264 可用")
            else:
                report(WARN, "H.264 编码器", "未检测到 libx264，仍可能使用其他 H.264 编码器")
        except Exception as exc:  # noqa: BLE001
            report(WARN, "H.264 编码器", f"检查失败：{exc}")


def check_latex() -> None:
    for name in ("latex", "dvisvgm"):
        found = shutil.which(name)
        if not found:
            report(
                FAIL,
                name,
                "未找到；MathTex 公式渲染需要它。可安装 MiKTeX 并加入 PATH。",
            )
            continue
        ok, line = run_version([found, "--version"])
        report(OK if ok else FAIL, name, f"{found} | {line}")
    # 这是 Manim MathTex 常用类文件；缺失时会在渲染时暴露。
    try:
        completed = subprocess.run(
            ["kpsewhich", "standalone.cls"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if completed.returncode == 0 and (completed.stdout or "").strip():
            report(OK, "LaTeX standalone.cls", (completed.stdout or "").strip())
        else:
            report(WARN, "LaTeX standalone.cls", "未定位到；首次渲染时 MiKTeX 可能需要自动安装")
    except FileNotFoundError:
        report(WARN, "LaTeX standalone.cls", "kpsewhich 不可用，跳过类文件检查")


def check_chinese_fonts() -> None:
    try:
        import manimpango

        fonts = set(manimpango.list_fonts())
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "中文字体", f"无法读取 Pango 字体列表：{exc}")
        return

    for family in (config.FONT_CN, config.FONT_CN_UI, config.FONT_MONO):
        if family in fonts:
            report(OK, f"字体 {family}", "Pango 可见")
        else:
            report(FAIL, f"字体 {family}", "Pango 字体列表中不存在")

    yahei = Path(r"C:\Windows\Fonts\msyh.ttc")
    if yahei.is_file():
        report(OK, "字体文件 msyh.ttc", str(yahei))
    else:
        report(FAIL, "字体文件 msyh.ttc", "未找到，中文字符可能无法显示")


def check_paths() -> None:
    for path in (config.OUTPUT_DIR, config.AUDIO_DIR, config.ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)

    probe = config.OUTPUT_DIR / ".write_probe"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        report(OK, "项目可写", str(config.OUTPUT_DIR))
    except OSError as exc:
        report(FAIL, "项目可写", f"无法写入 output/：{exc}")

    added = config.ensure_tool_paths()
    ffmpeg = shutil.which("ffmpeg")
    if added:
        report(OK, "项目工具 PATH", "已加入: " + "; ".join(str(p) for p in added))
    elif ffmpeg:
        report(OK, "项目工具 PATH", f"已生效（ffmpeg = {ffmpeg}）")
    else:
        report(WARN, "项目工具 PATH", "没有新增目录，且 PATH 中仍无 ffmpeg")


def check_video_spec() -> None:
    report(
        OK,
        "目标视频规格",
        f"{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}, "
        f"{config.FRAME_RATE} FPS, 16:9, 目标 {config.TARGET_DURATION_SECONDS:.0f}s",
    )


def main() -> int:
    print("=" * 78)
    print("Transformer 教学视频 - 环境检查")
    print("=" * 78)
    print(f"操作系统: {platform.platform()}")
    print(f"工作目录: {Path.cwd()}")
    print(f"项目目录: {ROOT_DIR}")
    print("-" * 78)

    check_python()
    check_python_packages()
    check_video_spec()
    check_ffmpeg()
    check_latex()
    check_chinese_fonts()
    check_paths()

    failures = [item for item in _results if item[0] == FAIL]
    warnings = [item for item in _results if item[0] == WARN]
    print("-" * 78)
    print(f"结果: {len(_results) - len(failures) - len(warnings)} 项通过, "
          f"{len(warnings)} 项警告, {len(failures)} 项失败")
    if failures:
        print("未通过项：")
        for _, name, detail in failures:
            print(f"  - {name}: {detail}")
        print("\n请先修复失败项，再进行 Manim 渲染验收。")
        return 1

    print("环境检查通过：可以进行最小 Manim 渲染。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
