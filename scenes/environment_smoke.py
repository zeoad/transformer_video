"""阶段 1 的最小 Manim 验收场景。

它同时验证：
* 1920x1080 / 30 FPS 的渲染参数（由渲染命令传入）；
* Microsoft YaHei 中文字体；
* LaTeX 数学公式；
* 从 math_utils 读取的真实 Attention 数值；
* 简短电影感版式与统一配色。

运行：见项目根目录 README.md 中的“最小渲染验收”命令。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manim import (  # noqa: E402
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

import config  # noqa: E402
from math_utils import build_demo_trace  # noqa: E402


class EnvironmentSmoke(Scene):
    """约 9 秒的环境检查片头；不是最终正片。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND

        # 低对比度网格，只提供空间感，不喧宾夺主。
        grid = VGroup()
        for x in range(-7, 8, 2):
            grid.add(
                Line(
                    [x, -4.5, 0],
                    [x, 4.5, 0],
                    stroke_color=config.GRID_COLOR,
                    stroke_width=0.6,
                    stroke_opacity=0.20,
                )
            )
        for y in range(-4, 5, 2):
            grid.add(
                Line(
                    [-8, y, 0],
                    [8, y, 0],
                    stroke_color=config.GRID_COLOR,
                    stroke_width=0.6,
                    stroke_opacity=0.20,
                )
            )
        self.add(grid)

        frame_border = Rectangle(
            width=13.4,
            height=7.4,
            stroke_color=config.BLUE_DARK,
            stroke_width=0.8,
            stroke_opacity=0.45,
        )
        self.add(frame_border)

        eyebrow = Text(
            "TRANSFORMER  /  VISUAL EXPLAINER",
            font=config.FONT_MONO,
            font_size=17,
            color=config.CYAN,
        ).to_corner(UP + LEFT, buff=0.55)

        title = Text(
            "注意力，是 Transformer 的核心",
            font=config.FONT_CN,
            font_size=48,
            weight="BOLD",
            color=config.TEXT_COLOR,
        )
        title.move_to([0, 2.72, 0])

        accent = Line(
            [-4.7, 2.18, 0],
            [4.7, 2.18, 0],
            stroke_color=config.CYAN,
            stroke_width=1.6,
        )
        subtitle = Text(
            "中文渲染 · LaTeX 公式 · 真实矩阵计算",
            font=config.FONT_CN,
            font_size=24,
            color=config.TEXT_MUTED,
        ).next_to(accent, DOWN, buff=0.20)

        tokens = ["The", "cat", "sat"]
        token_group = VGroup()
        for token in tokens:
            box = RoundedRectangle(
                width=1.48,
                height=0.68,
                corner_radius=0.12,
                stroke_color=config.BLUE,
                stroke_width=1.8,
                fill_color=config.SURFACE,
                fill_opacity=0.85,
            )
            label = Text(
                token,
                font=config.FONT_MONO,
                font_size=30,
                color=config.TEXT_COLOR,
            )
            token_group.add(VGroup(box, label))
        token_group.arrange(RIGHT, buff=0.45)
        token_group.move_to([0, 1.15, 0])

        formula = MathTex(
            r"\mathrm{Attention}(Q,K,V)"
            r"=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
            font_size=56,
            color=config.TEXT_COLOR,
        )
        formula.move_to([0, -0.25, 0])

        trace, _ = build_demo_trace()
        row_sum_text = " / ".join(f"{value:.4f}" for value in trace.row_sums)
        # 状态行混有中文与数学符号，使用支持完整字形的 YaHei；
        # Cascadia Mono 缺少部分 CJK/上标字形，会渲染成缺字方块。
        status = Text(
            "QK^T 形状 3x3    Softmax 每行之和  "
            f"{row_sum_text}    d_k = {trace.d_k}",
            font=config.FONT_CN,
            font_size=21,
            color=config.GREEN,
        )
        status.move_to([0, -1.55, 0])

        check = Text(
            "阶段 1 验收：中文、公式、数值链路与渲染参数",
            font=config.FONT_CN,
            font_size=22,
            color=config.TEXT_MUTED,
        ).move_to([0, -2.55, 0])

        footer = Text(
            "NUMBERS FROM math_utils.py  ·  NUMPY FLOAT64",
            font=config.FONT_MONO,
            font_size=14,
            color=config.CYAN_DIM,
        ).move_to([0, -3.45, 0])

        self.play(
            FadeIn(eyebrow, shift=DOWN * 0.15),
            FadeIn(title, shift=DOWN * 0.2),
            Create(accent),
            FadeIn(subtitle, shift=DOWN * 0.1),
            run_time=1.3,
        )
        self.play(FadeIn(token_group, shift=UP * 0.25, lag_ratio=0.12), run_time=1.2)
        self.play(Write(formula), run_time=1.6)
        self.play(FadeIn(status, shift=UP * 0.12), FadeIn(check), run_time=1.0)
        self.play(FadeIn(footer), run_time=0.4)
        self.wait(2.6)
        self.play(FadeOut(VGroup(grid, frame_border, eyebrow, title, accent, subtitle,
                                 token_group, formula, status, check, footer)),
                  run_time=0.8)
