"""第 3 章：Transformer Block 的结构与作用。

约 40 秒。结构图顺序固定：
输入 → Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm → 输出。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

from manim import (  # noqa: E402
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    UP,
    VGroup,
)

from components import (  # noqa: E402
    TransformerBlockDiagram,
    cn_text,
    formula,
    make_backdrop,
    make_caption,
    make_section_header,
    mono_text,
)

BLACK = "#000000"


class TransformerBlockScene(Scene):
    """Transformer Block 的结构讲解。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND
        self.add(make_backdrop())

        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.add(black)
        self.header = make_section_header(
            3, "Transformer Block", tag_text="TRANSFORMER / 03"
        )
        self.play(
            FadeOut(black, run_time=0.55),
            FadeIn(self.header, shift=DOWN * 0.08),
            run_time=0.55,
        )

        self._opening()
        self._build_diagram()
        self._focus_multihead()
        self._focus_residual()
        self._focus_layernorm()
        self._focus_ffn()
        self._recap_and_teaser()
        self._exit_to_black()

    # ------------------------------------------------------------------
    # 布局辅助
    # ------------------------------------------------------------------
    def _focus_panel(
        self,
        title: str,
        body: str,
        color: str,
        *,
        extra: VGroup | None = None,
    ) -> VGroup:
        card = RoundedRectangle(
            width=5.35,
            height=3.55,
            corner_radius=0.14,
            stroke_color=color,
            stroke_width=1.0,
            fill_color=config.SURFACE,
            fill_opacity=0.58,
        ).move_to([3.55, 0.22, 0])
        kicker = mono_text("BLOCK COMPONENT", 10, color)
        kicker.move_to([3.55, 1.75, 0])
        title_text = cn_text(title, 21, color, weight="SEMIBOLD")
        title_text.move_to([3.55, 1.36, 0])
        body_text = cn_text(body, 15, config.TEXT_MUTED)
        body_text.move_to([3.55, 0.94, 0])
        accent = Line(
            [1.20, 1.93, 0],
            [1.85, 1.93, 0],
            stroke_color=color,
            stroke_width=1.5,
            stroke_opacity=0.8,
        )
        group = VGroup(card, kicker, title_text, body_text, accent)
        if extra is not None:
            extra.move_to([3.55, -0.33, 0])
            group.add(extra)
        return group

    def _outline(self, node_index: int, color: str):
        return SurroundingRectangle(
            self.diagram.node_boxes[node_index],
            color=color,
            buff=0.045,
            stroke_width=1.8,
        )

    # ------------------------------------------------------------------
    # 开场
    # ------------------------------------------------------------------
    def _opening(self) -> None:
        title = cn_text(
            "一个 Block，如何把上下文变成表示？",
            34,
            config.TEXT_COLOR,
            weight="BOLD",
        ).move_to([0, 0.85, 0])
        subtitle = cn_text(
            "Attention 汇总信息，FFN 变换特征，残差与归一化保证可训练。",
            19,
            config.TEXT_MUTED,
        ).move_to([0, 0.16, 0])
        accent = Line(
            [-2.1, -0.35, 0],
            [2.1, -0.35, 0],
            stroke_color=config.CYAN,
            stroke_width=1.5,
        )
        caption = make_caption(
            "Block 内部顺序固定：Attention → 残差与归一化 → FFN → 残差与归一化。"
        )
        opening = VGroup(title, subtitle, accent, caption)
        self.play(
            FadeIn(title, shift=UP * 0.10),
            FadeIn(subtitle, shift=UP * 0.06),
            Create(accent),
            FadeIn(caption, shift=UP * 0.05),
            run_time=0.9,
        )
        self.wait(2.3)
        self.play(FadeOut(opening), run_time=0.5)

    # ------------------------------------------------------------------
    # 结构图
    # ------------------------------------------------------------------
    def _build_diagram(self) -> None:
        self.diagram = TransformerBlockDiagram(
            x=-3.45, top_y=2.12, pitch=0.88, node_width=3.15
        )
        sequence = []
        for index, node in enumerate(self.diagram.node_groups):
            sequence.append(FadeIn(node, shift=UP * 0.07))
            if index < len(self.diagram.arrows):
                sequence.append(Create(self.diagram.arrows[index]))
        self.play(LaggedStart(*sequence, lag_ratio=0.08, run_time=1.45))
        self.play(Create(self.diagram.residuals), run_time=0.65)
        self.overview_caption = make_caption(
            "输入进入 Attention，经过两次 Add & Norm，再输出给下一层。"
        )
        self.play(
            FadeIn(self.overview_caption, shift=UP * 0.05),
            run_time=0.45,
        )
        self.wait(1.3)

    # ------------------------------------------------------------------
    # 多头注意力
    # ------------------------------------------------------------------
    def _focus_multihead(self) -> None:
        heads = VGroup()
        for index, color in enumerate(
            (config.BLUE, config.CYAN, config.GREEN, config.AMBER), start=1
        ):
            box = RoundedRectangle(
                width=1.02,
                height=0.48,
                corner_radius=0.08,
                stroke_color=color,
                stroke_width=1.0,
                fill_color=color,
                fill_opacity=0.16,
            )
            label = mono_text(f"H{index}", 14, color).move_to(box)
            heads.add(VGroup(box, label))
        heads.arrange(RIGHT, buff=0.16)
        head_caption = cn_text("并行学习不同关系 → 拼接", 14, config.TEXT_MUTED)
        head_caption.next_to(heads, DOWN, buff=0.18)
        extra = VGroup(heads, head_caption)

        card = self._focus_panel(
            "Multi-Head Attention",
            "从不同表示子空间学习信息关系",
            config.BLUE,
            extra=extra,
        )
        outline = self._outline(1, config.BLUE)
        caption = make_caption(
            "多个注意力头并行关注不同关系，再拼接成统一表示。"
        )

        self.play(FadeIn(card, shift=RIGHT * 0.10), run_time=0.7)
        self.play(Create(outline), run_time=0.4)
        self.play(
            LaggedStart(
                *[FadeIn(head, shift=UP * 0.05) for head in heads],
                lag_ratio=0.12,
            ),
            FadeIn(head_caption, shift=UP * 0.05),
            run_time=0.75,
        )
        self.wait(2.1)
        self.play(
            FadeOut(self.overview_caption),
            FadeIn(caption, shift=UP * 0.05),
            run_time=0.45,
        )
        self.play(Indicate(self.diagram.node_boxes[1], color=config.BLUE), run_time=0.9)
        self.wait(1.4)

        self.play(FadeOut(VGroup(card, outline, caption)), run_time=0.5)

    # ------------------------------------------------------------------
    # 残差连接
    # ------------------------------------------------------------------
    def _focus_residual(self) -> None:
        extra = formula(
            r"x_{\mathrm{out}} = x + \mathrm{Sublayer}(x)",
            20,
            config.CYAN,
        )
        extra2 = cn_text(
            "子层输入直接加到输出，形成信息与梯度通路",
            13,
            config.TEXT_MUTED,
        )
        extra2.next_to(extra, DOWN, buff=0.28)
        card = self._focus_panel(
            "Residual Connection",
            "保留原始信息，帮助信息和梯度流动",
            config.CYAN,
            extra=VGroup(extra, extra2),
        )
        outline_a = self._outline(2, config.CYAN)
        outline_b = self._outline(4, config.CYAN)
        caption = make_caption(
            "残差把子层输入直接加回输出，避免深层网络丢失原始信息。"
        )
        self.play(FadeIn(card, shift=RIGHT * 0.10), run_time=0.7)
        self.play(
            Create(outline_a),
            Create(outline_b),
            self.diagram.residuals.animate.set_stroke(width=2.0, opacity=1.0),
            run_time=0.7,
        )
        self.wait(1.4)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.45)
        self.wait(2.1)
        self.play(
            Indicate(self.diagram.node_boxes[2], color=config.CYAN),
            Indicate(self.diagram.node_boxes[4], color=config.CYAN),
            run_time=1.0,
        )
        self.wait(1.4)
        self.play(FadeOut(VGroup(card, outline_a, outline_b, caption)), run_time=0.5)

    # ------------------------------------------------------------------
    # LayerNorm
    # ------------------------------------------------------------------
    def _focus_layernorm(self) -> None:
        extra = formula(
            r"\mathrm{LayerNorm}\left(x+\mathrm{Sublayer}(x)\right)",
            18,
            config.GREEN,
        )
        extra2 = cn_text(
            "沿特征维归一化：稳定尺度，改善优化",
            13,
            config.TEXT_MUTED,
        )
        extra2.next_to(extra, DOWN, buff=0.28)
        card = self._focus_panel(
            "Layer Normalization",
            "对特征做归一化，稳定数值尺度",
            config.GREEN,
            extra=VGroup(extra, extra2),
        )
        outline_a = self._outline(2, config.GREEN)
        outline_b = self._outline(4, config.GREEN)
        caption = make_caption(
            "LayerNorm 对每个位置的特征做归一化，让训练更稳定。"
        )
        self.play(FadeIn(card, shift=RIGHT * 0.10), run_time=0.7)
        self.play(Create(outline_a), Create(outline_b), run_time=0.6)
        self.wait(1.4)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.45)
        self.wait(2.0)
        self.play(
            Indicate(self.diagram.node_boxes[2], color=config.GREEN),
            Indicate(self.diagram.node_boxes[4], color=config.GREEN),
            run_time=1.0,
        )
        self.wait(1.4)
        self.play(FadeOut(VGroup(card, outline_a, outline_b, caption)), run_time=0.5)

    # ------------------------------------------------------------------
    # FFN
    # ------------------------------------------------------------------
    def _focus_ffn(self) -> None:
        extra = formula(
            r"\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2",
            17,
            config.AMBER,
        )
        extra2 = cn_text(
            "同一个两层网络独立作用于每个位置",
            13,
            config.TEXT_MUTED,
        )
        extra2.next_to(extra, DOWN, buff=0.28)
        card = self._focus_panel(
            "Feed-Forward Network",
            "对每个位置的特征做进一步变换",
            config.AMBER,
            extra=VGroup(extra, extra2),
        )
        outline = self._outline(3, config.AMBER)
        caption = make_caption(
            "FFN 对每个位置分别作用，不混合不同 Token 的信息。"
        )
        self.play(FadeIn(card, shift=RIGHT * 0.10), run_time=0.7)
        self.play(Create(outline), run_time=0.4)
        self.wait(1.4)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.45)
        self.wait(2.0)
        self.play(Indicate(self.diagram.node_boxes[3], color=config.AMBER), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(VGroup(card, outline, caption)), run_time=0.5)

    # ------------------------------------------------------------------
    # 汇总与堆叠引导
    # ------------------------------------------------------------------
    def _recap_and_teaser(self) -> None:
        recap = make_caption(
            "一个 Block 完成：Attention → Add & Norm → FFN → Add & Norm。"
        )
        self.play(
            FadeIn(recap, shift=UP * 0.05),
            Indicate(self.diagram.node_boxes[1], color=config.BLUE),
            Indicate(self.diagram.node_boxes[3], color=config.AMBER),
            run_time=1.1,
        )
        self.wait(2.0)

        stack_title = cn_text(
            "把 Block 堆叠 N 次，得到深层上下文表示", 22, config.CYAN
        ).move_to([0, 0.35, 0])
        mini_blocks = VGroup()
        for index in range(3):
            block = RoundedRectangle(
                width=3.7,
                height=0.55,
                corner_radius=0.09,
                stroke_color=config.BLUE if index < 2 else config.CYAN,
                stroke_width=1.1,
                fill_color=config.SURFACE,
                fill_opacity=0.55,
            )
            label = mono_text("Transformer Block", 14, config.TEXT_COLOR)
            label.move_to(block)
            mini_blocks.add(VGroup(block, label))
        mini_blocks.arrange(DOWN, buff=0.18).move_to([0, -1.05, 0])
        stack_caption = make_caption(
            "下一章：堆叠后的模型如何输出下一个 Token。"
        )
        self.play(
            FadeOut(self.diagram),
            FadeOut(recap),
            FadeIn(stack_title, shift=UP * 0.08),
            LaggedStart(
                *[FadeIn(block, shift=UP * 0.06) for block in mini_blocks],
                lag_ratio=0.12,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(stack_caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.5)
        self.play(FadeOut(VGroup(stack_title, mini_blocks, stack_caption)), run_time=0.5)

    def _exit_to_black(self) -> None:
        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(FadeOut(self.header, run_time=0.3), run_time=0.3)
        self.play(FadeIn(black, run_time=0.55))
