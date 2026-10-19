"""第 4 章：从 Transformer 到下一个 Token；第 5 章：总结。

生成段中的所有 logits 与概率都是人为构造的教学示例，由
``math_utils.build_next_token_example`` 通过真实 Softmax 计算得到，
不是任何真实语言模型的推理结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

from manim import (  # noqa: E402
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
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
    ProbabilityBars,
    TokenChip,
    cn_text,
    formula,
    make_backdrop,
    make_caption,
    make_section_header,
    mono_text,
)
from math_utils import build_next_token_example  # noqa: E402

BLACK = "#000000"


class GenerationScene(Scene):
    """堆叠、Logits、Softmax 概率、选择 Token 与架构局限。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND
        self.example = build_next_token_example()
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
            4, "从上下文到下一个 Token", tag_text="TRANSFORMER / 04"
        )
        self.play(
            FadeOut(black, run_time=0.55),
            FadeIn(self.header, shift=DOWN * 0.08),
            run_time=0.55,
        )

        self._opening()
        self._stack()
        self._probabilities()
        self._decode()
        self._architecture_caveat()
        self._exit_black()

    # ------------------------------------------------------------------
    def _step_heading(self, step: int, title: str) -> VGroup:
        tag = mono_text(f"STEP {step}", 12, config.CYAN)
        heading = cn_text(title, 20, config.TEXT_COLOR, weight="SEMIBOLD")
        group = VGroup(tag, heading).arrange(
            RIGHT, buff=0.22, aligned_edge=DOWN
        )
        group.to_edge(LEFT, buff=0.55)
        group.shift(UP * 2.70)
        return group

    # ------------------------------------------------------------------
    def _opening(self) -> None:
        title = cn_text(
            "模型如何写出下一个 Token？", 34, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 0.85, 0])
        subtitle = cn_text(
            "生成是逐步进行的：每次根据已有上下文预测一个 Token。",
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
            "本节使用固定教学示例；概率不是真实模型的推理结果。"
        )
        opening = VGroup(title, subtitle, accent, caption)
        self.play(
            FadeIn(title, shift=UP * 0.10),
            FadeIn(subtitle, shift=UP * 0.06),
            Create(accent),
            FadeIn(caption, shift=UP * 0.05),
            run_time=0.9,
        )
        self.wait(2.2)
        self.play(FadeOut(opening), run_time=0.5)

    # ------------------------------------------------------------------
    def _stack(self) -> None:
        heading = self._step_heading(1, "N 层 Transformer：逐层更新上下文")
        prompt_box = RoundedRectangle(
            width=4.85,
            height=0.56,
            corner_radius=0.10,
            stroke_color=config.BLUE,
            stroke_width=1.0,
            fill_color=config.SURFACE,
            fill_opacity=0.7,
        ).move_to([-2.90, -1.78, 0])
        prompt_text = mono_text("The cat sat on the", 21, config.TEXT_COLOR)
        prompt_text.move_to(prompt_box)

        blocks = VGroup()
        for y in [.97, .30, -.37, -1.04]:
            block = RoundedRectangle(
                width=3.15,
                height=0.46,
                corner_radius=0.08,
                stroke_color=config.BLUE,
                stroke_width=1.0,
                fill_color=config.SURFACE,
                fill_opacity=0.58,
            ).move_to([-2.90, y, 0])
            label = mono_text("Transformer Block", 13, config.TEXT_COLOR)
            label.move_to(block)
            blocks.add(VGroup(block, label))

        block_arrows = VGroup()
        for index in range(len(blocks) - 1):
            block_arrows.add(
                Arrow(
                    blocks[index].get_bottom() + DOWN * 0.02,
                    blocks[index + 1].get_top() + UP * 0.02,
                    buff=0.02,
                    stroke_width=1.2,
                    color=config.TEXT_MUTED,
                    max_tip_length_to_length_ratio=0.35,
                )
            )
        prompt_arrow = Arrow(
            prompt_box.get_top(),
            blocks[-1].get_bottom(),
            buff=0.04,
            stroke_width=1.2,
            color=config.CYAN_DIM,
            max_tip_length_to_length_ratio=0.30,
        )
        output_box = RoundedRectangle(
            width=3.25,
            height=0.50,
            corner_radius=0.09,
            stroke_color=config.GREEN,
            stroke_width=1.0,
            fill_color=config.SURFACE,
            fill_opacity=0.6,
        ).move_to([-2.90, 1.62, 0])
        output_text = cn_text("最后一层的上下文表示", 13, config.GREEN)
        output_text.move_to(output_box)
        output_arrow = Arrow(
            blocks[0].get_top(),
            output_box.get_bottom(),
            buff=0.04,
            stroke_width=1.2,
            color=config.GREEN,
            max_tip_length_to_length_ratio=0.30,
        )

        notes_head = cn_text("N 层串联", 18, config.CYAN, weight="SEMIBOLD")
        notes_head.move_to([2.75, 1.35, 0])
        notes = VGroup(
            cn_text("Attention 交换上下文", 15, config.TEXT_MUTED),
            cn_text("FFN 变换每个位置的特征", 15, config.TEXT_MUTED),
            cn_text("残差与 Norm 稳定深层训练", 15, config.TEXT_MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        notes.move_to([2.75, 0.15, 0])
        layer_note = mono_text("4 LAYERS SHOWN / N CAN BE LARGE", 10, config.CYAN_DIM)
        layer_note.move_to([2.75, -1.10, 0])

        caption = make_caption(
            "多层 Block 把 Token 表示逐层更新为上下文表示。"
        )
        self.play(FadeIn(heading, shift=DOWN * 0.06), run_time=0.45)
        self.play(
            FadeIn(prompt_box),
            FadeIn(prompt_text),
            run_time=0.55,
        )
        self.play(
            LaggedStart(
                *[FadeIn(block, shift=UP * 0.06) for block in reversed(blocks)],
                lag_ratio=0.12,
            ),
            run_time=0.9,
        )
        self.play(
            GrowArrow(prompt_arrow),
            LaggedStart(
                *[GrowArrow(arrow) for arrow in block_arrows],
                lag_ratio=0.10,
            ),
            GrowArrow(output_arrow),
            FadeIn(output_box),
            FadeIn(output_text),
            run_time=0.6,
        )
        self.play(
            FadeIn(notes_head),
            LaggedStart(
                *[FadeIn(note, shift=RIGHT * 0.05) for note in notes],
                lag_ratio=0.12,
            ),
            FadeIn(layer_note),
            run_time=0.6,
        )
        self.wait(1.3)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(1.3)

        section = VGroup(
            heading, prompt_box, prompt_text, blocks, block_arrows,
            prompt_arrow, output_box, output_text, output_arrow, notes_head,
            notes, layer_note, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    def _probabilities(self) -> None:
        heading = self._step_heading(2, "Logits → Softmax → 下一个 Token 概率")
        softmax_line = formula(
            r"p_i=\mathrm{softmax}(z_i)", 25, config.GREEN
        ).move_to([0, 2.20, 0])
        disclaimer = cn_text(
            "教学构造数据 · 非真实推理", 12, config.AMBER
        ).move_to([4.90, 2.20, 0])

        logits_card = RoundedRectangle(
            width=3.30,
            height=3.15,
            corner_radius=0.12,
            stroke_color=config.BLUE,
            stroke_width=0.9,
            fill_color=config.SURFACE,
            fill_opacity=0.55,
        ).move_to([-4.40, -0.10, 0])
        logits_title = cn_text("LOGITS / 教学示例", 11, config.AMBER)
        logits_title.move_to([-4.40, 1.22, 0])
        logit_rows = VGroup()
        for index, (token, logit) in enumerate(
            zip(self.example.candidates, self.example.logits)
        ):
            y = 0.55 - index * 0.50
            token_text = mono_text(token, 17, config.TEXT_COLOR)
            value_text = mono_text(f"{logit:+.1f}", 17, config.CYAN)
            token_text.move_to([-5.25, y, 0])
            value_text.move_to([-3.55, y, 0])
            logit_rows.add(token_text, value_text)
        logits = VGroup(logits_card, logits_title, logit_rows)

        bar_title = cn_text(
            "下一个 Token 概率（教学示例）", 16, config.CYAN
        ).move_to([2.75, 1.50, 0])
        bars = ProbabilityBars(
            self.example.candidates,
            self.example.probabilities,
            selected_index=self.example.selected_index,
            max_height=2.30,
            bar_width=0.70,
            spacing=1.10,
            baseline_y=-1.05,
            label_font_size=18,
            value_font_size=15,
        )
        bars.move_to([2.75, -0.22, 0])
        sum_label = mono_text("sum p = 1.000", 13, config.GREEN)
        sum_label.move_to([2.75, -2.02, 0])

        softmax_arrow = Arrow(
            [-2.55, -0.20, 0],
            [0.78, -0.20, 0],
            buff=0.0,
            stroke_width=1.4,
            color=config.GREEN,
            max_tip_length_to_length_ratio=0.10,
        )
        softmax_label = mono_text("softmax", 12, config.GREEN)
        softmax_label.next_to(softmax_arrow, UP, buff=0.10)

        selected_bar = bars.bar_rects[self.example.selected_index]
        selected_highlight = SurroundingRectangle(
            selected_bar,
            color=config.GREEN,
            buff=0.06,
            stroke_width=2.0,
        )
        argmax_label = mono_text(
            f"argmax = {self.example.selected_token}", 14, config.GREEN
        )
        argmax_label.move_to([2.75, -2.32, 0])
        caption = make_caption(
            "logits 与概率均为教学示例，不是真实语言模型的推理结果。"
        )

        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(softmax_line, shift=DOWN * 0.06),
            FadeIn(disclaimer),
            run_time=0.55,
        )
        self.play(FadeIn(logits, shift=RIGHT * 0.08), run_time=0.75)
        self.wait(1.0)
        self.play(
            GrowArrow(softmax_arrow),
            FadeIn(softmax_label),
            FadeIn(bar_title),
            FadeIn(bars.baseline),
            run_time=0.55,
        )
        self.play(
            LaggedStart(
                *[
                    GrowFromEdge(bar, DOWN)
                    for bar in bars.bar_rects
                ],
                lag_ratio=0.14,
            ),
            run_time=0.9,
        )
        self.play(
            FadeIn(bars.token_labels),
            FadeIn(bars.value_labels),
            run_time=0.45,
        )
        self.wait(1.4)
        self.play(
            Create(selected_highlight),
            FadeIn(argmax_label, shift=LEFT * 0.06),
            FadeIn(sum_label),
            run_time=0.6,
        )
        self.wait(1.0)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.55)
        self.wait(1.7)

        section = VGroup(
            heading, softmax_line, disclaimer, logits, softmax_arrow,
            softmax_label, bar_title, bars, selected_highlight, argmax_label,
            sum_label, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    def _decode(self) -> None:
        heading = self._step_heading(3, "解码：选择 Token，并追加到上下文")
        prefix = mono_text("The cat sat on the", 29, config.TEXT_COLOR)
        chip = TokenChip(
            self.example.selected_token,
            color=config.GREEN,
            width=1.20,
            font_size=24,
        )
        sentence = VGroup(prefix, chip).arrange(RIGHT, buff=0.36)
        sentence.move_to([0, 0.92, 0])

        choice = mono_text(
            f"greedy  /  argmax  =  {self.example.selected_token}",
            14,
            config.GREEN,
        ).move_to([0, 1.65, 0])

        strategy_head = cn_text("解码策略", 17, config.CYAN)
        strategy_head.move_to([0, -0.20, 0])
        pills = VGroup()
        for label, active in [
            ("贪心：取概率最大", True),
            ("随机采样：按概率抽", False),
            ("Top-k：限制候选", False),
        ]:
            box = RoundedRectangle(
                width=2.55,
                height=0.44,
                corner_radius=0.08,
                stroke_color=config.GREEN if active else config.GRID_COLOR,
                stroke_width=1.0,
                fill_color=config.GREEN if active else config.SURFACE,
                fill_opacity=0.14 if active else 0.45,
            )
            text = cn_text(
                label, 12, config.GREEN if active else config.TEXT_MUTED
            )
            text.move_to(box)
            pills.add(VGroup(box, text))
        pills.arrange(RIGHT, buff=0.22).move_to([0, -0.80, 0])
        strategy_note = cn_text(
            "本片用贪心解码演示；真实系统还可以按 temperature 或采样策略生成。",
            13,
            config.TEXT_MUTED,
        ).move_to([0, -1.42, 0])
        caption = make_caption(
            "模型根据上下文计算概率分布，再按解码策略生成 Token。"
        )

        self.play(FadeIn(heading, shift=DOWN * 0.06), run_time=0.45)
        self.play(
            FadeIn(choice, shift=UP * 0.06),
            FadeIn(prefix, shift=UP * 0.06),
            run_time=0.65,
        )
        self.play(
            FadeIn(chip, shift=LEFT * 0.08),
            run_time=0.55,
        )
        self.wait(1.0)
        self.play(
            FadeIn(strategy_head),
            LaggedStart(
                *[FadeIn(pill, shift=UP * 0.05) for pill in pills],
                lag_ratio=0.10,
            ),
            FadeIn(strategy_note),
            run_time=0.7,
        )
        self.wait(1.3)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.0)

        section = VGroup(
            heading, prefix, chip, choice, strategy_head, pills,
            strategy_note, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    def _architecture_caveat(self) -> None:
        tag = mono_text("WHAT ARCHITECTURE ALONE DOES NOT DO", 11, config.AMBER)
        tag.move_to([0, 1.55, 0])
        title = cn_text(
            "Transformer 是核心架构，但不是全部能力。",
            27,
            config.AMBER,
            weight="BOLD",
        ).move_to([0, 0.85, 0])
        body = cn_text(
            "ChatGPT 的能力还来自大规模训练数据、算力、指令微调与对齐。",
            18,
            config.TEXT_COLOR,
        ).move_to([0, 0.05, 0])
        note = cn_text(
            "画面中的 logits 和概率是教学构造，不是真实推理。",
            14,
            config.TEXT_MUTED,
        ).move_to([0, -0.70, 0])
        accent = Line(
            [-2.2, 0.45, 0],
            [2.2, 0.45, 0],
            stroke_color=config.CYAN,
            stroke_width=1.2,
            stroke_opacity=0.7,
        )
        caveat = VGroup(tag, title, body, note, accent)
        self.play(FadeIn(caveat, shift=UP * 0.10), run_time=0.9)
        self.wait(3.0)
        self.play(FadeOut(caveat), run_time=0.5)

    def _exit_black(self) -> None:
        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(FadeOut(self.header, run_time=0.3), run_time=0.3)
        self.play(FadeIn(black, run_time=0.55))


class SummaryScene(Scene):
    """第 5 章：10 秒全景回顾与结束语。"""

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
        header = make_section_header(
            5, "全景回顾", tag_text="TRANSFORMER / 05"
        )
        self.play(
            FadeOut(black, run_time=0.55),
            FadeIn(header, shift=DOWN * 0.08),
            run_time=0.55,
        )

        title = cn_text(
            "一条完整的生成链路", 30, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 2.10, 0])

        items = [
            ("文本", "The cat sat...", config.TEXT_MUTED),
            ("Token / 向量", "X", config.BLUE),
            ("Attention", "上下文混合", config.CYAN),
            ("Block × N", "深层表示", config.GREEN),
            ("下一个 Token", "概率 + 解码", config.AMBER),
        ]
        nodes = VGroup()
        for label, sub, color in items:
            box = RoundedRectangle(
                width=2.20,
                height=1.25,
                corner_radius=0.10,
                stroke_color=color,
                stroke_width=1.0,
                fill_color=config.SURFACE,
                fill_opacity=0.55,
            )
            title_text = cn_text(label, 15, color)
            sub_text = cn_text(sub, 11, config.TEXT_MUTED)
            title_text.move_to([0, 0.18, 0])
            sub_text.move_to([0, -0.22, 0])
            content = VGroup(title_text, sub_text)
            content.move_to(box)
            nodes.add(VGroup(box, content))
        nodes.arrange(RIGHT, buff=0.32).move_to([0, 0.15, 0])

        arrows = VGroup()
        for index in range(len(nodes) - 1):
            arrows.add(
                Arrow(
                    nodes[index].get_right() + RIGHT * 0.04,
                    nodes[index + 1].get_left() + LEFT * 0.04,
                    buff=0.02,
                    stroke_width=1.0,
                    color=config.CYAN_DIM,
                    max_tip_length_to_length_ratio=0.35,
                )
            )

        self.play(
            FadeIn(title, shift=UP * 0.08),
            run_time=0.6,
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(node, shift=UP * 0.06)
                    for node in nodes
                ],
                lag_ratio=0.10,
            ),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[GrowArrow(arrow) for arrow in arrows],
                lag_ratio=0.10,
            ),
            run_time=0.6,
        )
        self.wait(1.7)

        flow = VGroup(title, nodes, arrows)
        self.play(FadeOut(flow), run_time=0.5)

        quote_one = cn_text(
            "理解 Attention，", 34, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 0.75, 0])
        quote_two = cn_text(
            "就迈出了理解大语言模型的重要一步。",
            34,
            config.CYAN,
            weight="BOLD",
        ).move_to([0, -0.05, 0])
        quote_rule = Line(
            [-2.2, 0.28, 0],
            [2.2, 0.28, 0],
            stroke_color=config.CYAN,
            stroke_width=1.4,
            stroke_opacity=0.8,
        )
        credit = mono_text(
            "TRANSFORMER  /  A 3-MINUTE VISUAL EXPLAINER",
            11,
            config.CYAN_DIM,
        ).move_to([0, -2.10, 0])
        quote = VGroup(quote_one, quote_two, quote_rule, credit)
        self.play(FadeIn(quote, shift=UP * 0.10), run_time=0.85)
        self.wait(3.2)

        final_black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(
            FadeOut(quote, run_time=0.4),
            FadeOut(header, run_time=0.4),
            run_time=0.4,
        )
        self.play(FadeIn(final_black, run_time=0.6))
