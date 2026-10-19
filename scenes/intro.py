"""第 1 章：文本如何变成机器能处理的数字。

约 20 秒。包含：电影感标题、句子分词、教学示意 Token ID、前 3 个 Token
的二维向量与 X 矩阵、抽象 Transformer 网络空间、下一章引导。
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
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    LaggedStart,
    LEFT,
    Line,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Scene,
    UP,
    VGroup,
)

from components import (  # noqa: E402
    MatrixPanel,
    TokenChip,
    cn_text,
    formula,
    make_backdrop,
    make_caption,
    make_section_header,
    mono_text,
)
from math_utils import demo_qkv  # noqa: E402

BLACK = "#000000"


class IntroScene(Scene):
    """开场：语言 → Token → 向量 → Transformer。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND
        self.X, _, _, _ = demo_qkv()
        self.add(make_backdrop())
        self._opening()
        self._text_tokens_vectors()
        self._network_space()

    def _opening(self) -> None:
        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.add(black)

        brand = mono_text(
            "TRANSFORMER  /  VISUAL EXPLAINER", 13, config.CYAN
        ).to_corner(UP + LEFT, buff=0.55)
        tag = mono_text(
            "A 3-MINUTE EXPLAINER  /  01", 12, config.CYAN_DIM
        ).to_corner(UP + RIGHT, buff=0.55)
        kicker = mono_text(
            "CHAPTER 01  /  FROM TEXT TO NUMBERS", 13, config.CYAN
        ).move_to([0, 1.55, 0])
        title = cn_text(
            "语言，如何变成向量？", 50, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 0.55, 0])
        rule = Line(
            [-1.6, -0.15, 0],
            [1.6, -0.15, 0],
            stroke_color=config.CYAN,
            stroke_width=1.7,
            stroke_opacity=0.9,
        )
        subtitle = cn_text(
            "从一句话开始，进入 Transformer 的世界。",
            22,
            config.TEXT_MUTED,
        ).move_to([0, -0.78, 0])
        footer = mono_text(
            "TEXT  /  TOKENS  /  EMBEDDINGS  /  TRANSFORMER",
            11,
            config.CYAN_DIM,
        ).move_to([0, -2.30, 0])

        opening = VGroup(brand, tag, kicker, title, rule, subtitle, footer)
        self.play(
            FadeOut(black, run_time=1.0),
            FadeIn(brand, shift=DOWN * 0.08),
            FadeIn(tag),
            FadeIn(kicker, shift=UP * 0.08),
            FadeIn(title, shift=UP * 0.12),
            Create(rule),
            FadeIn(subtitle, shift=UP * 0.06),
            FadeIn(footer),
            run_time=1.35,
        )
        self.wait(2.0)
        self.play(FadeOut(opening), run_time=0.45)

    def _text_tokens_vectors(self) -> None:
        header = make_section_header(
            1, "文本 → Token → 向量", tag_text="TRANSFORMER / 01"
        )
        self.play(FadeIn(header, shift=DOWN * 0.08), run_time=0.4)

        source_tag = mono_text("INPUT TEXT", 12, config.TEXT_MUTED)
        sentence = mono_text("The cat sat on the mat.", 36, config.TEXT_COLOR)
        sentence.move_to([0, 1.82, 0])
        source_tag.next_to(sentence, UP, buff=0.16)
        caption = make_caption(
            "语言模型首先需要将文本转换为机器可以处理的数字表示。"
        )
        self.play(
            FadeIn(source_tag, shift=UP * 0.05),
            FadeIn(sentence, shift=UP * 0.08),
            FadeIn(caption, shift=UP * 0.05),
            run_time=0.7,
        )
        self.wait(0.8)

        words = ["The", "cat", "sat", "on", "the", "mat", "."]
        widths = [1.10, 1.10, 1.10, 0.95, 1.10, 1.15, 0.62]
        chips = VGroup(
            *[
                TokenChip(word, width=width, font_size=22)
                for word, width in zip(words, widths)
            ]
        ).arrange(RIGHT, buff=0.22)
        chips.move_to([0, 0.85, 0])

        ids = VGroup()
        for index, chip in enumerate(chips):
            id_label = mono_text(f"id {101 + index}", 12, config.CYAN_DIM)
            id_label.next_to(chip.box, DOWN, buff=0.13)
            ids.add(id_label)

        token_caption = make_caption(
            "教学示意 ID，不代表任何真实分词器的编号。"
        )
        self.play(
            FadeOut(sentence, shift=DOWN * 0.08),
            FadeOut(source_tag),
            FadeIn(chips, shift=DOWN * 0.08, lag_ratio=0.08),
            run_time=0.75,
        )
        self.play(FadeIn(ids, lag_ratio=0.04), run_time=0.35)
        self.play(
            FadeOut(caption),
            FadeIn(token_caption, shift=UP * 0.05),
            run_time=0.4,
        )
        self.wait(1.1)

        # 保留前三个 Token，散开排列，再给它们加上二维向量。
        focus_chips = VGroup(*chips[:3])
        focus_targets = [[-3.85, 0.85, 0], [-2.10, 0.85, 0], [-0.35, 0.85, 0]]
        x_panel = self._matrix_panel(
            self.X, label="X", color=config.BLUE
        )
        x_panel.move_to([3.70, 0.60, 0])

        embed_arrow = Arrow(
            [0.60, 0.95, 0],
            [2.55, 0.82, 0],
            buff=0.0,
            stroke_width=1.3,
            color=config.BLUE,
            max_tip_length_to_length_ratio=0.13,
        )
        embed_label = cn_text("嵌入", 14, config.TEXT_MUTED)
        embed_label.next_to(embed_arrow, UP, buff=0.08)

        focus_label = cn_text(
            "取前 3 个 Token、每个 2 维，作为可手算示例",
            16,
            config.TEXT_MUTED,
        )
        focus_label.move_to([-1.2, 1.95, 0])
        focus_caption = make_caption(
            "得到 3×2 的矩阵 X；后续 Attention 使用这里真实的 NumPy 数值。"
        )

        self.play(
            FadeOut(VGroup(*chips[3:])),
            FadeOut(ids),
            FadeOut(token_caption),
            FadeIn(focus_label, shift=DOWN * 0.05),
            *[
                chip.animate.move_to(target)
                for chip, target in zip(focus_chips, focus_targets)
            ],
            run_time=0.6,
        )

        focus_vectors = VGroup()
        vector_arrows = VGroup()
        for chip, row in zip(focus_chips, self.X):
            vector = mono_text(
                f"[{row[0]:.1f}, {row[1]:.1f}]", 16, config.BLUE
            )
            vector.next_to(chip.box, DOWN, buff=0.36)
            focus_vectors.add(vector)
            vector_arrows.add(
                Arrow(
                    chip.box.get_bottom(),
                    vector.get_top(),
                    buff=0.03,
                    stroke_width=0.9,
                    color=config.CYAN_DIM,
                    max_tip_length_to_length_ratio=0.35,
                )
            )

        self.play(
            FadeIn(focus_vectors, shift=UP * 0.06),
            LaggedStart(
                *[GrowArrow(arrow) for arrow in vector_arrows],
                lag_ratio=0.08,
            ),
            GrowArrow(embed_arrow),
            FadeIn(embed_label),
            FadeIn(x_panel, shift=LEFT * 0.12),
            run_time=0.85,
        )
        self.play(FadeIn(focus_caption, shift=UP * 0.05), run_time=0.45)
        self.wait(1.35)

        content = VGroup(
            header, focus_label, focus_chips, focus_vectors,
            vector_arrows, x_panel, embed_arrow, embed_label, focus_caption,
        )
        self.play(FadeOut(content), run_time=0.45)

    def _matrix_panel(self, values, *, label: str, color: str) -> MatrixPanel:
        return MatrixPanel(
            values,
            label=label,
            color=color,
            digits=1,
            cell_width=0.62,
            cell_height=0.46,
            value_font_size=17,
            label_font_size=24,
        )

    def _network_space(self) -> None:
        heading = cn_text(
            "进入抽象的神经网络空间", 24, config.CYAN, weight="SEMIBOLD"
        ).move_to([0, 2.60, 0])

        nodes = VGroup()
        for y, token in zip([1.35, 0.0, -1.35], ["The", "cat", "sat"]):
            chip = TokenChip(token, width=1.15, font_size=22)
            chip.move_to([-5.1, y, 0])
            nodes.add(chip)

        panel = RoundedRectangle(
            width=4.35,
            height=3.7,
            corner_radius=0.16,
            stroke_color=config.BLUE,
            stroke_width=1.2,
            fill_color=config.SURFACE,
            fill_opacity=0.55,
        ).move_to([0.0, 0.0, 0])

        net_dots = VGroup()
        net_edges = VGroup()
        layer_xs = [-1.4, -0.48, 0.44, 1.36]
        layer_ys = [-0.85, 0.0, 0.85]
        points = []
        for layer_x in layer_xs:
            layer_points = []
            for layer_y in layer_ys:
                point = [layer_x, layer_y, 0]
                layer_points.append(point)
                dot = Dot(point, radius=0.045, color=config.CYAN)
                dot.set_opacity(0.75)
                net_dots.add(dot)
            points.append(layer_points)
        for layer_index in range(len(points) - 1):
            for start in points[layer_index]:
                for end in points[layer_index + 1]:
                    net_edges.add(
                        Line(
                            start,
                            end,
                            stroke_color=config.BLUE_DARK,
                            stroke_width=0.55,
                            stroke_opacity=0.34,
                        )
                    )
        net = VGroup(net_edges, net_dots).move_to([0.0, -0.30, 0])
        panel_title = mono_text("TRANSFORMER", 13, config.CYAN)
        panel_title.move_to([0.0, 1.32, 0])
        panel_label = cn_text("层叠表示", 15, config.TEXT_MUTED)
        panel_label.move_to([0.0, 1.02, 0])

        output_box = RoundedRectangle(
            width=2.25,
            height=1.25,
            corner_radius=0.12,
            stroke_color=config.GREEN,
            stroke_width=1.0,
            fill_color=config.SURFACE,
            fill_opacity=0.55,
        ).move_to([4.65, 0.0, 0])
        output_title = mono_text("CONTEXT", 13, config.GREEN)
        output_title.next_to(output_box, UP, buff=0.12)
        output_text = cn_text("新的 Token 表示", 15, config.TEXT_COLOR)
        output_text.move_to(output_box)

        in_arrows = VGroup()
        for node in nodes:
            in_arrows.add(
                Arrow(
                    node.get_right() + RIGHT * 0.10,
                    [
                        panel.get_left()[0] + 0.10,
                        node.get_center()[1],
                        0,
                    ],
                    buff=0.05,
                    stroke_width=1.0,
                    color=config.CYAN_DIM,
                    max_tip_length_to_length_ratio=0.10,
                )
            )
        out_arrow = Arrow(
            panel.get_right() + RIGHT * 0.08,
            output_box.get_left() + LEFT * 0.08,
            buff=0.05,
            stroke_width=1.2,
            color=config.GREEN,
            max_tip_length_to_length_ratio=0.15,
        )

        caption = make_caption(
            "多个 Transformer Block 让 Token 在上下文中不断更新表示。"
        )
        self.play(FadeIn(heading, shift=DOWN * 0.08), run_time=0.45)
        self.play(
            FadeIn(nodes, shift=RIGHT * 0.10, lag_ratio=0.08),
            FadeIn(panel),
            FadeIn(panel_title),
            FadeIn(panel_label),
            Create(net_edges, lag_ratio=0.01),
            FadeIn(net_dots, lag_ratio=0.01),
            LaggedStart(
                *[GrowArrow(arrow) for arrow in in_arrows],
                lag_ratio=0.10,
            ),
            run_time=1.2,
        )
        self.play(
            GrowArrow(out_arrow),
            FadeIn(output_box),
            FadeIn(output_title),
            FadeIn(output_text),
            FadeIn(caption, shift=UP * 0.05),
            run_time=0.65,
        )
        self.wait(1.45)

        network = VGroup(
            heading, nodes, panel, panel_title, panel_label, net, in_arrows,
            out_arrow, output_box, output_title, output_text, caption,
        )
        self.play(FadeOut(network), run_time=0.45)

        # 引出下一章。
        chapter_tag = mono_text("NEXT  /  02", 13, config.CYAN)
        chapter_title = cn_text(
            "让 Token 互相看见", 40, config.TEXT_COLOR, weight="BOLD"
        )
        chapter_formula = formula(
            r"\mathrm{Attention}(Q,K,V)"
            r"=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
            27,
            config.TEXT_MUTED,
        )
        teaser = VGroup(chapter_tag, chapter_title, chapter_formula)
        chapter_tag.move_to([0, 1.4, 0])
        chapter_title.move_to([0, 0.55, 0])
        chapter_formula.move_to([0, -0.75, 0])
        rule = Line(
            [-1.4, 0.02, 0],
            [1.4, 0.02, 0],
            stroke_color=config.CYAN,
            stroke_width=1.4,
        )
        teaser.add(rule)
        self.play(FadeIn(teaser, shift=UP * 0.10), run_time=0.75)
        self.wait(2.0)
        self.play(FadeOut(teaser), run_time=0.45)

        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(FadeIn(black, run_time=0.55))
