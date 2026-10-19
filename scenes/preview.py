"""阶段 2：约 30 秒的 Transformer 技术预览。

预览包含：电影感开场、Token/向量、Q/K/V 投影、一次真实 Attention 数值计算、
简洁 Transformer Block 结构，以及预览结束卡。

所有矩阵数值来自 ``math_utils.build_demo_trace()``，没有手填近似结果。
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
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    Rectangle,
    RIGHT,
    Scene,
    SurroundingRectangle,
    UP,
    VGroup,
    Write,
)

from components import (  # noqa: E402
    MatrixPanel,
    TokenChip,
    TransformerBlockDiagram,
    cn_text,
    formula,
    make_backdrop,
    make_caption,
    make_legend_item,
    make_section_header,
    mono_text,
)
from math_utils import build_demo_trace, demo_qkv  # noqa: E402

BLACK = "#000000"


class TransformerPreview(Scene):
    """阶段 2 的约 30 秒单场景预览。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND
        self.trace, self.tokens = build_demo_trace()
        self.X, _, _, _ = demo_qkv()
        self.add(make_backdrop())
        self._intro()
        self._tokens_to_vectors()
        self._qkv()
        self._attention_math()
        self._transformer_block()
        self._closing()

    # ------------------------------------------------------------------
    # 0. 开场
    # ------------------------------------------------------------------
    def _intro(self) -> None:
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
        tag = mono_text("TECHNICAL PREVIEW  /  30 s", 12, config.CYAN_DIM)
        tag.to_corner(UP + RIGHT, buff=0.55)

        kicker = mono_text(
            "A JOURNEY THROUGH SELF-ATTENTION", 13, config.CYAN
        ).move_to([0, 1.65, 0])
        title = cn_text(
            "一次注意力的旅程", 58, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 0.62, 0])
        rule = Line(
            [-1.5, -0.12, 0],
            [1.5, -0.12, 0],
            stroke_color=config.CYAN,
            stroke_width=1.7,
            stroke_opacity=0.9,
        )
        subtitle = cn_text(
            "从一句话，到下一个 Token", 24, config.TEXT_MUTED
        ).move_to([0, -0.75, 0])
        footer = mono_text(
            "SELF-ATTENTION  /  TRANSFORMER  /  NEXT TOKEN",
            12,
            config.CYAN_DIM,
        ).move_to([0, -2.35, 0])

        intro = VGroup(brand, tag, kicker, title, rule, subtitle, footer)
        self.play(
            FadeOut(black, run_time=1.1),
            FadeIn(brand, shift=DOWN * 0.1),
            FadeIn(tag, shift=DOWN * 0.1),
            FadeIn(kicker, shift=UP * 0.1),
            FadeIn(title, shift=UP * 0.15),
            Create(rule),
            FadeIn(subtitle, shift=UP * 0.08),
            FadeIn(footer),
            run_time=1.45,
        )
        self.wait(2.35)
        self.play(FadeOut(intro), run_time=0.45)

    # ------------------------------------------------------------------
    # 1. 文本 → Token → 向量
    # ------------------------------------------------------------------
    def _tokens_to_vectors(self) -> None:
        header = make_section_header(1, "文本 → Token → 向量")
        self.play(FadeIn(header, shift=DOWN * 0.08), run_time=0.45)

        sentence = mono_text("The cat sat", 40, config.TEXT_COLOR)
        sentence.move_to([0, 2.12, 0])
        source_tag = mono_text("INPUT TEXT", 12, config.TEXT_MUTED)
        source_tag.next_to(sentence, UP, buff=0.18)
        self.play(
            FadeIn(source_tag, shift=UP * 0.08),
            FadeIn(sentence, shift=UP * 0.10),
            run_time=0.65,
        )
        self.wait(0.35)

        chips = VGroup(
            *[
                TokenChip(token, color=config.BLUE, width=1.35)
                for token in self.tokens
            ]
        ).arrange(RIGHT, buff=0.38)
        chips.move_to([-3.35, 0.98, 0])

        vectors = VGroup()
        down_arrows = VGroup()
        # Token 的向量表示来自输入嵌入 X，而不是 Query 投影 Q。
        for chip, row in zip(chips, self.X):
            vector_label = mono_text(
                f"[{row[0]:.1f}, {row[1]:.1f}]", 17, config.BLUE
            )
            vector_label.next_to(chip.box, DOWN, buff=0.34)
            vectors.add(vector_label)
            down_arrows.add(
                Arrow(
                    chip.box.get_bottom(),
                    vector_label.get_top(),
                    buff=0.04,
                    stroke_width=1.0,
                    color=config.CYAN_DIM,
                    max_tip_length_to_length_ratio=0.4,
                )
            )

        self.play(
            FadeOut(sentence, shift=DOWN * 0.08),
            FadeOut(source_tag),
            FadeIn(chips, shift=DOWN * 0.10, lag_ratio=0.12),
            run_time=0.75,
        )
        self.play(
            FadeIn(vectors, shift=UP * 0.08),
            LaggedStart(
                *[GrowArrow(arrow) for arrow in down_arrows],
                lag_ratio=0.12,
            ),
            run_time=0.55,
        )

        x_panel = MatrixPanel(
            self.X,
            label="X",
            color=config.BLUE,
            digits=1,
            cell_width=0.72,
            cell_height=0.55,
            value_font_size=19,
            label_font_size=26,
        )
        x_panel.move_to([3.62, 0.50, 0])

        embed_arrow = Arrow(
            [-0.55, 0.98, 0],
            [2.52, 0.82, 0],
            buff=0.0,
            stroke_width=1.5,
            color=config.BLUE,
            max_tip_length_to_length_ratio=0.12,
        )
        embed_label = cn_text("嵌入", 15, config.TEXT_MUTED)
        embed_label.next_to(embed_arrow, UP, buff=0.08)

        caption = make_caption(
            "Token 被映射为向量矩阵 X；这是固定教学示例，不是真实模型输出。"
        )
        self.play(
            GrowArrow(embed_arrow),
            FadeIn(embed_label),
            FadeIn(x_panel, shift=LEFT * 0.15),
            FadeIn(caption, shift=UP * 0.08),
            run_time=0.85,
        )
        self.wait(0.85)

        section = VGroup(
            header, chips, vectors, down_arrows, x_panel,
            embed_arrow, embed_label, caption,
        )
        self.play(FadeOut(section), run_time=0.45)

    # ------------------------------------------------------------------
    # 2. 线性投影得到 Q / K / V
    # ------------------------------------------------------------------
    def _qkv(self) -> None:
        header = make_section_header(2, "Q / K / V 线性投影")
        self.play(FadeIn(header, shift=DOWN * 0.08), run_time=0.4)

        subheading = cn_text(
            "同一个 X，乘以三组可学习权重", 18, config.TEXT_MUTED
        ).move_to([0, 2.68, 0])
        self.play(FadeIn(subheading, shift=DOWN * 0.06), run_time=0.35)

        x_panel = MatrixPanel(
            self.X,
            label="X",
            color=config.BLUE,
            digits=1,
            cell_width=0.58,
            cell_height=0.31,
            value_font_size=17,
            label_font_size=21,
        )
        x_panel.move_to([-5.05, -0.08, 0])

        colors = [config.BLUE, config.CYAN, config.AMBER]
        labels = ["Q", "K", "V"]
        weights = [config.BLUE_DARK, config.CYAN_DIM, config.AMBER]
        row_y = [1.52, -0.12, -1.76]
        equations = VGroup()
        matrices = VGroup()
        arrows = VGroup()
        for index, (letter, color, y) in enumerate(
            zip(labels, colors, row_y)
        ):
            equation = formula(
                rf"{letter} = XW_{{{letter}}}", 30, color
            ).move_to([-2.30, y, 0])
            panel = MatrixPanel(
                [self.trace.Q, self.trace.K, self.trace.V][index],
                label=letter,
                color=color,
                digits=1,
                cell_width=0.58,
                cell_height=0.31,
                value_font_size=17,
                label_font_size=21,
            )
            panel.move_to([3.10, y, 0])
            arrow = Arrow(
                equation.get_right() + RIGHT * 0.10,
                panel.get_left() + LEFT * 0.10,
                buff=0.0,
                stroke_width=1.3,
                color=weights[index],
                max_tip_length_to_length_ratio=0.08,
            )
            equations.add(equation)
            matrices.add(panel)
            arrows.add(arrow)

        self.play(
            FadeIn(x_panel, shift=RIGHT * 0.10),
            FadeIn(equations[0], shift=RIGHT * 0.10),
            FadeIn(matrices[0]),
            GrowArrow(arrows[0]),
            run_time=0.7,
        )
        self.play(
            FadeIn(equations[1], shift=RIGHT * 0.10),
            FadeIn(matrices[1]),
            GrowArrow(arrows[1]),
            run_time=0.65,
        )
        self.play(
            FadeIn(equations[2], shift=RIGHT * 0.10),
            FadeIn(matrices[2]),
            GrowArrow(arrows[2]),
            run_time=0.65,
        )

        # 只在可读的位置展示简短角色说明。
        role_line = VGroup(
            cn_text("Q 查询", 16, config.BLUE),
            mono_text("  /  ", 16, config.TEXT_MUTED),
            cn_text("K 键", 16, config.CYAN),
            mono_text("  /  ", 16, config.TEXT_MUTED),
            cn_text("V 值", 16, config.AMBER),
            cn_text("   —— 同一 X 的三种投影", 15, config.TEXT_MUTED),
        ).arrange(RIGHT, buff=0.04)
        role_line.move_to([0, -2.70, 0])

        caption = make_caption(
            "三个矩阵的形状均为 3×2；数值由 X @ W_Q、X @ W_K、X @ W_V 实际计算。"
        )
        self.play(
            FadeIn(role_line, shift=UP * 0.08),
            FadeIn(caption, shift=UP * 0.08),
            run_time=0.55,
        )
        self.wait(0.95)

        section = VGroup(
            header, subheading, x_panel, equations, matrices, arrows,
            role_line, caption,
        )
        self.play(FadeOut(section), run_time=0.45)

    # ------------------------------------------------------------------
    # 3. QK^T → 缩放 → Softmax（真实数值）
    # ------------------------------------------------------------------
    def _attention_math(self) -> None:
        header = make_section_header(3, "真实计算：从 QK^T 到 Softmax")
        self.play(FadeIn(header, shift=DOWN * 0.08), run_time=0.4)

        attention_formula = formula(
            r"\mathrm{Attention}(Q,K,V)"
            r"=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
            28,
        ).move_to([0, 2.62, 0])

        step1 = mono_text("1  QK^T", 14, config.CYAN)
        arrow_12 = mono_text(">", 14, config.CYAN_DIM)
        step2 = mono_text("2  / sqrt(dk)", 14, config.TEXT_MUTED)
        arrow_23 = mono_text(">", 14, config.CYAN_DIM)
        step3 = mono_text("3  softmax", 14, config.TEXT_MUTED)
        steps = VGroup(step1, arrow_12, step2, arrow_23, step3)
        steps.arrange(RIGHT, buff=0.22)
        steps.move_to([0, 2.14, 0])

        self.play(
            Write(attention_formula),
            FadeIn(steps, shift=DOWN * 0.05),
            run_time=0.75,
        )

        q_panel = MatrixPanel(
            self.trace.Q,
            label="Q",
            color=config.BLUE,
            digits=1,
            cell_width=0.55,
            cell_height=0.45,
            value_font_size=17,
            label_font_size=21,
        )
        q_panel.move_to([-5.15, -0.42, 0])

        kt_panel = MatrixPanel(
            self.trace.K.T,
            label=r"K^{\top}",
            color=config.CYAN,
            digits=1,
            cell_width=0.55,
            cell_height=0.45,
            value_font_size=17,
            label_font_size=21,
        )
        kt_panel.move_to([-2.62, -0.42, 0])

        times = formula(r"\times", 30, config.TEXT_MUTED)
        times.move_to([-4.03, -0.42, 0])
        equals = formula("=", 34, config.TEXT_MUTED)
        equals.move_to([-0.52, -0.42, 0])

        s_panel = MatrixPanel(
            self.trace.raw_scores,
            label=r"S = QK^{\top}",
            color=config.BLUE,
            digits=0,
            cell_width=0.84,
            cell_height=0.50,
            value_font_size=18,
            label_font_size=21,
        )
        s_panel.move_to([1.72, -0.42, 0])

        # 预构造同一位置的缩放矩阵与权重矩阵。阶段切换时不修改面板内部
        # MathTex（那会触发不期望的整块染色），而是干净地淡出/淡入新面板。
        scaled_panel = MatrixPanel(
            self.trace.scaled_scores,
            label=r"S/\sqrt{d_k}",
            color=config.CYAN,
            digits=3,
            cell_width=0.84,
            cell_height=0.50,
            value_font_size=18,
            label_font_size=21,
        )
        scaled_panel.move_to([1.72, -0.42, 0])

        weights_panel = MatrixPanel(
            self.trace.weights,
            label="A",
            color=config.GREEN,
            digits=3,
            cell_width=0.84,
            cell_height=0.50,
            value_font_size=18,
            label_font_size=25,
        )
        weights_panel.move_to([1.72, -0.42, 0])

        shape_note = mono_text(
            "(3x2)  x  (2x3)  ->  (3x3)", 13, config.TEXT_MUTED
        ).move_to([-1.90, -2.24, 0])
        sqrt_note = formula(
            r"d_k=2,\quad \sqrt{d_k}\approx 1.414", 18, config.TEXT_MUTED
        ).move_to([2.25, -2.22, 0])

        self.play(
            FadeIn(q_panel, shift=RIGHT * 0.10),
            FadeIn(kt_panel, shift=RIGHT * 0.10),
            FadeIn(times),
            FadeIn(equals),
            FadeIn(s_panel, shift=LEFT * 0.10),
            FadeIn(shape_note),
            run_time=0.9,
        )
        self.wait(0.9)

        caption = make_caption(
            "教学示例：先缩放，再按行 Softmax；所有数值均来自 NumPy 实际计算。"
        )
        self.play(FadeIn(caption, shift=UP * 0.06), run_time=0.4)

        self.play(
            step1.animate.set_color(config.CYAN_DIM),
            step2.animate.set_color(config.CYAN),
            FadeOut(s_panel, run_time=0.20),
            run_time=0.20,
        )
        self.play(
            FadeIn(scaled_panel, run_time=0.38),
            FadeIn(sqrt_note, run_time=0.38),
            run_time=0.38,
        )
        self.wait(0.75)

        self.play(
            step2.animate.set_color(config.CYAN_DIM),
            step3.animate.set_color(config.GREEN),
            FadeOut(scaled_panel, run_time=0.20),
            run_time=0.20,
        )
        self.play(FadeIn(weights_panel, run_time=0.38), run_time=0.38)
        self.wait(0.35)

        last_row = VGroup(*weights_panel.cell_rects[6:9])
        row_highlight = SurroundingRectangle(
            last_row,
            color=config.GREEN,
            buff=0.035,
            stroke_width=1.8,
            fill_color=config.GREEN,
            fill_opacity=0.06,
        )
        row_sum_group = VGroup()
        row_sum_title = cn_text("行和", 12, config.TEXT_MUTED)
        sum_x = weights_panel.get_right()[0] + 0.38
        first_row_center = VGroup(*weights_panel.cell_rects[0:3]).get_center()
        row_sum_title.move_to([sum_x, first_row_center[1] + 0.32, 0])
        row_sum_group.add(row_sum_title)
        for row_index, value in enumerate(self.trace.row_sums):
            row_center = VGroup(
                *weights_panel.cell_rects[row_index * 3 : row_index * 3 + 3]
            ).get_center()
            sum_label = mono_text(f"{value:.3f}", 13, config.GREEN)
            sum_label.move_to([sum_x, row_center[1], 0])
            row_sum_group.add(sum_label)

        self.play(
            Create(row_highlight),
            FadeIn(row_sum_group, shift=LEFT * 0.08),
            run_time=0.6,
        )
        self.wait(0.95)

        section = VGroup(
            header, attention_formula, steps, q_panel, kt_panel, times,
            equals, weights_panel, shape_note, sqrt_note, caption,
            row_highlight, row_sum_group,
        )
        self.play(FadeOut(section), run_time=0.45)

    # ------------------------------------------------------------------
    # 4. Transformer Block 结构
    # ------------------------------------------------------------------
    def _transformer_block(self) -> None:
        header = make_section_header(4, "Transformer Block")
        self.play(FadeIn(header, shift=DOWN * 0.08), run_time=0.4)

        diagram = TransformerBlockDiagram(
            x=-2.45, top_y=2.08, pitch=0.90, node_width=3.25
        )
        sequence = []
        for index, node in enumerate(diagram.node_groups):
            sequence.append(FadeIn(node, shift=UP * 0.08))
            if index < len(diagram.arrows):
                sequence.append(Create(diagram.arrows[index]))
        self.play(
            LaggedStart(*sequence, lag_ratio=0.08, run_time=1.45)
        )
        self.play(Create(diagram.residuals), run_time=0.7)

        heading = cn_text(
            "四个关键部件", 18, config.CYAN, weight="SEMIBOLD"
        ).move_to([3.35, 2.38, 0])
        items = VGroup(
            make_legend_item(
                "Multi-Head Attention",
                "从多个表示子空间学习信息关系",
                config.BLUE,
            ),
            make_legend_item(
                "Feed-Forward Network",
                "进一步变换每个位置的特征",
                config.AMBER,
            ),
            make_legend_item(
                "Residual Connection",
                "为信息与梯度提供传播通路",
                config.CYAN,
            ),
            make_legend_item(
                "Layer Normalization",
                "对特征进行归一化",
                config.CYAN,
            ),
        )
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        items.move_to([3.35, 0.45, 0])

        self.play(
            FadeIn(heading, shift=DOWN * 0.08),
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.08) for item in items],
                lag_ratio=0.15,
                run_time=1.0,
            ),
        )

        caption = make_caption(
            "结构固定：Attention → 残差与归一化 → FFN → 残差与归一化。"
        )
        self.play(FadeIn(caption, shift=UP * 0.06), run_time=0.4)
        self.play(
            Indicate(diagram.node_boxes[1], color=config.BLUE, scale_factor=1.05),
            Indicate(diagram.node_boxes[3], color=config.AMBER, scale_factor=1.05),
            Indicate(diagram.node_boxes[2], color=config.CYAN, scale_factor=1.03),
            Indicate(diagram.node_boxes[4], color=config.CYAN, scale_factor=1.03),
            run_time=1.0,
        )
        self.wait(1.0)

        section = VGroup(header, diagram, heading, items, caption)
        self.play(FadeOut(section), run_time=0.45)

    # ------------------------------------------------------------------
    # 5. 结尾卡
    # ------------------------------------------------------------------
    def _closing(self) -> None:
        tag = mono_text(
            "PREVIEW COMPLETE  /  MATH VERIFIED", 13, config.CYAN
        ).move_to([0, 1.55, 0])
        title = cn_text(
            "30 秒预览完成", 46, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 0.55, 0])
        rule = Line(
            [-1.8, 0.0, 0],
            [1.8, 0.0, 0],
            stroke_color=config.CYAN,
            stroke_width=1.6,
            stroke_opacity=0.9,
        )
        subtitle = cn_text(
            "Token 与向量 · Q/K/V · Softmax · Transformer Block",
            21,
            config.TEXT_MUTED,
        ).move_to([0, -0.72, 0])
        footnote = cn_text(
            "完整 3 分钟正片将在预览验收后继续制作。",
            17,
            config.CYAN_DIM,
        ).move_to([0, -2.15, 0])
        closing = VGroup(tag, title, rule, subtitle, footnote)

        self.play(FadeIn(closing, shift=UP * 0.12), run_time=0.9)
        self.wait(1.8)

        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(FadeIn(black, run_time=0.75))
