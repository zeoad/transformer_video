"""第 2 章：Self-Attention 的完整数值流程。

约 70 秒。使用 3 个 Token、2 维向量的固定教学示例，所有矩阵数字来自
``math_utils.build_demo_trace`` / ``demo_qkv``，不手工编造中间结果。

结构：
1. 动机：Token 需要上下文
2. 线性投影：Q、K、V
3. QK^T 相关性分数
4. 缩放
5. 逐行 Softmax
6. 用权重汇总 V
7. 上下文表示更新与公式回顾
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

from manim import (  # noqa: E402
    AnimationGroup,
    Arrow,
    Create,
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
    SurroundingRectangle,
    UP,
    VGroup,
    Write,
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
from math_utils import build_demo_trace, demo_qkv  # noqa: E402

BLACK = "#000000"


class AttentionScene(Scene):
    """Self-Attention 全流程教学。"""

    def construct(self) -> None:
        self.camera.background_color = config.BACKGROUND
        self.trace, self.tokens = build_demo_trace()
        self.X, self.W_q, self.W_k, self.W_v = demo_qkv()
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
            2, "Self-Attention", tag_text="TRANSFORMER / 02"
        )
        self.play(
            FadeOut(black, run_time=0.6),
            FadeIn(self.header, shift=DOWN * 0.08),
            run_time=0.6,
        )

        self._opening()
        self._projections()
        self._scores()
        self._scaling()
        self._softmax()
        self._weighted_sum()
        self._context_update()
        self._recap_and_exit()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _panel(
        self,
        values,
        label: str,
        color: str,
        *,
        digits: int = 1,
        cell_width: float = 0.54,
        cell_height: float = 0.33,
        value_font_size: float = 16,
        label_font_size: float = 18,
        show_card: bool = True,
    ) -> MatrixPanel:
        return MatrixPanel(
            values,
            label=label,
            color=color,
            digits=digits,
            cell_width=cell_width,
            cell_height=cell_height,
            value_font_size=value_font_size,
            label_font_size=label_font_size,
            show_card=show_card,
        )

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
    # 开场
    # ------------------------------------------------------------------
    def _opening(self) -> None:
        kicker = mono_text(
            "SELF-ATTENTION  /  CONTEXTUAL REPRESENTATION",
            12,
            config.CYAN,
        ).move_to([0, 2.05, 0])
        title = cn_text(
            "让每个 Token 看见上下文", 38, config.TEXT_COLOR, weight="BOLD"
        ).move_to([0, 1.22, 0])
        subtitle = cn_text(
            "Attention 让每个位置从整句话中收集相关信息。",
            20,
            config.TEXT_MUTED,
        ).move_to([0, 0.60, 0])

        chips = VGroup(
            *[TokenChip(token, width=1.25, font_size=24) for token in self.tokens]
        ).arrange(RIGHT, buff=0.72).move_to([0, -0.55, 0])
        arcs = VGroup(
            Arrow(
                chips[0].get_right() + RIGHT * 0.05,
                chips[1].get_left() + LEFT * 0.05,
                buff=0.02,
                stroke_width=0.9,
                color=config.CYAN_DIM,
                max_tip_length_to_length_ratio=0.25,
            ),
            Arrow(
                chips[1].get_right() + RIGHT * 0.05,
                chips[2].get_left() + LEFT * 0.05,
                buff=0.02,
                stroke_width=0.9,
                color=config.CYAN_DIM,
                max_tip_length_to_length_ratio=0.25,
            ),
        )
        caption = make_caption(
            "每个输出位置都会整合整句话的信息，而不只看自己。"
        )
        content = VGroup(kicker, title, subtitle, chips, arcs, caption)
        self.play(
            FadeIn(kicker, shift=UP * 0.06),
            FadeIn(title, shift=UP * 0.10),
            FadeIn(subtitle, shift=UP * 0.06),
            FadeIn(chips, shift=UP * 0.10, lag_ratio=0.10),
            Create(arcs, lag_ratio=0.15),
            FadeIn(caption, shift=UP * 0.05),
            run_time=1.1,
        )
        self.wait(2.6)
        self.play(FadeOut(content), run_time=0.45)

    # ------------------------------------------------------------------
    # Step 1：线性投影
    # ------------------------------------------------------------------
    def _projections(self) -> None:
        heading = self._step_heading(
            1, "线性投影：从 X 得到 Q、K、V"
        )
        rows = [
            (
                "Q",
                self.trace.Q,
                self.W_q,
                config.BLUE,
                "Query（查询）",
                "当前 Token 用来查询相关信息",
            ),
            (
                "K",
                self.trace.K,
                self.W_k,
                config.CYAN,
                "Key（键）",
                "提供用于匹配的键表示",
            ),
            (
                "V",
                self.trace.V,
                self.W_v,
                config.AMBER,
                "Value（值）",
                "携带被加权汇总的信息",
            ),
        ]
        row_y = [1.45, -0.10, -1.65]

        row_groups = VGroup()
        animation_groups = []
        for (letter, result_values, weight_values, color, role_title, role_desc), y in zip(
            rows, row_y
        ):
            x_panel = self._panel(
                self.X, "X", config.BLUE, digits=1, cell_width=0.54, cell_height=0.32
            )
            x_panel.move_to([-4.80, y, 0])
            times = formula(r"\times", 22, config.TEXT_MUTED).move_to(
                [-3.70, y, 0]
            )
            w_panel = self._panel(
                weight_values,
                rf"W_{{{letter}}}",
                color,
                digits=1,
                cell_width=0.54,
                cell_height=0.34,
            )
            w_panel.move_to([-2.70, y, 0])
            equals = formula("=", 24, config.TEXT_MUTED).move_to(
                [-1.65, y, 0]
            )
            result_panel = self._panel(
                result_values,
                letter,
                color,
                digits=1,
                cell_width=0.56,
                cell_height=0.32,
            )
            result_panel.move_to([0.20, y, 0])

            role_heading = VGroup(
                mono_text(letter, 17, color),
                cn_text(role_title, 16, color),
            ).arrange(RIGHT, buff=0.14)
            role_detail = cn_text(role_desc, 14, config.TEXT_MUTED)
            role = VGroup(role_heading, role_detail).arrange(
                DOWN, aligned_edge=LEFT, buff=0.06
            )
            role.move_to([4.35, y, 0])
            if role.get_right()[0] > 6.5:
                role.scale_to_fit_width(3.3).move_to([4.72, y, 0])

            row_group = VGroup(
                x_panel, times, w_panel, equals, result_panel, role
            )
            row_groups.add(row_group)
            animation_groups.append(
                AnimationGroup(
                    FadeIn(x_panel, shift=RIGHT * 0.08),
                    FadeIn(times),
                    FadeIn(w_panel, shift=RIGHT * 0.08),
                    FadeIn(equals),
                    FadeIn(result_panel, shift=LEFT * 0.08),
                    FadeIn(role, shift=RIGHT * 0.06),
                )
            )

        caption = make_caption(
            "同一个 3×2 的 X 乘三组 2×2 权重；Q、K、V 仍为 3×2。"
        )
        self.play(FadeIn(heading, shift=DOWN * 0.06), run_time=0.45)
        self.play(animation_groups[0], run_time=0.9)
        self.wait(2.0)
        self.play(animation_groups[1], run_time=0.8)
        self.wait(1.7)
        self.play(animation_groups[2], run_time=0.8)
        self.wait(1.7)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(3.0)

        section = VGroup(heading, row_groups, caption)
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # Step 2：QK^T
    # ------------------------------------------------------------------
    def _scores(self) -> None:
        heading = self._step_heading(
            2, "QKᵀ：每个 Token 与每个 Token 的相关性"
        )
        formula_line = formula(
            r"S = QK^{\top}", 30, config.CYAN
        ).move_to([0, 2.20, 0])

        q_panel = self._panel(
            self.trace.Q,
            "Q",
            config.BLUE,
            digits=1,
            cell_width=0.62,
            cell_height=0.48,
            value_font_size=18,
            label_font_size=21,
        ).move_to([-4.95, -0.15, 0])
        times = formula(r"\times", 26, config.TEXT_MUTED).move_to(
            [-3.75, -0.15, 0]
        )
        kt_panel = self._panel(
            self.trace.K.T,
            r"K^{\top}",
            config.CYAN,
            digits=1,
            cell_width=0.62,
            cell_height=0.48,
            value_font_size=18,
            label_font_size=21,
        ).move_to([-2.35, -0.15, 0])
        equals = formula("=", 30, config.TEXT_MUTED).move_to(
            [-0.35, -0.15, 0]
        )
        s_panel = self._panel(
            self.trace.raw_scores,
            r"S = QK^{\top}",
            config.GREEN,
            digits=0,
            cell_width=0.82,
            cell_height=0.60,
            value_font_size=20,
            label_font_size=21,
        ).move_to([2.45, -0.15, 0])

        shape_note = mono_text(
            "(3x2)  x  (2x3)  ->  (3x3)", 14, config.TEXT_MUTED
        ).move_to([-0.60, -2.18, 0])

        q_row = VGroup(*q_panel.cell_rects[4:6])
        k_column = VGroup(
            kt_panel.cell_rects[1],
            kt_panel.cell_rects[4],
        )
        s_cell = s_panel.cell_rects[7]
        q_highlight = SurroundingRectangle(
            q_row, color=config.AMBER, buff=0.035, stroke_width=1.8
        )
        k_highlight = SurroundingRectangle(
            k_column, color=config.AMBER, buff=0.035, stroke_width=1.8
        )
        s_highlight = SurroundingRectangle(
            s_cell,
            color=config.AMBER,
            buff=0.035,
            stroke_width=1.8,
            fill_color=config.AMBER,
            fill_opacity=0.08,
        )
        dot_note = formula(
            r"S_{3,2}=Q_{\mathrm{sat}}\cdot K_{\mathrm{cat}}"
            r"=[1,1]\cdot[1,0]=1",
            21,
            config.TEXT_COLOR,
        ).move_to([1.05, -2.68, 0])

        caption = make_caption(
            "分数矩阵 S 的每一格，都是一个 Query 与一个 Key 的点积。"
        )
        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(formula_line, shift=DOWN * 0.06),
            run_time=0.55,
        )
        self.play(
            FadeIn(q_panel, shift=RIGHT * 0.10),
            FadeIn(times),
            FadeIn(kt_panel, shift=RIGHT * 0.10),
            FadeIn(equals),
            FadeIn(s_panel, shift=LEFT * 0.10),
            FadeIn(shape_note),
            run_time=0.95,
        )
        self.wait(1.5)
        self.play(
            Create(q_highlight),
            Create(k_highlight),
            Create(s_highlight),
            run_time=0.6,
        )
        self.play(FadeIn(dot_note, shift=UP * 0.06), run_time=0.65)
        self.wait(2.5)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.3)

        section = VGroup(
            heading, formula_line, q_panel, times, kt_panel, equals,
            s_panel, shape_note, q_highlight, k_highlight, s_highlight,
            dot_note, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # Step 3：缩放
    # ------------------------------------------------------------------
    def _scaling(self) -> None:
        heading = self._step_heading(
            3, "缩放：除以 √d_k，控制数值范围"
        )
        formula_line = formula(
            r"\frac{QK^{\top}}{\sqrt{d_k}}", 30, config.CYAN
        ).move_to([0, 2.20, 0])

        raw_panel = self._panel(
            self.trace.raw_scores,
            "S",
            config.GREEN,
            digits=0,
            cell_width=0.82,
            cell_height=0.50,
            value_font_size=19,
            label_font_size=21,
        ).move_to([-3.90, -0.25, 0])
        raw_tag = cn_text("原始分数", 14, config.TEXT_MUTED).next_to(
            raw_panel, DOWN, buff=0.14
        )

        scaled_panel = self._panel(
            self.trace.scaled_scores,
            r"S/\sqrt{d_k}",
            config.CYAN,
            digits=3,
            cell_width=0.82,
            cell_height=0.50,
            value_font_size=17,
            label_font_size=21,
        ).move_to([2.75, -0.25, 0])
        scaled_tag = cn_text("缩放后", 14, config.TEXT_MUTED).next_to(
            scaled_panel, DOWN, buff=0.14
        )

        arrow = Arrow(
            raw_panel.get_right() + RIGHT * 0.18,
            scaled_panel.get_left() + LEFT * 0.18,
            buff=0.0,
            stroke_width=1.7,
            color=config.CYAN,
            max_tip_length_to_length_ratio=0.12,
        )
        scale_label = formula(
            r"\div \sqrt{d_k}", 22, config.CYAN
        ).next_to(arrow, UP, buff=0.10)
        sqrt_note = formula(
            r"d_k=2,\quad \sqrt{d_k}\approx 1.414",
            18,
            config.TEXT_MUTED,
        ).move_to([0, -2.28, 0])

        caption = make_caption(
            "点积会随维度增大而变大；缩放让 Softmax 的输入保持稳定。"
        )
        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(formula_line, shift=DOWN * 0.06),
            run_time=0.5,
        )
        self.play(
            FadeIn(raw_panel, shift=RIGHT * 0.08),
            FadeIn(raw_tag),
            run_time=0.65,
        )
        self.wait(0.7)
        self.play(
            GrowArrow(arrow),
            FadeIn(scale_label),
            FadeIn(scaled_panel, shift=LEFT * 0.08),
            FadeIn(scaled_tag),
            run_time=0.85,
        )
        self.play(FadeIn(sqrt_note, shift=UP * 0.05), run_time=0.4)
        self.wait(1.8)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.0)

        section = VGroup(
            heading, formula_line, raw_panel, raw_tag, arrow, scale_label,
            scaled_panel, scaled_tag, sqrt_note, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # Step 4：Softmax
    # ------------------------------------------------------------------
    def _softmax(self) -> None:
        heading = self._step_heading(
            4, "Softmax：把每一行变成概率分布"
        )
        formula_line = formula(
            r"A_{ij}=\frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}",
            26,
            config.GREEN,
        ).move_to([0, 2.22, 0])

        left_panel = self._panel(
            self.trace.scaled_scores,
            r"S/\sqrt{d_k}",
            config.CYAN,
            digits=3,
            cell_width=0.82,
            cell_height=0.50,
            value_font_size=17,
            label_font_size=20,
        ).move_to([-4.18, -0.35, 0])
        arrow = Arrow(
            left_panel.get_right() + RIGHT * 0.15,
            [-0.35, -0.35, 0],
            buff=0.0,
            stroke_width=1.8,
            color=config.GREEN,
            max_tip_length_to_length_ratio=0.12,
        )
        arrow_label = mono_text("softmax  /  row by row", 11, config.GREEN)
        arrow_label.next_to(arrow, UP, buff=0.10)

        weights_panel = self._panel(
            self.trace.weights,
            "A",
            config.GREEN,
            digits=3,
            cell_width=0.82,
            cell_height=0.52,
            value_font_size=17,
            label_font_size=24,
        ).move_to([1.42, -0.35, 0])

        row_sum_group = VGroup()
        title = cn_text("行和", 12, config.TEXT_MUTED)
        first_row_center = VGroup(
            *weights_panel.cell_rects[0:3]
        ).get_center()
        sum_x = weights_panel.get_right()[0] + 0.38
        title.move_to([sum_x, first_row_center[1] + 0.34, 0])
        row_sum_group.add(title)
        for row_index, value in enumerate(self.trace.row_sums):
            row_center = VGroup(
                *weights_panel.cell_rects[row_index * 3 : row_index * 3 + 3]
            ).get_center()
            label = mono_text(f"{value:.3f}", 13, config.GREEN)
            label.move_to([sum_x, row_center[1], 0])
            row_sum_group.add(label)

        row_one = VGroup(*weights_panel.cell_rects[0:3])
        highlight = SurroundingRectangle(
            row_one,
            color=config.GREEN,
            buff=0.035,
            stroke_width=1.8,
            fill_color=config.GREEN,
            fill_opacity=0.06,
        )
        caption = make_caption(
            "Softmax 按行归一化：每一行权重和为 1，表示注意力如何分配。"
        )

        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(formula_line, shift=DOWN * 0.06),
            run_time=0.5,
        )
        self.play(
            FadeIn(left_panel, shift=RIGHT * 0.08),
            run_time=0.7,
        )
        self.wait(0.9)
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label),
            FadeIn(weights_panel, shift=LEFT * 0.08),
            run_time=0.9,
        )
        self.wait(1.1)
        self.play(FadeIn(row_sum_group, shift=LEFT * 0.05), run_time=0.5)
        self.play(Create(highlight), run_time=0.5)
        self.wait(1.8)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.6)

        section = VGroup(
            heading, formula_line, left_panel, arrow, arrow_label,
            weights_panel, row_sum_group, highlight, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # Step 5：加权汇总 V
    # ------------------------------------------------------------------
    def _weighted_sum(self) -> None:
        heading = self._step_heading(
            5, "用注意力权重汇总 V，得到新的 Token 表示"
        )
        formula_line = formula(
            r"Z = AV", 30, config.GREEN
        ).move_to([0, 2.20, 0])

        a_panel = self._panel(
            self.trace.weights,
            "A",
            config.GREEN,
            digits=3,
            cell_width=0.82,
            cell_height=0.48,
            value_font_size=17,
            label_font_size=22,
        ).move_to([-4.60, -0.15, 0])
        times = formula(r"\times", 26, config.TEXT_MUTED).move_to(
            [-2.78, -0.15, 0]
        )
        v_panel = self._panel(
            self.trace.V,
            "V",
            config.AMBER,
            digits=1,
            cell_width=0.65,
            cell_height=0.48,
            value_font_size=18,
            label_font_size=20,
        ).move_to([-1.55, -0.15, 0])
        equals = formula("=", 28, config.TEXT_MUTED).move_to(
            [0.15, -0.15, 0]
        )
        z_panel = self._panel(
            self.trace.output,
            "Z",
            config.BLUE,
            digits=3,
            cell_width=0.92,
            cell_height=0.50,
            value_font_size=18,
            label_font_size=22,
        ).move_to([2.15, -0.15, 0])

        a_row = VGroup(*a_panel.cell_rects[0:3])
        z_row = VGroup(*z_panel.cell_rects[0:2])
        a_highlight = SurroundingRectangle(
            a_row,
            color=config.GREEN,
            buff=0.035,
            stroke_width=1.8,
            fill_color=config.GREEN,
            fill_opacity=0.06,
        )
        z_highlight = SurroundingRectangle(
            z_row,
            color=config.BLUE,
            buff=0.035,
            stroke_width=1.8,
            fill_color=config.BLUE,
            fill_opacity=0.06,
        )
        row_note = formula(
            r"z_1=0.198v_1+0.401v_2+0.401v_3=[1.102,\ 1.000]",
            21,
            config.TEXT_COLOR,
        ).move_to([0.0, -2.35, 0])
        shape_note = mono_text(
            "A: 3x3    V: 3x2    Z = AV: 3x2", 13, config.TEXT_MUTED
        ).move_to([0.0, -1.65, 0])

        caption = make_caption(
            "Z 的每一行，是用这一行的注意力权重对所有 Value 做加权求和。"
        )
        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(formula_line, shift=DOWN * 0.06),
            run_time=0.5,
        )
        self.play(
            FadeIn(a_panel, shift=RIGHT * 0.08),
            FadeIn(times),
            FadeIn(v_panel, shift=RIGHT * 0.08),
            FadeIn(equals),
            FadeIn(z_panel, shift=LEFT * 0.08),
            FadeIn(shape_note),
            run_time=1.0,
        )
        self.wait(1.4)
        self.play(
            Create(a_highlight),
            Create(z_highlight),
            run_time=0.6,
        )
        self.play(FadeIn(row_note, shift=UP * 0.06), run_time=0.8)
        self.wait(2.2)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(2.8)

        section = VGroup(
            heading, formula_line, a_panel, times, v_panel, equals,
            z_panel, shape_note, a_highlight, z_highlight, row_note, caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # Step 6：上下文更新
    # ------------------------------------------------------------------
    def _context_update(self) -> None:
        heading = self._step_heading(
            6, "上下文更新：每个 Token 都吸收了整句话的信息"
        )
        input_label = cn_text("输入表示", 15, config.TEXT_MUTED).move_to(
            [-4.45, 2.05, 0]
        )
        output_label = cn_text(
            "上下文表示", 15, config.TEXT_MUTED
        ).move_to([4.30, 2.05, 0])

        inputs = VGroup()
        outputs = VGroup()
        for y, token in zip([1.25, 0.0, -1.25], self.tokens):
            inputs.add(
                TokenChip(token, width=1.25, font_size=22).move_to(
                    [-4.45, y, 0]
                )
            )
            outputs.add(
                TokenChip(
                    token, width=1.25, font_size=22, color=config.GREEN
                ).move_to([4.30, y, 0])
            )

        attention_box = RoundedRectangle(
            width=3.20,
            height=3.25,
            corner_radius=0.14,
            stroke_color=config.CYAN,
            stroke_width=1.2,
            fill_color=config.SURFACE,
            fill_opacity=0.65,
        ).move_to([0.0, 0.0, 0])
        attention_text = mono_text("SELF-ATTENTION", 16, config.CYAN)
        attention_text.move_to(attention_box)

        input_arrows = VGroup()
        output_arrows = VGroup()
        for node, out_node in zip(inputs, outputs):
            input_arrows.add(
                Arrow(
                    node.get_right() + RIGHT * 0.08,
                    [
                        attention_box.get_left()[0] + 0.08,
                        node.get_center()[1],
                        0,
                    ],
                    buff=0.04,
                    stroke_width=1.1,
                    color=config.CYAN_DIM,
                    max_tip_length_to_length_ratio=0.12,
                )
            )
            output_arrows.add(
                Arrow(
                    [
                        attention_box.get_right()[0] - 0.08,
                        out_node.get_center()[1],
                        0,
                    ],
                    out_node.get_left() + LEFT * 0.08,
                    buff=0.04,
                    stroke_width=1.1,
                    color=config.GREEN,
                    max_tip_length_to_length_ratio=0.12,
                )
            )

        caption = make_caption(
            "输出的每一行，都是融合上下文后的新 Token 表示。"
        )
        self.play(
            FadeIn(heading, shift=DOWN * 0.06),
            FadeIn(input_label),
            FadeIn(output_label),
            FadeIn(inputs, shift=RIGHT * 0.08, lag_ratio=0.08),
            FadeIn(outputs, shift=LEFT * 0.08, lag_ratio=0.08),
            FadeIn(attention_box),
            FadeIn(attention_text),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[GrowArrow(arrow) for arrow in input_arrows],
                lag_ratio=0.08,
            ),
            LaggedStart(
                *[GrowArrow(arrow) for arrow in output_arrows],
                lag_ratio=0.08,
            ),
            run_time=0.8,
        )
        self.wait(1.2)
        self.play(FadeIn(caption, shift=UP * 0.05), run_time=0.5)
        self.wait(1.8)

        section = VGroup(
            heading, input_label, output_label, inputs, outputs,
            attention_box, attention_text, input_arrows, output_arrows,
            caption,
        )
        self.play(FadeOut(section), run_time=0.5)

    # ------------------------------------------------------------------
    # 回顾与离场
    # ------------------------------------------------------------------
    def _recap_and_exit(self) -> None:
        title = cn_text(
            "一次完整的缩放点积注意力", 22, config.CYAN, weight="SEMIBOLD"
        ).move_to([0, 1.25, 0])
        recap = formula(
            r"X\ \rightarrow\ Q,K,V\ \rightarrow\ "
            r"\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"
            r"\ \rightarrow\ Z",
            28,
            config.TEXT_COLOR,
        ).move_to([0, 0.25, 0])
        note = make_caption(
            "下一章：把这些计算封装成一个可堆叠的 Transformer Block。"
        )
        self.play(
            FadeIn(title, shift=UP * 0.08),
            Write(recap),
            FadeIn(note, shift=UP * 0.05),
            run_time=1.0,
        )
        self.wait(2.3)
        content = VGroup(title, recap, note, self.header)
        self.play(FadeOut(content), run_time=0.5)

        black = Rectangle(
            width=14.5,
            height=8.1,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1.0,
        )
        self.play(FadeIn(black, run_time=0.55))
