# -*- coding: utf-8 -*-
"""
report.py — 图表生成与综合风险报告
=====================================

本模块将 :class:`~var_garch.portfolio.Portfolio` 与
:class:`~var_garch.stress.StressTester` 的计算结果可视化，
生成以下图表：

1. **收益率分布直方图** — 带 VaR/ES 切割线（各资产分面）
2. **GARCH 条件波动率时序图** — 多资产叠加
3. **多方法 VaR 对比图** — 参数法/历史法/FHS/MC 横向对比
4. **成分 VaR 条形图** — 组合风险贡献分解
5. **相关矩阵热力图** — 标注数值
6. **历史情景压力测试图** — 损失 vs 基准 VaR
7. **因子敏感性曲线** — P&L 关于因子冲击的函数
8. **相关性压力测试曲线** — VaR 与分散化收益随 ρ 的变化

Notes
-----
matplotlib 中文字体依赖系统安装的中文字体（SimHei/Heiti SC/WenQuanYi），
若无法渲染，图表标题将回退至英文。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from var import VaRResult
from .stress     import StressTester


# 全局 matplotlib 中文字体配置（自动回退至 DejaVu Sans）
_CN_FONTS = ["SimHei", "Heiti SC", "WenQuanYi Micro Hei", "DejaVu Sans"]


def _set_mpl_style() -> None:
    """配置 matplotlib 绘图风格：中文字体 + 简洁样式。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif":        _CN_FONTS,
        "axes.unicode_minus":     False,    # 负号显示修正
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.grid":              True,
        "grid.alpha":             0.25,
        "grid.linestyle":         "--",
        "figure.dpi":             120,
        "axes.titlesize":         11,
        "axes.labelsize":         9,
    })


class RiskReport:
    """综合风险报告生成器。

    Parameters
    ----------
    portfolio : Portfolio
        已完成 ``fit(prices)`` 的组合对象。

    Examples
    --------
    >>> report = RiskReport(port)
    >>>
    >>> # 生成全部 VaR 图表
    >>> report.plot_all(output_dir="./figures", show=False)
    >>>
    >>> # 一键运行完整压力测试套件
    >>> report.stress_report(output_dir="./figures", show=False)
    >>>
    >>> # 导出 CSV
    >>> report.to_csv("var_results.csv")
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio
        self._results: Dict[str, VaRResult] = portfolio._per_asset

    # ------------------------------------------------------------------
    # 文字摘要
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """打印组合风险摘要（调用 Portfolio.summary()）。"""
        print(self.portfolio.summary())
        print("\n单资产 VaR 详情：")
        for result in self._results.values():
            print(result)

    def to_csv(self, path: str = "var_results.csv") -> pd.DataFrame:
        """将风险指标导出为 CSV 文件。

        Parameters
        ----------
        path : str
            输出路径。

        Returns
        -------
        pd.DataFrame
            导出的 DataFrame。
        """
        df = self.portfolio.to_dataframe()
        df.to_csv(path, encoding="utf-8-sig")   # utf-8-sig 确保 Excel 正常显示中文
        print(f"已保存：{path}")
        return df

    # ------------------------------------------------------------------
    # VaR 图表套件
    # ------------------------------------------------------------------

    def plot_all(
        self,
        output_dir:    str   = ".",
        show:          bool  = True,
        figsize_base:  tuple = (10, 5),
    ) -> List[str]:
        """生成并保存所有 VaR 分析图表。

        Parameters
        ----------
        output_dir : str
            图表保存目录（不存在时自动创建）。
        show : bool
            是否调用 plt.show() 交互展示。
        figsize_base : tuple
            基础图幅尺寸（宽度, 高度），部分图表会基于此缩放。

        Returns
        -------
        list of str
            所有已保存图表的路径列表。
        """
        import matplotlib.pyplot as plt

        _set_mpl_style()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = []

        saved.append(self._plot_return_distributions(plt, output_dir, show, figsize_base))
        saved.append(self._plot_garch_vol(plt, output_dir, show, figsize_base))
        saved.append(self._plot_var_comparison(plt, output_dir, show, figsize_base))
        saved.append(self._plot_component_var(plt, output_dir, show))
        saved.append(self._plot_correlation(plt, output_dir, show))

        return [s for s in saved if s]

    def _plot_return_distributions(self, plt, output_dir, show, figsize_base):
        """各资产收益率分布直方图，标注 VaR（红虚线）与 ES（橙点线）切割位置。"""
        n    = len(self.portfolio.tickers)
        cols = min(n, 2)
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(figsize_base[0], figsize_base[1] * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for ax, ticker in zip(axes, self.portfolio.tickers):
            result = self._results[ticker]
            rets   = self.portfolio._returns[ticker].values
            var_p  = result.var_parametric

            # 核密度估计曲线
            ax.hist(rets * 100, bins=60, color="#378ADD", alpha=0.65,
                    edgecolor="none", density=True, label="经验分布")
            ax.axvline(-var_p * 100, color="#E24B4A", lw=1.5, linestyle="--",
                       label=f"GARCH VaR {result.confidence*100:.0f}%")
            ax.axvline(-result.es_historical * 100, color="#BA7517", lw=1.2,
                       linestyle=":", label="ES (CVaR)")
            ax.set_title(f"{ticker} — 收益率分布")
            ax.set_xlabel("日收益率 (%)")
            ax.set_ylabel("密度")
            ax.legend(fontsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle("各资产收益率分布与 VaR/ES 切割线",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = str(Path(output_dir) / "return_distributions.png")
        fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{path}")
        return path

    def _plot_garch_vol(self, plt, output_dir, show, figsize_base):
        """各资产 GARCH 条件波动率时序图（年化百分比）。"""
        fig, ax = plt.subplots(figsize=figsize_base)
        colors  = ["#E24B4A", "#378ADD", "#3B6D11", "#BA7517", "#534AB7", "#0F6E56"]

        for i, ticker in enumerate(self.portfolio.tickers):
            g   = self._results[ticker].garch
            vol = g.conditional_vol * np.sqrt(252) * 100   # 年化%
            ax.plot(vol.index, vol.values,
                    label=ticker, color=colors[i % len(colors)],
                    lw=1.2, alpha=0.85)

        ax.set_title("GARCH 条件波动率（年化）", fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("波动率 (%/年)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = str(Path(output_dir) / "garch_volatility.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{path}")
        return path

    def _plot_var_comparison(self, plt, output_dir, show, figsize_base):
        """四种 VaR 方法横向对比条形图。"""
        methods = ["参数法 GARCH", "历史模拟法", "FHS", "蒙特卡洛"]
        tickers = self.portfolio.tickers
        x       = np.arange(len(tickers))
        width   = 0.20
        colors  = ["#E24B4A", "#378ADD", "#3B6D11", "#BA7517"]

        fig, ax = plt.subplots(figsize=figsize_base)
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = []
            for t in tickers:
                r = self._results[t]
                v = {
                    "参数法 GARCH": r.var_parametric,
                    "历史模拟法":   r.var_historical,
                    "FHS":          r.var_fhs,
                    "蒙特卡洛":     r.var_montecarlo,
                }[method]
                vals.append(v * 100)
            ax.bar(x + i * width, vals, width, label=method, color=color, alpha=0.82)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(tickers)
        ax.set_title("VaR 四方法对比", fontweight="bold")
        ax.set_ylabel("VaR (%)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = str(Path(output_dir) / "var_comparison.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{path}")
        return path

    def _plot_component_var(self, plt, output_dir, show):
        """成分 VaR（Component VaR）水平条形图。"""
        rc     = self.portfolio.risk_contributions * 100
        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in rc.values]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars    = ax.barh(rc.index, rc.values, color=colors, alpha=0.85)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("成分 VaR (%)")
        ax.set_title("成分 VaR 贡献分解", fontweight="bold")
        for bar, val in zip(bars, rc.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}%", va="center", fontsize=9)
        fig.tight_layout()
        path = str(Path(output_dir) / "component_var.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{path}")
        return path

    def _plot_correlation(self, plt, output_dir, show):
        """收益率相关矩阵热力图（标注数值）。"""
        corr   = self.portfolio.correlation_matrix

        fig, ax = plt.subplots(figsize=(6, 5))
        im      = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.index, fontsize=9)

        # 在每个格子内标注相关系数值
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                rho   = corr.iloc[i, j]
                color = "white" if abs(rho) > 0.5 else "black"
                ax.text(j, i, f"{rho:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label="相关系数 ρ")
        ax.set_title("资产收益率相关矩阵", fontweight="bold")
        fig.tight_layout()
        path = str(Path(output_dir) / "correlation_matrix.png")
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{path}")
        return path

    # ------------------------------------------------------------------
    # 完整压力测试报告套件
    # ------------------------------------------------------------------

    def stress_report(
        self,
        output_dir:       str   = ".",
        show:             bool  = False,
        custom_scenarios: Optional[List[Dict]] = None,
    ) -> Dict:
        """一键运行完整压力测试套件并生成全部图表。

        执行流程：
        1. 全部 8 个内置历史情景
        2. 可选的自定义假设情景
        3. 股票市场与利率两个因子的敏感性扫描
        4. 因子梯度分析（五大宏观因子各 -5% 冲击）
        5. 反向压力测试（10% 损失阈值）
        6. 相关性压力测试（压力相关系数至 0.95）

        Parameters
        ----------
        output_dir : str
            所有图表的保存目录。
        show : bool
            是否交互展示（批量运行时建议 False）。
        custom_scenarios : list of dict, optional
            额外假设情景，格式同
            :meth:`~var_garch.stress.StressTester.run_multiple_hypothetical`。

        Returns
        -------
        dict
            键：``historical``、``custom``、``sensitivity_equity``、
            ``sensitivity_rates``、``factor_ladder``、
            ``reverse_10pct``、``correlation_stress``。
        """
        from pathlib import Path as _P
        _set_mpl_style()
        _P(output_dir).mkdir(parents=True, exist_ok=True)

        st = StressTester(self.portfolio)

        # --- 历史情景 ---
        print("\n══ 1. 历史情景分析 ══")
        hist = st.run_historical_scenarios()
        st.print_scenario_summary(hist)
        st.plot_scenarios(hist,
                          output_path=str(_P(output_dir) / "stress_historical.png"),
                          show=show)

        # --- 自定义假设情景 ---
        custom = {}
        if custom_scenarios:
            print("\n══ 2. 自定义假设情景 ══")
            custom = st.run_multiple_hypothetical(custom_scenarios)
            st.print_scenario_summary(custom)
            st.plot_scenarios(custom,
                              output_path=str(_P(output_dir) / "stress_custom.png"),
                              show=show)

        # --- 敏感性：股票市场 ---
        print("\n══ 3. 单因子敏感性：股票市场 ══")
        sens_eq = st.sensitivity_analysis("股票市场", sweep_range=(-0.20, 0.05))
        st.plot_sensitivity(sens_eq,
                            output_path=str(_P(output_dir) / "sensitivity_equity.png"),
                            show=show)

        # --- 敏感性：利率（利率上行对各资产的 DV01 风格影响）---
        print("\n══ 4. 单因子敏感性：利率 ══")
        classes    = st._classify_tickers()
        # 利率上行对不同资产的 beta：长债 -0.8（高久期），投资级 -0.4，权益 -0.1
        rate_betas = {
            t: -0.80 if classes.get(t) == "fi_long"  else
               -0.40 if classes.get(t) in ("fi_ig", "fi_hy") else
               -0.10
            for t in self.portfolio.tickers
        }
        sens_rates = st.sensitivity_analysis(
            "利率",
            sweep_range=(-0.03, 0.03),
            asset_betas=rate_betas,
        )
        st.plot_sensitivity(sens_rates,
                            output_path=str(_P(output_dir) / "sensitivity_rates.png"),
                            show=show)

        # --- 因子梯度 ---
        print("\n══ 5. 因子梯度分析 ══")
        ladder = st.factor_ladder(
            ["股票市场", "利率", "信用利差", "汇率(美元)", "通胀预期"],
            shock_magnitude=-0.05,
        )
        print(ladder)
        st.plot_factor_ladder(ladder,
                              output_path=str(_P(output_dir) / "factor_ladder.png"),
                              show=show)

        # --- 反向压力测试 ---
        print("\n══ 6. 反向压力测试（损失阈值 10%）══")
        rev = st.reverse_stress_test(loss_threshold=0.10, mode="uniform_shock")
        st.print_reverse_stress(rev)

        # --- 相关性压力测试 ---
        print("\n══ 7. 相关性压力测试 ══")
        corr_df = st.correlation_stress(target_correlation=0.95, steps=20)
        st.plot_correlation_stress(
            corr_df,
            output_path=str(_P(output_dir) / "correlation_stress.png"),
            show=show,
        )

        print(f"\n压力测试完成。所有图表已保存至 '{output_dir}'。")

        return {
            "historical":         hist,
            "custom":             custom,
            "sensitivity_equity": sens_eq,
            "sensitivity_rates":  sens_rates,
            "factor_ladder":      ladder,
            "reverse_10pct":      rev,
            "correlation_stress": corr_df,
        }
