# -*- coding: utf-8 -*-
"""
portfolio.py — 多资产组合层面 VaR 聚合



"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from garch import GARCHResult
from var import VaREstimator, VaRResult
from utils.returns import log_returns


class Portfolio:
    r"""多资产组合 VaR 估计器。

    基于 GARCH 单步预测条件方差与历史相关系数矩阵（DCC 近似），
    计算组合层面 VaR、ES、成分 VaR 及分散化收益。

    Parameters
    ----------
    tickers : list of str
        资产代码列表，须与价格 DataFrame 的列名一致。
    weights : dict, optional
        权重字典 ``{ticker: weight}``，各权重之和归一化为 1。
        默认等权重 :math:`w_k = 1/K`。
    confidence : float
        置信水平 :math:`1-\alpha`。
    horizon : int
        持有期 :math:`h`（交易日）。
    garch_p, garch_q : int
        GARCH 阶数。
    vol_process : str
        波动率过程类型。
    dist : str
        创新项分布。
    portfolio_value : float
        组合市值（美元），用于计算美元 VaR/ES。

    Examples
    --------
    >>> from var_garch import fetch_prices, Portfolio
    >>> prices = fetch_prices(["AAPL", "MSFT", "AGG", "TLT"],
    ...                        start="2020-01-01")
    >>> port = Portfolio(
    ...     tickers=["AAPL", "MSFT", "AGG", "TLT"],
    ...     weights={"AAPL": 0.30, "MSFT": 0.30, "AGG": 0.20, "TLT": 0.20},
    ...     confidence=0.95,
    ...     horizon=1,
    ...     vol_process="GJR-GARCH",
    ...     dist="t",
    ...     portfolio_value=1_000_000,
    ... )
    >>> port.fit(prices)
    >>> print(port.summary())
    """

    def __init__(
        self,
        tickers:         List[str],
        weights:         Optional[Dict[str, float]] = None,
        confidence:      float = 0.95,
        horizon:         int   = 1,
        garch_p:         int   = 1,
        garch_q:         int   = 1,
        vol_process:     str   = "GARCH",
        dist:            str   = "normal",
        portfolio_value: float = 1_000_000,
    ) -> None:
        self.tickers         = tickers
        self.confidence      = confidence
        self.horizon         = horizon
        self.portfolio_value = portfolio_value

        # 权重归一化
        K = len(tickers)
        if weights is None:
            self.weights = {t: 1.0 / K for t in tickers}
        else:
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("权重之和须为正数。")
            self.weights = {t: weights.get(t, 0.0) / total for t in tickers}

        # 各资产使用相同的 GARCH 规格（生产中可允许逐资产配置）
        self._estimator = VaREstimator(
            confidence=confidence,
            horizon=horizon,
            garch_p=garch_p,
            garch_q=garch_q,
            vol_process=vol_process,
            dist=dist,
        )

        # 拟合后填充的属性
        self._per_asset:     Dict[str, VaRResult]  = {}
        self._returns:       Optional[pd.DataFrame] = None
        self._garch_results: Dict[str, GARCHResult] = {}
        self._corr_matrix:   Optional[pd.DataFrame] = None
        self._portfolio_var: float = 0.0
        self._portfolio_es:  float = 0.0
        self._diversif_benefit: float = 0.0

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> Dict[str, VaRResult]:
        """对价格矩阵拟合 GARCH 并聚合组合 VaR。

        Parameters
        ----------
        prices : pd.DataFrame
            日价格矩阵，列名须包含 ``tickers`` 中的所有代码。

        Returns
        -------
        dict
            ``{ticker: VaRResult}`` — 每个资产的单资产 VaR 结果。
        """
        missing = [t for t in self.tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"价格 DataFrame 中缺少以下代码：{missing}")

        # 计算对数收益率，删除含 NaN 的行（对齐日历）
        self._returns = log_returns(prices[self.tickers]).dropna()

        # 逐资产拟合 GARCH + VaR
        for ticker in self.tickers:
            w = self.weights[ticker]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._estimator.estimate(
                    self._returns[ticker],
                    ticker=ticker,
                    portfolio_value=self.portfolio_value * w,
                )
            self._per_asset[ticker]     = result
            self._garch_results[ticker] = result.garch

        # 聚合组合层面 VaR（考虑相关性）
        self._compute_portfolio_var()
        return self._per_asset

    # ------------------------------------------------------------------
    # 组合 VaR 聚合（DCC 近似）
    # ------------------------------------------------------------------

    def _compute_portfolio_var(self) -> None:
        r"""基于 GARCH 条件协方差矩阵计算组合 VaR。

        实现步骤：

        1. 计算历史收益率相关矩阵 :math:`\hat{\mathbf{C}}`（DCC 近似用历史相关替代）。
        2. 从各资产 GARCH 预测提取单步条件标准差向量 :math:`\boldsymbol{\sigma}_{T+1}`。
        3. 构造条件协方差矩阵：
           :math:`\hat{\boldsymbol{\Sigma}} = \mathbf{D}\hat{\mathbf{C}}\mathbf{D}`。
        4. 计算组合方差：
           :math:`\hat{\sigma}_p^2 = \mathbf{w}^\top \hat{\boldsymbol{\Sigma}} \mathbf{w}`。
        5. 乘以正态分位数与持有期得到 VaR/ES。
        """
        w     = np.array([self.weights[t] for t in self.tickers])
        sigma = np.array([
            np.sqrt(self._garch_results[t].forecast_variance)
            for t in self.tickers
        ])

        # 相关矩阵：使用历史样本估计（DCC 近似）
        self._corr_matrix = self._returns.corr()
        C   = self._corr_matrix.values
        D   = np.diag(sigma)
        Cov = D @ C @ D                          # K×K 条件协方差矩阵

        # 组合方差 σ²_p = w⊤ Σ w
        port_var = float(w @ Cov @ w)
        port_vol = np.sqrt(port_var) * np.sqrt(self.horizon)   # √h 时间尺度

        z   = abs(stats.norm.ppf(1.0 - self.confidence))
        self._portfolio_var = float(port_vol * z)

        # 正态 ES（闭合解）：E[-r | -r > VaR] = φ(z_α)/α · σ_p · √h
        alpha = 1.0 - self.confidence
        self._portfolio_es = float(
            stats.norm.pdf(stats.norm.ppf(alpha)) / alpha * port_vol
        )

        # 分散化收益 = 个体 VaR 加权和 - 组合 VaR
        undiversified = sum(
            self.weights[t] * self._per_asset[t].var_parametric
            for t in self.tickers
        )
        self._diversif_benefit = undiversified - self._portfolio_var

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def portfolio_var(self) -> float:
        """组合 VaR（小数形式）。"""
        return self._portfolio_var

    @property
    def portfolio_var_dollar(self) -> float:
        """组合美元 VaR。"""
        return self._portfolio_var * self.portfolio_value

    @property
    def portfolio_es(self) -> float:
        """组合预期损失 ES（小数形式）。"""
        return self._portfolio_es

    @property
    def portfolio_es_dollar(self) -> float:
        """组合美元 ES。"""
        return self._portfolio_es * self.portfolio_value

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """历史收益率相关矩阵 :math:`\\hat{\\mathbf{C}}`。"""
        return self._corr_matrix

    @property
    def risk_contributions(self) -> pd.Series:
        r"""成分 VaR（Component VaR）向量。

        成分 VaR 的计算公式：

        .. math::

            \mathrm{CVaR}_k
            = w_k \cdot \frac{(\hat{\boldsymbol{\Sigma}}\mathbf{w})_k}
              {\hat{\sigma}_p} \cdot |z_\alpha| \cdot \sqrt{h}

        满足 :math:`\sum_k \mathrm{CVaR}_k = \mathrm{VaR}_p`（加和性）。
        """
        w     = np.array([self.weights[t] for t in self.tickers])
        sigma = np.array([
            np.sqrt(self._garch_results[t].forecast_variance)
            for t in self.tickers
        ])
        C   = self._corr_matrix.values
        D   = np.diag(sigma)
        Cov = D @ C @ D

        port_var_scalar = float(w @ Cov @ w)
        if port_var_scalar <= 0:
            return pd.Series(0.0, index=self.tickers, name="component_var")

        # 边际 VaR 向量：∂VaR_p/∂w = Σw/σ_p · |z_α|·√h
        marginal = Cov @ w / np.sqrt(port_var_scalar)
        z        = abs(stats.norm.ppf(1.0 - self.confidence))
        cvars    = w * marginal * z * np.sqrt(self.horizon)

        return pd.Series(cvars, index=self.tickers, name="component_var")

    # ------------------------------------------------------------------
    # 报告与导出
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """格式化打印组合风险摘要。"""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║          组合 VaR 摘要报告（GARCH 条件协方差）          ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"  置信水平        : {self.confidence*100:.0f}%",
            f"  持有期          : {self.horizon} 交易日",
            f"  组合市值        : ${self.portfolio_value:,.0f}",
            "──────────────────────────────────────────────────────────",
            "  组合层面风险",
            f"  VaR ({self.confidence*100:.0f}%)    : "
            f"{self._portfolio_var*100:.4f}%  (${self.portfolio_var_dollar:,.0f})",
            f"  预期损失 ES     : "
            f"{self._portfolio_es*100:.4f}%  (${self.portfolio_es_dollar:,.0f})",
            f"  分散化收益      : "
            f"{self._diversif_benefit*100:.4f}%  (${self._diversif_benefit*self.portfolio_value:,.0f})",
            "──────────────────────────────────────────────────────────",
            "  单资产 VaR（参数法）",
        ]
        for t in self.tickers:
            r = self._per_asset[t]
            w = self.weights[t]
            lines.append(
                f"  {t:<8}  权重={w:.1%}  VaR={r.var_parametric*100:.4f}%"
                f"  年化波动率={r.garch.long_run_vol*100:.2f}%"
            )

        rc = self.risk_contributions
        lines += [
            "──────────────────────────────────────────────────────────",
            "  成分 VaR（Component VaR）贡献",
        ]
        for t in self.tickers:
            pct = rc[t] / self._portfolio_var * 100 if self._portfolio_var else 0.0
            lines.append(
                f"  {t:<8}  CVaR={rc[t]*100:.4f}%  占比={pct:.1f}%"
            )

        lines += [
            "──────────────────────────────────────────────────────────",
            "  GARCH 参数摘要",
        ]
        for t in self.tickers:
            g = self._garch_results[t]
            lines.append(
                f"  {t:<8}  α={g.alpha[0]:.4f}  β={g.beta[0]:.4f}"
                f"  持续性={g.persistence:.4f}"
                f"  1步预测波动率={g.forecast_vol*100:.3f}%/yr"
            )

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """将组合风险指标导出为整洁 DataFrame。"""
        rc   = self.risk_contributions
        rows = []
        for t in self.tickers:
            r = self._per_asset[t]
            g = r.garch
            rows.append({
                "ticker":           t,
                "weight":           self.weights[t],
                "var_garch_%":      r.var_parametric  * 100,
                "var_historical_%": r.var_historical   * 100,
                "var_fhs_%":        r.var_fhs          * 100,
                "var_mc_%":         r.var_montecarlo   * 100,
                "es_historical_%":  r.es_historical    * 100,
                "annual_vol_%":     g.long_run_vol     * 100,
                "forecast_vol_%":   g.forecast_vol     * 100,
                "garch_alpha":      g.alpha[0],
                "garch_beta":       g.beta[0],
                "persistence":      g.persistence,
                "component_var_%":  rc[t]              * 100,
            })
        return pd.DataFrame(rows).set_index("ticker")
