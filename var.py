# -*- coding: utf-8 -*-
"""
var.py — 风险价值（VaR）与预期损失（ES）估计
==============================================

理论框架
--------

**风险价值定义**

给定置信水平 :math:`1-\\alpha` 与持有期 :math:`h`，组合收益率
:math:`R_{t+1:t+h}` 的 VaR 定义为

.. math::

    \\mathrm{VaR}_{1-\\alpha}(h)
    = -\\inf\\{x \\in \\mathbb{R} : F_R(x) > \\alpha\\}
    = -F_R^{-1}(\\alpha)

即收益率分布左侧 :math:`\\alpha` 分位数的绝对值（正数表示损失）。

**预期损失（Expected Shortfall / CVaR）**

VaR 无法刻画尾部损失的严重程度；ES 定义为

.. math::

    \\mathrm{ES}_{1-\\alpha}(h)
    = -\\mathbb{E}[R_{t+1:t+h} \\mid R_{t+1:t+h} \\leq -\\mathrm{VaR}_{1-\\alpha}]

ES 满足次可加性（subadditivity），是一致风险度量（coherent risk measure）。

**时间尺度变换（平方根时间法则）**

在 i.i.d. 正态收益率假设下，:math:`h` 期 VaR 满足

.. math::

    \\mathrm{VaR}_{1-\\alpha}(h) = \\mathrm{VaR}_{1-\\alpha}(1) \\cdot \\sqrt{h}

GARCH 框架下此法则不严格成立，应使用多步递推预测；
但在实践中（Basel 框架），平方根法则仍被广泛使用。

四种估计方法
------------

1. **参数法（Parametric GARCH VaR）**

   .. math::

      \\mathrm{VaR}^{\\mathrm{GARCH}}_{1-\\alpha}(h)
      = -z_\\alpha \\cdot \\hat{\\sigma}_{T+1|T} \\cdot \\sqrt{h}

   其中 :math:`z_\\alpha = F_Z^{-1}(\\alpha)` 为标准化分布的 :math:`\\alpha` 分位数，
   :math:`\\hat{\\sigma}_{T+1|T}` 为 GARCH 单步预测条件标准差。

2. **历史模拟法（Historical Simulation, HS）**

   直接取历史收益率序列的 :math:`\\alpha` 经验分位数：

   .. math::

      \\mathrm{VaR}^{\\mathrm{HS}}_{1-\\alpha} = -\\hat{r}_{(\\lfloor\\alpha T\\rfloor)}

   其中 :math:`\\hat{r}_{(k)}` 为升序排列后第 :math:`k` 个观测值。

3. **过滤历史模拟法（Filtered Historical Simulation, FHS）**

   先以 GARCH 标准化残差：:math:`\\tilde{z}_t = r_t / \\hat{\\sigma}_t`，
   再对标准化残差重采样，乘以预测波动率：

   .. math::

      r_{T+1}^{\\mathrm{sim}} = \\hat{\\sigma}_{T+1|T} \\cdot \\tilde{z}^*

   FHS 综合了历史模拟的非参数灵活性与 GARCH 的波动率时变性。

4. **蒙特卡洛法（Monte Carlo, MC）**

   基于 GARCH 方差递推方程生成 :math:`N_\\mathrm{sim}` 条路径：

   .. math::

      r_{t+1} = \\sigma_{t+1} z_{t+1},
      \\quad z_{t+1} \\overset{\\mathrm{i.i.d.}}{\\sim} \\mathcal{N}(0,1)

   VaR 取模拟分布的 :math:`\\alpha` 经验分位数。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from garch import GARCHModel, GARCHResult

Method = Literal["parametric", "historical", "fhs", "montecarlo"]


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """单资产 VaR / ES 估计结果容器。

    所有 VaR 与 ES 数值均为**正数**（损失端），以小数表示（如 0.025 = 2.5%）。
    """

    ticker:      str
    confidence:  float       # 置信水平 1-α
    horizon:     int         # 持有期 h（交易日）

    # --- 四种 VaR 估计（正数 = 损失） ---
    var_parametric:  float   # GARCH 参数法
    var_historical:  float   # 历史模拟法
    var_fhs:         float   # 过滤历史模拟法
    var_montecarlo:  float   # 蒙特卡洛法

    # --- 预期损失 ---
    es_parametric:   float   # 正态 ES = σ·φ(z_α)/α · √h
    es_historical:   float   # 历史尾部均值

    # --- 美元金额（可选） ---
    dollar_var:      float = 0.0
    dollar_es:       float = 0.0

    # --- 底层 GARCH 拟合结果 ---
    garch: Optional[GARCHResult] = field(default=None, repr=False)

    @property
    def var_summary(self) -> Dict[str, float]:
        """返回所有 VaR 估计的百分比字典。"""
        return {
            "参数法 GARCH":      self.var_parametric  * 100,
            "历史模拟法":        self.var_historical   * 100,
            "过滤历史模拟 FHS":  self.var_fhs          * 100,
            "蒙特卡洛 MC":       self.var_montecarlo   * 100,
        }

    def __str__(self) -> str:
        cl = self.confidence * 100
        lines = [
            f"\n{'─'*52}",
            f"  VaR 估计报告 — {self.ticker}",
            f"  置信水平: {cl:.0f}%  |  持有期: {self.horizon}d",
            f"{'─'*52}",
            f"  参数法 GARCH VaR   : {self.var_parametric*100:>8.4f} %",
            f"  历史模拟法 VaR     : {self.var_historical*100:>8.4f} %",
            f"  过滤历史模拟 VaR   : {self.var_fhs*100:>8.4f} %",
            f"  蒙特卡洛 VaR       : {self.var_montecarlo*100:>8.4f} %",
            f"{'─'*52}",
            f"  历史法 ES (CVaR)   : {self.es_historical*100:>8.4f} %",
            f"  参数法 ES          : {self.es_parametric*100:>8.4f} %",
        ]
        if self.dollar_var:
            lines += [
                f"{'─'*52}",
                f"  美元 VaR (参数法)  : ${self.dollar_var:>14,.0f}",
                f"  美元 ES  (参数法)  : ${self.dollar_es:>14,.0f}",
            ]
        lines.append(f"{'─'*52}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# VaR 估计器
# ---------------------------------------------------------------------------

class VaREstimator:
    r"""单资产 VaR 估计器：整合四种方法与滚动回测。

    Parameters
    ----------
    confidence : float
        置信水平 :math:`1-\alpha`，如 ``0.95``（95% VaR）或 ``0.99``（99% VaR）。
    horizon : int
        持有期 :math:`h`（交易日数）。
    garch_p, garch_q : int
        GARCH(p,q) 阶数。
    vol_process : str
        波动率过程，见 :class:`~var_garch.garch.GARCHModel`。
    dist : str
        创新项分布。
    n_simulations : int
        蒙特卡洛模拟路径数 :math:`N_\\mathrm{sim}`，越大精度越高（但更耗时）。

    Examples
    --------
    >>> est = VaREstimator(confidence=0.99, horizon=10, vol_process="GJR-GARCH", dist="t")
    >>> result = est.estimate(rets["AAPL"], ticker="AAPL", portfolio_value=1_000_000)
    >>> print(result)
    """

    def __init__(
        self,
        confidence:    float = 0.95,
        horizon:       int   = 1,
        garch_p:       int   = 1,
        garch_q:       int   = 1,
        vol_process:   str   = "GARCH",
        dist:          str   = "normal",
        n_simulations: int   = 10_000,
    ) -> None:
        if not (0.5 < confidence < 1.0):
            raise ValueError(f"置信水平须在 (0.5, 1.0) 范围内，当前值: {confidence}")
        if horizon < 1:
            raise ValueError("持有期须为正整数。")

        self.confidence    = confidence
        self.horizon       = horizon
        self.n_simulations = n_simulations

        # 底层 GARCH 估计器（四种 VaR 方法共享同一 GARCH 拟合结果）
        self._garch = GARCHModel(
            p=garch_p, q=garch_q, vol=vol_process, dist=dist
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def estimate(
        self,
        returns:         pd.Series,
        ticker:          str   = "asset",
        portfolio_value: float = 0.0,
    ) -> VaRResult:
        """估计单资产的四种 VaR 及 ES。

        Parameters
        ----------
        returns : pd.Series
            日对数收益率序列（小数）。
        ticker : str
            资产代码。
        portfolio_value : float
            组合市值（美元）；非零时计算美元 VaR/ES。

        Returns
        -------
        VaRResult
        """
        # 先拟合 GARCH，所有方法共用同一条件方差估计
        garch_res = self._garch.fit(returns, ticker=ticker)

        var_p, es_p  = self._parametric_var(garch_res)
        var_h, es_h  = self._historical_var(returns)
        var_fhs      = self._fhs_var(returns, garch_res)
        var_mc       = self._montecarlo_var(garch_res)

        dv = portfolio_value * var_p if portfolio_value else 0.0
        de = portfolio_value * es_p  if portfolio_value else 0.0

        return VaRResult(
            ticker          = ticker,
            confidence      = self.confidence,
            horizon         = self.horizon,
            var_parametric  = var_p,
            var_historical  = var_h,
            var_fhs         = var_fhs,
            var_montecarlo  = var_mc,
            es_parametric   = es_p,
            es_historical   = es_h,
            dollar_var      = dv,
            dollar_es       = de,
            garch           = garch_res,
        )

    def backtest(
        self,
        returns: pd.Series,
        ticker:  str = "asset",
        window:  int = 252,
    ) -> pd.DataFrame:
        r"""滚动 VaR 回测（Kupiec 无条件覆盖率检验）。

        在每个时间点 :math:`t`，以 :math:`[t-w+1, t]` 窗口重新估计 GARCH，
        对下一期收益率 :math:`r_{t+1}` 检验是否超过 VaR：

        .. math::

            \\text{breach}_t = \\mathbf{1}\\{r_{t+1} < -\\mathrm{VaR}_{1-\\alpha,t}\\}

        **Kupiec (1995) POF 检验** 原假设：

        .. math::

            H_0: \\mathbb{E}[\\text{breach}_t] = \\alpha
            \\quad \\text{（VaR 覆盖率正确）}

        检验统计量：

        .. math::

            LR_{\\mathrm{POF}}
            = -2\\ln\\frac{(1-\\alpha)^{T-N}\\alpha^N}{(1-p)^{T-N}p^N}
            \\xrightarrow{H_0} \\chi^2_1

        其中 :math:`N` = 实际违约次数，:math:`T` = 总观测数，:math:`p = N/T`。

        Parameters
        ----------
        returns : pd.Series
            完整历史收益率序列。
        ticker : str
        window : int
            滚动估计窗口 :math:`w`（交易日数）。

        Returns
        -------
        pd.DataFrame
            列：``var``（预测 VaR，正数）、``actual_return``、``breach``。
            同时打印 Kupiec 检验统计量。
        """
        records = []
        idx     = returns.index

        for i in range(len(returns) - window):
            train  = returns.iloc[i : i + window]
            actual = returns.iloc[i + window]
            date   = idx[i + window]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    g = self._garch.fit(train, ticker=ticker)
                    var_1d, _ = self._parametric_var(g)
                except Exception:
                    var_1d = float("nan")

            records.append({
                "date":          date,
                "var":           var_1d,      # 正数 = 预测损失上界
                "actual_return": actual,
                "breach":        actual < -var_1d if not np.isnan(var_1d) else False,
            })

        df = pd.DataFrame(records).set_index("date")

        # --- Kupiec POF 检验 ---
        alpha     = 1.0 - self.confidence
        T         = len(df)
        N         = int(df["breach"].sum())
        p_hat     = N / T if T else 0.0
        p_hat     = max(min(p_hat, 1 - 1e-10), 1e-10)

        # LR 统计量
        lr = -2.0 * (
            (T - N) * np.log((1 - alpha) / (1 - p_hat))
            + N     * np.log(alpha        / p_hat)
        ) if N > 0 and N < T else float("nan")

        p_val = float(stats.chi2.sf(lr, df=1)) if not np.isnan(lr) else float("nan")

        print(
            f"\nKupiec POF 检验 [{ticker}]  置信水平 {self.confidence*100:.0f}%\n"
            f"  样本量 T         = {T}\n"
            f"  实际违约次数 N   = {N}  (期望 ≈ {alpha*T:.1f})\n"
            f"  实际违约率 p̂    = {p_hat*100:.3f}%  (理论 α = {alpha*100:.1f}%)\n"
            f"  LR_POF 统计量   = {lr:.4f}  (χ²₁ p 值 = {p_val:.4f})\n"
            f"  {'未拒绝' if p_val > 0.05 else '拒绝'} H₀ (α=0.05)\n"
        )
        return df

    # ------------------------------------------------------------------
    # 私有：参数法
    # ------------------------------------------------------------------

    def _parametric_var(self, garch: GARCHResult):
        r"""GARCH 参数法 VaR 与 ES。

        计算公式（正态创新项）：

        .. math::

            \mathrm{VaR}^{\mathrm{GARCH}}_{1-\alpha}(h)
            &= |z_\alpha| \cdot \hat{\sigma}_{T+1|T} \cdot \sqrt{h} \\
            \mathrm{ES}^{\mathrm{GARCH}}_{1-\alpha}(h)
            &= \frac{\phi(z_\alpha)}{\alpha} \cdot \hat{\sigma}_{T+1|T} \cdot \sqrt{h}

        其中 :math:`\phi` 为标准正态密度，:math:`z_\alpha = \Phi^{-1}(\alpha)`。
        """
        sigma = np.sqrt(garch.forecast_variance)   # 单日条件标准差（小数）
        alpha = 1.0 - self.confidence

        # 分位数：按创新项分布选取（Student-t 分位数 > 正态分位数，保守估计）
        if garch.distribution == "t":
            nu = getattr(garch, "_nu", 8.0)        # 自由度（如有）
            z  = float(stats.t.ppf(alpha, df=nu))
        else:
            z  = float(stats.norm.ppf(alpha))      # 默认正态

        var = abs(sigma * z) * np.sqrt(self.horizon)

        # 正态 ES 公式（闭合解）：E[X | X < z_α] = -φ(z_α)/α
        es_raw = (
            stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            * sigma * np.sqrt(self.horizon)
        )
        return float(var), float(es_raw)

    # ------------------------------------------------------------------
    # 私有：历史模拟法
    # ------------------------------------------------------------------

    def _historical_var(self, returns: pd.Series):
        r"""历史模拟法 VaR 与尾部 ES。

        VaR 取经验分布 :math:`\alpha` 分位数（线性插值）：

        .. math::

            \mathrm{VaR}^{\mathrm{HS}} = -\hat{r}_{(\lfloor \alpha T \rfloor)}

        ES 取尾部均值：

        .. math::

            \mathrm{ES}^{\mathrm{HS}} = -\frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} r_t,
            \quad \mathcal{T} = \{t : r_t \leq -\mathrm{VaR}^{\mathrm{HS}}\}
        """
        alpha    = 1.0 - self.confidence
        sorted_r = np.sort(returns.values)         # 升序排列

        # 线性插值分位数：位置 = α·(T-1)（同 numpy quantile 默认方式）
        q_pos  = alpha * (len(sorted_r) - 1)
        lo, hi = int(np.floor(q_pos)), int(np.ceil(q_pos))
        frac   = q_pos - lo
        q_val  = sorted_r[lo] * (1 - frac) + sorted_r[hi] * frac

        var = abs(q_val) * np.sqrt(self.horizon)

        # 尾部样本（含边界）
        tail = sorted_r[sorted_r <= q_val]
        es   = abs(float(tail.mean())) * np.sqrt(self.horizon) if len(tail) else var

        return float(var), float(es)

    # ------------------------------------------------------------------
    # 私有：过滤历史模拟法（FHS）
    # ------------------------------------------------------------------

    def _fhs_var(self, returns: pd.Series, garch: GARCHResult) -> float:
        r"""过滤历史模拟法（Barone-Adesi et al., 1999）。

        步骤：
        1. 计算标准化残差 :math:`\tilde{z}_t = r_t / \hat{\sigma}_t`（去除 GARCH 波动率结构）。
        2. 对 :math:`\{\tilde{z}_t\}` 进行有放回重采样（bootstrap）。
        3. 将重采样残差乘以预测标准差 :math:`\hat{\sigma}_{T+1|T}` 与 :math:`\sqrt{h}` 得到模拟收益率。
        4. 取模拟分布的 :math:`\alpha` 分位数作为 VaR。
        """
        cond_vol = garch.conditional_vol
        aligned  = returns.loc[cond_vol.index]

        # 标准化残差（去 GARCH 效应后理论上接近 i.i.d.）
        std_resid = (aligned / cond_vol.replace(0, np.nan)).dropna()

        rng      = np.random.default_rng(seed=42)
        z_boot   = rng.choice(std_resid.values, size=self.n_simulations, replace=True)
        sigma_f  = np.sqrt(garch.forecast_variance)

        # 多期：平方根时间法则（简化，生产中建议多步递推）
        sim_rets = sigma_f * z_boot * np.sqrt(self.horizon)

        alpha = 1.0 - self.confidence
        return float(abs(np.quantile(sim_rets, alpha)))

    # ------------------------------------------------------------------
    # 私有：蒙特卡洛法
    # ------------------------------------------------------------------

    def _montecarlo_var(self, garch: GARCHResult) -> float:
        r"""基于 GARCH(1,1) 递推方程的蒙特卡洛 VaR。

        对每条路径，从 :math:`t=T+1` 开始递推 :math:`h` 步：

        .. math::

            \varepsilon_{t+1} &= \hat{\sigma}_{t+1} z_{t+1}, \quad z_{t+1} \sim \mathcal{N}(0,1) \\
            \hat{\sigma}_{t+2}^2 &= \omega + \alpha \varepsilon_{t+1}^2 + \beta \hat{\sigma}_{t+1}^2

        路径累计收益率 :math:`R = \sum_{k=1}^h \varepsilon_{T+k}` 的
        :math:`\alpha` 分位数即为 MC VaR。
        """
        rng   = np.random.default_rng(seed=0)
        omega = garch.omega
        alpha = float(garch.alpha[0]) if len(garch.alpha) > 0 else 0.09
        beta  = float(garch.beta[0])  if len(garch.beta)  > 0 else 0.85

        paths    = np.empty(self.n_simulations)
        init_var = garch.forecast_variance

        for i in range(self.n_simulations):
            sigma2  = init_var
            cum_ret = 0.0
            for _ in range(self.horizon):
                z        = rng.standard_normal()
                eps      = np.sqrt(sigma2) * z
                cum_ret += eps
                # GARCH(1,1) 递推
                sigma2   = omega + alpha * eps**2 + beta * sigma2
                sigma2   = max(sigma2, 1e-12)   # 数值截断防止退化
            paths[i] = cum_ret

        alpha_q = 1.0 - self.confidence
        return float(abs(np.quantile(paths, alpha_q)))
