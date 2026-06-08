# -*- coding: utf-8 -*-
"""
garch.py — GARCH 族模型拟合模块
================================



**信息准则**

- AIC：:math:`-2\\ell(\\hat{\\theta}) + 2k`
- BIC：:math:`-2\\ell(\\hat{\\theta}) + k\\ln T`

其中 :math:`k` 为参数个数，:math:`T` 为样本量。AIC 倾向于选择较复杂模型；
BIC 对参数数量惩罚更重，适合大样本。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


# 类型别名，方便 IDE 提示
VolProcess = Literal["GARCH", "EGARCH", "GJR-GARCH"]
ErrorDist  = Literal["normal", "t", "skewt"]




@dataclass
class GARCHResult:
    """GARCH 模型拟合结果的数据容器。

    Attributes
    ----------
    ticker : str
        资产代码（标识用）。
    model_type : str
        波动率过程类型（GARCH / EGARCH / GJR-GARCH）。
    p, q : int
        滞后阶数，:math:`p`（GARCH 项），:math:`q`（ARCH 项）。
    distribution : str
        创新项分布（normal / t / skewt）。
    omega : float
        常数项 :math:`\\omega > 0`。
    alpha : np.ndarray
        ARCH 系数 :math:`(\\alpha_1,\\ldots,\\alpha_q)`，长度 :math:`q`。
    beta : np.ndarray
        GARCH 系数 :math:`(\\beta_1,\\ldots,\\beta_p)`，长度 :math:`p`。
    gamma : np.ndarray 或 None
        非对称系数（GJR-GARCH/EGARCH），长度 :math:`q`；对称模型为 ``None``。
    log_likelihood : float
        对数似然值 :math:`\\ell(\\hat{\\theta})`。
    aic, bic : float
        赤池信息准则与贝叶斯信息准则。
    conditional_vol : pd.Series
        样本内条件标准差序列 :math:`\\{\\hat{\\sigma}_t\\}`（小数，非百分比）。
    forecast_variance : float
        单步超前预测方差 :math:`\\hat{\\sigma}_{T+1|T}^2`。
    """

    ticker:         str
    model_type:     str
    p:              int
    q:              int
    distribution:   str
    omega:          float
    alpha:          np.ndarray
    beta:           np.ndarray
    gamma:          Optional[np.ndarray]
    log_likelihood: float
    aic:            float
    bic:            float
    conditional_vol:   pd.Series = field(repr=False)
    forecast_variance: float     = 0.0
    _fitted:           object    = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    @property
    def persistence(self) -> float:
        r"""波动率持续性参数 :math:`\sum_i \alpha_i + \sum_j \beta_j`.

        取值范围 (0, 1)：越接近 1 表示波动率冲击衰减越慢（长记忆性）。
        若持续性 ≥ 1 则过程非平稳（IGARCH 或爆炸过程）。
        """
        return float(self.alpha.sum() + self.beta.sum())

    @property
    def long_run_vol(self) -> float:
        r"""无条件（长期）年化波动率 :math:`\bar{\sigma} = \sqrt{\omega/(1-\alpha-\beta)} \cdot \sqrt{252}`.

        协方差平稳条件下有限；若持续性 = 1 则为无穷大（IGARCH）。
        """
        denom = 1.0 - self.persistence
        if denom <= 1e-10:
            return float("inf")
        return float(np.sqrt(self.omega / denom) * np.sqrt(252))

    @property
    def forecast_vol(self) -> float:
        """单步超前条件波动率年化值 :math:`\\hat{\\sigma}_{T+1|T} \\cdot \\sqrt{252}`。"""
        return float(np.sqrt(self.forecast_variance) * np.sqrt(252))

    # ------------------------------------------------------------------
    # 多步预测
    # ------------------------------------------------------------------

    def multi_step_forecast(self, horizon: int = 10) -> pd.Series:
        r"""前向递推多步条件方差预测。

        对于 GARCH(1,1)，:math:`h` 步超前方差满足均值回归递推：

        .. math::

            \hat{\sigma}_{T+h|T}^2 = \bar{\sigma}^2
            + (\alpha + \beta)^{h-1}(\hat{\sigma}_{T+1|T}^2 - \bar{\sigma}^2)

        即条件方差以速率 :math:`(\alpha+\beta)^h` 向长期均值 :math:`\bar{\sigma}^2` 回归。

        Parameters
        ----------
        horizon : int
            预测步数 :math:`h`。

        Returns
        -------
        pd.Series
            索引为 1..h，值为对应步骤的年化条件波动率（小数）。
        """
        if self._fitted is not None:
            # 使用 arch 库的精确多步预测（含路径积分）
            fc = self._fitted.forecast(horizon=horizon, reindex=False)
            var_series = fc.variance.iloc[-1]
            vol_series = np.sqrt(var_series) * np.sqrt(252)
            vol_series.index = range(1, horizon + 1)
            vol_series.name = "forecast_vol_ann"
            return vol_series

        # 退化路径：GARCH(1,1) 解析递推
        # σ²_{T+h} = ω/(1-α-β) + (α+β)^{h-1} · (σ²_{T+1} - ω/(1-α-β))
        denom = max(1.0 - self.persistence, 1e-10)
        lr_var = self.omega / denom
        forecasts: list[float] = []
        var_h = self.forecast_variance
        for h in range(1, horizon + 1):
            if h == 1:
                forecasts.append(var_h)
            else:
                var_h = self.omega + self.persistence * var_h
                forecasts.append(var_h)

        return pd.Series(
            np.sqrt(forecasts) * np.sqrt(252),
            index=range(1, horizon + 1),
            name="forecast_vol_ann",
        )

    # ------------------------------------------------------------------
    # 参数摘要
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """格式化输出模型参数与诊断统计量。"""
        sep = "=" * 54
        lines = [
            sep,
            f"  {self.model_type}({self.p},{self.q})  [{self.ticker}]"
            f"  创新项分布: {self.distribution}",
            sep,
            f"  ω (omega)         = {self.omega:.6e}",
        ]
        for i, a in enumerate(self.alpha):
            lines.append(f"  α_{i+1} (alpha)      = {a:.6f}")
        for i, b in enumerate(self.beta):
            lines.append(f"  β_{i+1} (beta)       = {b:.6f}")
        if self.gamma is not None:
            for i, g in enumerate(self.gamma):
                lines.append(f"  γ_{i+1} (gamma)      = {g:.6f}  [非对称项]")
        lines += [
            "-" * 54,
            f"  持续性 Σα+Σβ       = {self.persistence:.6f}",
            f"  长期年化波动率       = {self.long_run_vol*100:.4f} %",
            f"  1步预测年化波动率    = {self.forecast_vol*100:.4f} %",
            "-" * 54,
            f"  对数似然 ℓ(θ̂)     = {self.log_likelihood:.4f}",
            f"  AIC               = {self.aic:.4f}",
            f"  BIC               = {self.bic:.4f}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模型估计器
# ---------------------------------------------------------------------------

class GARCHModel:
    r"""GARCH 族模型估计器。

    通过调用 ``arch`` 库实现准最大似然估计（QML），
    支持 GARCH(p,q)、EGARCH(p,q)、GJR-GARCH(p,q) 三种波动率过程，
    以及正态、Student-t、偏斜 Student-t 三种创新项分布。

    Parameters
    ----------
    p : int
        GARCH 滞后阶数（:math:`\\beta` 项数量），通常取 1。
    q : int
        ARCH 滞后阶数（:math:`\\alpha` 项数量），通常取 1。
    vol : VolProcess
        波动率过程：``"GARCH"``（对称）| ``"EGARCH"``（对数条件方差）|
        ``"GJR-GARCH"``（非对称阈值 GARCH，刻画杠杆效应）。
    dist : ErrorDist
        创新项分布：``"normal"``（正态）| ``"t"``（厚尾 Student-t）|
        ``"skewt"``（偏斜 t，同时刻画偏度与尖峰）。
    mean : str
        均值方程：``"Zero"``（零均值，适合日收益率）| ``"Constant"`` |
        ``"AR"``（自回归均值）。

    Notes
    -----
    **模型选择建议**

    - 权益日收益率：GJR-GARCH(1,1) + Student-t，可同时刻画杠杆效应与厚尾。
    - 固定收益 ETF：GARCH(1,1) + normal 通常足够，持续性较高（β > 0.9）。
    - 模型比较：以 AIC/BIC 为准则，见 :meth:`compare_models`。

    Examples
    --------
    >>> from var_garch import fetch_prices, log_returns, GARCHModel
    >>> prices = fetch_prices("AAPL", start="2019-01-01")
    >>> rets   = log_returns(prices)["AAPL"]
    >>> model  = GARCHModel(p=1, q=1, vol="GJR-GARCH", dist="t")
    >>> result = model.fit(rets, ticker="AAPL")
    >>> print(result.summary())
    >>> fc = result.multi_step_forecast(horizon=10)
    """

    def __init__(
        self,
        p:    int        = 1,
        q:    int        = 1,
        vol:  VolProcess = "GARCH",
        dist: ErrorDist  = "normal",
        mean: str        = "Zero",
    ) -> None:
        if p < 1 or q < 1:
            raise ValueError("p 与 q 均须为正整数。")
        self.p    = p
        self.q    = q
        self.vol  = vol
        self.dist = dist
        self.mean = mean

    def fit(
        self,
        returns: pd.Series,
        ticker:  str = "asset",
    ) -> GARCHResult:
        r"""对收益率序列进行 GARCH 模型拟合（QML 估计）。

        估计过程
        --------
        1. 将日收益率乘以 100（百分比化）以改善数值条件数。
        2. 调用 ``arch.arch_model`` 构造模型对象。
        3. 通过 BFGS 拟牛顿法最大化对数似然 :math:`\\ell(\\theta)`。
        4. 提取参数估计量，并将单位还原为小数。
        5. 计算单步超前条件方差预测 :math:`\\hat{\\sigma}_{T+1|T}^2`。

        Parameters
        ----------
        returns : pd.Series
            日对数收益率序列（小数，非百分比），长度至少 100。
        ticker : str
            资产标识，用于结果展示。

        Returns
        -------
        GARCHResult
            包含估计参数、诊断统计量及预测值的结果容器。

        Raises
        ------
        ImportError
            若未安装 ``arch`` 包。
        RuntimeError
            若 MLE 优化不收敛（极少见，通常由极端收益率序列引起）。
        """
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("需要安装 arch 包：pip install arch") from exc

        if len(returns) < 50:
            raise ValueError(f"收益率序列长度 {len(returns)} 过短，建议至少 50 个观测值。")

        # arch 库对百分比收益率数值更稳定（避免 ω 量级过小导致收敛困难）
        scaled = returns * 100.0

        # GJR-GARCH 在 arch 中通过 o 参数（非对称项阶数）实现
        is_gjr = (self.vol == "GJR-GARCH")
        vol_str = "EGARCH" if is_gjr else self.vol

        dist_map = {"normal": "normal", "t": "t", "skewt": "skewt"}

        am = arch_model(
            scaled,
            mean=self.mean,
            vol=vol_str,
            p=self.p,
            o=self.q if is_gjr else 0,   # GJR 非对称阶数
            q=self.q,
            dist=dist_map[self.dist],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)

        # ----- 参数提取 -----
        params = res.params

        # ω 从百分比² 还原为小数²（乘以 1e-4）
        omega_key = [k for k in params.index if "omega" in k.lower()][0]
        omega = float(params[omega_key]) * 1e-4

        # α（ARCH 项）
        alpha_keys = sorted([k for k in params.index if k.lower().startswith("alpha")])
        alpha = np.array([float(params[k]) for k in alpha_keys])

        # β（GARCH 项）
        beta_keys  = sorted([k for k in params.index if k.lower().startswith("beta")])
        beta  = np.array([float(params[k]) for k in beta_keys])

        # γ（非对称项，GJR/EGARCH）
        gamma_keys = sorted([k for k in params.index if k.lower().startswith("gamma")])
        gamma = np.array([float(params[k]) for k in gamma_keys]) if gamma_keys else None

        # 条件标准差序列（还原为小数）
        cond_vol_pct  = res.conditional_volatility            # 百分比单位
        cond_vol_dec  = pd.Series(cond_vol_pct.values / 100.0,
                                  index=cond_vol_pct.index)   # 小数单位

        # 单步预测：arch 返回的是百分比²，需除以 10000
        fc       = res.forecast(horizon=1, reindex=False)
        fcast_var = float(fc.variance.iloc[-1, 0]) / 10_000.0

        return GARCHResult(
            ticker          = ticker,
            model_type      = "GJR-GARCH" if is_gjr else self.vol,
            p               = self.p,
            q               = self.q,
            distribution    = self.dist,
            omega           = omega,
            alpha           = alpha,
            beta            = beta,
            gamma           = gamma,
            log_likelihood  = float(res.loglikelihood),
            aic             = float(res.aic),
            bic             = float(res.bic),
            conditional_vol = cond_vol_dec,
            forecast_variance = fcast_var,
            _fitted         = res,
        )

    @staticmethod
    def compare_models(
        returns:  pd.Series,
        ticker:   str = "asset",
        configs:  list[dict] | None = None,
    ) -> pd.DataFrame:
        """在多种 GARCH 规格间进行 AIC/BIC 模型比较。

        Parameters
        ----------
        returns : pd.Series
            日对数收益率序列。
        ticker : str
            资产代码。
        configs : list of dict, optional
            每个 dict 可含键 ``vol``、``dist``、``p``、``q``。
            默认对比 8 种常用规格。

        Returns
        -------
        pd.DataFrame
            按 AIC 升序排列，包含参数数量、对数似然、AIC、BIC 及持续性。

        Examples
        --------
        >>> tbl = GARCHModel.compare_models(rets["AAPL"], ticker="AAPL")
        >>> print(tbl.head(3))
        """
        if configs is None:
            # 覆盖最常见的规格组合
            configs = [
                {"vol": "GARCH",     "dist": "normal"},
                {"vol": "GARCH",     "dist": "t"},
                {"vol": "GARCH",     "dist": "skewt"},
                {"vol": "GJR-GARCH", "dist": "normal"},
                {"vol": "GJR-GARCH", "dist": "t"},
                {"vol": "GJR-GARCH", "dist": "skewt"},
                {"vol": "EGARCH",    "dist": "t"},
                {"vol": "EGARCH",    "dist": "skewt"},
            ]

        rows = []
        for cfg in configs:
            vol  = cfg.get("vol",  "GARCH")
            dist = cfg.get("dist", "normal")
            p    = cfg.get("p",    1)
            q    = cfg.get("q",    1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = GARCHModel(p=p, q=q, vol=vol, dist=dist).fit(
                        returns, ticker=ticker
                    )
                rows.append({
                    "model":       f"{vol}({p},{q})-{dist}",
                    "log_lik":     round(result.log_likelihood, 2),
                    "aic":         round(result.aic, 2),
                    "bic":         round(result.bic, 2),
                    "persistence": round(result.persistence, 4),
                    "lr_vol_%":    round(result.long_run_vol * 100, 3),
                })
            except Exception as e:
                # 极少数情况下某些规格不收敛，跳过
                rows.append({
                    "model": f"{vol}({p},{q})-{dist}",
                    "log_lik": float("nan"), "aic": float("nan"),
                    "bic": float("nan"), "persistence": float("nan"),
                    "lr_vol_%": float("nan"),
                })

        df = pd.DataFrame(rows).set_index("model")
        return df.sort_values("aic")
