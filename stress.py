# -*- coding: utf-8 -*-
"""
stress.py — 情景压力测试与敏感性分析
======================================

理论背景
--------

**历史情景法**

直接将历史危机期间观测到的收益率冲击施加于当前组合持仓，
得到"假如现在发生同等冲击，组合损失为多少"的直觉估计。
损失计算为

.. math::

    L^{\\text{hist}} = -\\sum_{k=1}^K w_k \\delta_k

其中 :math:`\\delta_k` 为情景中资产 :math:`k` 的单日对数收益率冲击（负数表示下跌）。

**假设情景法（Hypothetical Scenario）**

允许分析人员自定义各资产冲击向量 :math:`\\boldsymbol{\\delta}`，
适用于监管情景（如 DFAST、EBA 压力测试）与内部风险管理。

**敏感性分析（单因子扫描）**

设单一风险因子（如股票市场指数）的冲击幅度为 :math:`f`，
资产 :math:`k` 对该因子的敏感性（beta）为 :math:`\\beta_k`，则

.. math::

    \\delta_k(f) = \\beta_k \\cdot f

扫描 :math:`f` 的取值范围，得到组合损失关于因子冲击的函数：

.. math::

    L(f) = -\\sum_k w_k \\beta_k f = -\\mathbf{w}^\\top\\boldsymbol{\\beta} \\cdot f

**反向压力测试（Reverse Stress Testing）**

已知损失阈值 :math:`L^*`，反向求解最小冲击幅度：

.. math::

    f^* = \\arg\\min_{f} |f|
    \\quad \\text{s.t.} \\quad L(f) \\geq L^*

对均匀冲击情形，解析解为 :math:`f^* = -L^*/\\bar{\\beta}`，
其中 :math:`\\bar{\\beta} = \\sum_k w_k = 1`（等权资产敏感性），
故 :math:`f^* = -L^*`（即需要 :math:`|f^*|=L^*` 量级的均匀跌幅）。

**相关性压力测试**

在危机期间，资产间相关系数趋向于 1（correlation breakdown），
此时分散化收益消失。设压力相关矩阵为

.. math::

    \\mathbf{C}^{\\text{stress}}(\\lambda)
    = (1-\\lambda) \\hat{\\mathbf{C}} + \\lambda \\mathbf{C}^{\\text{target}}

其中 :math:`\\lambda \\in [0,1]` 为压力强度，
:math:`\\mathbf{C}^{\\text{target}}` 为高相关目标矩阵。

在压力相关矩阵下，组合 VaR 为

.. math::

    \\mathrm{VaR}^{\\text{stress}}_p
    = |z_\\alpha| \\sqrt{\\mathbf{w}^\\top \\mathbf{D}\\mathbf{C}^{\\text{stress}}\\mathbf{D} \\mathbf{w}}
    \\cdot \\sqrt{h}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 历史情景库（8 个主要市场危机事件）
# ---------------------------------------------------------------------------

#: 内置历史情景：以资产类别为键的单日对数收益率冲击（负值 = 损失）。
#: ``vol_multiplier`` 为在情景下对 GARCH 条件波动率的倍数放大系数，
#: 用于估计情景下的"压力 VaR"（Stressed VaR，Basel III 要求）。
HISTORICAL_SCENARIOS: Dict[str, Dict] = {
    "GFC_2008": {
        "label":       "全球金融危机（2008年10月）",
        "description": "雷曼兄弟破产，史上最大单月股市跌幅（自1929年后）",
        "shocks":      {"equity": -0.085, "fi_long": +0.012, "fi_ig": -0.018, "fi_hy": -0.045},
        "vol_multiplier": 3.5,
    },
    "COVID_2020": {
        "label":       "新冠疫情冲击（2020年3月）",
        "description": "史上最快熊市；2020年3月16日标普500单日跌12%",
        "shocks":      {"equity": -0.120, "fi_long": +0.020, "fi_ig": -0.030, "fi_hy": -0.060},
        "vol_multiplier": 4.0,
    },
    "DOTCOM_2000": {
        "label":       "互联网泡沫破裂（2000年4月）",
        "description": "纳斯达克4月单月跌34%；科技股估值泡沫崩溃",
        "shocks":      {"equity": -0.060, "fi_long": +0.005, "fi_ig": -0.005, "fi_hy": -0.015},
        "vol_multiplier": 2.2,
    },
    "TAPER_TANTRUM_2013": {
        "label":       "缩表恐慌（2013年5-6月）",
        "description": "美联储暗示缩减QE引发债市抛售；TLT数周内跌约15%",
        "shocks":      {"equity": -0.035, "fi_long": -0.055, "fi_ig": -0.020, "fi_hy": -0.015},
        "vol_multiplier": 1.8,
    },
    "RATE_SHOCK_2022": {
        "label":       "加息冲击（2022年）",
        "description": "美联储史上最快加息周期；股债同跌，60/40组合失效",
        "shocks":      {"equity": -0.045, "fi_long": -0.060, "fi_ig": -0.025, "fi_hy": -0.030},
        "vol_multiplier": 2.0,
    },
    "BLACK_MONDAY_1987": {
        "label":       "黑色星期一（1987年10月19日）",
        "description": "道琼斯单日跌22.6%；程序化交易引发连锁踩踏",
        "shocks":      {"equity": -0.205, "fi_long": +0.015, "fi_ig": -0.008, "fi_hy": -0.020},
        "vol_multiplier": 6.0,
    },
    "EURO_DEBT_2011": {
        "label":       "欧债危机（2011年8月）",
        "description": "标普下调美国主权评级 + 欧元区外围国家主权风险骤升",
        "shocks":      {"equity": -0.065, "fi_long": +0.018, "fi_ig": -0.012, "fi_hy": -0.030},
        "vol_multiplier": 2.5,
    },
    "FLASH_CRASH_2010": {
        "label":       "闪崩（2010年5月6日）",
        "description": "道琼斯盘中暴跌9%；流动性骤然枯竭后迅速回弹",
        "shocks":      {"equity": -0.090, "fi_long": +0.008, "fi_ig": -0.010, "fi_hy": -0.025},
        "vol_multiplier": 3.0,
    },
}

# Yahoo Finance 代码 → 资产大类的映射
_FI_LONG_TICKERS = {"TLT", "EDV", "VGLT", "ZROZ", "IEF", "GOVT"}
_FI_IG_TICKERS   = {"AGG", "BND", "LQD", "VCIT", "IGSB", "SCHZ"}
_FI_HY_TICKERS   = {"HYG", "JNK", "USHY", "FALN", "SHYG"}


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """单一压力情景的损益结果容器。"""

    scenario_name: str
    label:         str
    description:   str

    # 组合层面
    portfolio_loss_pct:    float    # 组合损失（小数正数表示损失）
    portfolio_loss_dollar: float
    stressed_var:          float    # 压力 VaR（放大波动率后重估）
    stressed_var_dollar:   float

    # 单资产层面
    asset_losses:          pd.Series    # {ticker: 损失分数（正数）}
    asset_losses_dollar:   pd.Series

    # 与基准的比较
    baseline_var:  float
    var_multiple:  float    # 情景损失 / 基准 VaR（直觉：超出几倍 VaR）


@dataclass
class SensitivityResult:
    """单因子敏感性扫描结果容器。"""

    factor:          str
    values:          np.ndarray       # 扫描的因子冲击序列
    portfolio_losses: np.ndarray      # 对应组合损失（小数）
    asset_losses:    pd.DataFrame     # 形状 (n_steps, K)：各资产加权损失
    baseline_loss:   float


@dataclass
class ReverseStressResult:
    """反向压力测试结果容器。"""

    threshold_pct:       float    # 目标损失阈值（小数）
    threshold_dollar:    float
    min_shock:           float    # 均匀冲击模式下的最小冲击幅度
    min_vol_multiplier:  float    # 波动率倍数模式下的最小倍数
    asset_shocks:        pd.Series
    iterations:          int


# ---------------------------------------------------------------------------
# 压力测试引擎
# ---------------------------------------------------------------------------

class StressTester:
    r"""多资产组合压力测试引擎。

    支持五类分析：
    1. 历史情景重演（Historical Scenario Replay）
    2. 假设情景分析（Hypothetical Scenario Analysis）
    3. 单因子敏感性扫描（Single-Factor Sensitivity Sweep）
    4. 反向压力测试（Reverse Stress Testing）
    5. 相关性压力测试（Correlation Stress Testing）

    Parameters
    ----------
    portfolio : Portfolio
        已完成 ``fit(prices)`` 的 :class:`~var_garch.portfolio.Portfolio` 实例。

    Notes
    -----
    Basel III/IV 框架下，压力 VaR（Stressed VaR）基于金融危机期间的历史窗口，
    用于补充当前 VaR，以捕捉模型参数在平静期低估尾部风险的问题。
    本模块通过 ``vol_multiplier`` 近似模拟这一机制。

    Examples
    --------
    >>> from var_garch import StressTester
    >>> st = StressTester(port)
    >>>
    >>> # 1. 全部内置历史情景
    >>> hist = st.run_historical_scenarios()
    >>> st.print_scenario_summary(hist)
    >>>
    >>> # 2. 自定义假设情景（如利率上行200bps）
    >>> rate_shock = st.run_hypothetical_scenario(
    ...     name="利率上行200bp",
    ...     shocks={"TLT": -0.08, "AGG": -0.04, "AAPL": -0.02},
    ...     vol_multiplier=1.8,
    ...     description="模拟美联储意外加息200bps对组合的冲击",
    ... )
    >>>
    >>> # 3. 股票市场敏感性扫描
    >>> sens = st.sensitivity_analysis(
    ...     "股票市场",
    ...     sweep_range=(-0.20, 0.05),
    ...     asset_betas={"AAPL": 1.3, "MSFT": 1.1, "AGG": 0.05, "TLT": -0.10},
    ... )
    >>> st.plot_sensitivity(sens, show=False)
    >>>
    >>> # 4. 反向压力测试：找到产生 10% 损失的最小均匀冲击
    >>> rev = st.reverse_stress_test(loss_threshold=0.10)
    >>> st.print_reverse_stress(rev)
    >>>
    >>> # 5. 相关性压力测试
    >>> corr_df = st.correlation_stress(target_correlation=0.95)
    >>> st.plot_correlation_stress(corr_df, show=False)
    """

    def __init__(self, portfolio) -> None:
        self._port           = portfolio
        self._tickers        = portfolio.tickers
        self._weights        = np.array([portfolio.weights[t] for t in self._tickers])
        self._portfolio_value = portfolio.portfolio_value
        self._baseline_var   = portfolio._portfolio_var
        self._garch          = portfolio._garch_results
        self._corr           = portfolio.correlation_matrix.values
        self._returns        = portfolio._returns
        self._confidence     = portfolio.confidence

    # ------------------------------------------------------------------
    # 1. 历史情景
    # ------------------------------------------------------------------

    def run_historical_scenarios(
        self,
        scenarios:         Optional[List[str]]       = None,
        custom_ticker_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ScenarioResult]:
        """对所有内置历史危机情景进行损益测算。

        Parameters
        ----------
        scenarios : list of str, optional
            情景键列表，取自 :data:`HISTORICAL_SCENARIOS`；默认运行全部8个。
        custom_ticker_map : dict, optional
            覆盖自动资产大类分类，例如 ``{"MYETF": "fi_long"}``。
            有效大类：``"equity"``、``"fi_long"``、``"fi_ig"``、``"fi_hy"``。

        Returns
        -------
        dict of {str: ScenarioResult}
        """
        keys    = scenarios or list(HISTORICAL_SCENARIOS.keys())
        classes = self._classify_tickers(custom_ticker_map)
        results: Dict[str, ScenarioResult] = {}

        for key in keys:
            if key not in HISTORICAL_SCENARIOS:
                raise ValueError(
                    f"未知情景键 '{key}'，可用情景：{list(HISTORICAL_SCENARIOS.keys())}"
                )
            spec     = HISTORICAL_SCENARIOS[key]
            shocks_k = spec["shocks"]
            # 按资产大类取冲击值；若无精确大类则使用权益冲击作为保守估计
            asset_rets = {
                t: shocks_k.get(classes.get(t, "equity"), shocks_k["equity"])
                for t in self._tickers
            }
            results[key] = self._compute_scenario(
                scenario_name=key,
                label=spec["label"],
                description=spec["description"],
                asset_returns=asset_rets,
                vol_multiplier=spec["vol_multiplier"],
            )
        return results

    # ------------------------------------------------------------------
    # 2. 假设情景
    # ------------------------------------------------------------------

    def run_hypothetical_scenario(
        self,
        name:          str,
        shocks:        Dict[str, float],
        vol_multiplier: float = 1.0,
        description:   str   = "",
        fill_missing:  float = 0.0,
    ) -> ScenarioResult:
        """对自定义冲击向量进行组合损益测算。

        Parameters
        ----------
        name : str
            情景名称（用于展示与图表标题）。
        shocks : dict
            各资产单日对数收益率冲击 ``{ticker: δ_k}``，负数表示下跌。
            未指定的资产使用 ``fill_missing`` 冲击。
        vol_multiplier : float
            波动率放大系数，用于计算压力 VaR；
            若仅关心 P&L 估计，保持默认 1.0。
        description : str
            情景的文字描述，存储于结果对象中。
        fill_missing : float
            未出现在 ``shocks`` 中的资产的默认冲击值。

        Returns
        -------
        ScenarioResult

        Examples
        --------
        >>> r = st.run_hypothetical_scenario(
        ...     name="股债双杀",
        ...     shocks={"AAPL": -0.06, "MSFT": -0.06, "AGG": -0.04, "TLT": -0.08},
        ...     vol_multiplier=2.5,
        ...     description="美联储意外加息 + 经济衰退预期同步冲击",
        ... )
        >>> print(f"情景损失: {r.portfolio_loss_pct*100:.2f}%  "
        ...       f"({r.var_multiple:.1f}×基准VaR)")
        """
        asset_rets = {t: shocks.get(t, fill_missing) for t in self._tickers}
        return self._compute_scenario(
            scenario_name  = name,
            label          = name,
            description    = description or f"假设情景：{name}",
            asset_returns  = asset_rets,
            vol_multiplier = vol_multiplier,
        )

    def run_multiple_hypothetical(
        self,
        scenarios: List[Dict],
    ) -> Dict[str, ScenarioResult]:
        """批量执行多个假设情景。

        Parameters
        ----------
        scenarios : list of dict
            每个 dict 须包含 ``name``、``shocks``，可选项：
            ``vol_multiplier``、``description``、``fill_missing``。

        Returns
        -------
        dict of {str: ScenarioResult}

        Examples
        --------
        >>> results = st.run_multiple_hypothetical([
        ...     {
        ...         "name": "利率上行200bp",
        ...         "shocks": {"TLT": -0.08, "AGG": -0.04, "AAPL": -0.02},
        ...         "vol_multiplier": 1.8,
        ...         "description": "货币政策意外收紧",
        ...     },
        ...     {
        ...         "name": "科技股崩盘-30%",
        ...         "shocks": {"AAPL": -0.30, "MSFT": -0.30},
        ...         "vol_multiplier": 4.0,
        ...     },
        ...     {
        ...         "name": "滞胀情景",
        ...         "shocks": {"AAPL": -0.04, "MSFT": -0.04,
        ...                     "AGG": -0.03, "TLT": -0.06},
        ...         "vol_multiplier": 2.0,
        ...         "description": "增长停滞 + 通胀持续，股债均受压",
        ...     },
        ... ])
        """
        return {
            spec["name"]: self.run_hypothetical_scenario(
                name          = spec["name"],
                shocks        = spec["shocks"],
                vol_multiplier = spec.get("vol_multiplier", 1.0),
                description   = spec.get("description", ""),
                fill_missing  = spec.get("fill_missing",  0.0),
            )
            for spec in scenarios
        }

    # ------------------------------------------------------------------
    # 3. 单因子敏感性分析
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        factor:      str,
        sweep_range: Tuple[float, float] = (-0.20, 0.10),
        n_steps:     int                 = 60,
        asset_betas: Optional[Dict[str, float]] = None,
    ) -> SensitivityResult:
        r"""单因子冲击扫描：计算组合损益关于因子冲击的函数。

        各资产对因子 :math:`f` 的冲击响应为 :math:`\delta_k = \beta_k \cdot f`，
        组合损失为

        .. math::

            L(f) = -\sum_k w_k \beta_k f

        Parameters
        ----------
        factor : str
            因子名称（用于图表标注），如 ``"股票市场"``、``"利率"``、``"信用利差"``。
        sweep_range : tuple of float
            因子冲击扫描区间 :math:`(f_{\min}, f_{\max})`，以日对数收益率计。
        n_steps : int
            扫描步数。
        asset_betas : dict, optional
            各资产对因子的敏感性系数 :math:`\beta_k`。
            默认：权益资产 :math:`\beta=1.0`，固定收益 :math:`\beta=0.1`。

        Returns
        -------
        SensitivityResult
        """
        factor_values = np.linspace(sweep_range[0], sweep_range[1], n_steps)
        classes       = self._classify_tickers()

        if asset_betas is None:
            # 根据资产大类赋予默认 beta：权益 1.0，固收 0.1
            asset_betas = {
                t: 1.0 if classes.get(t, "equity") == "equity" else 0.1
                for t in self._tickers
            }

        portfolio_losses: list[float] = []
        asset_rows:       list[dict]  = []

        for fval in factor_values:
            asset_rets = {t: asset_betas.get(t, 1.0) * fval for t in self._tickers}
            portfolio_losses.append(self._portfolio_loss(asset_rets))
            # 各资产加权损失（用于堆叠面积图）
            asset_rows.append({
                t: -asset_rets[t] * self._weights[i]
                for i, t in enumerate(self._tickers)
            })

        return SensitivityResult(
            factor           = factor,
            values           = factor_values,
            portfolio_losses = np.array(portfolio_losses),
            asset_losses     = pd.DataFrame(asset_rows),
            baseline_loss    = 0.0,
        )

    def factor_ladder(
        self,
        factors:          List[str],
        shock_magnitude:  float                      = -0.05,
        asset_factor_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """对多个因子逐一施加固定量级冲击（因子梯度/瀑布图）。

        每次对单一因子施加 ``shock_magnitude`` 量级的冲击，
        其余因子保持不变，从而量化各因子的独立贡献。

        Parameters
        ----------
        factors : list of str
            因子名称列表。
        shock_magnitude : float
            冲击幅度（负数表示下行冲击）。
        asset_factor_map : dict, optional
            ``{因子名: {ticker: beta}}``；若为 None 则使用默认 beta。

        Returns
        -------
        pd.DataFrame
            索引为因子名，列：``shock_%``、``portfolio_loss_%``、``portfolio_loss_$``。
        """
        classes = self._classify_tickers()
        rows    = []

        for factor in factors:
            betas = (
                (asset_factor_map or {}).get(factor)
                or {
                    t: 1.0 if classes.get(t, "equity") == "equity" else 0.1
                    for t in self._tickers
                }
            )
            asset_rets = {t: betas.get(t, 1.0) * shock_magnitude for t in self._tickers}
            loss       = self._portfolio_loss(asset_rets)
            rows.append({
                "factor":            factor,
                "shock_%":           shock_magnitude * 100,
                "portfolio_loss_%":  round(loss * 100, 4),
                "portfolio_loss_$":  round(loss * self._portfolio_value, 2),
            })

        return pd.DataFrame(rows).set_index("factor")

    # ------------------------------------------------------------------
    # 4. 反向压力测试
    # ------------------------------------------------------------------

    def reverse_stress_test(
        self,
        loss_threshold: float,
        mode:           str   = "uniform_shock",
        max_iterations: int   = 200,
        tolerance:      float = 1e-6,
    ) -> ReverseStressResult:
        r"""寻找恰好达到损失阈值的最小冲击（二分法求解）。

        **均匀冲击模式（uniform_shock）**

        求解

        .. math::

            f^* = \arg\min_f |f|
            \quad \text{s.t.} \quad L(f \cdot \mathbf{1}) \geq L^*

        其中 :math:`\mathbf{1}` 表示所有资产同比例受冲击。

        **波动率倍数模式（vol_multiplier）**

        求解使组合压力 VaR 超过阈值的最小 GARCH 波动率倍数 :math:`m^*`：

        .. math::

            m^* = \arg\min_m m
            \quad \text{s.t.} \quad \mathrm{VaR}^{\text{stress}}_p(m) \geq L^*

        Parameters
        ----------
        loss_threshold : float
            损失阈值（小数形式，例如 ``0.10`` = 10%）。
        mode : str
            ``"uniform_shock"`` 或 ``"vol_multiplier"``。
        max_iterations : int
            二分法最大迭代次数，50 次通常已足够精确。
        tolerance : float
            收敛判据：:math:`|L(f) - L^*| < \\varepsilon`。

        Returns
        -------
        ReverseStressResult

        Examples
        --------
        >>> # 找出导致 10% 损失的最小均匀跌幅
        >>> r = st.reverse_stress_test(0.10, mode="uniform_shock")
        >>> print(f"需要均匀跌幅: {r.min_shock*100:.2f}%")
        >>>
        >>> # 找出需要 VaR 超过 5% 的最小波动率倍数
        >>> r2 = st.reverse_stress_test(0.05, mode="vol_multiplier")
        >>> print(f"需要波动率倍数: {r2.min_vol_multiplier:.2f}×")
        """
        if mode == "uniform_shock":
            return self._reverse_uniform_shock(loss_threshold, max_iterations, tolerance)
        elif mode == "vol_multiplier":
            return self._reverse_vol_multiplier(loss_threshold, max_iterations, tolerance)
        else:
            raise ValueError(f"mode 须为 'uniform_shock' 或 'vol_multiplier'，当前：'{mode}'")

    # ------------------------------------------------------------------
    # 5. 相关性压力测试
    # ------------------------------------------------------------------

    def correlation_stress(
        self,
        target_correlation: float = 0.90,
        steps:              int   = 15,
    ) -> pd.DataFrame:
        r"""将相关矩阵从基准值线性插值至目标值，测量组合 VaR 的变化。

        危机期间，资产间相关系数趋向于 1，分散化效益消失。
        压力相关矩阵定义为

        .. math::

            \mathbf{C}^{(\lambda)} = (1-\lambda)\hat{\mathbf{C}}
            + \lambda \mathbf{C}^{\text{target}},
            \quad \lambda \in [0,1]

        其中 :math:`C^{\text{target}}_{ij} = \rho^* \; (i \ne j)`,
        :math:`C^{\text{target}}_{ii} = 1`。

        Parameters
        ----------
        target_correlation : float
            非对角元素的目标相关系数 :math:`\\rho^*`（通常取 0.85–0.95）。
        steps : int
            插值步数（:math:`\\lambda` 从 0 到 1 的离散化点数）。

        Returns
        -------
        pd.DataFrame
            列：``correlation_level``（平均非对角相关系数）、
            ``portfolio_var_%``、``portfolio_var_$``、``diversif_benefit_%``。
        """
        K        = len(self._tickers)
        baseline = self._corr.copy()
        # 目标相关矩阵：非对角元素均为 target_correlation
        C_target = (
            target_correlation * np.ones((K, K))
            + (1.0 - target_correlation) * np.eye(K)
        )

        z    = abs(stats.norm.ppf(1.0 - self._confidence))
        rows = []

        for lam in np.linspace(0.0, 1.0, steps):
            # 线性插值生成压力相关矩阵
            C_stress = (1.0 - lam) * baseline + lam * C_target
            np.fill_diagonal(C_stress, 1.0)   # 对角线固定为 1

            pv     = self._portfolio_var_from_corr(C_stress)
            avg_rho = float(C_stress[np.triu_indices(K, k=1)].mean())

            # 无分散化情形的 VaR（完全相关，即个体 VaR 加总）
            sigma_k = np.array([
                np.sqrt(self._garch[t].forecast_variance) for t in self._tickers
            ])
            undiv   = float((self._weights * sigma_k).sum() * z)

            rows.append({
                "correlation_level": round(avg_rho, 4),
                "portfolio_var_%":   round(pv * 100, 4),
                "portfolio_var_$":   round(pv * self._portfolio_value, 0),
                "diversif_benefit_%": round((undiv - pv) * 100, 4),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def scenario_summary_table(
        self, results: Dict[str, ScenarioResult]
    ) -> pd.DataFrame:
        """将情景结果导出为按损失降序排列的汇总 DataFrame。"""
        rows = [
            {
                "情景":          r.label,
                "组合损失_%":    round(r.portfolio_loss_pct    * 100, 3),
                "组合损失_$":    round(r.portfolio_loss_dollar, 0),
                "压力VaR_%":     round(r.stressed_var          * 100, 3),
                "压力VaR_$":     round(r.stressed_var_dollar,   0),
                "VaR倍数_x":     round(r.var_multiple,          2),
            }
            for r in results.values()
        ]
        df = pd.DataFrame(rows).set_index("情景")
        return df.sort_values("组合损失_%", ascending=False)

    def print_scenario_summary(
        self, results: Dict[str, ScenarioResult]
    ) -> None:
        """在标准输出中打印格式化的情景摘要表。"""
        bv = self._baseline_var
        print("\n" + "=" * 82)
        print("  压力测试结果摘要")
        print(
            f"  基准 {self._confidence*100:.0f}% VaR：{bv*100:.4f}%"
            f"  (${bv * self._portfolio_value:,.0f})"
        )
        print("=" * 82)
        print(f"  {'情景':<42} {'损失%':>7} {'损失$':>13} {'VaR倍数':>8}")
        print("-" * 82)
        for r in sorted(results.values(),
                        key=lambda x: x.portfolio_loss_pct, reverse=True):
            flag = " ⚡" if r.var_multiple > 3 else (" ★" if r.var_multiple > 2 else "   ")
            print(
                f"  {r.label:<42}"
                f"{r.portfolio_loss_pct*100:>7.2f}%"
                f"  ${r.portfolio_loss_dollar:>11,.0f}"
                f"  {r.var_multiple:>5.1f}×{flag}"
            )
        print("=" * 82)
        print("  ⚡ 损失 > 3×VaR    ★ 损失 > 2×VaR\n")

    def print_reverse_stress(self, result: ReverseStressResult) -> None:
        """打印反向压力测试结果。"""
        print("\n" + "=" * 60)
        print("  反向压力测试结果")
        print("=" * 60)
        print(
            f"  损失阈值       ：{result.threshold_pct*100:.2f}%"
            f"  (${result.threshold_dollar:,.0f})"
        )
        if result.min_shock:
            print(f"  最小均匀冲击   ：{result.min_shock*100:.4f}%")
        if result.min_vol_multiplier > 1.0:
            print(f"  最小波动率倍数 ：{result.min_vol_multiplier:.3f}×")
        print(f"  收敛迭代次数   ：{result.iterations}")
        print("  各资产冲击详情：")
        for ticker, val in result.asset_shocks.items():
            print(f"    {ticker:<10}  {val*100:.4f}%")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 图表
    # ------------------------------------------------------------------

    def plot_scenarios(
        self,
        results:     Dict[str, ScenarioResult],
        output_path: str  = "stress_scenarios.png",
        show:        bool = True,
    ) -> str:
        """水平条形图：各情景损失 vs 基准 VaR。"""
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        labels  = [r.label  for r in results.values()]
        losses  = [r.portfolio_loss_pct * 100 for r in results.values()]
        baseline = self._baseline_var * 100

        order   = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)
        labels  = [labels[i] for i in order]
        losses  = [losses[i] for i in order]
        colors  = [
            "#8B1A1A" if l > baseline * 3 else
            "#E24B4A" if l > baseline * 2 else
            "#378ADD"
            for l in losses
        ]

        fig, ax = plt.subplots(figsize=(13, max(4, len(labels) * 0.7 + 1)))
        bars    = ax.barh(range(len(labels)), losses, color=colors, alpha=0.85, height=0.65)
        ax.axvline(baseline,     color="#BA7517", lw=1.5, linestyle="--",
                   label=f"基准VaR ({baseline:.2f}%)")
        ax.axvline(baseline * 2, color="#8B1A1A", lw=1.0, linestyle=":",
                   alpha=0.55, label=f"2×VaR ({baseline*2:.2f}%)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("组合损失 (%)")
        ax.set_title("压力测试——情景损失 vs 基准 VaR", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, losses):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{output_path}")
        return output_path

    def plot_sensitivity(
        self,
        result:      SensitivityResult,
        output_path: str  = "sensitivity.png",
        show:        bool = True,
    ) -> str:
        """双面板图：组合损益曲线 + 各资产加权损失贡献。"""
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        x = result.values * 100
        y = result.portfolio_losses * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        ax1.plot(x, y, color="#E24B4A", lw=2)
        ax1.axhline(self._baseline_var * 100, color="#BA7517", lw=1.2,
                    linestyle="--", label="基准VaR")
        ax1.axvline(0, color="gray", lw=0.7, alpha=0.4)
        ax1.fill_between(x, 0, y, where=(np.array(y) > 0),
                         alpha=0.08, color="#E24B4A")
        ax1.set_xlabel(f"{result.factor}冲击 (%)")
        ax1.set_ylabel("组合损失 (%)")
        ax1.set_title(f"P&L 曲线——{result.factor}", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.spines[["top", "right"]].set_visible(False)

        clrs = ["#E24B4A", "#378ADD", "#3B6D11", "#BA7517", "#534AB7", "#0F6E56"]
        for i, col in enumerate(result.asset_losses.columns):
            ax2.plot(x, result.asset_losses[col].values * 100,
                     label=col, color=clrs[i % len(clrs)], lw=1.3, alpha=0.85)
        ax2.axvline(0, color="gray", lw=0.7, alpha=0.4)
        ax2.set_xlabel(f"{result.factor}冲击 (%)")
        ax2.set_ylabel("加权资产损失 (%)")
        ax2.set_title("各资产损失贡献", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"单因子敏感性分析——{result.factor}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{output_path}")
        return output_path

    def plot_correlation_stress(
        self,
        df:          pd.DataFrame,
        output_path: str  = "correlation_stress.png",
        show:        bool = True,
    ) -> str:
        """双纵轴图：压力 VaR 与分散化收益随相关系数的变化。"""
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()

        ax1.plot(df["correlation_level"], df["portfolio_var_%"],
                 color="#E24B4A", lw=2, label="组合VaR%")
        ax1.fill_between(df["correlation_level"], df["portfolio_var_%"],
                         alpha=0.08, color="#E24B4A")
        ax1.axhline(self._baseline_var * 100, color="#BA7517", lw=1.2,
                    linestyle="--", alpha=0.8, label="基准VaR")
        ax1.set_xlabel("平均非对角相关系数 ρ")
        ax1.set_ylabel("组合VaR (%)", color="#E24B4A")
        ax1.tick_params(axis="y", colors="#E24B4A")

        ax2.plot(df["correlation_level"], df["diversif_benefit_%"],
                 color="#378ADD", lw=1.5, linestyle="-.", label="分散化收益%")
        ax2.set_ylabel("分散化收益 (%)", color="#378ADD")
        ax2.tick_params(axis="y", colors="#378ADD")

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, fontsize=9)

        ax1.set_title("相关性压力测试——VaR 与分散化收益", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.25)
        ax1.spines[["top"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{output_path}")
        return output_path

    def plot_factor_ladder(
        self,
        df:          pd.DataFrame,
        output_path: str  = "factor_ladder.png",
        show:        bool = True,
    ) -> str:
        """因子梯度图：各因子单独冲击下的组合损失。"""
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        factors = df.index.tolist()
        losses  = df["portfolio_loss_%"].values
        colors  = ["#E24B4A" if l > 0 else "#378ADD" for l in losses]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(factors)), losses, color=colors, alpha=0.85, width=0.6)
        ax.axhline(self._baseline_var * 100, color="#BA7517", lw=1.2,
                   linestyle="--", label="基准VaR")
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels(factors, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("组合损失 (%)")
        ax.set_title("因子梯度分析（各因子单独5%冲击）", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.005, f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"已保存：{output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _classify_tickers(
        self, overrides: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """根据代码自动判断资产大类（权益 / 长债 / 投资级信用 / 高收益信用）。"""
        out = {}
        for t in self._tickers:
            if overrides and t in overrides:
                out[t] = overrides[t]
            elif t in _FI_LONG_TICKERS:
                out[t] = "fi_long"
            elif t in _FI_IG_TICKERS:
                out[t] = "fi_ig"
            elif t in _FI_HY_TICKERS:
                out[t] = "fi_hy"
            else:
                out[t] = "equity"
        return out

    def _portfolio_loss(self, asset_returns: Dict[str, float]) -> float:
        r"""计算组合损失 :math:`L = -\mathbf{w}^\top \boldsymbol{\delta}`（正数表示损失）。"""
        return -float(
            sum(self._weights[i] * asset_returns.get(t, 0.0)
                for i, t in enumerate(self._tickers))
        )

    def _stressed_portfolio_var(self, vol_multiplier: float) -> float:
        r"""在 GARCH 波动率放大 vol_multiplier 倍后重估组合 VaR。

        压力 VaR 定义为

        .. math::

            \mathrm{VaR}^{\text{stress}}_p
            = |z_\alpha| \sqrt{\mathbf{w}^\top \mathbf{D}^* \mathbf{C} \mathbf{D}^* \mathbf{w}}
            \cdot \sqrt{h}

        其中 :math:`\mathbf{D}^* = m \cdot \mathbf{D}`，:math:`m` = vol_multiplier。
        """
        sigma  = np.array([
            np.sqrt(self._garch[t].forecast_variance) * vol_multiplier
            for t in self._tickers
        ])
        D      = np.diag(sigma)
        Cov    = D @ self._corr @ D
        pv     = float(self._weights @ Cov @ self._weights)
        z      = abs(stats.norm.ppf(1.0 - self._confidence))
        return  float(np.sqrt(pv) * z)

    def _portfolio_var_from_corr(self, corr: np.ndarray) -> float:
        """使用自定义相关矩阵计算组合 VaR（供相关性压力测试使用）。"""
        sigma = np.array([
            np.sqrt(self._garch[t].forecast_variance) for t in self._tickers
        ])
        D   = np.diag(sigma)
        Cov = D @ corr @ D
        pv  = float(self._weights @ Cov @ self._weights)
        z   = abs(stats.norm.ppf(1.0 - self._confidence))
        return float(np.sqrt(pv) * z)

    def _compute_scenario(
        self,
        scenario_name:  str,
        label:          str,
        description:    str,
        asset_returns:  Dict[str, float],
        vol_multiplier: float,
    ) -> ScenarioResult:
        """核心计算：给定资产收益率冲击，输出组合损益与压力 VaR。"""
        port_loss = self._portfolio_loss(asset_returns)

        # 各资产损失（正数表示损失）
        asset_losses = pd.Series(
            {t: -asset_returns[t] for t in self._tickers},
            name="loss_pct",
        )
        asset_losses_dollar = pd.Series(
            {
                t: asset_losses[t] * self._weights[i] * self._portfolio_value
                for i, t in enumerate(self._tickers)
            },
            name="loss_dollar",
        )
        stressed_var = self._stressed_portfolio_var(vol_multiplier)

        return ScenarioResult(
            scenario_name          = scenario_name,
            label                  = label,
            description            = description,
            portfolio_loss_pct     = port_loss,
            portfolio_loss_dollar  = port_loss * self._portfolio_value,
            stressed_var           = stressed_var,
            stressed_var_dollar    = stressed_var * self._portfolio_value,
            asset_losses           = asset_losses,
            asset_losses_dollar    = asset_losses_dollar,
            baseline_var           = self._baseline_var,
            var_multiple           = (port_loss / self._baseline_var)
                                     if self._baseline_var else 0.0,
        )

    def _reverse_uniform_shock(
        self, threshold: float, max_iter: int, tol: float
    ) -> ReverseStressResult:
        """二分法求解均匀冲击下的反向压力临界值。"""
        lo, hi = -threshold * 5.0, 0.0
        mid    = lo

        for i in range(max_iter):
            mid  = (lo + hi) / 2.0
            loss = self._portfolio_loss({t: mid for t in self._tickers})
            if abs(loss - threshold) < tol:
                break
            if loss < threshold:
                hi = mid
            else:
                lo = mid

        return ReverseStressResult(
            threshold_pct      = threshold,
            threshold_dollar   = threshold * self._portfolio_value,
            min_shock          = mid,
            min_vol_multiplier = 1.0,
            asset_shocks       = pd.Series(
                {t: mid for t in self._tickers}, name="均匀冲击"
            ),
            iterations         = i + 1,
        )

    def _reverse_vol_multiplier(
        self, threshold: float, max_iter: int, tol: float
    ) -> ReverseStressResult:
        """二分法求解波动率倍数模式下的反向压力临界值。"""
        lo, hi = 1.0, 30.0
        mid    = hi

        for i in range(max_iter):
            mid = (lo + hi) / 2.0
            sv  = self._stressed_portfolio_var(mid)
            if abs(sv - threshold) < tol:
                break
            if sv < threshold:
                lo = mid
            else:
                hi = mid

        return ReverseStressResult(
            threshold_pct      = threshold,
            threshold_dollar   = threshold * self._portfolio_value,
            min_shock          = 0.0,
            min_vol_multiplier = mid,
            asset_shocks       = pd.Series(
                {
                    t: np.sqrt(self._garch[t].forecast_variance) * mid
                    for t in self._tickers
                },
                name="压力条件波动率",
            ),
            iterations         = i + 1,
        )
