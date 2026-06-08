# -*- coding: utf-8 -*-
"""
var_garch — 基于GARCH族模型的风险价值（VaR）估计框架
=====================================================

理论基础
--------
设资产价格过程 :math:`S_t` 满足几何布朗运动的离散近似，对数收益率定义为

.. math::

    r_t = \\ln\\frac{S_t}{S_{t-1}}

在GARCH(p,q)框架下，条件方差 :math:`\\sigma_t^2` 服从递推方程

.. math::

    \\sigma_t^2 = \\omega
    + \\sum_{i=1}^{q} \\alpha_i \\varepsilon_{t-i}^2
    + \\sum_{j=1}^{p} \\beta_j \\sigma_{t-j}^2

其中 :math:`\\varepsilon_t = \\sigma_t z_t`，:math:`z_t \\overset{\\text{i.i.d.}}{\\sim} D(0,1)`。

模块结构
--------
- ``data``      : Yahoo Finance 行情数据获取
- ``returns``   : 收益率序列计算与统计量
- ``garch``     : GARCH族模型拟合（GARCH / EGARCH / GJR-GARCH）
- ``var``       : VaR 与 ES 四种估计方法
- ``portfolio`` : 多资产组合层面 VaR 聚合（含相关性修正）
- ``stress``    : 情景压力测试与敏感性分析
- ``report``    : 图表生成与报告输出

版本: 2.0.0
"""

from utils.data import fetch_prices, fetch_yields
from utils.returns import log_returns, pct_returns, annualise_vol, rolling_vol
from garch import GARCHModel, GARCHResult
from var import VaREstimator, VaRResult
from .portfolio import Portfolio
from .stress    import StressTester, HISTORICAL_SCENARIOS
from .report    import RiskReport

__version__ = "2.0.0"
__author__  = "var_garch contributors"

__all__ = [
    # 数据层
    "fetch_prices",
    "fetch_yields",
    # 收益率
    "log_returns",
    "pct_returns",
    "annualise_vol",
    "rolling_vol",
    # GARCH 模型
    "GARCHModel",
    "GARCHResult",
    # VaR 估计
    "VaREstimator",
    "VaRResult",
    # 组合层面
    "Portfolio",
    # 压力测试
    "StressTester",
    "HISTORICAL_SCENARIOS",
    # 报告
    "RiskReport",
]
