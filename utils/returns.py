# -*- coding: utf-8 -*-
"""
returns.py — 收益率序列计算与统计矩估计
========================================

理论背景
--------
设资产价格序列 :math:`\\{P_t\\}_{t=0}^T`，定义两种收益率：

**对数收益率（连续复利收益率）**

.. math::

    r_t^{\\log} = \\ln P_t - \\ln P_{t-1} = \\ln\\frac{P_t}{P_{t-1}}

对数收益率具有时间可加性（time-additivity），即多期对数收益率等于单期之和：

.. math::

    r_{t_1 \\to t_k}^{\\log} = \\sum_{i=1}^{k} r_{t_i}^{\\log}

**简单收益率**

.. math::

    r_t^{\\text{simple}} = \\frac{P_t - P_{t-1}}{P_{t-1}}

两者关系：:math:`r_t^{\\log} = \\ln(1 + r_t^{\\text{simple}})`，
当 :math:`|r_t^{\\text{simple}}| \\ll 1` 时两者近似相等。

**年化波动率**

若日收益率方差为 :math:`\\sigma_d^2`，则年化波动率为

.. math::

    \\sigma_{\\text{ann}} = \\sigma_d \\sqrt{N}

其中 :math:`N` 为年交易日数（权益市场通常取 252）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 收益率计算
# ---------------------------------------------------------------------------

def log_returns(
    prices: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """计算对数收益率序列 :math:`r_t = \\ln(P_t / P_{t-1})`。

    Parameters
    ----------
    prices : pd.DataFrame 或 pd.Series
        以交易日为索引的价格序列。多资产时每列对应一个资产。

    Returns
    -------
    pd.DataFrame 或 pd.Series
        对数收益率序列，第一行（NaN）已删除，共 :math:`T-1` 个观测值。

    Notes
    -----
    实现等价于 :math:`r_t = \\ln P_t - \\ln P_{t-1}`，
    利用 ``numpy.log`` 的数值稳定性优于直接做差。
    """
    # np.log(P_t / P_{t-1}) 等价于 np.log(P_t) - np.log(P_{t-1})
    # 后者在 P_t 极小时数值更稳定，但两者精度相当
    return np.log(prices / prices.shift(1)).dropna()


def pct_returns(
    prices: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """计算简单收益率序列 :math:`r_t = (P_t - P_{t-1}) / P_{t-1}`。

    Parameters
    ----------
    prices : pd.DataFrame 或 pd.Series
        价格序列。

    Returns
    -------
    pd.DataFrame 或 pd.Series
        简单收益率序列，第一行 NaN 已删除。
    """
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# 统计量
# ---------------------------------------------------------------------------

def annualise_vol(
    returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """计算年化波动率 :math:`\\hat{\\sigma}_{\\text{ann}} = \\hat{\\sigma}_d \\sqrt{N}`。

    Parameters
    ----------
    returns : pd.Series
        日收益率序列（小数形式，非百分比）。
    trading_days : int
        年交易日数 :math:`N`；权益 252，外汇通常取 260。

    Returns
    -------
    float
        年化波动率（小数形式）。

    Notes
    -----
    此处使用无偏样本标准差（分母 :math:`T-1`）估计日波动率，
    再乘以 :math:`\\sqrt{N}` 进行时间尺度变换。
    该变换隐含收益率序列 i.i.d. 假设，GARCH 模型放宽了此假设。
    """
    # pandas std() 默认 ddof=1，即无偏估计量
    sigma_d = float(returns.std(ddof=1))
    return sigma_d * np.sqrt(trading_days)


def rolling_vol(
    returns: pd.Series,
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """计算滚动已实现波动率（Rolling Realised Volatility）。

    .. math::

        \\hat{\\sigma}_{t,\\text{roll}}
        = \\sqrt{\\frac{1}{m-1} \\sum_{i=0}^{m-1}(r_{t-i} - \\bar{r})^2} \\cdot \\sqrt{N}

    Parameters
    ----------
    returns : pd.Series
        日收益率序列。
    window : int
        滚动窗口 :math:`m`（交易日数）。常用值：21（月）、63（季）、252（年）。
    trading_days : int
        年化系数 :math:`N`。

    Returns
    -------
    pd.Series
        年化滚动波动率序列，前 ``window-1`` 个观测为 NaN。
    """
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(trading_days)


def sample_moments(returns: pd.Series) -> dict:
    """计算收益率序列的样本矩：均值、方差、偏度、峰度。

    .. math::

        \\hat{\\mu} &= \\frac{1}{T}\\sum_{t=1}^T r_t \\\\
        \\hat{\\sigma}^2 &= \\frac{1}{T-1}\\sum_{t=1}^T (r_t - \\hat{\\mu})^2 \\\\
        \\hat{\\gamma}_1 &= \\frac{\\hat{\\mu}_3}{\\hat{\\sigma}^3},
        \\quad \\hat{\\mu}_3 = \\frac{1}{T}\\sum_t (r_t-\\hat{\\mu})^3 \\\\
        \\hat{\\kappa} &= \\frac{\\hat{\\mu}_4}{\\hat{\\sigma}^4} - 3
        \\quad (\\text{超额峰度，正态分布下为0})

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    dict
        键：``mean``, ``variance``, ``std``, ``skewness``, ``excess_kurtosis``,
        ``annual_vol``, ``sharpe_ratio``（假设无风险利率为0）。
    """
    T = len(returns)
    mu     = float(returns.mean())
    var    = float(returns.var(ddof=1))
    sigma  = float(np.sqrt(var))
    # 偏度：分布不对称性，负偏（左尾肥大）是金融收益率常见特征
    skew   = float(returns.skew())
    # 超额峰度：尖峰厚尾（leptokurtosis）程度，正态分布下超额峰度为 0
    kurt   = float(returns.kurt())
    ann_v  = sigma * np.sqrt(252)
    sharpe = (mu * 252) / ann_v if ann_v > 0 else float("nan")

    return {
        "mean":              mu,
        "variance":          var,
        "std":               sigma,
        "skewness":          skew,
        "excess_kurtosis":   kurt,
        "annual_vol":        ann_v,
        "sharpe_ratio":      sharpe,
    }
