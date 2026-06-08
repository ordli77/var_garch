# -*- coding: utf-8 -*-
"""
data.py — Yahoo Finance 行情数据获取模块
========================================

通过 ``yfinance`` 包下载调整后收盘价，支持股票与固定收益 ETF。
调整逻辑包含分拆复权（split adjustment）与股息再投资（dividend reinvestment）。


"""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: Union[str, List[str]],
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """从 Yahoo Finance 下载复权收盘价序列。


    ------------
    设下载得到的价格矩阵为

    .. math::

        \\mathbf{P} = (P_{t,k})_{t=1,\\ldots,T;\\; k=1,\\ldots,K}

    其中 :math:`T` 为观测天数，:math:`K` 为资产数量。

    Parameters
    ----------
    tickers : str 或 list of str
        Yahoo Finance 代码，例如 ``["AAPL", "TLT"]``。
    start : str
        起始日期，格式 ``"YYYY-MM-DD"``。
    end : str, optional
        截止日期；默认取当日。
    auto_adjust : bool
        是否启用 yfinance 自动复权（含分拆与股息）。

    Returns
    -------
    pd.DataFrame
        以交易日为索引、以代码为列名的价格 DataFrame :math:`\\mathbf{P}`。
        缺失日期（非交易日）已自动剔除。

    Raises
    ------
    ImportError
        若未安装 ``yfinance``。
    ValueError
        若返回数据为空（代码有误或日期超出范围）。

    Examples
    --------
    >>> # 同时获取权益与固定收益 ETF
    >>> prices = fetch_prices(["AAPL", "MSFT", "AGG", "TLT"],
    ...                        start="2020-01-01")
    >>> prices.shape
    (n_trading_days, 4)
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "需要安装 yfinance：pip install yfinance"
        ) from exc

    if isinstance(tickers, str):
        tickers = [tickers]

    # 多代码批量下载，启用多线程以加速
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    # yfinance 对单代码与多代码返回结构不同，统一处理
    col = "Close"                          # auto_adjust=True 时列名为 "Close"
    if len(tickers) == 1:
        prices = raw[[col]].copy()
        prices.columns = tickers
    else:
        prices = raw[col].copy()

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    if prices.empty:
        raise ValueError(
            f"未能获取数据，请检查代码 {tickers} 及日期范围 [{start}, {end}]。"
        )

    return prices


def fetch_yields(
    tickers: Union[str, List[str]],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """获取固定收益 ETF 价格序列（``fetch_prices`` 的语义封装）。

    常用固定收益 ETF 代码参考
    -------------------------
    .. list-table::
       :header-rows: 1

       * - 代码
         - 描述
         - 久期特征
       * - AGG
         - iShares 核心美国综合债券 ETF
         - 中等久期 (~6 年)
       * - TLT
         - iShares 20+ 年期美国国债 ETF
         - 长久期 (~17 年)
       * - IEF
         - iShares 7-10 年期美国国债 ETF
         - 中长久期 (~8 年)
       * - LQD
         - iShares 投资级企业债 ETF
         - 中等久期 + 信用利差风险
       * - HYG
         - iShares 高收益企业债 ETF
         - 高信用风险，久期较短
       * - BND
         - Vanguard 总债券市场 ETF
         - 宽基，中等久期

    Parameters
    ----------
    tickers : str 或 list of str
        固定收益 ETF 代码。
    start : str
        起始日期。
    end : str, optional
        截止日期。

    Returns
    -------
    pd.DataFrame
        价格矩阵，同 :func:`fetch_prices`。
    """
    # 固定收益 ETF 与权益的数据结构相同，价格驱动力不同
    # ETF 价格隐含了久期风险（利率敏感性）与信用利差变动
    return fetch_prices(tickers, start=start, end=end)
