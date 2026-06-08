# -*- coding: utf-8 -*-
"""
examples/main.py — 端到端使用示例
==========================================

运行方式：
    python examples/main.py

依赖：
    pip install var-garch yfinance arch scipy matplotlib
"""

from var_garch.utils import (
    fetch_prices,
    fetch_yields,
    log_returns
)

from var_garch import(
    GARCHModel,
    VaREstimator,
    Portfolio,
    StressTester,
    RiskReport,
    HISTORICAL_SCENARIOS
)

# ── 1. 获取行情数据 ────────────────────────────────────────────────────

print("正在从 Yahoo Finance 下载行情数据…")

equity_tickers = ["AAPL", "MSFT", "JPM"]
fi_tickers     = ["AGG", "TLT"]                 # 投资级综合 + 长期国债 ETF
all_tickers    = equity_tickers + fi_tickers

prices = fetch_prices(all_tickers, start="2021-01-01")
print(prices.tail())

# ── 2. 计算对数收益率 ──────────────────────────────────────────────────

rets = log_returns(prices)
print(f"\n收益率矩阵形状：{rets.shape}")
print(rets.describe().round(6))

# ── 3. 单资产 GARCH 拟合 ───────────────────────────────────────────────

print("\n── GARCH 模型对比（AAPL）──")
tbl = GARCHModel.compare_models(rets["AAPL"], ticker="AAPL")
print(tbl)

print("\n── GJR-GARCH(1,1) + Student-t 拟合（AAPL）──")
model = GARCHModel(p=1, q=1, vol="GJR-GARCH", dist="t")
g     = model.fit(rets["AAPL"], ticker="AAPL")
print(g.summary())

# 10 步超前条件波动率预测（均值回归至长期波动率）
fc = g.multi_step_forecast(horizon=10)
print("10日条件波动率预测（年化%）：")
print((fc * 100).round(3))

# ── 4. 单资产 VaR（固定收益 AGG） ─────────────────────────────────────

print("\n── AGG 的四方法 VaR ──")
est    = VaREstimator(confidence=0.99, horizon=1, vol_process="GARCH", dist="t")
var_fi = est.estimate(rets["AGG"], ticker="AGG", portfolio_value=500_000)
print(var_fi)

# ── 5. 多资产组合 VaR ────────────────────────────────────────────────

print("\n── 组合 VaR（60/40 变体）──")
port = Portfolio(
    tickers  = all_tickers,
    weights  = {"AAPL": 0.25, "MSFT": 0.25, "JPM": 0.10, "AGG": 0.25, "TLT": 0.15},
    confidence      = 0.95,
    horizon         = 1,
    vol_process     = "GJR-GARCH",
    dist            = "t",
    portfolio_value = 1_000_000,
)
port.fit(prices)
print(port.summary())
print("\n风险指标 DataFrame：")
print(port.to_dataframe().round(4))

# ── 6. 回测（Kupiec POF 检验） ───────────────────────────────────────

print("\n── 滚动 VaR 回测（AAPL，252日窗口）──")
bt = est.backtest(rets["AAPL"], ticker="AAPL", window=252)
breach_rate = bt["breach"].mean()
print(f"实际违约率：{breach_rate*100:.3f}%（理论 α = 1.00%）")

# ── 7. 压力测试 ──────────────────────────────────────────────────────

st = StressTester(port)

print("\n── 历史情景压力测试 ──")
hist = st.run_historical_scenarios()
st.print_scenario_summary(hist)

print("\n── 自定义假设情景 ──")
custom = st.run_multiple_hypothetical([
    {
        "name":        "利率上行200bp",
        "shocks":      {"TLT": -0.08, "AGG": -0.04, "AAPL": -0.02, "MSFT": -0.02, "JPM": -0.01},
        "vol_multiplier": 1.8,
        "description": "美联储意外加息200bps，货币政策转鹰",
    },
    {
        "name":        "科技股崩盘-30%",
        "shocks":      {"AAPL": -0.30, "MSFT": -0.30},
        "vol_multiplier": 4.0,
        "description": "监管打压或盈利暴雷引发科技板块深度回调",
    },
    {
        "name":        "滞胀情景",
        "shocks":      {"AAPL": -0.04, "MSFT": -0.04,
                         "JPM": -0.03, "AGG": -0.03, "TLT": -0.06},
        "vol_multiplier": 2.0,
        "description": "经济停滞 + 通胀持续，股债同步承压",
    },
])
st.print_scenario_summary(custom)

print("\n── 反向压力测试 ──")
rev_shock = st.reverse_stress_test(0.10, mode="uniform_shock")
st.print_reverse_stress(rev_shock)

rev_vol = st.reverse_stress_test(0.05, mode="vol_multiplier")
st.print_reverse_stress(rev_vol)

# ── 8. 完整报告生成 ──────────────────────────────────────────────────

print("\n── 生成完整风险报告 ──")
report = RiskReport(port)
report.print_summary()
report.to_csv("var_results.csv")
report.plot_all(output_dir="./figures", show=False)
report.stress_report(output_dir="./figures", show=False)

print("\n✓ 全部流程完成。")
