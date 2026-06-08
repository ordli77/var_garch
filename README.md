# var-garch

**基于 GARCH 族模型的风险价值（VaR）估计框架，含压力测试与情景分析**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 理论概述

### 风险价值（Value at Risk）

给定置信水平 $1-\alpha$ 与持有期 $h$，组合收益率 $R_{t+1:t+h}$ 的 VaR 定义为

$$
\mathrm{VaR}_{1-\alpha}(h) = -F_R^{-1}(\alpha)
$$

即收益率分布左侧 $\alpha$ 分位数的绝对值（正数表示损失）。

### GARCH(1,1) 条件方差方程

$$
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,
\quad \varepsilon_t = \sigma_t z_t,\quad z_t \overset{\text{i.i.d.}}{\sim} D(0,1)
$$

协方差平稳性条件：$\alpha + \beta < 1$。无条件方差：$\bar\sigma^2 = \omega\,/\,(1-\alpha-\beta)$。

### 组合层 VaR（DCC 近似）

$$
\hat{\boldsymbol{\Sigma}}_{T+1|T} = \mathbf{D}_{T+1}\,\hat{\mathbf{C}}\,\mathbf{D}_{T+1},
\quad \hat{\sigma}_p^2 = \mathbf{w}^\top\hat{\boldsymbol{\Sigma}}_{T+1|T}\mathbf{w}
$$

其中 $\mathbf{D}_{T+1} = \mathrm{diag}(\hat\sigma_{1,T+1|T},\ldots,\hat\sigma_{K,T+1|T})$，
$\hat{\mathbf{C}}$ 为历史样本相关矩阵（DCC 近似）。

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/var-garch.git
cd var-garch

# 安装运行时依赖
pip install -r requirements.txt

# 或直接从源码安装包
pip install -e .

# 开发模式（含测试工具链）
pip install -r requirements-dev.txt
```

**依赖项说明**

| 包 | 最低版本 | 用途 |
|---|---|---|
| `yfinance` | 0.2.40 | Yahoo Finance 行情数据下载 |
| `arch` | 6.3 | GARCH 族模型 QML 估计（BFGS） |
| `numpy` | 1.24 | 矩阵运算、随机数生成（MC 路径） |
| `pandas` | 2.0 | 时间序列 DataFrame、日历索引对齐 |
| `scipy` | 1.10 | 分位数函数 $F_Z^{-1}(\alpha)$、密度 $\phi(z_\alpha)$ |
| `matplotlib` | 3.7 | 图表生成（8 类风险图表） |

---

## 快速开始

```python
from var_garch import fetch_prices, Portfolio, RiskReport, StressTester

# 1. 获取数据（股票 + 固定收益 ETF）
prices = fetch_prices(
    ["AAPL", "MSFT", "AGG", "TLT"],
    start="2020-01-01",
)

# 2. 构建组合并估计 VaR
port = Portfolio(
    tickers  = ["AAPL", "MSFT", "AGG", "TLT"],
    weights  = {"AAPL": 0.30, "MSFT": 0.30, "AGG": 0.20, "TLT": 0.20},
    confidence      = 0.95,
    horizon         = 1,
    vol_process     = "GJR-GARCH",   # 捕捉杠杆效应
    dist            = "t",           # Student-t 创新项（厚尾）
    portfolio_value = 1_000_000,
)
port.fit(prices)
print(port.summary())

# 3. 生成图表
report = RiskReport(port)
report.plot_all(output_dir="./figures", show=False)

# 4. 压力测试套件
report.stress_report(output_dir="./figures", show=False)
```

---

## 模块说明

### `data.py` — 行情数据获取

```python
from var_garch import fetch_prices, fetch_yields

# 权益
prices = fetch_prices(["AAPL", "MSFT", "JPM"], start="2021-01-01")

# 固定收益 ETF（AGG 投资级综合 / TLT 长期国债 / LQD 信用债 / HYG 高收益）
fi = fetch_yields(["AGG", "TLT", "BND", "LQD"], start="2021-01-01")
```

返回以交易日为索引的复权价格 DataFrame $\mathbf{P} \in \mathbb{R}^{T \times K}$。

---

### `returns.py` — 收益率与统计矩

```python
from var_garch import log_returns, pct_returns, annualise_vol
from var_garch.returns import sample_moments

rets  = log_returns(prices)       # r_t = ln(P_t/P_{t-1})
pcts  = pct_returns(prices)       # r_t = (P_t - P_{t-1})/P_{t-1}
sigma = annualise_vol(rets["AAPL"])  # σ_ann = σ_d · √252

# 样本矩（偏度、超额峰度用于诊断厚尾）
m = sample_moments(rets["AAPL"])
print(f"偏度: {m['skewness']:.3f}  超额峰度: {m['excess_kurtosis']:.3f}")
```

---

### `garch.py` — GARCH 族模型拟合

支持三种波动率过程与三种创新项分布：

| 过程 | 公式特征 | 适用场景 |
|---|---|---|
| `"GARCH"` | 对称 $\alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ | 固定收益 ETF、低杠杆效应资产 |
| `"GJR-GARCH"` | 非对称 $(\alpha+\gamma\mathbf{1}_{\varepsilon<0})\varepsilon^2$ | 权益（负冲击 → 更大波动率上升） |
| `"EGARCH"` | $\ln\sigma_t^2$ 的线性方程 | 天然保证方差非负；模型稳健性测试 |

| 分布 | 特征 | 适用场景 |
|---|---|---|
| `"normal"` | 正态创新项 | 基准模型；固定收益 |
| `"t"` | Student-t（厚尾） | 权益日收益率 |
| `"skewt"` | 偏斜 t（偏度 + 厚尾） | 高收益债、新兴市场 |

```python
from var_garch import GARCHModel

# 单资产拟合
model  = GARCHModel(p=1, q=1, vol="GJR-GARCH", dist="t")
result = model.fit(rets["AAPL"], ticker="AAPL")
print(result.summary())
# 输出：ω、α、β、γ、持续性 α+β、长期波动率、AIC/BIC

# 多步超前预测（条件方差均值回归）
# σ²_{T+h} ≈ σ̄² + (α+β)^{h-1} · (σ²_{T+1} - σ̄²)
fc = result.multi_step_forecast(horizon=10)

# 模型选择（AIC/BIC 比较 8 种规格）
tbl = GARCHModel.compare_models(rets["AAPL"], ticker="AAPL")
print(tbl.head(3))
```

---

### `var.py` — 四方法 VaR 估计

| 方法 | 属性 | 公式 |
|---|---|---|
| 参数法 GARCH | `var_parametric` | $\|z_\alpha\| \cdot \hat\sigma_{T+1\|T} \cdot \sqrt{h}$ |
| 历史模拟法 | `var_historical` | $-\hat{r}_{(\lfloor\alpha T\rfloor)}$ |
| 蒙特卡洛 | `var_montecarlo` | GARCH 递推 $N_\text{sim}=10^4$ 条路径的 $\alpha$ 分位数 |

```python
from var_garch import VaREstimator

est = VaREstimator(
    confidence    = 0.99,
    horizon       = 10,          # 巴塞尔框架 10 日持有期
    vol_process   = "GJR-GARCH",
    dist          = "t",
    n_simulations = 50_000,
)
result = est.estimate(rets["TLT"], ticker="TLT", portfolio_value=500_000)
print(result)

# Kupiec POF 回测检验
# H₀: 实际违约率 = α   LR_POF ~ χ²₁
bt = est.backtest(rets["AAPL"], ticker="AAPL", window=252)
```

---

### `portfolio.py` — 多资产组合 VaR

```python
from var_garch import Portfolio

port = Portfolio(
    tickers         = ["AAPL", "MSFT", "JPM", "AGG", "TLT"],
    weights         = {"AAPL": 0.25, "MSFT": 0.25, "JPM": 0.10,
                       "AGG": 0.25, "TLT": 0.15},
    confidence      = 0.95,
    horizon         = 1,
    vol_process     = "GJR-GARCH",
    dist            = "t",
    portfolio_value = 1_000_000,
)
port.fit(prices)

# 组合 VaR / ES（考虑相关性）
print(port.portfolio_var_dollar)   # 美元 VaR
print(port.portfolio_es_dollar)    # 美元 ES（CVaR）

# 成分 VaR（加和性：Σ CVaR_k = VaR_p）
print(port.risk_contributions)

# 分散化收益 = 个体 VaR 加权和 - 组合 VaR ≥ 0
print(port.to_dataframe())
```

---

### `stress.py` — 压力测试引擎

**五种分析模式：**

```python
from var_garch import StressTester

st = StressTester(port)


# 1. 假设情景（自定义冲击向量）
custom = st.run_multiple_hypothetical([
    {
        "name":           "利率上行200bp",
        "shocks":         {"TLT": -0.08, "AGG": -0.04, "AAPL": -0.02},
        "vol_multiplier": 1.8,
        "description":    "美联储意外加息，货币政策转鹰",
    },
    {
        "name":           "滞胀",
        "shocks":         {"AAPL": -0.04, "MSFT": -0.04,
                           "AGG":  -0.03, "TLT": -0.06},
        "vol_multiplier": 2.0,
    },
])

# 2. 单因子敏感性扫描（P&L 关于因子冲击的函数）
sens = st.sensitivity_analysis(
    "股票市场",
    sweep_range  = (-0.20, 0.05),
    asset_betas  = {"AAPL": 1.3, "MSFT": 1.1, "AGG": 0.05, "TLT": -0.10},
)
st.plot_sensitivity(sens, show=False)



# 3a. 反向压力测试——均匀冲击模式
#     求解 f* = argmin|f| s.t. L(f·1) ≥ 10%
rev = st.reverse_stress_test(0.10, mode="uniform_shock")
st.print_reverse_stress(rev)

# 3b. 反向压力测试——波动率倍数模式
#     求解 m* s.t. VaR^stress_p(m) ≥ 5%
rev2 = st.reverse_stress_test(0.05, mode="vol_multiplier")

# 4. 相关性压力测试（ρ → 0.95，模拟危机期 correlation breakdown）
corr_df = st.correlation_stress(target_correlation=0.95, steps=20)
st.plot_correlation_stress(corr_df, show=False)
```


---

### `report.py` — 图表生成

```python
from var_garch import RiskReport

report = RiskReport(port)

# VaR 分析图表套件（5 张图）
report.plot_all(output_dir="./figures", show=False)

# 一键压力测试套件（7 步，生成 6 张图）
report.stress_report(output_dir="./figures", show=False)

# 导出 CSV（utf-8-sig 编码，Excel 直接打开无乱码）
report.to_csv("var_results.csv")
```

生成图表清单：

| 文件名 | 内容 |
|---|---|
| `return_distributions.png` | 各资产收益率直方图 + VaR/ES 切割线 |
| `garch_volatility.png` | GARCH 条件波动率时序（年化%） |
| `var_comparison.png` | 四方法 VaR 横向对比 |
| `component_var.png` | 成分 VaR 贡献分解 |
| `correlation_matrix.png` | 资产相关矩阵热力图 |
| `stress_historical.png` | 8 个历史情景损失 vs 基准 VaR |
| `sensitivity_equity.png` | 股票市场因子敏感性曲线 |
| `sensitivity_rates.png` | 利率因子敏感性曲线 |
| `factor_ladder.png` | 五大因子梯度柱状图 |
| `correlation_stress.png` | VaR 与分散化收益 vs 相关系数 |

---



---

## 模块依赖关系

```
外部库层:   yfinance   arch   scipy.stats   numpy/pandas
              │          │          │            │
数据层:    data.py   returns.py ──────────────────┘
              │          │
计算层:    garch.py ←── │ ──→ var.py
              │                  │
组合层:       └──── portfolio.py ──→ stress.py
                          │              │
报告层:                   └──── report.py
```

依赖图为严格有向无环图（DAG），数据流单向自下而上。底层模块可独立测试，替换某一层实现不产生级联修改。

---

## 运行测试

```bash
# 运行全部单元测试
pytest tests/ -v

# 含覆盖率报告
pytest tests/ --cov=var_garch --cov-report=html

# 代码格式检查
ruff check var_garch/
black --check --line-length 100 var_garch/

# 类型检查
mypy var_garch/
```




## 许可证

MIT License © var-garch contributors
