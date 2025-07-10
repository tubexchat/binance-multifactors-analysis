# Binance Multi-Factors AI Analysis Platform

🚀 **专业的加密货币多因子分析平台，专注于IC（信息系数）相关性分析**

## 🎯 项目概述

这是一个基于Python的量化分析平台，专门用于分析加密货币市场的多种因子与未来收益的相关性。系统采用15分钟标准化数据，提供64种技术、市场和情绪因子的综合分析。

本项目Pro版本为基于机器学习模型对币价走势的预测，仅作为学习目的。

> ⚠️ 中国大陆用户需要启动VPN，并将VPN IP地址配置进币安API管理页面中。

## ✨ 核心功能

- **🔍 多因子分析**: 64种因子涵盖技术指标、市场数据和情绪分析
- **📊 IC相关性计算**: 使用多种相关性方法(Pearson、Spearman、Kendall)
- **⚡ 异步数据处理**: 高效的数据获取和批量计算
- **📈 智能缓存系统**: 优化性能，避免重复API调用
- **🎛️ 灵活配置**: 支持自定义分析参数和交易对

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥（创建.env文件）
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# 运行单币种分析
python main.py --symbol BTCUSDT

# 运行批量分析
python main.py --batch-symbols BTC,ETH,BNB

# Pro版本如下：
# 1. 演示系统（推荐首次使用）
python demo_strategy.py

# 2. 分析单个交易对
python strategies/strategy_main.py --symbol BTCUSDT

# 3. 批量分析主要交易对  
python strategies/strategy_main.py --major

# 4. 自定义参数分析
python strategies/strategy_main.py --symbol ETHUSDT --ic-threshold 0.08 --top-factors 3
```

## 📚 详细文档

## 🏗️ 系统架构

```
core/               # 核心模块（配置、数据管理、因子引擎）
factors/            # 因子库（技术、市场、情绪因子）
analysis/           # 分析接口
main.py            # 命令行入口
```

## 📊 分析结果示例

系统将输出每个因子的IC值、相关性强度、稳定性评估和投资建议，帮助识别最有效的量化因子。

## 🔬 因子体系详解

本系统包含64个精心设计的量化因子，分为三大类别：技术因子、市场因子和情绪因子。每个因子都经过专业的金融工程设计，用于捕捉不同维度的市场信息。

### 📈 技术因子 (Technical Factors)

技术因子基于价格和成交量的技术分析指标，是量化交易的基础因子类别。

#### **1. RSI系列因子 (7个)**
- `rsi_14/21/6`: 不同周期的相对强弱指数，衡量价格动量强度
- `rsi_oversold/overbought`: 超买超卖信号，识别价格极端状态
- `rsi_momentum`: RSI动量，捕捉RSI指标的变化趋势
- `rsi_cross_50`: RSI突破中轴线信号，判断趋势转换

#### **2. MACD系列因子 (8个)**
- `macd/macd_signal/macd_histogram`: 经典MACD三线，趋势跟踪核心指标
- `macd_cross_up/down`: MACD金叉死叉信号，趋势转换确认
- `macd_momentum/divergence`: MACD动量和背离，趋势强度分析
- `macd_fast`: 快速MACD，提升信号敏感度

#### **3. 移动平均线系列因子 (约25个)**
- `sma_5/10/20/50/100/200`: 多周期简单移动平均线
- `ema_5/10/20/50`: 指数移动平均线，对近期价格更敏感
- `*_ratio`: 价格相对均线偏离度，衡量超买超卖程度
- `*_cross_up`: 价格突破均线信号，确认趋势启动
- `ma_5_20_cross/ma_10_50_cross`: 双均线金叉系统

#### **4. 布林带系列因子 (8个)**
- `bollinger_upper/middle/lower`: 布林带三线，动态支撑阻力
- `bb_width`: 布林带宽度，衡量市场波动性
- `bb_position`: 价格在布林带中位置，相对强弱判断
- `bb_squeeze`: 布林带收缩，突破前的蓄势状态
- `bb_break_up/down`: 布林带突破信号，强势趋势确认

#### **5. 随机指标系列因子 (8个)**
- `stoch_k/d`: 慢速随机指标，经典超买超卖指标
- `stoch_fast_k/d`: 快速随机指标，提升信号敏感度
- `stoch_rsi`: 随机RSI，结合两种指标优势
- `stoch_oversold/overbought`: 随机指标超买超卖状态
- `stoch_cross_up`: 随机指标金叉信号

#### **6. 其他震荡指标 (威廉、CCI、ATR等，约15个)**
- `williams_r`: 威廉指标，反向随机指标
- `cci`: 商品通道指数，价格偏离统计均值程度
- `atr/true_range`: 真实波动幅度，衡量市场波动性
- `*_momentum`: 各指标动量，捕捉指标变化趋势

#### **7. 成交量技术指标 (7个)**
- `obv`: 能量潮指标，累积资金流向
- `mfi`: 资金流量指数，成交量版RSI
- `ad/adosc`: 累积分布线，基于价格位置的资金流向

### 🏛️ 市场因子 (Market Factors)

市场因子基于交易所提供的市场微观结构数据，反映机构行为和市场情绪。

#### **1. 资金费率系列因子 (9个)**
资金费率是期货合约特有的机制，反映市场对价格走向的预期和资金成本。
- `funding_rate`: 当前资金费率，正值表示多头付费
- `funding_rate_ma_8/24`: 不同周期均值，平滑短期波动
- `funding_rate_std_8`: 资金费率波动性
- `funding_rate_change/momentum`: 费率变化趋势
- `funding_rate_extreme_*`: 极端费率状态识别
- `funding_rate_neutral`: 中性费率状态

#### **2. 多空比系列因子 (9个)**
多空比反映市场参与者的仓位分布，是重要的情绪指标。
- `long_short_ratio`: 当前多空比，>1表示多头占优
- `ls_ratio_ma_12/48`: 多空比移动平均，情绪趋势
- `ls_ratio_std_12`: 多空比波动性，情绪稳定性
- `ls_ratio_extreme_*`: 极端情绪状态识别
- `ls_ratio_deviation`: 偏离中期均值程度
- `ls_ratio_trend`: 多空比变化趋势

#### **3. 持仓量相关因子 (4个)**
- `open_interest`: 当前持仓量
- `oi_change/momentum`: 持仓量变化趋势
- `oi_high`: 高持仓量标识

#### **4. 溢价相关因子 (4个)**
- `premium_index`: 溢价指数
- `mark_price/index_price`: 标记价格和指数价格
- `price_premium`: 价格溢价程度

#### **5. 价格相关市场因子 (15个)**
基于OHLCV数据的高级市场指标：
- `return_1/2/4/8`: 多周期收益率
- `volatility_12/48`: 不同周期波动率
- `skewness_12/kurtosis_12`: 收益率分布高阶特征
- `price_momentum_12/48`: 价格动量指标
- `volume_*`: 成交量相关衍生指标
- `vwap/price_to_vwap`: 成交量加权平均价格相关

### 🎭 情绪因子 (Sentiment Factors)

情绪因子通过多维度分析捕捉市场参与者的心理状态和行为模式。

#### **1. 动量类情绪因子 (9个)**
- `momentum_1/3/5/10/20`: 多周期价格动量
- `momentum_acceleration`: 动量加速度，情绪变化二阶导数
- `consecutive_up/down`: 连续涨跌期数，趋势持续性
- `momentum_strength`: 相对动量强度
- `trend_consistency`: 多时间框架趋势一致性
- `momentum_reversal`: 动量转折点识别

#### **2. 波动率类情绪因子 (12个)**
- `volatility_5/10/20`: 多周期收益率波动率
- `range_volatility_5/20`: 基于价格区间的波动率
- `volatility_trend/spike`: 波动率趋势和异常事件
- `conditional_volatility`: GARCH类条件波动率
- `volatility_skewness/kurtosis`: 波动率分布特征
- `vix_like`: 类VIX恐慌指数

#### **3. 成交量情绪因子 (8个)**
- `volume_change`: 成交量变化率
- `volume_price_divergence`: 量价背离信号
- `volume_surge`: 成交量异常放大
- `volume_price_cooperation`: 量价配合度
- `money_flow_strength`: 资金流向强度
- `cumulative_money_flow`: 累积资金流向
- `volume_relative_strength`: 成交量相对强度

#### **4. 价格行为情绪因子 (10个)**
基于K线形态的情绪分析：
- `gap_up/down`: 跳空缺口，情绪突变
- `long_upper/lower_shadow`: 长上下影线，市场犹豫
- `doji`: 十字星，市场不确定性
- `price_rejection_up/down`: 价格试探被拒绝
- `market_emotion_strength`: 市场情绪强度

#### **5. 市场强度因子 (7个)**
- `price_strength`: 价格强度指标
- `breakout_strength_up/down`: 突破强度
- `market_breadth`: 市场宽度
- `buying/selling_pressure`: 买卖压力
- `relative_strength`: 相对强弱

#### **6. 恐慌贪婪指数类因子 (8个)**
模拟CNN恐慌贪婪指数：
- `extreme_movement`: 极端价格变动
- `sentiment_index`: 综合情绪指数
- `panic_indicator`: 恐慌指标
- `greed_indicator`: 贪婪指标
- `market_overheated/oversold`: 市场过热过冷
- `sentiment_reversal`: 情绪反转信号

## 🎯 因子应用策略

### **趋势类策略**
- 使用移动平均线、MACD金叉死叉捕捉趋势
- 结合动量因子确认趋势强度
- 利用资金费率验证趋势持续性

### **均值回归策略**
- RSI、随机指标识别超买超卖
- 布林带边界作为入场点
- 情绪极端因子确认反转时机

### **突破策略**
- 布林带收缩后的突破
- 成交量配合的价格突破
- 波动率突增的趋势确认

### **套利策略**
- 资金费率异常的期现套利机会
- 溢价指数的跨市场套利
- 多空比极端的情绪套利

## 💡 因子选择建议

### **新手推荐因子**
🟢 **入门级**（易理解，信号明确）
- `rsi_14`, `rsi_oversold`, `rsi_overbought` - RSI超买超卖
- `macd_cross_up`, `macd_cross_down` - MACD金叉死叉
- `ma_5_20_cross`, `ma_10_50_cross` - 均线金叉
- `bb_break_up`, `bb_break_down` - 布林带突破
- `funding_rate_extreme_positive/negative` - 极端资金费率

### **进阶组合因子**
🟡 **中级**（需要一定理解和组合使用）
- 动量组合：`momentum_1` + `momentum_strength` + `trend_consistency`
- 波动率组合：`volatility_spike` + `vix_like` + `atr_high_volatility`
- 量价组合：`volume_surge` + `volume_price_cooperation` + `obv_momentum`
- 情绪组合：`sentiment_index` + `panic_indicator` + `greed_indicator`

### **专家级因子**
🔴 **高级**（需要深度理解市场微观结构）
- `conditional_volatility` + `volatility_skewness` - 高级波动率分析
- `ls_ratio_deviation` + `funding_rate_momentum` - 市场结构分析
- `price_rejection_up/down` + `market_emotion_strength` - 价格行为分析
- `buying_pressure` / `selling_pressure` - 资金流向分析

## 📋 因子使用最佳实践

### **1. 因子预处理**
```python
# 数据清洗
factors = factors.dropna()  # 移除缺失值
factors = factors.replace([np.inf, -np.inf], np.nan)  # 处理无穷值

# 异常值处理（3σ原则）
for col in factor_columns:
    mean = factors[col].mean()
    std = factors[col].std()
    factors[col] = factors[col].clip(mean - 3*std, mean + 3*std)
```

### **2. 因子标准化**
```python
# Z-score标准化
factors_normalized = (factors - factors.mean()) / factors.std()

# 排序标准化（适用于非正态分布）
factors_ranked = factors.rank(pct=True)
```

### **3. 因子有效性检验**
- **IC值**：|IC| > 0.05 为有效，|IC| > 0.1 为强有效
- **IC稳定性**：IC标准差 < IC均值，信号稳定
- **单调性**：因子分组收益应单调递增/递减
- **显著性**：p值 < 0.05，统计显著

### **4. 多因子组合策略**
```python
# 因子加权组合
combined_factor = (
    0.3 * rsi_factor +      # 技术因子权重30%
    0.4 * market_factor +   # 市场因子权重40%  
    0.3 * sentiment_factor  # 情绪因子权重30%
)
```

## ⚠️ 重要提示

### **风险警告**
1. **过拟合风险**：避免使用过多因子，建议单策略不超过5-8个主要因子
2. **数据挖掘陷阱**：历史有效不代表未来有效，需要样本外验证
3. **市场制度变化**：加密货币市场快速演变，因子有效性可能衰减
4. **交易成本**：考虑手续费、滑点、资金费率等实际交易成本

### **使用建议**
- 🔄 **定期回测**：每月更新因子有效性分析
- 📊 **分层验证**：不同市场环境下的因子表现
- 🎯 **风险控制**：设置止损、仓位控制、最大回撤限制
- 📈 **组合优化**：使用马科维茨优化或风险平价分配权重

---

**⚡ 开始使用**: 建议从RSI、MACD、移动平均线等经典技术因子开始，逐步加入市场和情绪因子，构建你的量化交易策略！
