# Binance Multi-Factor Analysis Platform

🚀 **专业的加密货币多因子分析平台，专注于IC（信息系数）相关性分析**

## 🎯 项目概述

这是一个基于Python的量化分析平台，专门用于分析加密货币市场的多种因子与未来收益的相关性。系统采用15分钟标准化数据，提供64种技术、市场和情绪因子的综合分析。

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
```

## 📚 详细文档

完整的使用指南、因子说明和配置选项请参考：
**[📖 完整文档 - README_MULTIFACTOR.md](./README_MULTIFACTOR.md)**

## 🏗️ 系统架构

```
core/               # 核心模块（配置、数据管理、因子引擎）
factors/            # 因子库（技术、市场、情绪因子）
analysis/           # 分析接口
main.py            # 命令行入口
```

## 📊 分析结果示例

系统将输出每个因子的IC值、相关性强度、稳定性评估和投资建议，帮助识别最有效的量化因子。

---

**⚡ 注意**: 使用前请确保已正确配置Binance API密钥，并查看详细文档了解完整功能。
