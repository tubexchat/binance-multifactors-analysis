"""
多因子分析模块

本模块提供加密货币量化分析中的各种因子计算和IC相关性分析功能
"""

from .technical_factors import *
from .market_factors import *
from .sentiment_factors import *
from .ic_analysis import *

__version__ = "1.0.0"
__author__ = "Binance Multi-Factor Analysis Team"

# 因子类别定义
FACTOR_CATEGORIES = {
    'technical': '技术指标因子',
    'market': '市场微观结构因子',
    'sentiment': '市场情绪因子',
    'volume': '成交量因子',
    'volatility': '波动率因子'
}

# 标准时间周期
STANDARD_INTERVAL = '15m'  # 标准15分钟周期 