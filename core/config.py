"""
多因子分析平台核心配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class DataConfig:
    """数据获取配置"""

    # 标准时间间隔（15分钟）
    STANDARD_INTERVAL: str = "15m"

    # 历史数据长度（条数）
    HISTORY_LENGTH: int = 1000

    # 数据更新频率（秒）
    UPDATE_FREQUENCY: int = 900  # 15分钟

    # 币安API相关
    BINANCE_BASE_URL: str = "https://fapi.binance.com"
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3

    # 最小交易量过滤（USDT）
    MIN_QUOTE_VOLUME: float = 100_000_000  # 1亿USDT

    # 排除的交易对
    EXCLUDED_SYMBOLS: List[str] = field(
        default_factory=lambda: [
            "USDC",
            "AGIX",
            "BNX",
            "ALPACA",
            "DOGE",
            "TRUMP",
            "DGB",
            "LINA",
            "KEY",
            "MDT",
            "LOOM",
            "REN",
            "WAVES",
            "BOND",
            "BLZ",
            "OMG",
            "KLAY",
            "SNT",
            "UNFI",
            "FTM",
            "COMBO",
            "STRAX",
            "NULS",
            "AMB",
            "TROY",
            "VIDT",
        ]
    )


@dataclass
class FactorConfig:
    """因子计算配置"""

    # 技术指标参数
    RSI_PERIOD: int = 14
    STOCH_RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # 移动平均线参数
    MA_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # 波动率参数
    VOLATILITY_WINDOW: int = 20

    # 成交量指标参数
    VOLUME_MA_PERIOD: int = 20

    # 因子计算的最小数据量
    MIN_DATA_POINTS: int = 200

    # 数据处理配置
    DEFAULT_HISTORY_LENGTH: int = 1000
    FILL_NA_METHOD: str = "forward"  # 'forward', 'zero', 'drop'

    # 缓存配置
    CACHE_TTL: int = 1800  # 30分钟
    MAX_CACHE_SIZE: int = 100


@dataclass
class ICAnalysisConfig:
    """IC分析配置"""

    # IC计算窗口
    IC_WINDOW: int = 20  # 20个周期

    # 收益率计算周期
    RETURN_PERIODS: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8]
    )  # 1,2,4,8个15分钟周期的收益率

    # IC有效性阈值
    IC_THRESHOLD: float = 0.05  # 绝对值大于0.05认为有效

    # 因子筛选阈值
    FACTOR_SELECTION_THRESHOLD: float = 0.02  # IC绝对值大于0.02的因子

    # 滚动IC计算窗口
    ROLLING_IC_WINDOW: int = 60  # 60个周期


@dataclass
class AnalysisConfig:
    """分析配置"""

    # 同时分析的币种数量
    MAX_SYMBOLS: int = 50

    # 优先分析的主要币种
    MAJOR_SYMBOLS: List[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOTUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "LUNAUSDT",
        ]
    )

    # 分析报告生成频率（秒）
    REPORT_FREQUENCY: int = 3600  # 1小时

    # 结果保存路径
    OUTPUT_DIR: str = "./analysis_results"

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./multifactor.db")

    # 相关性分析配置
    HIGH_CORRELATION_THRESHOLD: float = 0.8  # 高相关性阈值


class Config:
    """主配置类"""

    def __init__(self):
        self.data = DataConfig()
        self.factors = FactorConfig()
        self.ic_analysis = ICAnalysisConfig()
        self.analysis = AnalysisConfig()

        # 确保输出目录存在
        os.makedirs(self.analysis.OUTPUT_DIR, exist_ok=True)

    def get_api_endpoints(self) -> Dict[str, str]:
        """获取API端点配置"""
        base_url = self.data.BINANCE_BASE_URL
        return {
            "klines": f"{base_url}/fapi/v1/klines",
            "ticker_24hr": f"{base_url}/fapi/v1/ticker/24hr",
            "funding_rate": f"{base_url}/fapi/v1/fundingRate",
            "premium_index": f"{base_url}/fapi/v1/premiumIndex",
            "long_short_ratio": f"{base_url}/futures/data/topLongShortPositionRatio",
            "open_interest": f"{base_url}/fapi/v1/openInterest",
        }

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类配置"""
        return {
            "technical": [
                "rsi",
                "stoch_rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "sma_5",
                "sma_10",
                "sma_20",
                "sma_50",
                "sma_100",
                "sma_200",
                "ema_5",
                "ema_10",
                "ema_20",
                "bollinger_upper",
                "bollinger_lower",
                "atr",
                "williams_r",
                "cci",
                "stoch_k",
                "stoch_d",
            ],
            "market": [
                "funding_rate",
                "funding_rate_change",
                "premium_index",
                "long_short_ratio",
                "long_short_ratio_change",
                "open_interest",
                "open_interest_change",
            ],
            "volume": [
                "volume",
                "volume_ma",
                "volume_ratio",
                "vwap",
                "money_flow_index",
                "accumulation_distribution",
                "on_balance_volume",
                "volume_price_trend",
            ],
            "volatility": [
                "price_volatility",
                "volume_volatility",
                "return_volatility",
                "high_low_ratio",
                "true_range",
                "average_true_range",
            ],
            "sentiment": [
                "price_momentum_1",
                "price_momentum_2",
                "price_momentum_4",
                "price_momentum_8",
                "volume_momentum",
                "volatility_momentum",
                "market_beta",
            ],
        }


# 创建全局配置实例
config = Config()

# 导出主要配置对象
__all__ = ["config", "DataConfig", "FactorConfig", "ICAnalysisConfig", "AnalysisConfig"]
