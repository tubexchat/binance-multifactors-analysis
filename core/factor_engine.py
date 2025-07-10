"""
因子计算引擎

统一管理技术指标、市场因子、情绪因子的计算，提供批量计算和因子数据管理功能
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from core.config import config
from core.data_manager import DataManager
from factors.technical_factors import TechnicalFactors
from factors.market_factors import MarketFactors, create_market_factors
from factors.sentiment_factors import SentimentFactors
from factors.ic_analysis import ICAnalysis

logger = logging.getLogger(__name__)


class FactorEngine:
    """因子计算引擎"""

    def __init__(self):
        self.data_manager = None
        self.technical_factors = TechnicalFactors()
        self.market_factors = None  # 需要data_manager初始化
        self.sentiment_factors = SentimentFactors()
        self.ic_analyzer = ICAnalysis()

        # 因子缓存
        self.factor_cache = {}
        self.cache_expiry = {}

        # 因子配置
        self.factor_config = config.factors

    async def initialize(self):
        """初始化因子引擎"""
        try:
            self.data_manager = DataManager()
            await self.data_manager.__aenter__()

            # 初始化市场因子计算器
            self.market_factors = create_market_factors(self.data_manager)

            logger.info("因子计算引擎初始化完成")

        except Exception as e:
            logger.error(f"初始化因子引擎时出错: {e}")
            raise

    async def cleanup(self):
        """清理资源"""
        try:
            if self.data_manager:
                await self.data_manager.__aexit__(None, None, None)
            logger.info("因子计算引擎清理完成")
        except Exception as e:
            logger.error(f"清理因子引擎时出错: {e}")

    async def calculate_all_factors(
        self, symbol: str, limit: int = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        计算指定交易对的所有因子

        Args:
            symbol: 交易对符号
            limit: 数据条数限制
            use_cache: 是否使用缓存

        Returns:
            包含所有因子的DataFrame
        """
        try:
            logger.info(f"开始计算 {symbol} 的所有因子")

            # 检查缓存
            if use_cache and self._is_cache_valid(symbol):
                logger.info(f"使用缓存的 {symbol} 因子数据")
                return self.factor_cache[symbol]

            # 获取K线数据
            if limit is None:
                limit = self.factor_config.DEFAULT_HISTORY_LENGTH

            kline_data = await self.data_manager.get_kline_data(
                symbol=symbol, limit=limit
            )

            if kline_data is None or kline_data.empty:
                logger.warning(f"无法获取 {symbol} 的K线数据")
                return pd.DataFrame()

            # 确保数据长度足够
            if len(kline_data) < 200:
                logger.warning(f"{symbol} 数据长度不足: {len(kline_data)}")
                return pd.DataFrame()

            # 并行计算各类因子
            tasks = await asyncio.gather(
                self._calculate_technical_factors(kline_data),
                self._calculate_market_factors(symbol, kline_data),
                self._calculate_sentiment_factors(kline_data),
                return_exceptions=True,
            )

            # 合并因子数据
            all_factors = kline_data[["timestamp"]].copy()

            for i, task_result in enumerate(tasks):
                if isinstance(task_result, pd.DataFrame) and not task_result.empty:
                    # 根据时间戳合并
                    all_factors = self._merge_factors(all_factors, task_result)
                elif isinstance(task_result, Exception):
                    logger.warning(f"计算因子类型 {i} 时出错: {task_result}")

            # 数据清理和处理
            all_factors = self._clean_factor_data(all_factors)

            # 缓存结果
            if use_cache:
                self._cache_factors(symbol, all_factors)

            logger.info(f"成功计算 {symbol} 的 {len(all_factors.columns)-1} 个因子")
            return all_factors

        except Exception as e:
            logger.error(f"计算 {symbol} 所有因子时出错: {e}")
            return pd.DataFrame()

    async def calculate_factors_batch(
        self, symbols: List[str], limit: int = None, max_concurrent: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个交易对的因子

        Args:
            symbols: 交易对符号列表
            limit: 数据条数限制
            max_concurrent: 最大并发数

        Returns:
            包含各交易对因子数据的字典
        """
        try:
            logger.info(f"开始批量计算 {len(symbols)} 个交易对的因子")

            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrent)

            async def calculate_with_semaphore(symbol):
                async with semaphore:
                    return symbol, await self.calculate_all_factors(symbol, limit)

            # 批量执行
            tasks = [calculate_with_semaphore(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 整理结果
            factor_data = {}
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    symbol, factors = result
                    if not factors.empty:
                        factor_data[symbol] = factors
                elif isinstance(result, Exception):
                    logger.warning(f"批量计算中出错: {result}")

            logger.info(f"成功计算 {len(factor_data)} 个交易对的因子")
            return factor_data

        except Exception as e:
            logger.error(f"批量计算因子时出错: {e}")
            return {}

    async def _calculate_technical_factors(
        self, kline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算技术指标因子"""
        try:
            return self.technical_factors.calculate_all_factors(kline_data)
        except Exception as e:
            logger.error(f"计算技术因子时出错: {e}")
            return pd.DataFrame()

    async def _calculate_market_factors(
        self, symbol: str, kline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算市场因子"""
        try:
            if self.market_factors is None:
                logger.warning("市场因子计算器未初始化")
                return pd.DataFrame()

            return await self.market_factors.calculate_all_factors(symbol, kline_data)
        except Exception as e:
            logger.error(f"计算市场因子时出错: {e}")
            return pd.DataFrame()

    async def _calculate_sentiment_factors(
        self, kline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算情绪因子"""
        try:
            return self.sentiment_factors.calculate_all_factors(kline_data)
        except Exception as e:
            logger.error(f"计算情绪因子时出错: {e}")
            return pd.DataFrame()

    def _merge_factors(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """合并因子数据"""
        try:
            if df1.empty:
                return self._validate_merge_data(df2)
            if df2.empty:
                return self._validate_merge_data(df1)

            # 在合并前验证数据形状
            df1 = self._validate_merge_data(df1)
            df2 = self._validate_merge_data(df2)

            # 确保有相同的索引长度
            if len(df1) != len(df2):
                min_len = min(len(df1), len(df2))
                df1 = df1.iloc[:min_len].copy()
                df2 = df2.iloc[:min_len].copy()

            # 处理重复列名，避免多维数组问题
            df2_to_merge = df2.drop(columns=["timestamp"], errors="ignore")

            # 找出重复的列名
            duplicate_cols = set(df1.columns).intersection(set(df2_to_merge.columns))
            if duplicate_cols:
                logger.info(
                    f"检测到重复列: {list(duplicate_cols)}，将优先保留第一个数据源的版本"
                )
                # 从df2中移除重复的列，保留df1中的版本
                df2_to_merge = df2_to_merge.drop(
                    columns=list(duplicate_cols), errors="ignore"
                )

            # 合并DataFrame
            result = pd.concat([df1, df2_to_merge], axis=1)
            return result

        except Exception as e:
            logger.error(f"合并因子数据时出错: {e}")
            return df1

    def _validate_merge_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证合并前的数据形状"""
        try:
            if df.empty:
                return df

            # 检查每列是否为多维数据
            columns_to_drop = []
            for col in df.columns:
                if col == "timestamp":
                    continue

                col_data = df[col]
                if hasattr(col_data, "shape") and len(col_data.shape) > 1:
                    logger.warning(
                        f"发现多维因子数据 {col}，形状: {col_data.shape}，将在合并时移除"
                    )
                    columns_to_drop.append(col)

            # 移除多维列
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)

            return df

        except Exception as e:
            logger.error(f"验证合并数据时出错: {e}")
            return df

    def _clean_factor_data(self, factors: pd.DataFrame) -> pd.DataFrame:
        """清理因子数据"""
        try:
            # 移除全为NaN的列
            factors = factors.dropna(axis=1, how="all")

            # 确保所有因子列的形状一致
            factors = self._validate_factor_shapes(factors)

            # 处理无穷大值
            factors = factors.replace([np.inf, -np.inf], np.nan)

            # 填充NaN值（可以选择不同的填充策略）
            if self.factor_config.FILL_NA_METHOD == "forward":
                factors = factors.fillna(method="ffill")
            elif self.factor_config.FILL_NA_METHOD == "zero":
                factors = factors.fillna(0)

            # 数据类型优化
            for col in factors.columns:
                if (
                    col != "timestamp"
                    and hasattr(factors[col], "dtype")
                    and factors[col].dtype == "float64"
                ):
                    factors[col] = factors[col].astype("float32")

            return factors

        except Exception as e:
            logger.error(f"清理因子数据时出错: {e}")
            return factors

    def _validate_factor_shapes(self, factors: pd.DataFrame) -> pd.DataFrame:
        """验证因子数据形状一致性"""
        try:
            if factors.empty:
                return factors

            # 获取期望的数据长度（通常以DataFrame的索引为准）
            expected_length = len(factors)

            # 检查每列的长度和数据类型
            for col in factors.columns:
                if col == "timestamp":
                    continue

                col_data = factors[col]

                # 检查数据是否为多维
                if hasattr(col_data, "shape") and len(col_data.shape) > 1:
                    logger.warning(
                        f"因子 {col} 是多维数据，形状: {col_data.shape}，将其移除"
                    )
                    factors = factors.drop(columns=[col])
                    continue

                # 确保列数据是Series类型且与DataFrame索引对齐
                if not isinstance(col_data, pd.Series):
                    try:
                        # 检查是否为多维数组
                        if hasattr(col_data, "shape") and len(col_data.shape) > 1:
                            logger.warning(
                                f"因子 {col} 是多维数组，形状: {col_data.shape}，将其移除"
                            )
                            factors = factors.drop(columns=[col])
                            continue
                        # 尝试转换为Series
                        factors[col] = pd.Series(col_data, index=factors.index)
                    except Exception as e:
                        logger.warning(f"修复因子列 {col} 数据类型失败: {e}")
                        factors = factors.drop(columns=[col])
                        continue

                # 检查长度是否一致
                if len(col_data) != expected_length:
                    logger.warning(
                        f"因子 {col} 长度 {len(col_data)} 与期望长度 {expected_length} 不匹配，尝试修复"
                    )
                    try:
                        # 重新创建与索引对齐的Series
                        if len(col_data) > expected_length:
                            # 如果数据过长，截取前面部分
                            factors[col] = pd.Series(
                                col_data.iloc[:expected_length], index=factors.index
                            )
                        else:
                            # 如果数据过短，用NaN填充
                            new_data = np.full(expected_length, np.nan)
                            new_data[: len(col_data)] = col_data.values
                            factors[col] = pd.Series(new_data, index=factors.index)
                    except Exception:
                        logger.warning(f"无法修复因子 {col}，将其移除")
                        factors = factors.drop(columns=[col])

                # 确保数值型列，尝试转换非数值类型
                try:
                    factors[col] = pd.to_numeric(factors[col], errors="coerce")
                except Exception:
                    logger.warning(f"因子 {col} 无法转换为数值类型，将其移除")
                    factors = factors.drop(columns=[col])

            return factors

        except Exception as e:
            logger.error(f"验证因子数据形状时出错: {e}")
            return factors

    def _is_cache_valid(self, symbol: str) -> bool:
        """检查缓存是否有效"""
        try:
            if symbol not in self.factor_cache:
                return False

            if symbol not in self.cache_expiry:
                return False

            return datetime.now() < self.cache_expiry[symbol]

        except Exception:
            return False

    def _cache_factors(self, symbol: str, factors: pd.DataFrame):
        """缓存因子数据"""
        try:
            self.factor_cache[symbol] = factors.copy()
            self.cache_expiry[symbol] = datetime.now() + timedelta(
                seconds=self.factor_config.CACHE_TTL
            )

            # 限制缓存大小
            if len(self.factor_cache) > self.factor_config.MAX_CACHE_SIZE:
                # 移除最旧的缓存
                oldest_symbol = min(
                    self.cache_expiry.keys(), key=lambda k: self.cache_expiry[k]
                )
                del self.factor_cache[oldest_symbol]
                del self.cache_expiry[oldest_symbol]

        except Exception as e:
            logger.error(f"缓存因子数据时出错: {e}")

    def get_factor_info(self) -> Dict:
        """获取因子信息"""
        try:
            factor_info = {
                "technical_factors": {
                    "count": len(self._get_technical_factor_names()),
                    "factors": self._get_technical_factor_names(),
                    "description": "技术指标因子，包括RSI、MACD、布林带等",
                },
                "market_factors": {
                    "count": len(self._get_market_factor_names()),
                    "factors": self._get_market_factor_names(),
                    "description": "市场微观结构因子，包括资金费率、多空比、持仓量等",
                },
                "sentiment_factors": {
                    "count": len(self._get_sentiment_factor_names()),
                    "factors": self._get_sentiment_factor_names(),
                    "description": "市场情绪因子，包括动量、波动率、恐慌贪婪指数等",
                },
            }

            total_factors = sum(info["count"] for info in factor_info.values())
            factor_info["total_factors"] = total_factors

            return factor_info

        except Exception as e:
            logger.error(f"获取因子信息时出错: {e}")
            return {}

    def _get_technical_factor_names(self) -> List[str]:
        """获取技术因子名称列表"""
        return [
            "rsi_14",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "bb_position",
            "stoch_k",
            "stoch_d",
            "cci",
            "williams_r",
            "obv",
            "ad_line",
            "cmf",
            "mfi",
            "atr",
            "adx",
            "psar",
            "aroon_up",
            "aroon_down",
            "tsi",
            "ultimate_oscillator",
        ]

    def _get_market_factor_names(self) -> List[str]:
        """获取市场因子名称列表"""
        return [
            "funding_rate",
            "funding_rate_ma_8",
            "funding_rate_ma_24",
            "long_short_ratio",
            "ls_ratio_ma_12",
            "ls_ratio_deviation",
            "open_interest",
            "oi_change",
            "premium_index",
            "price_premium",
            "return_1",
            "return_4",
            "volatility_12",
            "volume_ratio",
            "vwap",
            "price_to_vwap",
            "high_low_ratio",
            "vpt_momentum",
        ]

    def _get_sentiment_factor_names(self) -> List[str]:
        """获取情绪因子名称列表"""
        return [
            "momentum_1",
            "momentum_5",
            "momentum_acceleration",
            "trend_consistency",
            "volatility_5",
            "volatility_trend",
            "volatility_spike",
            "vix_like",
            "volume_price_divergence",
            "volume_surge",
            "money_flow_strength",
            "gap_up",
            "gap_down",
            "doji",
            "close_position",
            "price_strength",
            "market_breadth",
            "sentiment_index",
            "panic_indicator",
            "greed_indicator",
            "market_overheated",
        ]

    async def save_factors_to_file(
        self, symbol: str, factors: pd.DataFrame, file_path: Optional[str] = None
    ) -> str:
        """保存因子数据到文件"""
        try:
            if file_path is None:
                # 创建默认文件路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"factor_data/{symbol}_factors_{timestamp}.parquet"

            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # 保存为parquet格式（压缩且高效）
            factors.to_parquet(file_path, compression="gzip", index=False)

            logger.info(f"因子数据已保存到: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"保存因子数据时出错: {e}")
            return ""

    async def load_factors_from_file(self, file_path: str) -> pd.DataFrame:
        """从文件加载因子数据"""
        try:
            if not Path(file_path).exists():
                logger.error(f"因子数据文件不存在: {file_path}")
                return pd.DataFrame()

            factors = pd.read_parquet(file_path)
            logger.info(f"从文件加载了 {len(factors)} 条因子数据")
            return factors

        except Exception as e:
            logger.error(f"加载因子数据时出错: {e}")
            return pd.DataFrame()


# 创建全局因子引擎实例
factor_engine = FactorEngine()

__all__ = ["FactorEngine", "factor_engine"]
