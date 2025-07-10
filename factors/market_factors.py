"""
市场微观结构因子模块

基于市场数据计算资金费率、持仓量、多空比等市场结构因子
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketFactors:
    """市场微观结构因子计算器"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    async def calculate_all_factors(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有市场因子

        Args:
            symbol: 交易对符号
            df: K线数据DataFrame

        Returns:
            包含所有市场因子的DataFrame
        """
        result_df = df[["timestamp"]].copy()

        try:
            # 获取各种市场数据
            tasks = await asyncio.gather(
                self._get_funding_rate_factors(symbol),
                self._get_long_short_ratio_factors(symbol),
                self._get_open_interest_factors(symbol),
                self._get_premium_factors(symbol),
                return_exceptions=True,
            )

            # 合并因子数据
            for task_result in tasks:
                if isinstance(task_result, pd.DataFrame) and not task_result.empty:
                    # 根据时间戳合并数据
                    result_df = self._merge_by_timestamp(result_df, task_result)
                elif isinstance(task_result, Exception):
                    logger.warning(f"计算市场因子时出错: {task_result}")

            # 计算价格相关的市场因子
            price_factors = self._calculate_price_factors(df)
            result_df = pd.concat([result_df, price_factors], axis=1)

            logger.info(f"成功计算 {len(result_df.columns)-1} 个市场因子")
            return result_df

        except Exception as e:
            logger.error(f"计算市场因子时出错: {e}")
            return pd.DataFrame()

    async def _get_funding_rate_factors(self, symbol: str) -> pd.DataFrame:
        """
        获取资金费率相关因子

        资金费率是期货合约维持价格锚定现货的机制，反映市场情绪和资金成本
        """
        try:
            funding_data = await self.data_manager.get_funding_rate(symbol)
            if funding_data is None or funding_data.empty:
                return pd.DataFrame()

            factors = pd.DataFrame()
            factors["timestamp"] = funding_data["fundingTime"]

            # 资金费率基础因子
            funding_rate = (
                funding_data["fundingRate"].astype(float) * 100
            )  # 转换为百分比
            factors["funding_rate"] = (
                funding_rate  # 当前资金费率，正值表示多头付费，负值表示空头付费
            )

            # 资金费率衍生因子
            factors["funding_rate_ma_8"] = funding_rate.rolling(
                8
            ).mean()  # 8期资金费率均值，平滑短期波动
            factors["funding_rate_ma_24"] = funding_rate.rolling(
                24
            ).mean()  # 24期资金费率均值，中期趋势
            factors["funding_rate_std_8"] = funding_rate.rolling(
                8
            ).std()  # 8期资金费率标准差，衡量波动性
            factors["funding_rate_change"] = funding_rate.diff(
                1
            )  # 资金费率变化，趋势方向判断
            factors["funding_rate_momentum"] = funding_rate.diff(
                3
            )  # 资金费率动量，中期变化趋势

            # 资金费率极端值标识
            factors["funding_rate_extreme_positive"] = (funding_rate > 0.1).astype(
                int
            )  # 极高正资金费率，强烈看涨情绪
            factors["funding_rate_extreme_negative"] = (funding_rate < -0.1).astype(
                int
            )  # 极高负资金费率，强烈看跌情绪
            factors["funding_rate_neutral"] = (  # 中性资金费率，市场平衡
                (funding_rate >= -0.01) & (funding_rate <= 0.01)
            ).astype(int)

            # 资金费率趋势
            factors["funding_rate_trend"] = np.where(  # 资金费率趋势方向
                factors["funding_rate_change"] > 0,
                1,
                np.where(factors["funding_rate_change"] < 0, -1, 0),
            )

            return factors

        except Exception as e:
            logger.error(f"计算资金费率因子时出错: {e}")
            return pd.DataFrame()

    async def _get_long_short_ratio_factors(self, symbol: str) -> pd.DataFrame:
        """
        获取多空比相关因子

        多空比反映市场参与者的仓位分布，是重要的情绪指标
        """
        try:
            # 获取5分钟多空比数据
            ls_data = await self.data_manager.get_long_short_ratio(
                symbol, period="5m", limit=500
            )
            if ls_data is None or ls_data.empty:
                return pd.DataFrame()

            factors = pd.DataFrame()
            factors["timestamp"] = ls_data["timestamp"]

            # 多空比基础因子
            long_short_ratio = ls_data["longShortRatio"].astype(float)
            factors["long_short_ratio"] = (
                long_short_ratio  # 多空比，>1表示多头占优，<1表示空头占优
            )

            # 多空比衍生因子
            factors["ls_ratio_ma_12"] = long_short_ratio.rolling(
                12
            ).mean()  # 1小时多空比均值，短期情绪趋势
            factors["ls_ratio_ma_48"] = long_short_ratio.rolling(
                48
            ).mean()  # 4小时多空比均值，中期情绪趋势
            factors["ls_ratio_std_12"] = long_short_ratio.rolling(
                12
            ).std()  # 多空比波动性，情绪稳定性指标
            factors["ls_ratio_change"] = long_short_ratio.diff(
                1
            )  # 多空比变化，情绪转换速度
            factors["ls_ratio_momentum"] = long_short_ratio.diff(
                6
            )  # 多空比动量，中期情绪变化

            # 多空比极端值
            factors["ls_ratio_extreme_long"] = (long_short_ratio > 4.0).astype(
                int
            )  # 极端多头情绪，可能反转信号
            factors["ls_ratio_extreme_short"] = (long_short_ratio < 0.25).astype(
                int
            )  # 极端空头情绪，可能反转信号
            factors["ls_ratio_balanced"] = (  # 多空平衡，市场中性情绪
                (long_short_ratio >= 0.8) & (long_short_ratio <= 1.2)
            ).astype(int)

            # 多空比偏离度
            factors["ls_ratio_deviation"] = (  # 多空比偏离中期均值的程度
                long_short_ratio - factors["ls_ratio_ma_48"]
            ) / factors["ls_ratio_ma_48"]

            # 多空比趋势
            factors["ls_ratio_trend"] = np.where(  # 多空比变化趋势方向
                factors["ls_ratio_change"] > 0,
                1,
                np.where(factors["ls_ratio_change"] < 0, -1, 0),
            )

            return factors

        except Exception as e:
            logger.error(f"计算多空比因子时出错: {e}")
            return pd.DataFrame()

    async def _get_open_interest_factors(self, symbol: str) -> pd.DataFrame:
        """获取持仓量相关因子"""
        try:
            oi_data = await self.data_manager.get_open_interest(symbol)
            if oi_data is None:
                return pd.DataFrame()

            # 注意：open_interest API返回的是当前值，需要历史数据来计算变化
            # 这里我们创建一个单行DataFrame作为示例
            factors = pd.DataFrame()
            current_oi = float(oi_data["openInterest"])

            # 创建当前时间戳
            factors["timestamp"] = [datetime.now()]
            factors["open_interest"] = [current_oi]

            # 注：实际应用中需要存储历史持仓量数据来计算这些因子
            factors["oi_change"] = [0]  # 需要历史数据计算
            factors["oi_momentum"] = [0]  # 需要历史数据计算
            factors["oi_high"] = [0]  # 需要历史数据标识

            return factors

        except Exception as e:
            logger.error(f"计算持仓量因子时出错: {e}")
            return pd.DataFrame()

    async def _get_premium_factors(self, symbol: str) -> pd.DataFrame:
        """获取溢价指数相关因子"""
        try:
            premium_data = await self.data_manager.get_premium_index()
            if premium_data is None:
                return pd.DataFrame()

            # 找到对应交易对的溢价数据
            symbol_premium = None
            for item in premium_data:
                if item.get("symbol") == symbol:
                    symbol_premium = item
                    break

            if symbol_premium is None:
                return pd.DataFrame()

            factors = pd.DataFrame()
            factors["timestamp"] = [datetime.now()]

            # 溢价指数基础因子
            factors["premium_index"] = [
                float(symbol_premium.get("lastFundingRate", 0)) * 100
            ]
            factors["mark_price"] = [float(symbol_premium.get("markPrice", 0))]
            factors["index_price"] = [float(symbol_premium.get("indexPrice", 0))]

            # 溢价相关因子
            if factors["index_price"].iloc[0] > 0:
                factors["price_premium"] = (
                    (factors["mark_price"] - factors["index_price"])
                    / factors["index_price"]
                    * 100
                )
            else:
                factors["price_premium"] = [0]

            return factors

        except Exception as e:
            logger.error(f"计算溢价因子时出错: {e}")
            return pd.DataFrame()

    def _calculate_price_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格相关的市场因子

        基于OHLCV数据计算的各种市场微观结构因子
        """
        factors = pd.DataFrame()

        try:
            # 创建基于原始数据索引的因子DataFrame
            factors = pd.DataFrame(index=df.index)

            # 收益率因子 - 不同周期的价格变化率
            close = df["close"]
            factors["return_1"] = close.pct_change(1)  # 1期收益率，短期价格动量
            factors["return_2"] = close.pct_change(2)  # 2期收益率，短期趋势
            factors["return_4"] = close.pct_change(4)  # 4期收益率，中短期趋势
            factors["return_8"] = close.pct_change(8)  # 8期收益率，中期趋势

            # 波动率因子 - 收益率的标准差，衡量价格波动程度
            factors["volatility_12"] = (
                factors["return_1"].rolling(12).std()
            )  # 12期波动率，短期波动性
            factors["volatility_48"] = (
                factors["return_1"].rolling(48).std()
            )  # 48期波动率，中期波动性

            # 偏度和峰度 - 收益率分布的高阶矩特征
            factors["skewness_12"] = (
                factors["return_1"].rolling(12).skew()
            )  # 12期偏度，价格分布的非对称性
            factors["kurtosis_12"] = (
                factors["return_1"].rolling(12).kurt()
            )  # 12期峰度，极端价格变动的频率

            # 价格动量因子 - 不同周期的累积收益率
            factors["price_momentum_12"] = (
                close / close.shift(12)
            ) - 1  # 12期价格动量，短期动量强度
            factors["price_momentum_48"] = (
                close / close.shift(48)
            ) - 1  # 48期价格动量，中期动量强度

            # 成交量相关因子 - 量价关系分析
            volume = df["volume"]
            factors["volume_ma_12"] = volume.rolling(
                12
            ).mean()  # 12期成交量均值，成交量基准线
            factors["volume_ratio"] = (
                volume / factors["volume_ma_12"]
            )  # 成交量相对强度，当前成交量的活跃程度

            # 修复volume_volatility计算，确保分母不为0
            volume_ma_12 = volume.rolling(12).mean()
            volume_std_12 = volume.rolling(12).std()
            # 直接计算并赋值，避免形状问题
            factors["volume_volatility"] = (
                volume_std_12 / volume_ma_12
            )  # 成交量波动率，成交量的稳定性
            factors["volume_volatility"] = factors["volume_volatility"].fillna(np.nan)

            # VWAP相关因子 - 成交量加权平均价格
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            volume_sum_12 = volume.rolling(12).sum()
            factors["vwap"] = np.where(  # 成交量加权平均价格，资金成本参考
                volume_sum_12 > 0,
                (typical_price * volume).rolling(12).sum() / volume_sum_12,
                np.nan,
            )
            factors["price_to_vwap"] = np.where(  # 价格相对VWAP的偏离度
                factors["vwap"] > 0, close / factors["vwap"] - 1, np.nan
            )

            # 高低价相关因子 - 价格区间分析
            factors["high_low_ratio"] = (
                df["high"] / df["low"] - 1
            )  # 高低价比率，单期价格波动幅度

            # 修复close_position计算，避免除零错误
            high_low_diff = df["high"] - df["low"]
            # 直接计算并赋值，避免形状问题
            factors["close_position"] = (
                df["close"] - df["low"]
            ) / high_low_diff  # 收盘价在日内区间的位置
            factors["close_position"] = factors["close_position"].fillna(np.nan)

            # 成交量价格趋势 - 量价配合度指标
            price_change_ratio = (df["close"] - df["close"].shift(1)) / df[
                "close"
            ].shift(1)
            factors["vpt"] = (
                price_change_ratio * volume
            ).cumsum()  # 成交量价格趋势，累积量价动量
            factors["vpt_momentum"] = factors["vpt"].diff(12)  # VPT动量，量价趋势变化

            # 重置索引以确保数据对齐
            factors = factors.reset_index(drop=True)

        except Exception as e:
            logger.error(f"计算价格市场因子时出错: {e}")

        return factors

    def _merge_by_timestamp(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """根据时间戳合并两个DataFrame"""
        try:
            if df1.empty:
                return df2
            if df2.empty:
                return df1

            # 确保时间戳列为datetime类型
            df1["timestamp"] = pd.to_datetime(df1["timestamp"])
            df2["timestamp"] = pd.to_datetime(df2["timestamp"])

            # 使用merge_asof进行时间对齐合并
            merged = pd.merge_asof(
                df1.sort_values("timestamp"),
                df2.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),  # 30分钟容差
            )

            return merged

        except Exception as e:
            logger.error(f"合并DataFrame时出错: {e}")
            return df1


# 创建全局市场因子计算器
# 注意：需要在使用时传入data_manager实例
def create_market_factors(data_manager):
    """创建市场因子计算器实例"""
    return MarketFactors(data_manager)


__all__ = ["MarketFactors", "create_market_factors"]
