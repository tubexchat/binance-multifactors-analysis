"""
技术指标因子模块

基于15分钟K线数据计算各种技术指标因子
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class TechnicalFactors:
    """技术指标因子计算器"""

    def __init__(self):
        self.required_periods = 200  # 最少需要200个数据点进行计算

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标因子

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含所有技术因子的DataFrame
        """
        if len(df) < self.required_periods:
            logger.warning(f"数据长度不足，需要至少{self.required_periods}个数据点")
            return pd.DataFrame()

        result_df = df[["timestamp"]].copy()

        try:
            # 基础数据
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            volume = df["volume"].values

            # 1. RSI相关指标
            result_df = pd.concat(
                [result_df, self._calculate_rsi_factors(close)], axis=1
            )

            # 2. MACD相关指标
            result_df = pd.concat(
                [result_df, self._calculate_macd_factors(close)], axis=1
            )

            # 3. 移动平均线相关指标
            result_df = pd.concat(
                [result_df, self._calculate_ma_factors(close)], axis=1
            )

            # 4. 布林带指标
            result_df = pd.concat(
                [result_df, self._calculate_bollinger_factors(close)], axis=1
            )

            # 5. 随机指标
            result_df = pd.concat(
                [result_df, self._calculate_stochastic_factors(high, low, close)],
                axis=1,
            )

            # 6. 威廉指标
            result_df = pd.concat(
                [result_df, self._calculate_williams_factors(high, low, close)], axis=1
            )

            # 7. CCI指标
            result_df = pd.concat(
                [result_df, self._calculate_cci_factors(high, low, close)], axis=1
            )

            # 8. ATR相关指标
            result_df = pd.concat(
                [result_df, self._calculate_atr_factors(high, low, close)], axis=1
            )

            # 9. 成交量相关技术指标
            result_df = pd.concat(
                [result_df, self._calculate_volume_factors(high, low, close, volume)],
                axis=1,
            )

            logger.info(f"成功计算 {len(result_df.columns)-1} 个技术因子")
            return result_df

        except Exception as e:
            logger.error(f"计算技术因子时出错: {e}")
            return pd.DataFrame()

    def _calculate_rsi_factors(self, close: np.ndarray) -> pd.DataFrame:
        """
        计算RSI（相对强弱指数）相关因子

        RSI是衡量价格变动速度和变化的震荡指标，用于识别超买超卖状态
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准RSI - 不同周期的相对强弱指数
            factors["rsi_14"] = talib.RSI(
                close, timeperiod=14
            )  # 14期RSI，经典参数，衡量中期超买超卖
            factors["rsi_21"] = talib.RSI(
                close, timeperiod=21
            )  # 21期RSI，更平滑的中期趋势指标
            factors["rsi_6"] = talib.RSI(
                close, timeperiod=6
            )  # 6期RSI，短期快速响应的超买超卖信号

            # RSI衍生指标 - 基于RSI的交易信号
            factors["rsi_oversold"] = (factors["rsi_14"] < 30).astype(
                int
            )  # RSI超卖信号（RSI<30），潜在买入机会
            factors["rsi_overbought"] = (factors["rsi_14"] > 70).astype(
                int
            )  # RSI超买信号（RSI>70），潜在卖出机会
            factors["rsi_momentum"] = factors["rsi_14"].diff(
                1
            )  # RSI动量，衡量RSI变化速度
            factors["rsi_cross_50"] = (  # RSI突破50中轴线，趋势转换信号
                (factors["rsi_14"] > 50) & (factors["rsi_14"].shift(1) <= 50)
            ).astype(int)

        except Exception as e:
            logger.error(f"计算RSI因子时出错: {e}")

        return factors

    def _calculate_macd_factors(self, close: np.ndarray) -> pd.DataFrame:
        """
        计算MACD（指数平滑移动平均收敛发散）相关因子

        MACD是趋势跟踪动量指标，通过比较快慢指数移动平均线识别趋势变化
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准MACD - 经典的趋势跟踪指标
            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            factors["macd"] = macd  # MACD主线，快慢均线差值，反映短期趋势
            factors["macd_signal"] = (
                macd_signal  # MACD信号线，MACD的平滑线，用于产生交易信号
            )
            factors["macd_histogram"] = (
                macd_hist  # MACD柱状图，主线与信号线差值，衡量趋势强度
            )

            # 转换为pandas Series以便使用shift和diff方法
            macd_series = pd.Series(macd, index=factors.index)
            macd_signal_series = pd.Series(macd_signal, index=factors.index)
            macd_hist_series = pd.Series(macd_hist, index=factors.index)

            # MACD衍生指标 - 基于MACD的交易信号
            factors["macd_cross_up"] = (  # MACD金叉信号，看涨信号
                (macd_series > macd_signal_series)
                & (macd_series.shift(1) <= macd_signal_series.shift(1))
            ).astype(int)
            factors["macd_cross_down"] = (  # MACD死叉信号，看跌信号
                (macd_series < macd_signal_series)
                & (macd_series.shift(1) >= macd_signal_series.shift(1))
            ).astype(int)
            factors["macd_momentum"] = macd_series.diff(1)  # MACD动量，衡量MACD变化速度
            factors["macd_divergence"] = macd_hist_series.diff(
                1
            )  # MACD柱状图变化，衡量趋势加速/减速

            # 不同参数的MACD - 更敏感的快速MACD
            macd_fast, _, _ = talib.MACD(
                close, fastperiod=5, slowperiod=13, signalperiod=9
            )
            factors["macd_fast"] = macd_fast  # 快速MACD，参数更敏感，适合短期交易

        except Exception as e:
            logger.error(f"计算MACD因子时出错: {e}")

        return factors

    def _calculate_ma_factors(self, close: np.ndarray) -> pd.DataFrame:
        """
        计算移动平均线相关因子

        移动平均线是最基础的趋势指标，用于识别趋势方向和支撑阻力位
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 简单移动平均线 - 不同周期的趋势线
            close_series = pd.Series(close, index=factors.index)
            for period in [5, 10, 20, 50, 100, 200]:
                ma = talib.SMA(close, timeperiod=period)
                ma_series = pd.Series(ma, index=factors.index)
                factors[f"sma_{period}"] = ma  # 简单移动平均线，平滑价格趋势
                factors[f"sma_{period}_ratio"] = (
                    close / ma - 1
                )  # 价格相对均线的偏离度，衡量超买超卖
                factors[f"sma_{period}_cross_up"] = (  # 价格突破均线信号，潜在买入机会
                    (close_series > ma_series)
                    & (close_series.shift(1) <= ma_series.shift(1))
                ).astype(int)

            # 指数移动平均线 - 对近期价格更敏感的均线
            for period in [5, 10, 20, 50]:
                ema = talib.EMA(close, timeperiod=period)
                factors[f"ema_{period}"] = ema  # 指数移动平均线，反应更快的趋势线
                factors[f"ema_{period}_ratio"] = close / ema - 1  # 价格相对EMA的偏离度

            # 移动平均线组合指标 - 双均线系统的金叉信号
            factors["ma_5_20_cross"] = (  # 5日线上穿20日线，短期看涨信号
                (factors["sma_5"] > factors["sma_20"])
                & (factors["sma_5"].shift(1) <= factors["sma_20"].shift(1))
            ).astype(int)
            factors["ma_10_50_cross"] = (  # 10日线上穿50日线，中期看涨信号
                (factors["sma_10"] > factors["sma_50"])
                & (factors["sma_10"].shift(1) <= factors["sma_50"].shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算移动平均线因子时出错: {e}")

        return factors

    def _calculate_bollinger_factors(self, close: np.ndarray) -> pd.DataFrame:
        """
        计算布林带（Bollinger Bands）相关因子

        布林带是基于标准差的波动性指标，用于识别价格的相对高低和波动性变化
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 布林带 - 由中轨（移动平均线）和上下轨（±2倍标准差）组成
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            factors["bollinger_upper"] = upper  # 布林带上轨，阻力位，价格接近时可能回调
            factors["bollinger_middle"] = middle  # 布林带中轨（20日均线），趋势参考线
            factors["bollinger_lower"] = lower  # 布林带下轨，支撑位，价格接近时可能反弹

            # 布林带衍生指标 - 基于布林带的交易信号
            close_series = pd.Series(close, index=factors.index)
            upper_series = pd.Series(upper, index=factors.index)
            lower_series = pd.Series(lower, index=factors.index)

            factors["bb_width"] = (upper - lower) / middle  # 布林带宽度，衡量市场波动性
            factors["bb_position"] = (close - lower) / (
                upper - lower
            )  # 价格在布林带中的位置（0-1）
            factors["bb_squeeze"] = (  # 布林带收缩，低波动性状态，可能预示突破
                factors["bb_width"] < factors["bb_width"].rolling(20).mean()
            ).astype(int)
            factors["bb_break_up"] = (  # 价格突破布林带上轨，强势信号
                (close_series > upper_series)
                & (close_series.shift(1) <= upper_series.shift(1))
            ).astype(int)
            factors["bb_break_down"] = (  # 价格跌破布林带下轨，弱势信号
                (close_series < lower_series)
                & (close_series.shift(1) >= lower_series.shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算布林带因子时出错: {e}")

        return factors

    def _calculate_stochastic_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """
        计算随机指标（Stochastic Oscillator）相关因子

        随机指标是动量震荡器，比较收盘价与过去一段时间内价格区间的关系
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准随机指标 - 经过平滑处理的慢速随机指标
            slowk, slowd = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )
            factors["stoch_k"] = slowk  # 慢速%K线，主要随机指标线
            factors["stoch_d"] = slowd  # 慢速%D线，%K的移动平均，信号线

            # 快速随机指标 - 未经平滑的原始随机指标
            fastk, fastd = talib.STOCHF(
                high, low, close, fastk_period=14, fastd_period=3
            )
            factors["stoch_fast_k"] = fastk  # 快速%K线，更敏感的随机指标
            factors["stoch_fast_d"] = fastd  # 快速%D线，快速%K的移动平均

            # 随机RSI - 结合RSI和随机指标的优势
            factors["stoch_rsi"] = self._calculate_stoch_rsi(
                close
            )  # 随机RSI，更敏感的超买超卖指标

            # 随机指标衍生 - 基于随机指标的交易信号
            slowk_series = pd.Series(slowk, index=factors.index)
            slowd_series = pd.Series(slowd, index=factors.index)

            factors["stoch_oversold"] = (slowk < 20).astype(
                int
            )  # 随机指标超卖（<20），潜在买入机会
            factors["stoch_overbought"] = (slowk > 80).astype(
                int
            )  # 随机指标超买（>80），潜在卖出机会
            factors["stoch_cross_up"] = (  # 随机指标金叉，%K上穿%D
                (slowk_series > slowd_series)
                & (slowk_series.shift(1) <= slowd_series.shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算随机指标因子时出错: {e}")

        return factors

    def _calculate_williams_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """
        计算威廉指标（Williams %R）相关因子

        威廉指标是动量震荡器，衡量收盘价在过去n期高低价区间中的位置
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 威廉%R - 反向随机指标，值域为-100到0
            factors["williams_r"] = talib.WILLR(
                high, low, close, timeperiod=14
            )  # 14期威廉%R，标准参数
            factors["williams_r_21"] = talib.WILLR(
                high, low, close, timeperiod=21
            )  # 21期威廉%R，更平滑

            # 威廉指标衍生 - 基于威廉指标的交易信号
            factors["williams_oversold"] = (factors["williams_r"] < -80).astype(
                int
            )  # 威廉%R超卖（<-80）
            factors["williams_overbought"] = (factors["williams_r"] > -20).astype(
                int
            )  # 威廉%R超买（>-20）

        except Exception as e:
            logger.error(f"计算威廉指标因子时出错: {e}")

        return factors

    def _calculate_cci_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """
        计算CCI指标（Commodity Channel Index）相关因子

        CCI是动量指标，衡量价格偏离其统计平均值的程度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # CCI - 商品通道指数
            factors["cci"] = talib.CCI(
                high, low, close, timeperiod=14
            )  # 14期CCI，标准参数
            factors["cci_20"] = talib.CCI(
                high, low, close, timeperiod=20
            )  # 20期CCI，更平滑

            # CCI衍生指标 - 基于CCI的交易信号
            factors["cci_extreme_high"] = (factors["cci"] > 100).astype(
                int
            )  # CCI超买信号（>100）
            factors["cci_extreme_low"] = (factors["cci"] < -100).astype(
                int
            )  # CCI超卖信号（<-100）
            factors["cci_momentum"] = factors["cci"].diff(1)  # CCI动量，衡量CCI变化速度

        except Exception as e:
            logger.error(f"计算CCI因子时出错: {e}")

        return factors

    def _calculate_atr_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """
        计算ATR（Average True Range）相关因子

        ATR是波动性指标，衡量价格的平均真实波动幅度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # ATR - 平均真实波动幅度
            factors["atr"] = talib.ATR(
                high, low, close, timeperiod=14
            )  # 14期ATR，衡量近期波动性
            factors["atr_21"] = talib.ATR(
                high, low, close, timeperiod=21
            )  # 21期ATR，更平滑的波动性

            # ATR衍生指标 - 基于ATR的波动性分析
            factors["atr_ratio"] = (
                factors["atr"] / close * 100
            )  # ATR占收盘价的百分比，标准化波动率
            factors["atr_momentum"] = factors["atr"].diff(1)  # ATR变化，波动性增减趋势
            factors["atr_high_volatility"] = (  # 高波动性标识
                factors["atr"] > factors["atr"].rolling(20).mean() * 1.5
            ).astype(int)

            # 真实范围 - ATR的基础组成部分
            factors["true_range"] = talib.TRANGE(
                high, low, close
            )  # 真实波动幅度，单日波动性

        except Exception as e:
            logger.error(f"计算ATR因子时出错: {e}")

        return factors

    def _calculate_volume_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """
        计算成交量相关技术因子

        成交量技术指标结合价格和成交量，分析资金流向和市场参与度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # OBV - 能量潮指标，累积成交量指标
            factors["obv"] = talib.OBV(close, volume)  # 成交量累积，反映资金流向

            # 资金流量指标 - 结合价格位置和成交量
            factors["mfi"] = talib.MFI(
                high, low, close, volume, timeperiod=14
            )  # 资金流量指数，成交量版RSI

            # 累积/分布线 - 基于收盘价位置的资金流向指标
            factors["ad"] = talib.AD(
                high, low, close, volume
            )  # 累积分布线，衡量资金流入流出
            factors["adosc"] = talib.ADOSC(  # 累积分布震荡器，AD线的MACD
                high, low, close, volume, fastperiod=3, slowperiod=10
            )

            # 成交量衍生指标 - 基于成交量指标的交易信号
            factors["obv_momentum"] = factors["obv"].diff(1)  # OBV动量，资金流向变化
            factors["mfi_oversold"] = (factors["mfi"] < 20).astype(int)  # MFI超卖信号
            factors["mfi_overbought"] = (factors["mfi"] > 80).astype(int)  # MFI超买信号

        except Exception as e:
            logger.error(f"计算成交量技术因子时出错: {e}")

        return factors

    def _calculate_stoch_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算随机RSI"""
        try:
            rsi = talib.RSI(close, timeperiod=period)
            rsi_series = pd.Series(rsi)

            lowest_rsi = rsi_series.rolling(window=period).min()
            highest_rsi = rsi_series.rolling(window=period).max()

            stoch_rsi = 100 * (rsi_series - lowest_rsi) / (highest_rsi - lowest_rsi)
            return stoch_rsi.values

        except Exception as e:
            logger.error(f"计算随机RSI时出错: {e}")
            return np.full(len(close), np.nan)


# 创建全局技术因子计算器实例
technical_factors = TechnicalFactors()

__all__ = ["TechnicalFactors", "technical_factors"]
