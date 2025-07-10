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
        """计算RSI相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准RSI
            factors["rsi_14"] = talib.RSI(close, timeperiod=14)
            factors["rsi_21"] = talib.RSI(close, timeperiod=21)
            factors["rsi_6"] = talib.RSI(close, timeperiod=6)

            # RSI衍生指标
            factors["rsi_oversold"] = (factors["rsi_14"] < 30).astype(int)
            factors["rsi_overbought"] = (factors["rsi_14"] > 70).astype(int)
            factors["rsi_momentum"] = factors["rsi_14"].diff(1)
            factors["rsi_cross_50"] = (
                (factors["rsi_14"] > 50) & (factors["rsi_14"].shift(1) <= 50)
            ).astype(int)

        except Exception as e:
            logger.error(f"计算RSI因子时出错: {e}")

        return factors

    def _calculate_macd_factors(self, close: np.ndarray) -> pd.DataFrame:
        """计算MACD相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            factors["macd"] = macd
            factors["macd_signal"] = macd_signal
            factors["macd_histogram"] = macd_hist

            # 转换为pandas Series以便使用shift和diff方法
            macd_series = pd.Series(macd, index=factors.index)
            macd_signal_series = pd.Series(macd_signal, index=factors.index)
            macd_hist_series = pd.Series(macd_hist, index=factors.index)

            # MACD衍生指标
            factors["macd_cross_up"] = (
                (macd_series > macd_signal_series)
                & (macd_series.shift(1) <= macd_signal_series.shift(1))
            ).astype(int)
            factors["macd_cross_down"] = (
                (macd_series < macd_signal_series)
                & (macd_series.shift(1) >= macd_signal_series.shift(1))
            ).astype(int)
            factors["macd_momentum"] = macd_series.diff(1)
            factors["macd_divergence"] = macd_hist_series.diff(1)

            # 不同参数的MACD
            macd_fast, _, _ = talib.MACD(
                close, fastperiod=5, slowperiod=13, signalperiod=9
            )
            factors["macd_fast"] = macd_fast

        except Exception as e:
            logger.error(f"计算MACD因子时出错: {e}")

        return factors

    def _calculate_ma_factors(self, close: np.ndarray) -> pd.DataFrame:
        """计算移动平均线相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 简单移动平均线
            close_series = pd.Series(close, index=factors.index)
            for period in [5, 10, 20, 50, 100, 200]:
                ma = talib.SMA(close, timeperiod=period)
                ma_series = pd.Series(ma, index=factors.index)
                factors[f"sma_{period}"] = ma
                factors[f"sma_{period}_ratio"] = close / ma - 1
                factors[f"sma_{period}_cross_up"] = (
                    (close_series > ma_series)
                    & (close_series.shift(1) <= ma_series.shift(1))
                ).astype(int)

            # 指数移动平均线
            for period in [5, 10, 20, 50]:
                ema = talib.EMA(close, timeperiod=period)
                factors[f"ema_{period}"] = ema
                factors[f"ema_{period}_ratio"] = close / ema - 1

            # 移动平均线组合指标
            factors["ma_5_20_cross"] = (
                (factors["sma_5"] > factors["sma_20"])
                & (factors["sma_5"].shift(1) <= factors["sma_20"].shift(1))
            ).astype(int)
            factors["ma_10_50_cross"] = (
                (factors["sma_10"] > factors["sma_50"])
                & (factors["sma_10"].shift(1) <= factors["sma_50"].shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算移动平均线因子时出错: {e}")

        return factors

    def _calculate_bollinger_factors(self, close: np.ndarray) -> pd.DataFrame:
        """计算布林带相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 布林带
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            factors["bollinger_upper"] = upper
            factors["bollinger_middle"] = middle
            factors["bollinger_lower"] = lower

            # 布林带衍生指标
            close_series = pd.Series(close, index=factors.index)
            upper_series = pd.Series(upper, index=factors.index)
            lower_series = pd.Series(lower, index=factors.index)

            factors["bb_width"] = (upper - lower) / middle
            factors["bb_position"] = (close - lower) / (upper - lower)
            factors["bb_squeeze"] = (
                factors["bb_width"] < factors["bb_width"].rolling(20).mean()
            ).astype(int)
            factors["bb_break_up"] = (
                (close_series > upper_series)
                & (close_series.shift(1) <= upper_series.shift(1))
            ).astype(int)
            factors["bb_break_down"] = (
                (close_series < lower_series)
                & (close_series.shift(1) >= lower_series.shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算布林带因子时出错: {e}")

        return factors

    def _calculate_stochastic_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """计算随机指标相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 标准随机指标
            slowk, slowd = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )
            factors["stoch_k"] = slowk
            factors["stoch_d"] = slowd

            # 快速随机指标
            fastk, fastd = talib.STOCHF(
                high, low, close, fastk_period=14, fastd_period=3
            )
            factors["stoch_fast_k"] = fastk
            factors["stoch_fast_d"] = fastd

            # 随机RSI
            factors["stoch_rsi"] = self._calculate_stoch_rsi(close)

            # 随机指标衍生
            slowk_series = pd.Series(slowk, index=factors.index)
            slowd_series = pd.Series(slowd, index=factors.index)

            factors["stoch_oversold"] = (slowk < 20).astype(int)
            factors["stoch_overbought"] = (slowk > 80).astype(int)
            factors["stoch_cross_up"] = (
                (slowk_series > slowd_series)
                & (slowk_series.shift(1) <= slowd_series.shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"计算随机指标因子时出错: {e}")

        return factors

    def _calculate_williams_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """计算威廉指标相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # 威廉%R
            factors["williams_r"] = talib.WILLR(high, low, close, timeperiod=14)
            factors["williams_r_21"] = talib.WILLR(high, low, close, timeperiod=21)

            # 威廉指标衍生
            factors["williams_oversold"] = (factors["williams_r"] < -80).astype(int)
            factors["williams_overbought"] = (factors["williams_r"] > -20).astype(int)

        except Exception as e:
            logger.error(f"计算威廉指标因子时出错: {e}")

        return factors

    def _calculate_cci_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """计算CCI指标相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # CCI
            factors["cci"] = talib.CCI(high, low, close, timeperiod=14)
            factors["cci_20"] = talib.CCI(high, low, close, timeperiod=20)

            # CCI衍生指标
            factors["cci_extreme_high"] = (factors["cci"] > 100).astype(int)
            factors["cci_extreme_low"] = (factors["cci"] < -100).astype(int)
            factors["cci_momentum"] = factors["cci"].diff(1)

        except Exception as e:
            logger.error(f"计算CCI因子时出错: {e}")

        return factors

    def _calculate_atr_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """计算ATR相关因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # ATR
            factors["atr"] = talib.ATR(high, low, close, timeperiod=14)
            factors["atr_21"] = talib.ATR(high, low, close, timeperiod=21)

            # ATR衍生指标
            factors["atr_ratio"] = factors["atr"] / close * 100  # ATR占收盘价的百分比
            factors["atr_momentum"] = factors["atr"].diff(1)
            factors["atr_high_volatility"] = (
                factors["atr"] > factors["atr"].rolling(20).mean() * 1.5
            ).astype(int)

            # 真实范围
            factors["true_range"] = talib.TRANGE(high, low, close)

        except Exception as e:
            logger.error(f"计算ATR因子时出错: {e}")

        return factors

    def _calculate_volume_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """计算成交量相关技术因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            # OBV
            factors["obv"] = talib.OBV(close, volume)

            # 资金流量指标
            factors["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)

            # 累积/分布线
            factors["ad"] = talib.AD(high, low, close, volume)
            factors["adosc"] = talib.ADOSC(
                high, low, close, volume, fastperiod=3, slowperiod=10
            )

            # 成交量衍生指标
            factors["obv_momentum"] = factors["obv"].diff(1)
            factors["mfi_oversold"] = (factors["mfi"] < 20).astype(int)
            factors["mfi_overbought"] = (factors["mfi"] > 80).astype(int)

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
