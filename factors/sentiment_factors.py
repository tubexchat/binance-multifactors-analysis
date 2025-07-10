"""
市场情绪因子模块

基于价格动量、波动率、成交量等计算市场情绪相关因子
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentFactors:
    """市场情绪因子计算器"""

    def __init__(self):
        self.required_periods = 100  # 最少需要100个数据点

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有情绪因子

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含所有情绪因子的DataFrame
        """
        if len(df) < self.required_periods:
            logger.warning(f"数据长度不足，需要至少{self.required_periods}个数据点")
            return pd.DataFrame()

        result_df = df[["timestamp"]].copy()

        try:
            # 基础数据
            open_price = df["open"].values
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            volume = df["volume"].values

            # 1. 动量类情绪因子
            result_df = pd.concat(
                [result_df, self._calculate_momentum_factors(close)], axis=1
            )

            # 2. 波动率类情绪因子
            result_df = pd.concat(
                [result_df, self._calculate_volatility_factors(high, low, close)],
                axis=1,
            )

            # 3. 成交量情绪因子
            result_df = pd.concat(
                [result_df, self._calculate_volume_sentiment_factors(close, volume)],
                axis=1,
            )

            # 4. 价格行为情绪因子
            result_df = pd.concat(
                [
                    result_df,
                    self._calculate_price_behavior_factors(
                        open_price, high, low, close
                    ),
                ],
                axis=1,
            )

            # 5. 市场强度因子
            result_df = pd.concat(
                [
                    result_df,
                    self._calculate_market_strength_factors(high, low, close, volume),
                ],
                axis=1,
            )

            # 6. 恐慌贪婪指数类因子
            result_df = pd.concat(
                [result_df, self._calculate_fear_greed_factors(close, volume)], axis=1
            )

            logger.info(f"成功计算 {len(result_df.columns)-1} 个情绪因子")
            return result_df

        except Exception as e:
            logger.error(f"计算情绪因子时出错: {e}")
            return pd.DataFrame()

    def _calculate_momentum_factors(self, close: np.ndarray) -> pd.DataFrame:
        """计算动量类情绪因子"""
        factors = pd.DataFrame(index=range(len(close)))
        close_series = pd.Series(close, index=factors.index)

        try:
            # 短期动量
            factors["momentum_1"] = close_series.pct_change(1)
            factors["momentum_3"] = close_series.pct_change(3)
            factors["momentum_5"] = close_series.pct_change(5)
            factors["momentum_10"] = close_series.pct_change(10)
            factors["momentum_20"] = close_series.pct_change(20)

            # 动量加速度
            factors["momentum_acceleration"] = factors["momentum_1"].diff(1)

            # 连续上涨/下跌天数
            returns = factors["momentum_1"]
            factors["consecutive_up"] = self._calculate_consecutive_periods(returns > 0)
            factors["consecutive_down"] = self._calculate_consecutive_periods(
                returns < 0
            )

            # 动量强度（相对于历史波动率）
            rolling_std = returns.rolling(20).std()
            factors["momentum_strength"] = returns / rolling_std

            # 趋势一致性（多个周期动量方向一致性）
            factors["trend_consistency"] = (
                (factors["momentum_1"] > 0).astype(int)
                + (factors["momentum_3"] > 0).astype(int)
                + (factors["momentum_5"] > 0).astype(int)
                + (factors["momentum_10"] > 0).astype(int)
            ) / 4

            # 动量转折点识别
            factors["momentum_reversal"] = (
                (factors["momentum_1"] > 0) & (factors["momentum_3"] < 0)
                | (factors["momentum_1"] < 0) & (factors["momentum_3"] > 0)
            ).astype(int)

        except Exception as e:
            logger.error(f"计算动量情绪因子时出错: {e}")

        return factors

    def _calculate_volatility_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """计算波动率类情绪因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)

            # 收益率波动率
            returns = close_series.pct_change()
            factors["volatility_5"] = returns.rolling(5).std()
            factors["volatility_10"] = returns.rolling(10).std()
            factors["volatility_20"] = returns.rolling(20).std()

            # 价格范围波动率
            price_range = (high_series - low_series) / close_series
            factors["range_volatility_5"] = price_range.rolling(5).mean()
            factors["range_volatility_20"] = price_range.rolling(20).mean()

            # 波动率趋势
            factors["volatility_trend"] = (
                factors["volatility_5"] / factors["volatility_20"] - 1
            )

            # 波动率突破
            factors["volatility_spike"] = (
                factors["volatility_5"] > factors["volatility_20"] * 1.5
            ).astype(int)

            # GARCH类似的条件波动率
            factors["conditional_volatility"] = self._calculate_garch_like_volatility(
                returns
            )

            # 波动率偏度（反映市场情绪的不对称性）
            factors["volatility_skewness"] = returns.rolling(20).skew()
            factors["volatility_kurtosis"] = returns.rolling(20).kurt()

            # VIX类似指标（基于价格范围）
            factors["vix_like"] = (
                price_range.rolling(20).mean() / price_range.rolling(100).mean()
            )

        except Exception as e:
            logger.error(f"计算波动率情绪因子时出错: {e}")

        return factors

    def _calculate_volume_sentiment_factors(
        self, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """计算成交量情绪因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)
            returns = close_series.pct_change()

            # 成交量变化率
            factors["volume_change"] = volume_series.pct_change()

            # 量价背离
            factors["volume_price_divergence"] = (
                (returns > 0) & (factors["volume_change"] < 0)
                | (returns < 0) & (factors["volume_change"] > 0)
            ).astype(int)

            # 成交量突增
            volume_ma = volume_series.rolling(20).mean()
            factors["volume_surge"] = (volume_series > volume_ma * 2).astype(int)

            # 价量配合度
            volume_return_corr = returns.rolling(20).corr(factors["volume_change"])
            factors["volume_price_cooperation"] = volume_return_corr

            # 资金流向强度
            factors["money_flow_strength"] = returns * volume_series
            factors["cumulative_money_flow"] = (
                factors["money_flow_strength"].rolling(20).sum()
            )

            # 成交量相对强度
            factors["volume_relative_strength"] = volume_series / volume_ma

            # 成交量波动率 - 确保索引对齐
            volume_volatility = factors["volume_change"].rolling(10).std()
            factors["volume_volatility"] = volume_volatility

        except Exception as e:
            logger.error(f"计算成交量情绪因子时出错: {e}")

        return factors

    def _calculate_price_behavior_factors(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """计算价格行为情绪因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            open_series = pd.Series(open_price, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)
            close_series = pd.Series(close, index=factors.index)

            # 跳空缺口
            gap = (open_series - close_series.shift(1)) / close_series.shift(1)
            factors["gap_up"] = (gap > 0.01).astype(int)  # 向上跳空超过1%
            factors["gap_down"] = (gap < -0.01).astype(int)  # 向下跳空超过1%

            # 长上影线和长下影线（反映市场犹豫情绪）
            body_size = abs(close_series - open_series)
            upper_shadow = high_series - np.maximum(open_series, close_series)
            lower_shadow = np.minimum(open_series, close_series) - low_series

            factors["long_upper_shadow"] = (upper_shadow > body_size * 2).astype(int)
            factors["long_lower_shadow"] = (lower_shadow > body_size * 2).astype(int)

            # 十字星（市场犹豫）
            factors["doji"] = (body_size < (high_series - low_series) * 0.1).astype(int)

            # 收盘价位置（反映多空力量对比）
            high_low_diff = high_series - low_series
            # 直接计算并赋值，避免形状问题
            factors["close_position"] = (close_series - low_series) / high_low_diff
            factors["close_position"] = factors["close_position"].fillna(np.nan)

            # 价格拒绝（价格试探但被拉回）
            factors["price_rejection_up"] = (
                (high_series > high_series.shift(1))
                & (close_series < open_series)
                & (upper_shadow > body_size)
            ).astype(int)

            factors["price_rejection_down"] = (
                (low_series < low_series.shift(1))
                & (close_series > open_series)
                & (lower_shadow > body_size)
            ).astype(int)

            # 市场情绪强度（基于实体大小）
            range_size = high_series - low_series
            factors["market_emotion_strength"] = body_size / range_size

        except Exception as e:
            logger.error(f"计算价格行为情绪因子时出错: {e}")

        return factors

    def _calculate_market_strength_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """计算市场强度因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)

            # 价格强度
            returns = close_series.pct_change()
            factors["price_strength"] = (
                returns.rolling(5).mean() / returns.rolling(5).std()
            )

            # 突破强度
            high_ma = high_series.rolling(20).max()
            low_ma = low_series.rolling(20).min()
            factors["breakout_strength_up"] = (close_series > high_ma).astype(int)
            factors["breakout_strength_down"] = (close_series < low_ma).astype(int)

            # 市场宽度（价格在区间中的位置）
            price_range_20 = (
                high_series.rolling(20).max() - low_series.rolling(20).min()
            )
            factors["market_breadth"] = (
                close_series - low_series.rolling(20).min()
            ) / price_range_20

            # 买卖压力
            typical_price = (high_series + low_series + close_series) / 3
            money_flow = typical_price * volume_series
            factors["buying_pressure"] = money_flow.rolling(10).apply(
                lambda x: x[
                    x.index[
                        close_series.loc[x.index] > close_series.loc[x.index].shift(1)
                    ]
                ].sum()
            )
            factors["selling_pressure"] = money_flow.rolling(10).apply(
                lambda x: x[
                    x.index[
                        close_series.loc[x.index] < close_series.loc[x.index].shift(1)
                    ]
                ].sum()
            )

            # 相对强弱
            factors["relative_strength"] = factors["buying_pressure"] / (
                factors["selling_pressure"] + 1e-8
            )

        except Exception as e:
            logger.error(f"计算市场强度因子时出错: {e}")

        return factors

    def _calculate_fear_greed_factors(
        self, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """计算恐慌贪婪指数类因子"""
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)
            returns = close_series.pct_change()

            # 极端移动（恐慌/贪婪的表现）
            return_std = returns.rolling(20).std()
            factors["extreme_movement"] = (abs(returns) > return_std * 2).astype(int)

            # 市场情绪指数（综合多个指标）
            momentum_score = (returns.rolling(5).mean() > 0).astype(int)
            volatility_score = (return_std < return_std.rolling(50).mean()).astype(int)
            volume_score = (volume_series > volume_series.rolling(20).mean()).astype(
                int
            )

            factors["sentiment_index"] = (
                momentum_score + volatility_score + volume_score
            ) / 3

            # 恐慌指标（大跌+高波动+高成交量）
            factors["panic_indicator"] = (
                (returns < -0.03)
                & (return_std > return_std.rolling(20).mean() * 1.5)
                & (volume_series > volume_series.rolling(20).mean() * 1.5)
            ).astype(int)

            # 贪婪指标（大涨+低波动+高成交量）
            factors["greed_indicator"] = (
                (returns > 0.03)
                & (return_std < return_std.rolling(20).mean())
                & (volume_series > volume_series.rolling(20).mean() * 1.2)
            ).astype(int)

            # 市场过热/过冷
            price_ma = close_series.rolling(20).mean()
            price_deviation = (close_series - price_ma) / price_ma
            factors["market_overheated"] = (price_deviation > 0.1).astype(int)
            factors["market_oversold"] = (price_deviation < -0.1).astype(int)

            # 情绪转换点
            sentiment_ma = factors["sentiment_index"].rolling(10).mean()
            factors["sentiment_reversal"] = (
                (factors["sentiment_index"] > 0.7) & (sentiment_ma < 0.5)
                | (factors["sentiment_index"] < 0.3) & (sentiment_ma > 0.5)
            ).astype(int)

        except Exception as e:
            logger.error(f"计算恐慌贪婪因子时出错: {e}")

        return factors

    def _calculate_consecutive_periods(self, condition: pd.Series) -> pd.Series:
        """计算连续满足条件的周期数"""
        try:
            groups = (condition != condition.shift()).cumsum()
            consecutive = condition.groupby(groups).cumsum()
            return consecutive * condition
        except Exception:
            return pd.Series(np.zeros(len(condition)), index=condition.index)

    def _calculate_garch_like_volatility(
        self, returns: pd.Series, alpha: float = 0.1, beta: float = 0.8
    ) -> pd.Series:
        """计算类似GARCH模型的条件波动率"""
        try:
            volatility = pd.Series(index=returns.index, dtype=float)
            volatility.iloc[0] = returns.std()

            for i in range(1, len(returns)):
                if pd.notna(returns.iloc[i]) and pd.notna(returns.iloc[i - 1]):
                    volatility.iloc[i] = (
                        alpha * returns.iloc[i - 1] ** 2
                        + beta * volatility.iloc[i - 1] ** 2
                        + (1 - alpha - beta) * returns.var()
                    ) ** 0.5
                else:
                    volatility.iloc[i] = volatility.iloc[i - 1]

            return volatility
        except Exception:
            return pd.Series(np.full(len(returns), returns.std()), index=returns.index)


# 创建全局情绪因子计算器实例
sentiment_factors = SentimentFactors()

__all__ = ["SentimentFactors", "sentiment_factors"]
