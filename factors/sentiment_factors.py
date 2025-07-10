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
        """
        计算动量类情绪因子

        动量因子反映价格变化的速度和持续性，是市场情绪的直接体现
        """
        factors = pd.DataFrame(index=range(len(close)))
        close_series = pd.Series(close, index=factors.index)

        try:
            # 短期动量 - 不同周期的价格变化率
            factors["momentum_1"] = close_series.pct_change(
                1
            )  # 1期动量，即日收益率，反映短期情绪
            factors["momentum_3"] = close_series.pct_change(3)  # 3期动量，短期趋势强度
            factors["momentum_5"] = close_series.pct_change(
                5
            )  # 5期动量，短期动量持续性
            factors["momentum_10"] = close_series.pct_change(
                10
            )  # 10期动量，中短期趋势判断
            factors["momentum_20"] = close_series.pct_change(
                20
            )  # 20期动量，中期趋势强度

            # 动量加速度 - 动量的变化率，反映情绪变化的加速度
            factors["momentum_acceleration"] = factors["momentum_1"].diff(
                1
            )  # 动量加速度，情绪变化的二阶导数

            # 连续上涨/下跌天数 - 趋势持续性指标
            returns = factors["momentum_1"]
            factors["consecutive_up"] = self._calculate_consecutive_periods(
                returns > 0
            )  # 连续上涨期数，多头情绪持续性
            factors["consecutive_down"] = (
                self._calculate_consecutive_periods(  # 连续下跌期数，空头情绪持续性
                    returns < 0
                )
            )

            # 动量强度（相对于历史波动率）- 标准化的动量指标
            rolling_std = returns.rolling(20).std()
            factors["momentum_strength"] = (
                returns / rolling_std
            )  # 相对动量强度，排除波动率影响的纯动量

            # 趋势一致性（多个周期动量方向一致性）- 多时间框架情绪共振
            factors["trend_consistency"] = (  # 趋势一致性得分，0-1之间
                (factors["momentum_1"] > 0).astype(int)
                + (factors["momentum_3"] > 0).astype(int)
                + (factors["momentum_5"] > 0).astype(int)
                + (factors["momentum_10"] > 0).astype(int)
            ) / 4

            # 动量转折点识别 - 短期与中期动量背离，可能的转折信号
            factors["momentum_reversal"] = (  # 动量转折信号，短中期方向不一致
                (factors["momentum_1"] > 0) & (factors["momentum_3"] < 0)
                | (factors["momentum_1"] < 0) & (factors["momentum_3"] > 0)
            ).astype(int)

        except Exception as e:
            logger.error(f"计算动量情绪因子时出错: {e}")

        return factors

    def _calculate_volatility_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> pd.DataFrame:
        """
        计算波动率类情绪因子

        波动率是市场不确定性和恐慌情绪的重要体现
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)

            # 收益率波动率 - 基于收益率的波动性测量
            returns = close_series.pct_change()
            factors["volatility_5"] = returns.rolling(
                5
            ).std()  # 5期波动率，短期市场不确定性
            factors["volatility_10"] = returns.rolling(
                10
            ).std()  # 10期波动率，中短期情绪波动
            factors["volatility_20"] = returns.rolling(
                20
            ).std()  # 20期波动率，中期市场风险水平

            # 价格范围波动率 - 基于高低价差的波动性
            price_range = (high_series - low_series) / close_series
            factors["range_volatility_5"] = price_range.rolling(
                5
            ).mean()  # 短期价格区间波动率
            factors["range_volatility_20"] = price_range.rolling(
                20
            ).mean()  # 中期价格区间波动率

            # 波动率趋势 - 波动率的相对变化
            factors["volatility_trend"] = (  # 波动率趋势，>1表示波动率上升
                factors["volatility_5"] / factors["volatility_20"] - 1
            )

            # 波动率突破 - 异常高波动率事件
            factors["volatility_spike"] = (  # 波动率突增信号，市场恐慌指标
                factors["volatility_5"] > factors["volatility_20"] * 1.5
            ).astype(int)

            # GARCH类似的条件波动率 - 自适应波动率预测
            factors["conditional_volatility"] = (
                self._calculate_garch_like_volatility(  # 条件波动率，考虑波动率聚集性
                    returns
                )
            )

            # 波动率偏度（反映市场情绪的不对称性）
            factors["volatility_skewness"] = returns.rolling(
                20
            ).skew()  # 收益率偏度，市场情绪偏向
            factors["volatility_kurtosis"] = returns.rolling(
                20
            ).kurt()  # 收益率峰度，极端事件频率

            # VIX类似指标（基于价格范围）- 恐慌指数的简化版本
            factors["vix_like"] = (  # 类VIX指标，基于价格区间的恐慌度量
                price_range.rolling(20).mean() / price_range.rolling(100).mean()
            )

        except Exception as e:
            logger.error(f"计算波动率情绪因子时出错: {e}")

        return factors

    def _calculate_volume_sentiment_factors(
        self, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """
        计算成交量情绪因子

        成交量反映市场参与度和情绪强度，是情绪分析的重要维度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)
            returns = close_series.pct_change()

            # 成交量变化率 - 参与度变化
            factors["volume_change"] = (
                volume_series.pct_change()
            )  # 成交量变化率，市场参与度变化

            # 量价背离 - 价格与成交量方向不一致的异常情况
            factors["volume_price_divergence"] = (  # 量价背离信号，可能的转折预警
                (returns > 0) & (factors["volume_change"] < 0)
                | (returns < 0) & (factors["volume_change"] > 0)
            ).astype(int)

            # 成交量突增 - 异常高参与度事件
            volume_ma = volume_series.rolling(20).mean()
            factors["volume_surge"] = (volume_series > volume_ma * 2).astype(
                int
            )  # 成交量突增，重大事件或情绪爆发

            # 价量配合度 - 价格与成交量的协同性
            volume_return_corr = returns.rolling(20).corr(factors["volume_change"])
            factors["volume_price_cooperation"] = (
                volume_return_corr  # 量价配合度，市场情绪一致性
            )

            # 资金流向强度 - 加权的价格变化
            factors["money_flow_strength"] = (
                returns * volume_series
            )  # 资金流向强度，考虑成交量的价格动量
            factors["cumulative_money_flow"] = (  # 累积资金流向，中期资金流向趋势
                factors["money_flow_strength"].rolling(20).sum()
            )

            # 成交量相对强度 - 标准化的成交量活跃度
            factors["volume_relative_strength"] = (
                volume_series / volume_ma
            )  # 成交量相对强度，当前活跃度水平

            # 成交量波动率 - 确保索引对齐
            volume_volatility = factors["volume_change"].rolling(10).std()
            factors["volume_volatility"] = (
                volume_volatility  # 成交量波动率，参与度稳定性
            )

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
        """
        计算价格行为情绪因子

        基于K线形态分析市场参与者的情绪和行为模式，这些因子反映市场心理状态
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            open_series = pd.Series(open_price, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)
            close_series = pd.Series(close, index=factors.index)

            # 跳空缺口 - 情绪突变的直接体现
            gap = (open_series - close_series.shift(1)) / close_series.shift(1)
            factors["gap_up"] = (gap > 0.01).astype(
                int
            )  # 向上跳空>1%，突发利好或FOMO情绪
            factors["gap_down"] = (gap < -0.01).astype(
                int
            )  # 向下跳空>1%，恐慌抛售或利空

            # 长上影线和长下影线（反映市场犹豫情绪）
            body_size = abs(close_series - open_series)
            upper_shadow = high_series - np.maximum(open_series, close_series)
            lower_shadow = np.minimum(open_series, close_series) - low_series

            factors["long_upper_shadow"] = (upper_shadow > body_size * 2).astype(
                int
            )  # 长上影线，高位遇阻力，卖压重
            factors["long_lower_shadow"] = (lower_shadow > body_size * 2).astype(
                int
            )  # 长下影线，低位有支撑，买盘强

            # 十字星（市场犹豫）- 多空力量均衡，方向不明
            factors["doji"] = (body_size < (high_series - low_series) * 0.1).astype(
                int
            )  # 十字星形态，市场犹豫不决

            # 收盘价位置（反映多空力量对比）
            high_low_diff = high_series - low_series
            # 直接计算并赋值，避免形状问题
            factors["close_position"] = (
                close_series - low_series
            ) / high_low_diff  # 收盘价在日内区间位置，0-1之间
            factors["close_position"] = factors["close_position"].fillna(np.nan)

            # 价格拒绝（价格试探但被拉回）- 关键位置的情绪反转
            factors["price_rejection_up"] = (  # 上方价格拒绝，冲高回落
                (high_series > high_series.shift(1))
                & (close_series < open_series)
                & (upper_shadow > body_size)
            ).astype(int)

            factors["price_rejection_down"] = (  # 下方价格拒绝，探底回升
                (low_series < low_series.shift(1))
                & (close_series > open_series)
                & (lower_shadow > body_size)
            ).astype(int)

            # 市场情绪强度（基于实体大小）- 多空争夺激烈程度
            range_size = high_series - low_series
            factors["market_emotion_strength"] = (
                body_size / range_size
            )  # 实体占比，反映情绪强度

        except Exception as e:
            logger.error(f"计算价格行为情绪因子时出错: {e}")

        return factors

    def _calculate_market_strength_factors(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """
        计算市场强度因子

        衡量市场参与者的力量对比和市场整体强弱程度，反映资金流向和情绪强度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            high_series = pd.Series(high, index=factors.index)
            low_series = pd.Series(low, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)

            # 价格强度 - 风险调整后的价格动量
            returns = close_series.pct_change()
            factors["price_strength"] = (  # 价格强度，考虑波动率的风险调整收益
                returns.rolling(5).mean() / returns.rolling(5).std()
            )

            # 突破强度 - 价格突破历史区间的能力
            high_ma = high_series.rolling(20).max()
            low_ma = low_series.rolling(20).min()
            factors["breakout_strength_up"] = (close_series > high_ma).astype(
                int
            )  # 向上突破强度，创新高能力
            factors["breakout_strength_down"] = (close_series < low_ma).astype(
                int
            )  # 向下突破强度，创新低程度

            # 市场宽度（价格在区间中的位置）- 相对强弱位置
            price_range_20 = (
                high_series.rolling(20).max() - low_series.rolling(20).min()
            )
            factors["market_breadth"] = (  # 市场宽度，价格在历史区间中位置
                close_series - low_series.rolling(20).min()
            ) / price_range_20

            # 买卖压力 - 基于成交量的资金流向分析
            typical_price = (high_series + low_series + close_series) / 3
            money_flow = typical_price * volume_series
            factors["buying_pressure"] = money_flow.rolling(
                10
            ).apply(  # 买入压力，上涨时的资金流入
                lambda x: x[
                    x.index[
                        close_series.loc[x.index] > close_series.loc[x.index].shift(1)
                    ]
                ].sum()
            )
            factors["selling_pressure"] = money_flow.rolling(
                10
            ).apply(  # 卖出压力，下跌时的资金流出
                lambda x: x[
                    x.index[
                        close_series.loc[x.index] < close_series.loc[x.index].shift(1)
                    ]
                ].sum()
            )

            # 相对强弱 - 买卖压力的比值
            factors["relative_strength"] = factors[
                "buying_pressure"
            ] / (  # 相对强弱，买卖力量对比
                factors["selling_pressure"] + 1e-8
            )

        except Exception as e:
            logger.error(f"计算市场强度因子时出错: {e}")

        return factors

    def _calculate_fear_greed_factors(
        self, close: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """
        计算恐慌贪婪指数类因子

        模拟CNN恐慌贪婪指数的计算逻辑，综合多维度衡量市场情绪极端程度
        """
        factors = pd.DataFrame(index=range(len(close)))

        try:
            close_series = pd.Series(close, index=factors.index)
            volume_series = pd.Series(volume, index=factors.index)
            returns = close_series.pct_change()

            # 极端移动（恐慌/贪婪的表现）- 异常大的价格变动
            return_std = returns.rolling(20).std()
            factors["extreme_movement"] = (abs(returns) > return_std * 2).astype(
                int
            )  # 极端价格变动，情绪极化事件

            # 市场情绪指数（综合多个指标）- 多维度情绪评分
            momentum_score = (returns.rolling(5).mean() > 0).astype(int)  # 动量得分
            volatility_score = (return_std < return_std.rolling(50).mean()).astype(
                int
            )  # 波动率得分（低波动=正面）
            volume_score = (
                volume_series > volume_series.rolling(20).mean()
            ).astype(  # 成交量得分
                int
            )

            factors["sentiment_index"] = (  # 综合情绪指数，0-1之间
                momentum_score + volatility_score + volume_score
            ) / 3

            # 恐慌指标（大跌+高波动+高成交量）- 市场恐慌状态识别
            factors["panic_indicator"] = (  # 恐慌指标，三重确认的恐慌信号
                (returns < -0.03)
                & (return_std > return_std.rolling(20).mean() * 1.5)
                & (volume_series > volume_series.rolling(20).mean() * 1.5)
            ).astype(int)

            # 贪婪指标（大涨+低波动+高成交量）- 市场贪婪状态识别
            factors["greed_indicator"] = (  # 贪婪指标，过度乐观的市场状态
                (returns > 0.03)
                & (return_std < return_std.rolling(20).mean())
                & (volume_series > volume_series.rolling(20).mean() * 1.2)
            ).astype(int)

            # 市场过热/过冷 - 价格偏离均值的程度
            price_ma = close_series.rolling(20).mean()
            price_deviation = (close_series - price_ma) / price_ma
            factors["market_overheated"] = (price_deviation > 0.1).astype(
                int
            )  # 市场过热，价格严重偏离均值
            factors["market_oversold"] = (price_deviation < -0.1).astype(
                int
            )  # 市场超卖，价格严重低估

            # 情绪转换点 - 情绪极端后的反转信号
            sentiment_ma = factors["sentiment_index"].rolling(10).mean()
            factors["sentiment_reversal"] = (  # 情绪反转信号，极端情绪的转折点
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
