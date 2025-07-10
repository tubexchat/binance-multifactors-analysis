"""
多因子分析器

提供完整的多因子分析功能，包括因子计算、IC分析、因子相关性分析等
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
from pathlib import Path

from core.config import config
from core.factor_engine import factor_engine
from factors.ic_analysis import ic_analyzer

logger = logging.getLogger(__name__)


class MultiFactorAnalyzer:
    """多因子分析器"""

    def __init__(self):
        self.factor_engine = factor_engine
        self.ic_analyzer = ic_analyzer
        self.analysis_config = config.analysis

    async def initialize(self):
        """初始化分析器"""
        try:
            await self.factor_engine.initialize()
            logger.info("多因子分析器初始化完成")
        except Exception as e:
            logger.error(f"初始化多因子分析器时出错: {e}")
            raise

    async def cleanup(self):
        """清理资源"""
        try:
            await self.factor_engine.cleanup()
            logger.info("多因子分析器清理完成")
        except Exception as e:
            logger.error(f"清理多因子分析器时出错: {e}")

    async def analyze_single_symbol(
        self, symbol: str, limit: int = None, save_results: bool = True
    ) -> Dict:
        """
        分析单个交易对的因子

        Args:
            symbol: 交易对符号
            limit: 数据条数限制
            save_results: 是否保存结果

        Returns:
            完整的分析结果字典
        """
        try:
            logger.info(f"开始分析交易对: {symbol}")

            # 1. 计算所有因子
            factors = await self.factor_engine.calculate_all_factors(symbol, limit)
            if factors.empty:
                logger.warning(f"无法获取 {symbol} 的因子数据")
                return {}

            # 2. 获取价格数据进行IC分析
            price_data = await self.factor_engine.data_manager.get_kline_data(
                symbol=symbol, limit=limit or config.data.HISTORY_LENGTH
            )

            if price_data.empty:
                logger.warning(f"无法获取 {symbol} 的价格数据")
                return {"factors": factors}

            # 3. 生成IC分析报告
            ic_report = self.ic_analyzer.generate_ic_report(symbol, factors, price_data)

            # 4. 计算因子相关性
            factor_correlation = self._calculate_factor_correlation(factors)

            # 5. 因子重要性排名
            factor_importance = self._calculate_factor_importance(ic_report)

            # 6. 生成分析摘要
            analysis_summary = self._generate_analysis_summary(
                symbol, factors, ic_report, factor_correlation
            )

            # 7. 整合结果
            analysis_result = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_period": {
                    "start_time": (
                        factors["timestamp"].iloc[0].isoformat()
                        if len(factors) > 0
                        else None
                    ),
                    "end_time": (
                        factors["timestamp"].iloc[-1].isoformat()
                        if len(factors) > 0
                        else None
                    ),
                    "total_periods": len(factors),
                },
                "factors": factors,
                "ic_analysis": ic_report,
                "factor_correlation": factor_correlation,
                "factor_importance": factor_importance,
                "analysis_summary": analysis_summary,
            }

            # 8. 保存结果
            if save_results:
                await self._save_analysis_results(symbol, analysis_result)

            logger.info(f"成功完成 {symbol} 的多因子分析")
            return analysis_result

        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {e}")
            return {}

    async def analyze_multiple_symbols(
        self,
        symbols: List[str],
        limit: int = None,
        max_concurrent: int = 3,
        save_results: bool = True,
    ) -> Dict[str, Dict]:
        """
        分析多个交易对的因子

        Args:
            symbols: 交易对符号列表
            limit: 数据条数限制
            max_concurrent: 最大并发数
            save_results: 是否保存结果

        Returns:
            包含各交易对分析结果的字典
        """
        try:
            logger.info(f"开始批量分析 {len(symbols)} 个交易对")

            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrent)

            async def analyze_with_semaphore(symbol):
                async with semaphore:
                    return symbol, await self.analyze_single_symbol(
                        symbol, limit, save_results
                    )

            # 批量执行分析
            tasks = [analyze_with_semaphore(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 整理结果
            analysis_results = {}
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    symbol, analysis = result
                    if analysis:  # 非空结果
                        analysis_results[symbol] = analysis
                elif isinstance(result, Exception):
                    logger.warning(f"批量分析中出错: {result}")

            # 生成批量分析报告
            if analysis_results:
                batch_report = self._generate_batch_analysis_report(analysis_results)
                if save_results:
                    await self._save_batch_results(batch_report)

            logger.info(f"成功完成 {len(analysis_results)} 个交易对的批量分析")
            return analysis_results

        except Exception as e:
            logger.error(f"批量分析时出错: {e}")
            return {}

    def _calculate_factor_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        try:
            # 排除时间戳列
            numeric_factors = factors.select_dtypes(include=[np.number])

            # 计算相关性矩阵
            correlation_matrix = numeric_factors.corr()

            # 处理NaN值
            correlation_matrix = correlation_matrix.fillna(0)

            return correlation_matrix

        except Exception as e:
            logger.error(f"计算因子相关性时出错: {e}")
            return pd.DataFrame()

    def _calculate_factor_importance(self, ic_report: Dict) -> pd.DataFrame:
        """计算因子重要性"""
        try:
            if "ic_statistics" not in ic_report or ic_report["ic_statistics"].empty:
                return pd.DataFrame()

            ic_stats = ic_report["ic_statistics"]
            importance_results = []

            for _, row in ic_stats.iterrows():
                return_period = row.get("return_period", "unknown")

                # 提取各因子的IC统计指标
                for col in row.index:
                    if col.endswith("_abs_mean") and not pd.isna(row[col]):
                        factor_name = col.replace("_abs_mean", "")

                        # 获取相关指标
                        abs_mean_ic = row[col]
                        mean_ic = row.get(f"{factor_name}_mean", 0)
                        ir_ratio = row.get(f"{factor_name}_ir", 0)

                        # 计算综合重要性分数
                        importance_score = abs_mean_ic * (1 + abs(ir_ratio))

                        importance_results.append(
                            {
                                "factor_name": factor_name,
                                "return_period": return_period,
                                "abs_mean_ic": abs_mean_ic,
                                "mean_ic": mean_ic,
                                "ir_ratio": ir_ratio,
                                "importance_score": importance_score,
                            }
                        )

            importance_df = pd.DataFrame(importance_results)

            # 按重要性分数排序
            if not importance_df.empty:
                importance_df = importance_df.sort_values(
                    "importance_score", ascending=False
                )
                importance_df["rank"] = range(1, len(importance_df) + 1)

            return importance_df

        except Exception as e:
            logger.error(f"计算因子重要性时出错: {e}")
            return pd.DataFrame()

    def _generate_analysis_summary(
        self,
        symbol: str,
        factors: pd.DataFrame,
        ic_report: Dict,
        factor_correlation: pd.DataFrame,
    ) -> Dict:
        """生成分析摘要"""
        try:
            summary = {
                "symbol": symbol,
                "total_factors": len(factors.columns) - 1,  # 排除timestamp列
                "data_quality": self._assess_data_quality(factors),
                "ic_analysis_summary": self._summarize_ic_analysis(ic_report),
                "correlation_analysis": self._summarize_correlation_analysis(
                    factor_correlation
                ),
                "recommendations": self._generate_recommendations(
                    ic_report, factor_correlation
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"生成分析摘要时出错: {e}")
            return {}

    def _assess_data_quality(self, factors: pd.DataFrame) -> Dict:
        """评估数据质量"""
        try:
            numeric_factors = factors.select_dtypes(include=[np.number])

            quality_metrics = {
                "completeness": (
                    1 - numeric_factors.isnull().sum().sum() / numeric_factors.size
                )
                * 100,
                "total_missing_values": numeric_factors.isnull().sum().sum(),
                "factors_with_missing_data": (numeric_factors.isnull().any()).sum(),
                "data_range": {
                    "start": (
                        factors["timestamp"].iloc[0].isoformat()
                        if len(factors) > 0
                        else None
                    ),
                    "end": (
                        factors["timestamp"].iloc[-1].isoformat()
                        if len(factors) > 0
                        else None
                    ),
                },
            }

            # 数据质量评级
            if quality_metrics["completeness"] >= 95:
                quality_metrics["rating"] = "Excellent"
            elif quality_metrics["completeness"] >= 90:
                quality_metrics["rating"] = "Good"
            elif quality_metrics["completeness"] >= 80:
                quality_metrics["rating"] = "Fair"
            else:
                quality_metrics["rating"] = "Poor"

            return quality_metrics

        except Exception as e:
            logger.error(f"评估数据质量时出错: {e}")
            return {}

    def _summarize_ic_analysis(self, ic_report: Dict) -> Dict:
        """总结IC分析结果"""
        try:
            if not ic_report or "effective_factors" not in ic_report:
                return {}

            effective_factors = ic_report.get("effective_factors", [])
            ic_stats = ic_report.get("ic_statistics", pd.DataFrame())

            summary = {
                "total_effective_factors": len(effective_factors),
                "effective_factors": effective_factors[:10],  # 显示前10个
                "best_performers": [],
                "overall_ic_quality": "Unknown",
            }

            # 分析最佳表现因子
            if not ic_stats.empty:
                # 找出IC绝对值最高的因子
                best_factors = []
                for _, row in ic_stats.iterrows():
                    return_period = row.get("return_period", "unknown")
                    for col in row.index:
                        if col.endswith("_abs_mean") and not pd.isna(row[col]):
                            factor_name = col.replace("_abs_mean", "")
                            best_factors.append(
                                {
                                    "factor": factor_name,
                                    "return_period": return_period,
                                    "abs_ic": row[col],
                                }
                            )

                # 排序并取前5个
                best_factors = sorted(
                    best_factors, key=lambda x: x["abs_ic"], reverse=True
                )[:5]
                summary["best_performers"] = best_factors

                # 评估整体IC质量
                avg_ic = np.mean([f["abs_ic"] for f in best_factors])
                if avg_ic >= 0.05:
                    summary["overall_ic_quality"] = "Excellent"
                elif avg_ic >= 0.03:
                    summary["overall_ic_quality"] = "Good"
                elif avg_ic >= 0.01:
                    summary["overall_ic_quality"] = "Fair"
                else:
                    summary["overall_ic_quality"] = "Poor"

            return summary

        except Exception as e:
            logger.error(f"总结IC分析时出错: {e}")
            return {}

    def _summarize_correlation_analysis(self, correlation_matrix: pd.DataFrame) -> Dict:
        """总结相关性分析"""
        try:
            if correlation_matrix.empty:
                return {}

            # 计算高相关性因子对
            high_corr_threshold = self.analysis_config.HIGH_CORRELATION_THRESHOLD
            high_corr_pairs = []

            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= high_corr_threshold:
                        high_corr_pairs.append(
                            {
                                "factor_1": correlation_matrix.columns[i],
                                "factor_2": correlation_matrix.columns[j],
                                "correlation": corr_value,
                            }
                        )

            # 按相关性绝对值排序
            high_corr_pairs = sorted(
                high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True
            )

            summary = {
                "total_factor_pairs": len(correlation_matrix.columns)
                * (len(correlation_matrix.columns) - 1)
                // 2,
                "high_correlation_pairs": len(high_corr_pairs),
                "top_correlated_pairs": high_corr_pairs[:10],  # 显示前10对
                "avg_correlation": correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix.values, k=1)
                ].mean(),
                "correlation_distribution": {
                    "high_positive": len(
                        [
                            p
                            for p in high_corr_pairs
                            if p["correlation"] > high_corr_threshold
                        ]
                    ),
                    "high_negative": len(
                        [
                            p
                            for p in high_corr_pairs
                            if p["correlation"] < -high_corr_threshold
                        ]
                    ),
                },
            }

            return summary

        except Exception as e:
            logger.error(f"总结相关性分析时出错: {e}")
            return {}

    def _generate_recommendations(
        self, ic_report: Dict, correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """生成分析建议"""
        recommendations = []

        try:
            # 基于IC分析的建议
            effective_factors = ic_report.get("effective_factors", [])
            if len(effective_factors) > 50:
                recommendations.append(
                    "建议考虑因子降维，当前有效因子数量较多，可能存在冗余"
                )
            elif len(effective_factors) < 10:
                recommendations.append(
                    "有效因子数量较少，建议增加更多类型的因子或调整IC阈值"
                )

            # 基于相关性的建议
            if not correlation_matrix.empty:
                high_corr_threshold = self.analysis_config.HIGH_CORRELATION_THRESHOLD
                high_corr_count = (
                    abs(correlation_matrix) >= high_corr_threshold
                ).sum().sum() - len(correlation_matrix)

                if high_corr_count > len(correlation_matrix) * 0.3:
                    recommendations.append(
                        "存在较多高相关性因子，建议进行因子去重或正交化处理"
                    )

            # 基于数据质量的建议
            if len(recommendations) == 0:
                recommendations.append("因子质量良好，可以进行下一步的策略构建")

            return recommendations

        except Exception as e:
            logger.error(f"生成建议时出错: {e}")
            return ["分析完成，请检查具体结果进行后续决策"]

    def _generate_batch_analysis_report(
        self, analysis_results: Dict[str, Dict]
    ) -> Dict:
        """生成批量分析报告"""
        try:
            report = {
                "batch_analysis_timestamp": datetime.now().isoformat(),
                "total_symbols": len(analysis_results),
                "successful_analyses": len(analysis_results),
                "summary_statistics": {},
                "cross_symbol_insights": {},
            }

            # 统计汇总
            total_factors = []
            effective_factors_counts = []

            for symbol, result in analysis_results.items():
                if "analysis_summary" in result:
                    summary = result["analysis_summary"]
                    total_factors.append(summary.get("total_factors", 0))

                    ic_summary = summary.get("ic_analysis_summary", {})
                    effective_factors_counts.append(
                        ic_summary.get("total_effective_factors", 0)
                    )

            if total_factors:
                report["summary_statistics"] = {
                    "avg_total_factors": np.mean(total_factors),
                    "avg_effective_factors": np.mean(effective_factors_counts),
                    "max_effective_factors": max(effective_factors_counts),
                    "min_effective_factors": min(effective_factors_counts),
                }

            return report

        except Exception as e:
            logger.error(f"生成批量分析报告时出错: {e}")
            return {}

    async def _save_analysis_results(self, symbol: str, analysis_result: Dict):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"analysis_results/{symbol}_analysis_{timestamp}.json"

            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # 转换DataFrame为JSON可序列化格式
            serializable_result = self._make_json_serializable(analysis_result)

            # 保存JSON文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            logger.info(f"分析结果已保存到: {file_path}")

        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")

    async def _save_batch_results(self, batch_report: Dict):
        """保存批量分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"analysis_results/batch_analysis_{timestamp}.json"

            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # 保存JSON文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(batch_report, f, ensure_ascii=False, indent=2)

            logger.info(f"批量分析报告已保存到: {file_path}")

        except Exception as e:
            logger.error(f"保存批量分析结果时出错: {e}")

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


# 创建全局多因子分析器实例
multi_factor_analyzer = MultiFactorAnalyzer()

__all__ = ["MultiFactorAnalyzer", "multi_factor_analyzer"]
