"""
IC分析模块

计算信息系数(Information Coefficient)，分析因子与未来收益率的相关性
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from core.config import config

logger = logging.getLogger(__name__)

class ICAnalysis:
    """IC分析器"""
    
    def __init__(self):
        self.ic_config = config.ic_analysis
        
    def calculate_forward_returns(
        self, 
        price_data: pd.DataFrame, 
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        计算前瞻收益率
        
        Args:
            price_data: 包含价格数据的DataFrame，必须有'close'列
            periods: 收益率计算周期列表
            
        Returns:
            包含各期前瞻收益率的DataFrame
        """
        if periods is None:
            periods = self.ic_config.RETURN_PERIODS
        
        returns_df = price_data[['timestamp']].copy()
        close_prices = price_data['close']
        
        for period in periods:
            # 计算前瞻收益率：(future_price - current_price) / current_price
            future_prices = close_prices.shift(-period)
            returns_df[f'forward_return_{period}'] = (future_prices - close_prices) / close_prices
        
        return returns_df
    
    def calculate_ic(
        self, 
        factor_data: pd.DataFrame, 
        return_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        计算IC值
        
        Args:
            factor_data: 因子数据DataFrame
            return_data: 收益率数据DataFrame  
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            包含IC值的DataFrame
        """
        try:
            # 合并数据
            merged_data = pd.merge(factor_data, return_data, on='timestamp', how='inner')
            
            # 获取因子列（排除timestamp）
            factor_columns = [col for col in factor_data.columns if col != 'timestamp']
            return_columns = [col for col in return_data.columns if col != 'timestamp']
            
            ic_results = []
            
            for return_col in return_columns:
                ic_row = {'return_period': return_col}
                
                for factor_col in factor_columns:
                    # 移除NaN值
                    valid_data = merged_data[[factor_col, return_col]].dropna()
                    
                    if len(valid_data) < 10:  # 最少需要10个有效数据点
                        ic_row[factor_col] = np.nan
                        continue
                    
                    try:
                        if method == 'pearson':
                            corr, p_value = stats.pearsonr(valid_data[factor_col], valid_data[return_col])
                        elif method == 'spearman':
                            corr, p_value = stats.spearmanr(valid_data[factor_col], valid_data[return_col])
                        elif method == 'kendall':
                            corr, p_value = stats.kendalltau(valid_data[factor_col], valid_data[return_col])
                        else:
                            raise ValueError(f"不支持的相关性计算方法: {method}")
                        
                        ic_row[factor_col] = corr
                        ic_row[f'{factor_col}_pvalue'] = p_value
                        
                    except Exception as e:
                        logger.warning(f"计算 {factor_col} 与 {return_col} 的IC时出错: {e}")
                        ic_row[factor_col] = np.nan
                        ic_row[f'{factor_col}_pvalue'] = np.nan
                
                ic_results.append(ic_row)
            
            return pd.DataFrame(ic_results)
            
        except Exception as e:
            logger.error(f"计算IC时出错: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_ic(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        window: int = None,
        method: str = 'pearson'
    ) -> Dict[str, pd.DataFrame]:
        """
        计算滚动IC
        
        Args:
            factor_data: 因子数据DataFrame
            return_data: 收益率数据DataFrame
            window: 滚动窗口大小
            method: 相关性计算方法
            
        Returns:
            包含各因子滚动IC的字典
        """
        if window is None:
            window = self.ic_config.ROLLING_IC_WINDOW
        
        try:
            # 合并数据
            merged_data = pd.merge(factor_data, return_data, on='timestamp', how='inner')
            merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
            
            factor_columns = [col for col in factor_data.columns if col != 'timestamp']
            return_columns = [col for col in return_data.columns if col != 'timestamp']
            
            rolling_ic_results = {}
            
            for return_col in return_columns:
                rolling_ic_df = merged_data[['timestamp']].copy()
                
                for factor_col in factor_columns:
                    ic_values = []
                    
                    for i in range(len(merged_data)):
                        start_idx = max(0, i - window + 1)
                        end_idx = i + 1
                        
                        if end_idx - start_idx < window:
                            ic_values.append(np.nan)
                            continue
                        
                        window_data = merged_data.iloc[start_idx:end_idx]
                        valid_data = window_data[[factor_col, return_col]].dropna()
                        
                        if len(valid_data) < max(10, window // 2):
                            ic_values.append(np.nan)
                            continue
                        
                        try:
                            if method == 'pearson':
                                corr, _ = stats.pearsonr(valid_data[factor_col], valid_data[return_col])
                            elif method == 'spearman':
                                corr, _ = stats.spearmanr(valid_data[factor_col], valid_data[return_col])
                            else:
                                corr, _ = stats.kendalltau(valid_data[factor_col], valid_data[return_col])
                            
                            ic_values.append(corr)
                            
                        except Exception:
                            ic_values.append(np.nan)
                    
                    rolling_ic_df[factor_col] = ic_values
                
                rolling_ic_results[return_col] = rolling_ic_df
            
            return rolling_ic_results
            
        except Exception as e:
            logger.error(f"计算滚动IC时出错: {e}")
            return {}
    
    def calculate_ic_statistics(self, ic_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算IC统计指标
        
        Args:
            ic_data: IC数据DataFrame
            
        Returns:
            包含IC统计指标的DataFrame
        """
        try:
            factor_columns = [col for col in ic_data.columns 
                            if col not in ['return_period', 'timestamp'] and not col.endswith('_pvalue')]
            
            stats_results = []
            
            for _, row in ic_data.iterrows():
                return_period = row.get('return_period', 'unknown')
                stats_row = {'return_period': return_period}
                
                for factor_col in factor_columns:
                    ic_values = row[factor_col]
                    
                    if pd.isna(ic_values):
                        stats_row[f'{factor_col}_mean'] = np.nan
                        stats_row[f'{factor_col}_std'] = np.nan
                        stats_row[f'{factor_col}_ir'] = np.nan
                        stats_row[f'{factor_col}_abs_mean'] = np.nan
                        continue
                    
                    # 如果是单个值，转换为列表
                    if not isinstance(ic_values, (list, np.ndarray, pd.Series)):
                        ic_values = [ic_values]
                    
                    ic_array = np.array(ic_values)
                    ic_array = ic_array[~np.isnan(ic_array)]  # 移除NaN
                    
                    if len(ic_array) == 0:
                        stats_row[f'{factor_col}_mean'] = np.nan
                        stats_row[f'{factor_col}_std'] = np.nan
                        stats_row[f'{factor_col}_ir'] = np.nan
                        stats_row[f'{factor_col}_abs_mean'] = np.nan
                        continue
                    
                    # 计算统计指标
                    ic_mean = np.mean(ic_array)
                    ic_std = np.std(ic_array)
                    ic_ir = ic_mean / ic_std if ic_std != 0 else 0  # Information Ratio
                    ic_abs_mean = np.mean(np.abs(ic_array))
                    
                    stats_row[f'{factor_col}_mean'] = ic_mean
                    stats_row[f'{factor_col}_std'] = ic_std
                    stats_row[f'{factor_col}_ir'] = ic_ir
                    stats_row[f'{factor_col}_abs_mean'] = ic_abs_mean
                
                stats_results.append(stats_row)
            
            return pd.DataFrame(stats_results)
            
        except Exception as e:
            logger.error(f"计算IC统计指标时出错: {e}")
            return pd.DataFrame()
    
    def rank_factors_by_ic(
        self, 
        ic_stats: pd.DataFrame, 
        ranking_method: str = 'abs_mean'
    ) -> pd.DataFrame:
        """
        根据IC指标对因子进行排名
        
        Args:
            ic_stats: IC统计指标DataFrame
            ranking_method: 排名方法 ('abs_mean', 'mean', 'ir')
            
        Returns:
            因子排名DataFrame
        """
        try:
            ranking_results = []
            
            for _, row in ic_stats.iterrows():
                return_period = row.get('return_period', 'unknown')
                
                # 提取该收益率周期下所有因子的排名指标
                factor_scores = {}
                
                for col in row.index:
                    if col.endswith(f'_{ranking_method}') and not pd.isna(row[col]):
                        factor_name = col.replace(f'_{ranking_method}', '')
                        factor_scores[factor_name] = abs(row[col]) if ranking_method != 'abs_mean' else row[col]
                
                # 排序
                sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (factor_name, score) in enumerate(sorted_factors, 1):
                    ranking_results.append({
                        'return_period': return_period,
                        'factor_name': factor_name,
                        'rank': rank,
                        'score': score,
                        'ranking_method': ranking_method
                    })
            
            return pd.DataFrame(ranking_results)
            
        except Exception as e:
            logger.error(f"因子排名时出错: {e}")
            return pd.DataFrame()
    
    def filter_effective_factors(
        self, 
        ic_stats: pd.DataFrame, 
        threshold: float = None
    ) -> List[str]:
        """
        筛选有效因子
        
        Args:
            ic_stats: IC统计指标DataFrame
            threshold: IC阈值
            
        Returns:
            有效因子名称列表
        """
        if threshold is None:
            threshold = self.ic_config.FACTOR_SELECTION_THRESHOLD
        
        try:
            effective_factors = set()
            
            for _, row in ic_stats.iterrows():
                for col in row.index:
                    if col.endswith('_abs_mean') and not pd.isna(row[col]):
                        if abs(row[col]) >= threshold:
                            factor_name = col.replace('_abs_mean', '')
                            effective_factors.add(factor_name)
            
            logger.info(f"筛选出 {len(effective_factors)} 个有效因子（阈值: {threshold}）")
            return list(effective_factors)
            
        except Exception as e:
            logger.error(f"筛选有效因子时出错: {e}")
            return []
    
    def analyze_factor_stability(
        self, 
        rolling_ic_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        分析因子稳定性
        
        Args:
            rolling_ic_results: 滚动IC结果字典
            
        Returns:
            因子稳定性分析DataFrame
        """
        try:
            stability_results = []
            
            for return_period, rolling_ic_df in rolling_ic_results.items():
                factor_columns = [col for col in rolling_ic_df.columns if col != 'timestamp']
                
                for factor_col in factor_columns:
                    ic_series = rolling_ic_df[factor_col].dropna()
                    
                    if len(ic_series) < 10:
                        continue
                    
                    # 计算稳定性指标
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    ic_positive_ratio = (ic_series > 0).mean()
                    ic_significant_ratio = (abs(ic_series) > self.ic_config.IC_THRESHOLD).mean()
                    
                    # 计算最大回撤（IC的角度）
                    cumulative_ic = ic_series.cumsum()
                    running_max = cumulative_ic.expanding().max()
                    drawdown = cumulative_ic - running_max
                    max_drawdown = drawdown.min()
                    
                    stability_results.append({
                        'return_period': return_period,
                        'factor_name': factor_col,
                        'ic_mean': ic_mean,
                        'ic_std': ic_std,
                        'ic_ir': ic_mean / ic_std if ic_std != 0 else 0,
                        'positive_ratio': ic_positive_ratio,
                        'significant_ratio': ic_significant_ratio,
                        'max_drawdown': max_drawdown,
                        'stability_score': ic_significant_ratio * (1 - abs(max_drawdown) / abs(cumulative_ic.iloc[-1]) if cumulative_ic.iloc[-1] != 0 else 0)
                    })
            
            return pd.DataFrame(stability_results)
            
        except Exception as e:
            logger.error(f"分析因子稳定性时出错: {e}")
            return pd.DataFrame()
    
    def generate_ic_report(
        self,
        symbol: str,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        生成完整的IC分析报告
        
        Args:
            symbol: 交易对符号
            factor_data: 因子数据
            price_data: 价格数据
            
        Returns:
            包含各种IC分析结果的字典
        """
        try:
            logger.info(f"开始生成 {symbol} 的IC分析报告")
            
            # 1. 计算前瞻收益率
            return_data = self.calculate_forward_returns(price_data)
            
            # 2. 计算IC
            ic_results = self.calculate_ic(factor_data, return_data)
            
            # 3. 计算滚动IC
            rolling_ic_results = self.calculate_rolling_ic(factor_data, return_data)
            
            # 4. 计算IC统计指标
            ic_stats = self.calculate_ic_statistics(ic_results)
            
            # 5. 因子排名
            factor_ranking = self.rank_factors_by_ic(ic_stats)
            
            # 6. 筛选有效因子
            effective_factors = self.filter_effective_factors(ic_stats)
            
            # 7. 稳定性分析
            stability_analysis = self.analyze_factor_stability(rolling_ic_results)
            
            report = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'return_data': return_data,
                'ic_results': ic_results,
                'rolling_ic_results': rolling_ic_results,
                'ic_statistics': ic_stats,
                'factor_ranking': factor_ranking,
                'effective_factors': effective_factors,
                'stability_analysis': stability_analysis
            }
            
            logger.info(f"成功生成 {symbol} 的IC分析报告")
            return report
            
        except Exception as e:
            logger.error(f"生成IC分析报告时出错: {e}")
            return {}

# 创建全局IC分析器实例
ic_analyzer = ICAnalysis()

__all__ = ['ICAnalysis', 'ic_analyzer'] 