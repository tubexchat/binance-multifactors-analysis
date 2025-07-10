#!/usr/bin/env python3
"""
加密货币多因子分析平台

主程序入口，提供完整的多因子分析功能
"""

import asyncio
import argparse
import logging
import json
from typing import List, Optional
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import config
from analysis.multi_factor_analyzer import multi_factor_analyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multifactor_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MultiFactorAnalysisApp:
    """多因子分析应用程序"""
    
    def __init__(self):
        self.analyzer = multi_factor_analyzer
        
    async def initialize(self):
        """初始化应用程序"""
        try:
            logger.info("正在初始化多因子分析平台...")
            await self.analyzer.initialize()
            logger.info("多因子分析平台初始化完成")
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.analyzer.cleanup()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")
    
    async def analyze_symbol(self, symbol: str, limit: int = None, save_results: bool = True):
        """
        分析单个交易对
        
        Args:
            symbol: 交易对符号（例如：BTCUSDT）
            limit: 数据条数限制
            save_results: 是否保存结果
        """
        try:
            logger.info(f"开始分析交易对: {symbol}")
            
            result = await self.analyzer.analyze_single_symbol(
                symbol=symbol,
                limit=limit,
                save_results=save_results
            )
            
            if result:
                self._print_analysis_summary(result)
                return result
            else:
                logger.warning(f"分析 {symbol} 失败或无数据")
                return None
                
        except Exception as e:
            logger.error(f"分析 {symbol} 时出错: {e}")
            return None
    
    async def analyze_batch(
        self, 
        symbols: List[str], 
        limit: int = None, 
        max_concurrent: int = 3,
        save_results: bool = True
    ):
        """
        批量分析多个交易对
        
        Args:
            symbols: 交易对符号列表
            limit: 数据条数限制
            max_concurrent: 最大并发数
            save_results: 是否保存结果
        """
        try:
            logger.info(f"开始批量分析 {len(symbols)} 个交易对")
            
            results = await self.analyzer.analyze_multiple_symbols(
                symbols=symbols,
                limit=limit,
                max_concurrent=max_concurrent,
                save_results=save_results
            )
            
            if results:
                self._print_batch_summary(results)
                return results
            else:
                logger.warning("批量分析无结果")
                return {}
                
        except Exception as e:
            logger.error(f"批量分析时出错: {e}")
            return {}
    
    async def analyze_major_symbols(self, limit: int = None, save_results: bool = True):
        """分析主要交易对"""
        major_symbols = config.analysis.MAJOR_SYMBOLS
        logger.info(f"分析主要交易对: {major_symbols}")
        
        return await self.analyze_batch(
            symbols=major_symbols,
            limit=limit,
            save_results=save_results
        )
    
    def _print_analysis_summary(self, result: dict):
        """打印单个分析结果摘要"""
        try:
            symbol = result.get('symbol', 'Unknown')
            print(f"\n{'='*60}")
            print(f"多因子分析报告 - {symbol}")
            print(f"{'='*60}")
            
            # 基本信息
            data_period = result.get('data_period', {})
            print(f"分析时间: {result.get('analysis_timestamp', 'Unknown')}")
            print(f"数据周期: {data_period.get('start_time', 'Unknown')} 到 {data_period.get('end_time', 'Unknown')}")
            print(f"数据点数: {data_period.get('total_periods', 0)}")
            
            # 分析摘要
            summary = result.get('analysis_summary', {})
            if summary:
                print(f"\n因子数量: {summary.get('total_factors', 0)}")
                
                # 数据质量
                data_quality = summary.get('data_quality', {})
                if data_quality:
                    print(f"数据完整性: {data_quality.get('completeness', 0):.2f}%")
                    print(f"数据质量评级: {data_quality.get('rating', 'Unknown')}")
                
                # IC分析摘要
                ic_summary = summary.get('ic_analysis_summary', {})
                if ic_summary:
                    print(f"\n有效因子数量: {ic_summary.get('total_effective_factors', 0)}")
                    print(f"IC分析质量: {ic_summary.get('overall_ic_quality', 'Unknown')}")
                    
                    # 最佳表现因子
                    best_performers = ic_summary.get('best_performers', [])
                    if best_performers:
                        print(f"\n前5个最佳因子:")
                        for i, factor in enumerate(best_performers[:5], 1):
                            print(f"  {i}. {factor['factor']} (收益期{factor['return_period']}, IC={factor['abs_ic']:.4f})")
                
                # 相关性分析
                corr_summary = summary.get('correlation_analysis', {})
                if corr_summary:
                    print(f"\n因子相关性分析:")
                    print(f"  高相关性因子对: {corr_summary.get('high_correlation_pairs', 0)}")
                    print(f"  平均相关性: {corr_summary.get('avg_correlation', 0):.4f}")
                
                # 建议
                recommendations = summary.get('recommendations', [])
                if recommendations:
                    print(f"\n分析建议:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"打印分析摘要时出错: {e}")
    
    def _print_batch_summary(self, results: dict):
        """打印批量分析结果摘要"""
        try:
            print(f"\n{'='*80}")
            print(f"批量多因子分析报告")
            print(f"{'='*80}")
            
            print(f"成功分析的交易对数量: {len(results)}")
            print(f"分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 统计信息
            total_factors_list = []
            effective_factors_list = []
            ic_quality_counts = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
            
            print(f"\n各交易对分析结果:")
            print(f"{'交易对':<12} {'因子数':<8} {'有效因子':<10} {'IC质量':<12} {'数据质量'}")
            print(f"{'-'*60}")
            
            for symbol, result in results.items():
                summary = result.get('analysis_summary', {})
                total_factors = summary.get('total_factors', 0)
                
                ic_summary = summary.get('ic_analysis_summary', {})
                effective_factors = ic_summary.get('total_effective_factors', 0)
                ic_quality = ic_summary.get('overall_ic_quality', 'Unknown')
                
                data_quality = summary.get('data_quality', {})
                data_rating = data_quality.get('rating', 'Unknown')
                
                print(f"{symbol:<12} {total_factors:<8} {effective_factors:<10} {ic_quality:<12} {data_rating}")
                
                total_factors_list.append(total_factors)
                effective_factors_list.append(effective_factors)
                if ic_quality in ic_quality_counts:
                    ic_quality_counts[ic_quality] += 1
            
            # 汇总统计
            if total_factors_list:
                print(f"\n汇总统计:")
                print(f"  平均因子数量: {sum(total_factors_list) / len(total_factors_list):.1f}")
                print(f"  平均有效因子: {sum(effective_factors_list) / len(effective_factors_list):.1f}")
                print(f"  IC质量分布: {dict(ic_quality_counts)}")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"打印批量分析摘要时出错: {e}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币多因子分析平台')
    parser.add_argument('--symbol', '-s', type=str, help='分析单个交易对 (例如: BTCUSDT)')
    parser.add_argument('--symbols', '-m', nargs='+', help='分析多个交易对')
    parser.add_argument('--major', '-M', action='store_true', help='分析主要交易对')
    parser.add_argument('--limit', '-l', type=int, help='数据条数限制')
    parser.add_argument('--concurrent', '-c', type=int, default=3, help='最大并发数')
    parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    parser.add_argument('--config-test', action='store_true', help='测试配置')
    
    args = parser.parse_args()
    
    # 测试配置
    if args.config_test:
        print("配置测试:")
        print(f"数据源: {config.data.BINANCE_BASE_URL}")
        print(f"标准间隔: {config.data.STANDARD_INTERVAL}")
        print(f"主要交易对: {config.analysis.MAJOR_SYMBOLS}")
        print(f"输出目录: {config.analysis.OUTPUT_DIR}")
        return
    
    app = MultiFactorAnalysisApp()
    
    try:
        # 初始化应用
        await app.initialize()
        
        save_results = not args.no_save
        
        # 根据参数执行不同的分析
        if args.symbol:
            # 分析单个交易对
            await app.analyze_symbol(args.symbol, args.limit, save_results)
            
        elif args.symbols:
            # 分析指定的多个交易对
            await app.analyze_batch(args.symbols, args.limit, args.concurrent, save_results)
            
        elif args.major:
            # 分析主要交易对
            await app.analyze_major_symbols(args.limit, save_results)
            
        else:
            # 默认分析主要交易对
            print("未指定分析目标，将分析主要交易对...")
            await app.analyze_major_symbols(args.limit, save_results)
    
    except KeyboardInterrupt:
        print("\n用户中断分析")
        logger.info("用户中断分析")
    
    except Exception as e:
        print(f"分析过程中出错: {e}")
        logger.error(f"分析过程中出错: {e}")
    
    finally:
        # 清理资源
        await app.cleanup()

def run_example():
    """运行示例分析"""
    print("运行多因子分析示例...")
    print("分析交易对: BTCUSDT")
    
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        app = MultiFactorAnalysisApp()
        loop.run_until_complete(app.initialize())
        result = loop.run_until_complete(app.analyze_symbol('BTCUSDT', limit=500))
        loop.run_until_complete(app.cleanup())
        return result
    finally:
        loop.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"程序启动失败: {e}")
        logger.error(f"程序启动失败: {e}")
        sys.exit(1) 