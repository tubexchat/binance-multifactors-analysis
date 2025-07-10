"""
数据管理器 - 负责统一的15分钟K线数据获取、处理和缓存
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time

from .config import config

logger = logging.getLogger(__name__)


class DataManager:
    """统一的数据管理器"""

    def __init__(self):
        self.session = None
        self.endpoints = config.get_api_endpoints()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.data.REQUEST_TIMEOUT)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def _fetch_with_retry(self, url: str, params: dict = None) -> Optional[dict]:
        """带重试机制的数据获取"""
        if not self.session:
            raise RuntimeError("DataManager must be used as async context manager")

        for attempt in range(config.data.MAX_RETRIES):
            try:
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                logger.warning(
                    f"请求超时 (尝试 {attempt+1}/{config.data.MAX_RETRIES}): {url}"
                )
                if attempt < config.data.MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)  # 指数退避

            except aiohttp.ClientError as e:
                logger.warning(
                    f"请求错误 (尝试 {attempt+1}/{config.data.MAX_RETRIES}): {e}"
                )
                if attempt < config.data.MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.error(f"未知错误: {e}")
                break

        return None

    async def get_kline_data(
        self,
        symbol: str,
        limit: int = None,
        start_time: datetime = None,
        end_time: datetime = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        获取15分钟K线数据

        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            limit: 数据条数，默认使用配置中的历史长度
            start_time: 开始时间
            end_time: 结束时间
            use_cache: 是否使用缓存

        Returns:
            包含OHLCV数据的DataFrame，列名为：timestamp, open, high, low, close, volume, quote_volume
        """
        if limit is None:
            limit = config.data.HISTORY_LENGTH

        # 构建请求参数
        params = {
            "symbol": symbol,
            "interval": config.data.STANDARD_INTERVAL,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        # 获取数据
        data = await self._fetch_with_retry(self.endpoints["klines"], params)
        if not data:
            logger.error(f"获取 {symbol} K线数据失败")
            return None

        # 转换为DataFrame
        try:
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )

            # 数据类型转换
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 只保留需要的列
            df = df[
                ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
            ]

            logger.info(f"成功获取 {symbol} 的 {len(df)} 条K线数据")
            return df

        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取资金费率数据"""
        try:
            params = {"symbol": symbol}
            data = await self._fetch_with_retry(self.endpoints["funding_rate"], params)
            if not data:
                return None

            df = pd.DataFrame(data)
            if not df.empty:
                df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
                df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            return df
        except Exception as e:
            logger.error(f"获取资金费率失败: {e}")
            return None

    async def get_long_short_ratio(
        self, symbol: str, period: str = "15m", limit: int = 30
    ) -> Optional[pd.DataFrame]:
        """获取多空比数据"""
        try:
            params = {"symbol": symbol, "period": period, "limit": limit}
            data = await self._fetch_with_retry(
                self.endpoints["long_short_ratio"], params
            )
            if not data:
                return None

            df = pd.DataFrame(data)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["longShortRatio"] = pd.to_numeric(
                    df["longShortRatio"], errors="coerce"
                )
            return df
        except Exception as e:
            logger.error(f"获取多空比失败: {e}")
            return None

    async def get_open_interest(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取持仓量数据"""
        try:
            params = {"symbol": symbol}
            data = await self._fetch_with_retry(self.endpoints["open_interest"], params)
            if not data:
                return None

            # 单个数据点，转换为DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
                df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
                return df
            return None
        except Exception as e:
            logger.error(f"获取持仓量失败: {e}")
            return None

    async def get_premium_index(self, symbol: str = None) -> Optional[list]:
        """获取溢价指数数据"""
        try:
            # 如果指定了symbol，只获取该symbol的数据
            params = {"symbol": symbol} if symbol else {}
            data = await self._fetch_with_retry(self.endpoints["premium_index"], params)
            if not data:
                return None

            # 如果是单个symbol，返回包含单个dict的列表
            if isinstance(data, dict):
                return [data]
            # 如果是多个symbol，返回list
            elif isinstance(data, list):
                return data
            return None
        except Exception as e:
            logger.error(f"获取溢价指数失败: {e}")
            return None

    async def get_active_symbols(self) -> List[str]:
        """获取活跃交易对列表"""

        # 获取24小时行情数据
        data = await self._fetch_with_retry(self.endpoints["ticker_24hr"])
        if not data:
            return []

        try:
            # 过滤活跃交易对
            active_symbols = []
            for item in data:
                symbol = item.get("symbol", "")
                quote_volume = float(item.get("quoteVolume", 0))

                # 应用过滤条件
                if (
                    quote_volume >= config.data.MIN_QUOTE_VOLUME
                    and symbol.endswith("USDT")
                    and not any(
                        excluded in symbol for excluded in config.data.EXCLUDED_SYMBOLS
                    )
                ):
                    active_symbols.append(symbol)

            logger.info(f"发现 {len(active_symbols)} 个活跃交易对")
            return active_symbols

        except Exception as e:
            logger.error(f"处理活跃交易对数据失败: {e}")
            return []

    async def get_batch_kline_data(
        self, symbols: List[str], limit: int = None
    ) -> Dict[str, pd.DataFrame]:
        """批量获取K线数据"""
        if limit is None:
            limit = config.data.HISTORY_LENGTH

        tasks = []
        for symbol in symbols:
            task = self.get_kline_data(symbol, limit=limit)
            tasks.append((symbol, task))

        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"获取 {symbol} 数据失败: {result}")
                continue
            if result is not None:
                results[symbol] = result

        logger.info(f"成功获取 {len(results)}/{len(symbols)} 个交易对的数据")
        return results


# 创建全局数据管理器实例
data_manager = DataManager()

__all__ = ["DataManager", "data_manager"]
