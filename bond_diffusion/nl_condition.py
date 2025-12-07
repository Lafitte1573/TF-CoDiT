import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from einops import rearrange, repeat
import json


class NaturalLanguageCondition:
    """自然语言条件描述生成器"""

    def __init__(self, config):
        self.config = config
        # 初始化描述模板
        self.templates = self._init_templates()

    def generate_condition_text(self, market_data, timeseries_metadata):
        """
        根据市场数据和时序元数据生成自然语言描述
        Args:
            market_data: dict, 包含利率、通胀等宏观数据
            timeseries_metadata: dict, 包含时序特征信息
        Returns:
            condition_text: str, 自然语言描述
        """
        # 1. 宏观状态描述
        macro_desc = self._describe_macro_state(market_data)

        # 2. 市场情绪描述
        sentiment_desc = self._describe_market_sentiment(market_data)

        # 3. 时序特征描述
        ts_desc = self._describe_timeseries_features(timeseries_metadata)

        # 4. 技术指标描述
        technical_desc = self._describe_technical_indicators(timeseries_metadata)

        # 5. 组合成完整的条件描述
        condition_text = f"""当前国债期货市场状态描述：
{macro_desc}
{sentiment_desc}
{ts_desc}
{technical_desc}

请基于以上市场状态生成相应的国债期货时序数据。
"""

        return condition_text.strip()

    def _describe_macro_state(self, market_data):
        """描述宏观状态"""
        rate = market_data.get('policy_rate', 0.0)
        inflation = market_data.get('inflation', 0.0)
        gdp_growth = market_data.get('gdp_growth', 0.0)

        desc = f"宏观环境：政策利率{rate:.2f}%，通胀率{inflation:.2f}%，GDP增长{gdp_growth:.2f}%。"

        # 添加定性判断
        if rate > 4.0:
            desc += "货币政策偏紧，"
        elif rate < 2.0:
            desc += "货币政策宽松，"
        else:
            desc += "货币政策中性，"

        if inflation > 3.0:
            desc += "通胀压力较大。"
        elif inflation < 1.0:
            desc += "通胀水平较低。"
        else:
            desc += "通胀处于温和水平。"

        return desc

    def _describe_market_sentiment(self, market_data):
        """描述市场情绪"""
        vix = market_data.get('vix', 20.0)
        bid_ask_spread = market_data.get('bid_ask_spread', 0.01)

        desc = f"市场情绪：波动率指数(VIX)为{vix:.1f}，买卖价差{bid_ask_spread * 100:.2f}bp。"

        if vix > 25:
            desc += "市场情绪紧张，避险需求较高。"
        elif vix < 15:
            desc += "市场情绪平稳，风险偏好较强。"
        else:
            desc += "市场情绪中性。"

        return desc

    def _describe_timeseries_features(self, ts_metadata):
        """描述时序特征"""
        trend = ts_metadata.get('trend', 'neutral')
        volatility = ts_metadata.get('volatility', 'medium')
        seasonality = ts_metadata.get('seasonality', 'weak')

        desc = f"时序特征：趋势方向{trend}，波动性{volatility}，季节性{seasonality}。"

        return desc

    def _describe_technical_indicators(self, ts_metadata):
        """描述技术指标"""
        rsi = ts_metadata.get('rsi', 50.0)
        macd_signal = ts_metadata.get('macd_signal', 'neutral')

        desc = f"技术指标：RSI指数{rsi:.1f}，MACD信号{macd_signal}。"

        if rsi > 70:
            desc += "市场可能处于超买状态。"
        elif rsi < 30:
            desc += "市场可能处于超卖状态。"
        else:
            desc += "市场处于正常状态。"

        return desc

    def _init_templates(self):
        """初始化描述模板"""
        return {
            'macro': [
                "在{policy_rate}%的政策利率和{inflation}%的通胀率环境下，",
                "经济增长率为{gdp_growth}%，",
                "货币供应量增长{money_supply}%，"
            ],
            'market': [
                "市场波动率为{volatility}%，",
                "流动性状况为{liquidity}，",
                "投资者情绪偏向{sentiment}，"
            ],
            'technical': [
                "技术指标显示{indicator_signal}信号，",
                "价格处于{support_resistance}位置，",
                "成交量呈现{volume_pattern}模式，"
            ]
        }
