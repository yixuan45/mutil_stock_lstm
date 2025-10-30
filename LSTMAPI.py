import os

import pandas as pd
import talib as ta
import torch
from sklearn.preprocessing import StandardScaler

from config import config
from models.lstm_model import LSTMStockPredictor



class LSTMPredictorAPI(object):
    def __init__(self, model_path=None):
        """初始化预测接口"""
        self.config = config
        self.data = None
        self.scaler_target = None
        self.window_detail = None
        self.X = None
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path):
        """加载训练好的模型"""
        if not model_path:
            model_path = os.path.join(config['save_dir'], f"{config['time_interval']}_best_model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 初始化模型
        model = LSTMStockPredictor()
        checkpoint = torch.load(model_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config['device'])
        model.eval()
        return model

    def _load_data(self, data):
        """加载数据"""
        df=data.copy()
        df['c_pred'] = df['c'].shift(-1)
        self.data = df

    def _preprocess_data(self):
        if self.data is None:
            raise ValueError("请先调用load_data()加载数据")
        # 构造特征
        self._create_feature()
        # 选择特征列（排除目标列）
        target_column = 'c_pred'  # 目标价格列
        feature_columns = [col for col in self.data.columns if col != target_column]  # 特征列

        # 数据标准化
        self._normalize_data(feature_columns, target_column)

        # 创建序列数据
        self._create_sequences(feature_columns)

    def _create_feature(self):
        """为数据构造相关特征"""
        cur_data = self.data.copy()

        # 构造rsi数据和rsi_ma数据
        self.data['rsi'] = ta.RSI(cur_data['c'], timeperiod=self.config['rsi_len'])
        self.data['rsi_mean'] = self.data['rsi'].rolling(window=self.config['rsi_mean_len']).mean()

        # 构造macd数据
        self.data['macd'], self.data['macd_signal'], _ = ta.MACD(cur_data['c'],
                                                                 fastperiod=self.config['macd_fastperiod'],
                                                                 slowperiod=self.config['macd_slowperiod'],
                                                                 signalperiod=self.config['macd_signalperiod'])

        self.data['atr'] = self._calculate_atr(cur_data, timeperiod=self.config['atr_period'])

        # 构造高低点斜率
        self.data['high_slope'] = (cur_data['h'] - cur_data['h'].shift(self.config['high_slope_len'])) / self.config[
            'high_slope_len']

        # 构造量价关系
        condition = (
            # 情况1：收盘价 > 开盘价（阳线）且成交量 > 前一日成交量（放量）
                (cur_data['c'] > cur_data['o']) & (cur_data['v'] > cur_data['v'].shift(1))
                |  # 或者
                # 情况2：收盘价 < 开盘价（阴线）且成交量 < 前一日成交量（缩量）
                (cur_data['c'] < cur_data['o']) & (cur_data['v'] < cur_data['v'].shift(1))
        )
        self.data['vol_price_sync'] = condition.astype(int)

        # 构造布林带数据
        self.data['bollinger_band_pos'] = self._calculate_bollinger_band_pos(cur_data,
                                                                             bb_timeperiod=self.config['bb_timeperiod'],
                                                                             bb_dev=self.config['bb_dev'])

        # 计算背离程度
        self.data['rsi_divergence'] = self._calculate_rsi_divergence(cur_data=self.data,
                                                                     rsi_divergence_length=self.config[
                                                                         'rsi_divergence_length'])

    def _create_sequences(self, feature_columns):
        self.X = self.data[feature_columns].values

    @staticmethod
    def _calculate_atr(cur_data, timeperiod=14):
        # 构造波动率数据
        tr1 = cur_data['h'] - cur_data['l']
        tr2 = abs(cur_data['h'] - cur_data['c'].shift(1))
        tr3 = abs(cur_data['c'].shift(1) - cur_data['l'])
        cur_data['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        cur_data['atr'] = cur_data['tr'].rolling(window=timeperiod).mean()
        return cur_data['atr']

    @staticmethod
    def _calculate_rsi_divergence(cur_data, rsi_divergence_length=5):
        """
            计算RSI背离程度
            :param divergence_window: 对比背离的周期（默认5日，即看5日内价格与RSI的变化）
            :return: 包含RSI背离程度的Series
        """
        # 价格涨跌幅：今日收盘价 - N日前收盘价
        cur_data['price_change'] = cur_data['c'] - cur_data['c'].shift(rsi_divergence_length)
        # RSI涨跌幅：今日RSI - N日前RSI
        cur_data['rsi_change'] = cur_data['rsi'] - cur_data['rsi'].shift(rsi_divergence_length)

        cur_data['rsi_divergence'] = cur_data['price_change'] - cur_data['rsi_change']

        return cur_data['rsi_divergence']

    @staticmethod
    def _calculate_bollinger_band_pos(cur_data, bb_timeperiod=20, bb_dev=2):
        """
        计算布林带位置
        :param cur_data: 包含收盘价（c）的DataFrame
        :param window: 布林带中轨周期（默认20日，常用值）
        :param window_dev: 标准差倍数（默认2，控制布林带宽度）
        :return: 包含布林带位置的Series
        """
        # 1. 计算中轨（window周期的收盘价移动平均）
        cur_data['bb_middle'] = cur_data['c'].rolling(window=bb_timeperiod).mean()

        # 2.计算收盘价与中轨的偏差（用于求标准差）
        cur_data['price_deviation'] = cur_data['c'] - cur_data['bb_middle']

        # 3.计算窗口内的标准差（衡量价格波动幅度）
        cur_data['rolling_std'] = cur_data['price_deviation'].rolling(window=bb_timeperiod).std(ddof=0)

        # 4.计算上轨和下轨
        cur_data['bb_upper'] = cur_data['bb_middle'] + bb_dev * cur_data['rolling_std']
        cur_data['bb_lower'] = cur_data['bb_middle'] - bb_dev * cur_data['rolling_std']

        # 5.计算布林带位置
        denominator = cur_data['bb_upper'] - cur_data['bb_lower']
        cur_data['bollinger_band_pos'] = (cur_data['c'] - cur_data['bb_lower']) / denominator

        return cur_data['bollinger_band_pos']

    def _normalize_data(self, feature_columns, target_column):
        window_size = self.config['sequence_length']
        scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()  # 创建特征的标准化和目标的标准化
        self.data = self.data[-window_size:]  # 把这个截取到滑动窗口的大小

        # 训练特征的标准化
        scaler_features.fit(self.data[feature_columns].values)
        self.data.loc[:, feature_columns] = scaler_features.fit_transform(self.data[feature_columns].values)

        # 训练目标的标准化
        self.scaler_target.fit(self.data[target_column].iloc[:-1].values.reshape(-1, 1))

    def main(self, data):
        """主函数"""
        # 加载数据
        self._load_data(data)

        # 数据处理过程
        self._preprocess_data()

        # 计算结果
        x_tensor=torch.tensor(self.X,dtype=torch.float32).to(self.config['device'])

        y=self.model(x_tensor).squeeze(0).detach().cpu().numpy()

        y=self.scaler_target.inverse_transform(y)
        return y
