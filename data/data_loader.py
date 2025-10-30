# -*- coding: utf-8 -*-

# import
import os
import torch
import json
import logging
import numpy as np
import talib as ta
import pandas as pd

# from import
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# own
from config import config

logger = logging.getLogger("data_loader")


class PriceDataset(Dataset):
    """价格预测数据集类"""

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class DataProcessor(object):
    """数据处理类，负责数据加载、预处理和数据集创建"""

    def __init__(self, data=None):
        self.config = config
        self.data = None
        self.X = None
        self.y = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = None
        self.scaler_target = None

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        for dir_path in [self.config['save_dir'], self.config['plot_dir'], os.path.dirname(self.config['log_file'])]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录：{dir_path}")

    def load_data(self):
        """加载原始数据"""
        try:
            logger.info(f"从{self.config['data_path']}加载数据")
            data = pd.read_csv(self.config['data_path'], index_col='t')
            data['c_pred'] = data['c'].shift(-1)
            self.data = data.dropna()
            logger.info(f"数据加载完成,形状:{self.data.shape}")

        except Exception as e:
            logger.error(f"数据加载失败:{str(e)}", exc_info=True)
            raise e

    def preprocess_data(self):
        """预处理数据，标准化和创建序列"""
        if self.data is None:
            raise ValueError("请先调用load_data()加载数据")

        logger.info("开始数据预处理...")

        # 构造特征
        self.create_feature()
        config['input_dim'] = len(self.data.columns) - 1  # 将构造的特征长度赋值给config配置项

        # 选择特征列（排除目标列）
        target_column = 'c_pred'  # 目标价格列
        feature_columns = [col for col in self.data.columns if col != target_column]  # 特征列

        # 先对数据进行一阶差分
        # self._time_series_differencer()

        # 数据标准化
        self._normalize_data(feature_columns, target_column)

        # 创建序列数据
        self._create_sequences(feature_columns, target_column)
        logger.info("数据预处理完成")

    def testprocess_data(self):
        """测试时候，处理数据，标准化和创建序列"""
        if self.data is None:
            raise ValueError("请先初始化对象，获取数据")

    def create_feature(self):
        """为数据构造相关特征"""
        cur_data = self.data.copy()

        logger.info(f"构造rsi和rsi_ma数据")
        # 构造rsi数据和rsi_ma数据
        self.data['rsi'] = ta.RSI(cur_data['c'], timeperiod=self.config['rsi_len'])
        self.data['rsi_mean'] = self.data['rsi'].rolling(window=self.config['rsi_mean_len']).mean()

        logger.info(f"构造macd数据")
        # 构造macd数据
        self.data['macd'], self.data['macd_signal'], _ = ta.MACD(cur_data['c'],
                                                                 fastperiod=self.config['macd_fastperiod'],
                                                                 slowperiod=self.config['macd_slowperiod'],
                                                                 signalperiod=self.config['macd_signalperiod'])
        logger.info(f"构造波动率atr数据")
        self.data['atr'] = self._calculate_atr(cur_data, timeperiod=self.config['atr_period'])

        logger.info(f"构造高低点斜率")
        # 构造高低点斜率
        self.data['high_slope'] = (cur_data['h'] - cur_data['h'].shift(self.config['high_slope_len'])) / self.config[
            'high_slope_len']

        logger.info(f"构造量价关系")
        # 构造量价关系
        condition = (
            # 情况1：收盘价 > 开盘价（阳线）且成交量 > 前一日成交量（放量）
                (cur_data['c'] > cur_data['o']) & (cur_data['v'] > cur_data['v'].shift(1))
                |  # 或者
                # 情况2：收盘价 < 开盘价（阴线）且成交量 < 前一日成交量（缩量）
                (cur_data['c'] < cur_data['o']) & (cur_data['v'] < cur_data['v'].shift(1))
        )
        self.data['vol_price_sync'] = condition.astype(int)

        logger.info(f"构造布林带数据")
        # 构造布林带数据
        self.data['bollinger_band_pos'] = self._calculate_bollinger_band_pos(cur_data,
                                                                             bb_timeperiod=self.config['bb_timeperiod'],
                                                                             bb_dev=self.config['bb_dev'])

        logger.info(f"计算背离程度")
        # 计算背离程度
        self.data['rsi_divergence'] = self._calculate_rsi_divergence(cur_data=self.data,
                                                                     rsi_divergence_length=self.config[
                                                                         'rsi_divergence_length'])
        self.data = self.data.dropna()

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

    def _time_series_differencer(self):
        """对有明显时序特征的数据进行差分处理"""
        logger.info("对有明显时序特征的数据进行差分进行处理")
        # 存储差分所需的起点值（用于还原）
        self.first_values = {}

        # 记录需要差分的特征
        diff_features = ['o', 'h', 'l', 'c', 'c_pred', 'v', 'qv']

        # 对当前的first_values进行初始化
        for feature in diff_features:
            self.first_values[feature] = self.data[feature].iloc[0]

        # 复制数据以避免修改原始数据
        diff_df = self.data.copy()

        # 对指定特征进行差分处理
        diff_df[diff_features] = self.data[diff_features].diff()

        # 移除第一行（因为差分后第一行为NaN）
        diff_df = diff_df.iloc[1:].copy()

        self.data[diff_features] = diff_df[diff_features]

        self.data.dropna(inplace=True)

        logger.info("对有明显时序特征的数据进行差分处理完成")

    def _normalize_data(self, feature_columns, target_column):
        """标准化特征和目标变量"""
        logger.info(f"使用{self.config['normalization']}滚动标准化所有特征和目标变量")

        window_size = self.config['sequence_length']  # 滑动标准化的窗口和输入长度相同
        norm_type = self.config['normalization']

        # 划分时序训练/验证/测试集的索引
        total_len = len(self.data)
        test_len = int(total_len * self.config['test_size'])
        val_len = int(total_len * self.config['val_size'])
        train_len = total_len - val_len - test_len

        # 时序索引：train→val→test（早期→中期→晚期）
        train_idx = range(train_len)
        val_idx = range(train_len, train_len + val_len)
        test_idx = range(train_len + val_len, total_len)

        # 保存最后一个滑动窗口的数据
        self.update_last_stats(data=self.data, window_size=window_size)

        # ---------------------- 特征滚动标准化 ----------------------
        self.scaler_features = {}
        for col in tqdm(feature_columns):
            # 对每个特征单独创建滚动标准化器
            scaler = RollingScaler(window_size=window_size, scaler_type=norm_type)
            all_data = self.data[col].values

            normalized_data = scaler.fit_transform(all_data)
            self.data.loc[:, col] = normalized_data
            self.scaler_features[col] = scaler  # 保存每个特征的标准化器

        # ---------------------- 目标列滚动标准化 ----------------------
        self.scaler_target = RollingScaler(window_size=window_size, scaler_type=norm_type)
        target_data = self.data[target_column].values
        normalized_target = self.scaler_target.fit_transform(target_data)
        self.data.loc[:, target_column] = normalized_target

        # 保存最后一个时间段的目标列标准化参数（用于后续推演）
        logger.info(f"训练集特征标准化完成")
        logger.info(f"验证集特征标准化完成")
        logger.info(f"测试集特征标准化完成")

        logger.info(f"训练集特征均值:{self.data.iloc[train_idx][feature_columns].mean().values}")
        logger.info(f"验证集特征均值:{self.data.iloc[val_idx][feature_columns].mean().values}")
        logger.info(f"测试集特征均值:{self.data.iloc[test_idx][feature_columns].mean().values}")

    @staticmethod
    def update_last_stats(data, window_size):
        """更新最后一个窗口的统计量（用于后续预测）"""
        start_idx = max(0, len(data) - window_size)
        window_data = data[start_idx:]
        window_data.to_csv("./data/window_data.csv", index=True)

    def _create_sequences(self, feature_columns, target_column):
        """创建输入序列和目标序列"""
        seq_len = self.config['sequence_length']
        pred_len = self.config['prediction_length']
        logger.info(f"创建序列 - 输入长度: {seq_len}, 预测长度: {pred_len}")
        X, y = [], []
        # 输入：[i, i+seq_len-1]（长度为 seq_len 的历史数据）
        # 目标：[i+seq_len, i+seq_len+pred_len-1]（长度为 pred_len 的未来数据）
        for i in tqdm(range(len(self.data) - seq_len - pred_len + 1)):
            # 输入序列
            seq = self.data[feature_columns].iloc[i:i + seq_len].values
            X.append(seq)

            # 目标序列(未来pred_len个时间步的价格)
            target = self.data[target_column].iloc[i + seq_len:i + seq_len + pred_len].values
            y.append(target)

        self.X = np.array(X)  # X:(list_num,seq_len,feature dimension)
        self.y = np.array(y)  # Y:(list_num,predict_len,)

        logger.info(f"序列创建完成 - 特征形状: {self.X.shape}, 目标形状: {self.y.shape}")

    def get_dataloaders(self):
        """创建并返回训练、验证和测试数据加载器"""
        if self.X is None or self.y is None:
            raise ValueError("请先调用 preprocess_data() 处理数据")

        # 划分训练集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config['test_size'],
            shuffle=False  # 时间序列不打乱顺序
        )

        # 从训练集中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config['val_size'] / (1 - self.config['test_size']),
            shuffle=False
        )

        # 创建数据集
        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        test_dataset = PriceDataset(X_test, y_test)

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle']
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        logger.info(
            f"数据加载器创建完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        return self.train_loader, self.val_loader, self.test_loader

    def inverse_transform_target(self, scaled_data):
        """将标准化的目标数据转换回原始尺度"""
        if self.scaler_target and self.config['normalization'] != 'none':
            return self.scaler_target.inverse_transform(scaled_data)
        return scaled_data


class RollingScaler(object):
    """滚动标准化器，支持华东窗口计算统计量"""

    def __init__(self, window_size, scaler_type='standard'):
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.all_stats = None  # 保存每个样本标准化时使用的统计量(用于反标准化)

    def fit_transform(self, data):
        """对数据进行滚动标准化并保存最后一个窗口的统计量"""
        normalized_data = []
        all_stats = []  # 存储每个样本对应的统计量
        n = len(data)

        for i in range(n):
            # 取当前位置前的窗口数据(不含当前值，避免数据泄露)
            start_idx = max(0, i - self.window_size)
            window_data = data[start_idx:i]

            # 计算窗口统计量
            if len(window_data) == 0:
                mean = 0.0
                std = 1.0
                stats = (mean, std)
            else:
                mean = window_data.mean()
                std = window_data.std() if window_data.std() != 0 else 1.0
                stats = (mean, std)

            # 标准化当前值
            if self.scaler_type == 'minmax':
                min_val = window_data.min() if len(window_data) > 0 else data[i]
                max_val = window_data.max() if len(window_data) > 0 else data[i]
                if max_val == min_val:
                    normalized = 0.0
                else:
                    normalized = (data[i] - min_val) / (max_val - min_val)
                # self.last_stats = (min_val, max_val)  # 保存min/max用于反归一化
            else:
                # 标准化/稳健标准化（使用窗口内的mean/std）
                mean, std = stats
                normalized = (data[i] - mean) / std
                # self.last_stats = (mean, std)  # 保存均值和标准差

            normalized_data.append(normalized)
            all_stats.append(stats)

        # 保存所有样本的统计量（用于反标准化）
        self.all_stats = np.array(all_stats)
        return np.array(normalized_data)

    def inverse_transform(self, normalized_data, indices=None):
        """使用最后保存的统计量反标准化"""
        if self.all_stats is None:
            raise ValueError("请先调用fit_transform进行拟合")

        # 确定需要反标准化的样本索引
        if indices is None:
            indices = len(normalized_data)
        # 提取这些样本对应的统计量
        stats = self.all_stats[-indices:]

        # 反标准化
        if self.scaler_type == 'minmax':
            min_vals = stats[:, 0]
            max_vals = stats[:, 1]
            # 处理min=max的情况
            mask = (max_vals == min_vals)
            result = np.where(
                mask,
                np.zeros_like(normalized_data),
                normalized_data * (max_vals - min_vals) + min_vals
            )
        else:
            means = stats[:, 0]
            stds = stats[:, 1]
            result = normalized_data * stds + means

        return result
