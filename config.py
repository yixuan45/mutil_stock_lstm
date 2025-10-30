# -*- coding: utf-8 -*-

import argparse
from datetime import datetime

import torch

current_time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def get_config():
    global current_time
    parser = argparse.ArgumentParser(description='LSTM_stock_predict')

    # 1.数据相关配置
    data_group = parser.add_argument_group('data_config')
    data_group.add_argument('--data_path', type=str, default='./data/1h_data.csv', help='data path')
    data_group.add_argument('--sequence_length', type=int, default=100, help='input sequence length')
    data_group.add_argument('--prediction_length', type=int, default=1, help='the length of prediction')
    data_group.add_argument('--test_size', type=float, default=0.2, help='test size')
    data_group.add_argument('--val_size', type=float, default=0.1, help='val size')
    data_group.add_argument('--batch_size', type=int, default=32, help='batch size')
    data_group.add_argument('--shuffle', action='store_true', default=True, help='whether shuffle')
    data_group.add_argument('--normalization', type=str, default='standard',
                            choices=['minmax', 'standard', 'robust', 'none'], help='normalization type')
    data_group.add_argument('--time_interval', type=int, default=60, help='time interval of minutes')

    # 2.特征参数相关配置
    feature_group = parser.add_argument_group('feature_config')
    feature_group.add_argument('--rsi_len', type=int, default=12, help='the length of rsi')
    feature_group.add_argument('--rsi_mean_len', type=int, default=15, help='the length of rsi mean')
    feature_group.add_argument('--macd_fastperiod', type=int, default=12, help='the fastperiod of macd')
    feature_group.add_argument('--macd_slowperiod', type=int, default=26, help='the slowperiod of macd')
    feature_group.add_argument('--macd_signalperiod', type=int, default=9, help='the signalperiod of macd')
    feature_group.add_argument('--atr_period', type=int, default=14, help='the period of atr')
    feature_group.add_argument('--high_slope_len', type=int, default=6, help='the length of high slope')
    feature_group.add_argument('--bb_timeperiod', type=int, default=12, help='the timeperiod of bb')
    feature_group.add_argument('--bb_dev', type=int, default=2, help='the dev period of bb')
    feature_group.add_argument('--rsi_divergence_length', type=int, default=5, help='the length of divergence')

    # 3.模型相关配置
    model_group = parser.add_argument_group('model_config')
    model_group.add_argument('--model_type', type=str, default='lstm', choices=['lstm'], help='model type')
    model_group.add_argument('--input_dim', type=int, default=17, help='the dimension of input')
    model_group.add_argument('--output_dim', type=int, default=1, help='the dimension of output')
    model_group.add_argument('--hidden_size', type=int, default=512, help='LSTM hidden size')
    model_group.add_argument('--num_layers', type=int, default=3, help='the number of LSTM layers')
    model_group.add_argument('--dropout', type=float, default=0.2, help='the dropout rate')

    # 4. 训练相关配置
    train_group = parser.add_argument_group('train_config')
    train_group.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    train_group.add_argument('--lr', type=float, default=1e-6, help='the learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    train_group.add_argument('--loss_fn', type=str, default='huber', choices=['mse', 'huber', 'mae'],
                             help='损失函数（Huber对异常值鲁棒）')
    train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='优化器')
    train_group.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                             help='学习率调度')
    train_group.add_argument('--is_early_stopping', type=bool, default=True, help='是否早停')
    train_group.add_argument('--early_stopping_patience', type=int, default=3, help='早停耐心值')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                             choices=['cpu', 'cuda'], help='训练设备')

    # 5. 日志和保存配置
    log_group = parser.add_argument_group('日志和保存配置')
    log_group.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='日志级别')
    log_group.add_argument('--log_file', type=str, default=f'./log/logs/{current_time}-training.log', help='日志路径')
    log_group.add_argument('--log_format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           help='日志格式')
    log_group.add_argument('--save_dir', type=str, default='./log/models/', help='模型保存路径')
    log_group.add_argument('--save_best', action='store_true', default=True, help='是否只保存最优模型')
    log_group.add_argument('--plot_dir', type=str, default='./log/plots/', help='图表保存路径')

    parser.add_argument('-f', '--file', help='Jupyter runtime file (ignored)', default=None)  # 添加这一行
    # 解析参数并返回字典
    args = parser.parse_args()
    config = vars(args)
    config['device'] = torch.device(config['device'])
    return config


# 创建全局配置实例
config = get_config()
