# -*- coding: utf-8 -*-

import torch
import argparse
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def get_config():
    global current_time
    parser = argparse.ArgumentParser(description='LSTM_stock_predict')

    # 1.数据相关配置（完全复用原框架）
    data_group = parser.add_argument_group('data_config')
    data_group.add_argument('--data_path', type=str, default='./data/min_eth_data.csv', help='data path')
    data_group.add_argument('--sequence_length', type=int, default=100, help='input sequence length')
    data_group.add_argument('--prediction_length', type=int, default=1, help='the length of prediction')
    data_group.add_argument('--test_size', type=float, default=0.2, help='test size')
    data_group.add_argument('--val_size', type=float, default=0.1, help='val size')
    data_group.add_argument('--batch_size', type=int, default=32, help='batch size')
    data_group.add_argument('--shuffle', action='store_true', default=True, help='whether shuffle')
    data_group.add_argument('--normalization', type=str, default='standard',
                            choices=['minmax', 'standard', 'robust', 'none'], help='normalization type')

    # 2.模型相关配置(新增LSTM专属参数，其余复用)
    model_group = parser.add_argument_group('model_config')
    model_group.add_argument('--model_type', type=str, default='lstm', choices=['lstm'], help='model type')
    model_group.add_argument('--input_dim', type=int, default=6, help='the dimension of input')
    model_group.add_argument('--output_dim', type=int, default=1, help='the dimension of output')
    model_group.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    model_group.add_argument('--num_layers', type=int, default=2, help='the number of LSTM layers')
    model_group.add_argument('--dropout', type=float, default=0.2, help='the dropout rate')

    # 3. 训练相关配置（完全复用原框架）
    train_group = parser.add_argument_group('train_config')
    train_group.add_argument('--epochs', type=int, default=30, help='the number of epochs')
    train_group.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
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

    # 4. 日志和保存配置（完全复用原框架）
    log_group = parser.add_argument_group('日志和保存配置')
    log_group.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='日志级别')
    log_group.add_argument('--log_file', type=str, default=f'./log/logs/{current_time}-training.log', help='日志路径')
    log_group.add_argument('--log_format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           help='日志格式')
    log_group.add_argument('--save_dir', type=str, default='./log/models/', help='模型保存路径')
    log_group.add_argument('--save_best', action='store_true', default=True, help='是否只保存最优模型')
    log_group.add_argument('--plot_dir', type=str, default='./log/plots/', help='图表保存路径')

    # 解析参数并返回字典
    args = parser.parse_args()
    config = vars(args)
    config['device'] = torch.device(config['device'])
    return config


# 创建全局配置实例
config = get_config()
