# -*- coding: utf-8 -*-

import torch
import logging
import torch.nn as nn

from config import config

logger = logging.getLogger("lstm_model")


class LSTMStockPredictor(nn.Module):
    def __init__(self):
        super(LSTMStockPredictor, self).__init__()

        # 1.LSTM层：处理时序特征
        self.lstm = nn.LSTM(
            input_size=config['input_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'],
            bidirectional=False
        )

        # 2.全连接层：将LSTM输出映射到预测维度
        self.fc = nn.Linear(
            in_features=config['hidden_size'],
            out_features=config['output_dim'] * config['prediction_length']  # 适配多部预测
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重，保证训练稳定性"""
        logger.info("开始初始化LSTM模型权重")
        # LSTM层权重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)  # 输入-隐藏层权重
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # 隐藏层-隐藏层权重（时序模型常用）
            elif 'bias' in name:
                param.data.fill_(0.1)  # 偏置初始化

        # 全连接层权重初始化
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.1)
        logger.info("LSTM模型权重初始化完成")

    def forward(self, x):
        """
        前向传播：适配原框架输入输出格式
        :param x: 输入序列，形状(batch_size,seq_len,input_dim)
        :return :预测结果，形状(batch_size,prediction_length,output_dim)
        """

        # LSTM层前向传播:output形状(batch_size,seq_len,hidden_size)
        lstm_out, _ = self.lstm(x)

        if lstm_out.dim()==3:
            # 取最后一个时间步的输出
            last_step_out = lstm_out[:, -1, :]  # 形状(batch_size,hidden_size)
        else:
            lstm_out = lstm_out.unsqueeze(0)
            last_step_out = lstm_out[:, -1, :]

        # 全连接层预测：映射到目标维度
        pred_flat = self.fc(last_step_out)

        # 调整输出形状为(batch_size, prediction_length, output_dim)
        pred = pred_flat.view(
            -1,
            config['prediction_length'],
            config['output_dim']
        )
        return pred


def get_lstm_model():
    logger.info(f"初始化LSTM模型：隐藏层维度{config['hidden_size']}，层数{config['num_layers']}")
    return LSTMStockPredictor().to(config['device'])
