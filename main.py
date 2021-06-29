# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
# 导入外部和本地包（省略若干行）
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import time
import datetime

import model
import evaluation# 导入外部和本地包（省略若干行）# 导入外部和本地包（省略若干行）

# PATH_TO_TRAIN = './PATH/TO/rsc15_train_full.txt'
# PATH_TO_TEST = './PATH/TO/rsc15_test.txt'

# PATH_TO_TRAIN = './PATH/TO/train_data_oneMonth.csv'
# PATH_TO_TEST = './PATH/TO/test_data_oneMonth.csv'
PATH_TO_TRAIN = r'./data/train_data_oneMonth_forSCAR.csv'
PATH_TO_TEST = r'./data/test_data_oneMonth_forSCAR.csv'

RESULT = r'./result.txt'

# 定义参数类
class Args():
    is_training = False
    layers = 1  # 层数
    rnn_size = 100
    n_epochs = 0
    batch_size = 128
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = True
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

# 定义命令行解析函数
def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--test', default=1, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='bpr', type=str)
    parser.add_argument('--dropout', default='0.8', type=float)

    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行传来的参数
    command_line = parseArgs()
    '''
    Adressa数据集
    '''
    # data_full = pd.read_csv(r"F:\Adressa\threeMonth\session_map_threeMonth.csv", sep='\t', names=['SessionId','ItemId','Time','country','city','region','deviceType','os'],dtype={'ItemId': np.int64,"region":np.int64})
    # 从指定目录中读取训练集和测试集
    data_full = pd.read_csv(r"./data/oneMonth_forSCAR_all.csv", header=0,
                            names=['SessionId', 'ItemId', 'Time', 'country', 'city', 'region', 'deviceType', 'os'],
                            dtype={'ItemId': np.int64, "region": np.int64})
    data = pd.read_csv(PATH_TO_TRAIN, header=0,
                       names=['SessionId', 'ItemId', 'Time', 'country', 'city', 'region', 'deviceType', 'os'],
                       dtype={'ItemId': np.int64, "region": np.int64})
    valid = pd.read_csv(PATH_TO_TEST, header=0,
                        names=['SessionId', 'ItemId', 'Time', 'country', 'city', 'region', 'deviceType', 'os'],
                        dtype={'ItemId': np.int64, "region": np.int64})
    '''
    rsc15数据集
    '''
    # data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    # valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})

    # 传递命令行参数
    args = Args()
    args.n_items = len(data_full['ItemId'].unique())  # 注意这里的数据集的选择
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)

    # 确保检查点的保存路径存在
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # 设置GPU的参数，允许显存动态增长
    #gpu_config = tf.ConfigProto()
    gpu_config = tf.compat.v1.ConfigProto()
    epoch = command_line.epoch
    gpu_config.gpu_options.allow_growth = True

    cut_off = 20

    # 创建TF会话
    with tf.Session(config=gpu_config) as sess:
        # 实例化模型/创建模型
        gru = model.GRU4Rec(sess, args)
        # 训练模型
        if args.is_training:
            print("训练开始:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            start = datetime.datetime.now()
            with open(RESULT, 'a') as file:
                file.write('\t' + "训练开始时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\t')
            gru.fit(data)

            print("训练结束:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            end = datetime.datetime.now()
            print("用时：" + str((end - start).seconds) + "秒")
            with open(RESULT, 'a') as file:
                file.write(
                    "训练结束时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n\t' + "训练用时" + str(
                        (end - start).seconds) + "秒" + '\n')
        else:
            # 模型预测
            print("测试")

            for i in range(epoch - 2, epoch):
                gru.restoreModel(i)
                res = evaluation.evaluate_sessions_batch(gru, data_full, valid, cut_off=cut_off)
                print('epoch: {}\tRecall@{}: {}\tMRR@{}: {}'.format(i + 1, cut_off, res[0], cut_off, res[1]))
                with open(RESULT, 'a') as file:
                    file.write(
                        '\nepoch: {}\tRecall@{}: {}\tMRR@{}: {}\n'.format(i + 1, cut_off, res[0], cut_off, res[1]))
            print('测试结束')
