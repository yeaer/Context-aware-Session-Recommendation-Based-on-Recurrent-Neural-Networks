# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tensorflow import set_random_seed
# set_random_seed(2)


# 定义模型类--GRU4Rec
class GRU4Rec:
    # 定义类的初始化函数：当main.py调用“gru = model.GRU4Rec(sess, args)”语句时，执行该函数
    def __init__(self, sess, args):
        # 从命令行窗口中传入函数参数
        self.sess = sess
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items  # 这个是item的数量
        self.test_model = args.test_model

        # 判断并选择模型的隐层激活函数(tanh or relu)
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        # 判断并选择模型的损失函数和最后一层的激活函数
        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        # 查找checkpoints文件的保存路径。如果路径不存在，返回错误信息
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        # 调用类方法build_model()，构建GRU4Rec模型
        self.build_model()
        # 初始化会话图中的所有变量
        self.sess.run(tf.global_variables_initializer())
        # 保存变量
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        # 判断命令行参数是否训练模型，如果训练模型，结束并返回
        if self.is_training:
            return

        # 如果是用模型进行预测，继续执行
        # 使用self.predict_state保存预测过程的模型隐状态

        # use self.predict_state to hold hidden states during prediction.
        # self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名。
        # ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        # 如果找到模型参数保存的文件，重新加载模型
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

    ########################ACTIVATION FUNCTIONS#########################
    # 激活函数
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    # 损失函数
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    ###构建GRU4Rec模型（本项目最核心的部分代码）
    def build_model(self):
        # 根据输入数据X和标签Y的格式，定义占位符的数据格式和数据维度。等待数据传入...
        self.X = tf.placeholder(tf.int32, [6, self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.flag = tf.placeholder(tf.bool, self.batch_size, name='session_reset_flag')
        # self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
        #               range(self.layers)]
        # 循环神经网络上一时刻的隐状态state必须传入下一时刻参与运算，定义占位符，维度是 (batch大小 * RNN隐状态的维度)
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
                      range(self.layers)]
        # 创建变量global_step......
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # 创建变量作用域'gru_layer'
        with tf.variable_scope('gru_layer'):
            # sigma指的是参数随机初始化的参数，如果命令行参数传入了sigma，则令sigma等于传入的值；
            # 否则，令sigma=np.sqrt(6.0 / (self.n_items + self.rnn_size))
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            # 变量初始化的方式：
            # 1. random_normal_initializer 服从正态分布的随机初始化
            # 2. random_uniform_initializer 服从均匀分布的随机初始化
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)

                # 定义embedding和softmax的参数矩阵
                '''
                tf.Variable()用于生成一个初始值为initial-value的变量；必须指定初始化值。
                tf.get_variable()获取已存在的变量(要求不仅名字，而且初始化方法等各个参数都一样)，如果不存在，就新建一个；
                可以用各种初始化方法，不用明确指定值。
                '''
            embedding = tf.get_variable(name='embedding', shape=[self.n_items, self.rnn_size],
                                        initializer=initializer)
            country_embedding = tf.get_variable(name='country_embedding', shape=[145, self.rnn_size],
                                                initializer=tf.random_normal_initializer(mean=0,
                                                                                         stddev=sigma))  # oneMonth:145 threeMonth174
            city_embedding = tf.get_variable(name='city_embedding', shape=[3808, self.rnn_size],
                                             initializer=tf.random_normal_initializer(mean=0,
                                                                                      stddev=sigma))  # oneMonth:3808 threeMonth6794
            region_embedding = tf.get_variable(name='region_embedding', shape=[897, self.rnn_size],
                                               initializer=tf.random_normal_initializer(mean=0,
                                                                                        stddev=sigma))  # oneMonth:897 threeMonth1299
            deviceType_embedding = tf.get_variable(name='deviceType', shape=[3, self.rnn_size],
                                                   initializer=tf.random_normal_initializer(mean=0,
                                                                                            stddev=sigma))  # oneMonth:3 threeMonth3
            os_embedding = tf.get_variable(name='os_embedding', shape=[9, self.rnn_size],
                                           initializer=tf.random_normal_initializer(mean=0,
                                                                                    stddev=sigma))  # oneMonth:9 threeMonth11

            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            # 创建循环神经元，GRUCell。所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。
            # 这是是一种有效的正则化方法，可以有效防止过拟合。
            # 在rnn中使用dropout的方法和cnn不同，推荐大家去把“recurrent neural network regularization”看一遍。
            cell = tf.nn.rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)

            # 定义多层RNN网络，层数由命令行参数决定
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell] * self.layers)

            item = tf.nn.embedding_lookup(embedding, self.X[0])

            country = tf.nn.embedding_lookup(country_embedding, self.X[1])
            city = tf.nn.embedding_lookup(city_embedding, self.X[2])
            region = tf.nn.embedding_lookup(region_embedding, self.X[3])
            deviceType = tf.nn.embedding_lookup(deviceType_embedding, self.X[4])
            os = tf.nn.embedding_lookup(os_embedding, self.X[5])

            # extraInformation = tf.layers.dense(tf.concat([country, city, region, deviceType, os], 1), self.rnn_size,
            #                                    activation=tf.nn.tanh)
            # extraInformation = country+city+region+deviceType+os
            extraInformation = tf.concat([country, city, region, deviceType, os], 1)  # 只拼接

            with tf.variable_scope('Session_Reset', reuse=tf.AUTO_REUSE):
                self.state[0] = tf.where(self.flag,
                                         tf.layers.dense(extraInformation, self.rnn_size, activation=tf.nn.tanh),
                                         self.state[
                                             0])  #

            # 使用embedding矩阵将输入数据嵌入到低维度；经过多层循环网络并输出output和隐状态state
            # inputs = tf.concat([item,extraInformation], 1)
            # inputs = tf.layers.dense(tf.concat([extraInformation,item],1), self.rnn_size,activation=tf.nn.tanh)
            # inputs = item+extraInformation
            inputs = item
            output, state = stacked_cell(inputs, tuple(self.state))

            # output = tf.layers.dense(tf.concat([extraInformation,output],1), self.rnn_size,activation=tf.nn.tanh)

            self.final_state = state

        # 搭建完模型
        # 如果是训练模型，选取同一个batch的其他item作为负采样的样例
        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W,
                               transpose_b=True) + sampled_b
            self.yhat = tf.nn.leaky_relu(logits)
            self.cost = self.loss_function(self.yhat)
        # 如果是用模型预测，使用最后一次output生成预测结果
        else:
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = tf.nn.leaky_relu(logits)

        # 如果是预测，程序运行结束
        if not self.is_training:  # 如果在测试阶段 就不执行下边的代码了
            return

        # 设置指数衰减的学习率，并且学习率大于0
        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay, staircase=True))

        # 尝试其他优化器
        '''
        Try different optimizers.
        '''
        # optimizer = tf.train.AdagradOptimizer(self.lr)  #不收敛
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.lr)  #不收敛
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()  # 只返回可训练的变量
        gvs = optimizer.compute_gradients(self.cost, tvars)  # 计算loss中可训练的var_list中的梯度。
        # 根据梯度的裁剪率判断是否需要裁减梯度
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
    def init(self, data):
        # 分别先后根据session_id和time_key排序
        data.sort_values([self.session_key, self.time_key], inplace=True)
        # offset_sessions的维度/元素个数等于session_id的个数
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)  # 创建一个长度为session个数的全0的np对象
        # 计算offset_sessions每个元素的值：根据每个Session的长度累加
        offset_sessions[1:] = data.groupby(
            self.session_key).size().cumsum()
        return offset_sessions

    # 传入数据，训练模型fit()
    def fit(self, data):
        # 模型训练时出现错误
        self.error_during_train = False

        # 获取每个session的相对于起始的偏移量
        offset_sessions = self.init(data)
        # 开始训练模型
        print('fitting model...')
        # 循环迭代
        for epoch in range(self.n_epochs):
            # 当前epoch的损失值
            epoch_cost = []  # 用于存储每个时间步的cost
            # 根据batch_size和rnn_size初始化state，state的元组数与网络层数有关
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in
                     range(self.layers)]  # 
            # session_idx_arr是session的索引数组，范围是 [0,1,...,session的个数-1]
            session_idx_arr = np.arange(len(offset_sessions) - 1)
            # iters是batch的索引，范围是[0, 1,...,batch_size-1]
            iters = np.arange(self.batch_size)  # 0到batch_size
            # maxiter = batch_size - 1
            maxiter = iters.max()  #
            # 在一个batch里，session的全部start位置和end位置，其中end位置指的是下一个session的start位置
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            flag = np.ones(self.batch_size, dtype=bool)
            # 会话结束标志位，初始化为False。表示当前没有session结束。
            finished = False
            while not finished:
                # minlen表示当前batch中的session最短长度
                minlen = (end - start).min()  #
                # in_idx表示当前时刻的输入数据，out_idx表示当前时刻的groud_truth/目标输出，
                out_idx = data.ItemId.values[start]
                # i：控制每个时刻的输入数据，如果i的位置是len(session)-1，可以停止。因为下一时刻，没有groudtruth值与之对应。
                for i in range(minlen - 1):
                    in_idx = out_idx

                    country_id = data.country.values[start + i]
                    city_id = data.city.values[start + i]
                    region_id = data.region.values[start + i]
                    deviceType_id = data.deviceType.values[start + i]
                    os_id = data.os.values[start + i]

                    out_idx = data.ItemId.values[start + i + 1]
                    # prepare inputs, targeted outputs and hidden states
                    inputs = np.vstack((in_idx, country_id, city_id, region_id, deviceType_id, os_id))
                    # inputs = np.vstack((in_idx, country_id,city_id,region_id,deviceType_id,os_id)) #拼接形成一个50×6的矩阵

                    # 准备输入数据，目标输出和隐状态
                    # fetches中的变量对应于sess.run()想要获得的变量
                    # feed_dict对应于输入数据和对应的目标输出
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: inputs, self.Y: out_idx, self.flag: flag}
                    # 如果网络是多层循环网络，需要保存底层GRU的state[j]以供上层GRU的训练。
                    # python中dict增加新的键值对{self.state[j]: state[j]}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    # 训练一个batch，得到batch_cost
                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    flag[:] = False  # 重置为False
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                # batch中的最短session训练结束，start中的所有元素移到第一个Session结束的位置
                start = start + minlen - 1  # 整体向后移动minlen
                # 统计本次最长训练步数下，有多少session已经结束。采用mask标记
                mask = np.arange(len(iters))[(end - start) <= 1]
                flag[mask] = True
                # 统计mask中session索引的数量，说明有多少session已经结束。maxiter相应向后移动多少数量的session，也就是将新的Session加入batch训练
                # 如果当前的maxiter超过了训练集中的最大偏置，说明训练已经结束。
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break

                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]
                # 如果mask的数量大于0，并且设置了Session结束以后重置mask。则执行以下操作。（没太明白这个操作的意思。。。）
                if len(mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][mask] = 0

            # 计算平均cost
            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            # 保存模型参数
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

    def restoreModel(self, test_model):
        '''
        加载训练好的模型
        :param test_model:将要启用的已经存储的模型的编号
        :return:
        '''
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, '{}/gru-model-{}'.format(self.checkpoint_dir, test_model))



    def predict_next_batch(self, session_ids, input_item_ids, input_coutry_ids, input_city_ids, input_region_ids,
                           input_deviceType_ids, input_os_ids,
                           itemidmap, countryidmap, cityidmap, regionidmap, deviceTypeidmap, osidmap, flag, batch=50):
        # def predict_next_batch(self, session_ids, input_item_ids, input_region_ids,input_os_ids,
        #
        #                        itemidmap, regionidmap, osidmap, batch=50):

        '''
                给出一组选定items的预测分数。可用于批处理模式，一次预测多个独立items（即不同session中的items），从而加快评估速度。
                如果在该函数的后续调用期间，session_ids参数的给定坐标处的会话ID保持不变，则网络的相应隐状态将保持不变（即，这就是如何可以预测会话中的项目的方式）。
                如果更改，则网络的隐状态将重置为零。

                参数
                --------
                session_ids : 1D array
                    包含batch项目中的所有session IDs，它的长度等于预测batch的大小

                input_item_ids : 1D array
                    包含batch中的item IDs。每一个item ID必须是网络中的训练数据。它的长度必须等于预测batch的大小

                batch : int
                    预测batch的尺寸大小

                Returns
                --------
                out : pandas.DataFrame
                    batch中每个session中预测items的得分
                    列：batch中的session; 行：items。行是通过Item IDs索引的。

                '''

        # batch的大小（在evaluate函数的参数中定义）与self.batch_size必须保持一致
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        # batch中的session顺序发生了改变
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        # 输入item_ids对应的索引值
        in_idxs = itemidmap[input_item_ids]

        country = countryidmap[input_coutry_ids]
        city = cityidmap[input_city_ids]
        region = regionidmap[input_region_ids]
        deviceType = deviceTypeidmap[input_deviceType_ids]
        os = osidmap[input_os_ids]

        # inputs = np.vstack((in_idxs, country, city, region, deviceType, os))
        inputs = np.vstack((in_idxs, country, city, region, deviceType, os))
        # 获取模型训练的“self.yhat”，“self.final_state”
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: inputs, self.flag: flag}
        # 分层获取模型的隐状态
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        # 模型预测，获取preds, self.final_state。
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        # 数组转置
        preds = np.asarray(preds).T
        # 返回预测值
        return pd.DataFrame(data=preds, index=itemidmap.index)
