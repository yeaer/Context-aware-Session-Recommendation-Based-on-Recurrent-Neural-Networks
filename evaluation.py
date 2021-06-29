# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
模型评估函数
"""
import numpy as np
import pandas as pd

def evaluate_sessions_batch(model, train_data, test_data, cut_off, batch_size=128, session_key='SessionId',
                            item_key='ItemId', time_key='Time'):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
   
    '''

    '''
        Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
        Parameters
        --------
        model : 训练完的模型
        train_data : 包含训练集的交互数据。
        test_data : 包含测试集的交互数据。 有一列是session ID, 一列是item IDs, 还有一列是events的时间戳.
        cut-off : int
            推荐列表的长度n
        batch_size : int
            Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
        session_key : string
            输入文件中session ID列名称
            Header of the session ID column in the input file (default: 'SessionId')
        item_key : string
            输入文件中item ID列名称
            Header of the item ID column in the input file (default: 'ItemId')
        time_key : string
            输入文件中时间戳对应的列名称
            Header of the timestamp column in the input file (default: 'Time')

        Returns
        --------
        out : tuple
            输出是衡量指标的元组（Recall@N, MRR@N）
            (Recall@N, MRR@N)

        '''

    model.predict = False
    # 为训练数据创建时间戳
    # Build itemidmap from train data.
    itemids = train_data[item_key].unique()  # 全集的itemid
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)  # 通过全集得到的map

    countryids = train_data['country'].unique()
    countryidmap = pd.Series(data=np.arange(len(countryids)), index=countryids)

    cityids = train_data['city'].unique()
    cityidmap = pd.Series(data=np.arange(len(cityids)), index=cityids)

    regionids = train_data['region'].unique()
    regionidmap = pd.Series(data=np.arange(len(regionids)), index=regionids)

    deviceTypeids = train_data['deviceType'].unique()
    deviceTypeidmap = pd.Series(data=np.arange(len(deviceTypeids)), index=deviceTypeids)

    osids = train_data['os'].unique()
    osidmap = pd.Series(data=np.arange(len(osids)), index=osids)

    # 对测试数据按session_ID和time排序
    test_data.sort_values([session_key, time_key], inplace=True)

    # 为测试集中的session生成偏置（这一段与model.fit()是一致的）
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)  # 获取训练集的session个数
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0
    # 如果最后一个batch的session数量小于batch_size，修改batch_size的值
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1

    # iters的取值范围[0,1,...,batch_size - 1]
    iters = np.arange(batch_size).astype(np.int32)

    # maxiter = batch_size-1
    maxiter = iters.max()

    # 生成batch中的session在训练集中的start和end偏置
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]

    # in_idx = [0, ...,0]，长度是batch_size
    in_idx = np.zeros(batch_size, dtype=np.int32)
    in_country_idx = np.zeros(batch_size, dtype=np.int32)  # 按照上边的方式初始化
    in_city_idx = np.zeros(batch_size, dtype=np.int32)  # 按照上边的方式初始化
    in_region_idx = np.zeros(batch_size, dtype=np.int32)  # 按照上边的方式初始化
    in_deviceType_idx = np.zeros(batch_size, dtype=np.int32)  # 按照上边的方式初始化
    in_os_idx = np.zeros(batch_size, dtype=np.int32)  # 按照上边的方式初始化

    np.random.seed(42)
    flag = np.ones(batch_size, dtype=bool)
    while True:
        # valid_mask记录进行测试的session_id标记，参与测试标为1，否则标为0
        valid_mask = iters >= 0  # valid_mask是布尔值
        # 如果batch_size=0，说明训练结束
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        # 计算当前训练的最短session长度
        minlen = (end[valid_mask] - start_valid).min()  # minlen是这个批次里的最短会话
        in_idx[valid_mask] = test_data[item_key].values[start_valid]

        in_country_idx[valid_mask] = test_data['country'].values[start_valid]
        in_city_idx[valid_mask] = test_data['city'].values[start_valid]
        in_region_idx[valid_mask] = test_data['region'].values[start_valid]
        in_deviceType_idx[valid_mask] = test_data['deviceType'].values[start_valid]
        in_os_idx[valid_mask] = test_data['os'].values[start_valid]

        # 根据当前Session的最短长度进行批量预测
        for i in range(minlen - 1):
            out_idx = test_data[item_key].values[start_valid + i + 1]
            out_country_idx = test_data['country'].values[start_valid + i + 1]
            out_city_idx = test_data['city'].values[start_valid + i + 1]
            out_region_idx = test_data['region'].values[start_valid + i + 1]
            out_deviceType_idx = test_data['deviceType'].values[start_valid + i + 1]
            out_os_idx = test_data['os'].values[start_valid + i + 1]

            preds = model.predict_next_batch(iters, in_idx, in_country_idx, in_city_idx, in_region_idx,
                                             in_deviceType_idx,
                                             in_os_idx, itemidmap, countryidmap, cityidmap, regionidmap,
                                             deviceTypeidmap, osidmap, flag, batch_size)
            flag[:] = False
            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx  # 到这里 in_idx表示的是真实值，用来与预测值比较
            in_country_idx[valid_mask] = out_country_idx
            in_city_idx[valid_mask] = out_city_idx
            in_region_idx[valid_mask] = out_region_idx
            in_deviceType_idx[valid_mask] = out_deviceType_idx
            in_os_idx[valid_mask] = out_os_idx

            # preds.values.T[valid_mask].T为了取出有效的【valid】数据
            # 取对角线的原因是，此时对角线的元素是真实值的分数
            # np.diag(preds.iloc[in_idx].values)[valid_mask])得到的是真实值在预测结果里得到的份数。大于它的越多，他的排名越靠后。
            ranks = (preds.values.T[valid_mask].T > np.diag(preds.iloc[in_idx].values)[valid_mask]).sum(
                axis=0) + 1  # ranks表示真实值在列表中的位置
            # 统计召回的个数recall和排序的次序mrr
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
        # 当前batch的最短session已经测试结束，屏蔽/mask掉已经训练结束的session
        start = start + minlen - 1
        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
        flag[mask] = True
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
    return recall / evalutation_point_count, mrr / evalutation_point_count
