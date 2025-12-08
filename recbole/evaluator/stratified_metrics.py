# -*- encoding: utf-8 -*-
"""
Stratified Metrics by Item Popularity
按 Item 交互次数分档评估

分档：
- new: [1, 3) - 交互次数 1-2 次（新 item）
- few: [3, 10) - 交互次数 3-9 次（少量交互）
- frequent: [10, +inf) - 交互次数 ≥10 次（频繁交互）
"""

import numpy as np
import torch
from recbole.evaluator.base_metric import TopkMetric
from recbole.utils import EvaluatorType


class StratifiedRecall(TopkMetric):
    r"""StratifiedRecall 按 item 交互次数分档计算 Recall
    
    分档：
    - new: 交互次数 [1, 3) 
    - few: 交互次数 [3, 10)
    - frequent: 交互次数 [10, +inf)
    
    对于每个分档，计算该档 items 的 Recall@K
    """
    
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.topk", "rec.items", "data.count_items"]
    
    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        
        # 分档阈值
        self.new_threshold = (1, 3)      # [1, 3)
        self.few_threshold = (3, 10)     # [3, 10)
        self.freq_threshold = (10, float('inf'))  # [10, +inf)
    
    def used_info(self, dataobject):
        """获取推荐结果、正例信息和 item 交互次数"""
        # rec.topk: [n_users, max_k+1] - 前 max_k 列是推荐结果，最后一列是正例数量
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        topk_idx = topk_idx.to(torch.bool).numpy()
        pos_len_list = pos_len_list.squeeze(-1).numpy()
        
        # rec.items: [n_users, max_k] - 推荐的 item IDs
        rec_items = dataobject.get("rec.items").numpy()
        
        # data.count_items: Counter - item 交互次数
        item_counter = dataobject.get("data.count_items")
        
        return topk_idx, pos_len_list, rec_items, item_counter
    
    def _get_item_stratum(self, item_id, item_counter):
        """判断 item 属于哪个分档"""
        count = item_counter.get(item_id, 0)
        
        if self.new_threshold[0] <= count < self.new_threshold[1]:
            return 'new'
        elif self.few_threshold[0] <= count < self.few_threshold[1]:
            return 'few'
        elif count >= self.freq_threshold[0]:
            return 'frequent'
        else:
            return None  # count < 1，不应该出现在推荐中
    
    def calculate_metric(self, dataobject):
        """计算分档 Recall"""
        topk_idx, pos_len_list, rec_items, item_counter = self.used_info(dataobject)
        
        # 初始化每个分档的统计
        strata_stats = {
            'new': {'hit': 0, 'total': 0},
            'few': {'hit': 0, 'total': 0},
            'frequent': {'hit': 0, 'total': 0}
        }
        
        # 遍历每个用户的推荐列表
        n_users = topk_idx.shape[0]
        
        for user_idx in range(n_users):
            # 获取该用户的推荐结果
            user_topk = topk_idx[user_idx]  # [max_k] - bool array
            user_rec_items = rec_items[user_idx]  # [max_k] - item IDs
            
            # 遍历推荐列表中的每个 item
            for k_idx in range(len(user_rec_items)):
                item_id = user_rec_items[k_idx]
                is_hit = user_topk[k_idx]  # 是否命中正例
                
                # 判断 item 所属分档
                stratum = self._get_item_stratum(item_id, item_counter)
                
                if stratum in strata_stats:
                    strata_stats[stratum]['total'] += 1
                    if is_hit:
                        strata_stats[stratum]['hit'] += 1
        
        # 计算每个分档的 Recall@K
        metric_dict = {}
        
        for stratum in ['new', 'few', 'frequent']:
            stats = strata_stats[stratum]
            
            for k in self.topk:
                # 计算该分档在 top-k 中的 recall
                # 注意：这里简化处理，统计所有出现在 top-k 的该档 item
                if stats['total'] > 0:
                    recall = stats['hit'] / stats['total']
                else:
                    recall = 0.0
                
                key = f"Recall_{stratum}@{k}"
                metric_dict[key] = round(recall, self.decimal_place)
        
        return metric_dict


class StratifiedNDCG(TopkMetric):
    r"""StratifiedNDCG 按 item 交互次数分档计算 NDCG
    
    分档：
    - new: 交互次数 [1, 3) 
    - few: 交互次数 [3, 10)
    - frequent: 交互次数 [10, +inf)
    """
    
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.topk", "rec.items", "data.count_items"]
    
    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        
        # 分档阈值
        self.new_threshold = (1, 3)
        self.few_threshold = (3, 10)
        self.freq_threshold = (10, float('inf'))
    
    def used_info(self, dataobject):
        """获取推荐结果、正例信息和 item 交互次数"""
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        topk_idx = topk_idx.to(torch.bool).numpy()
        pos_len_list = pos_len_list.squeeze(-1).numpy()
        
        rec_items = dataobject.get("rec.items").numpy()
        item_counter = dataobject.get("data.count_items")
        
        return topk_idx, pos_len_list, rec_items, item_counter
    
    def _get_item_stratum(self, item_id, item_counter):
        """判断 item 属于哪个分档"""
        count = item_counter.get(item_id, 0)
        
        if self.new_threshold[0] <= count < self.new_threshold[1]:
            return 'new'
        elif self.few_threshold[0] <= count < self.few_threshold[1]:
            return 'few'
        elif count >= self.freq_threshold[0]:
            return 'frequent'
        else:
            return None
    
    def _dcg_at_k(self, r, k):
        """计算 DCG@K"""
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0
    
    def _idcg_at_k(self, pos_len, k):
        """计算 IDCG@K"""
        idcg = np.sum(1.0 / np.log2(np.arange(2, min(pos_len, k) + 2)))
        return idcg
    
    def calculate_metric(self, dataobject):
        """计算分档 NDCG"""
        topk_idx, pos_len_list, rec_items, item_counter = self.used_info(dataobject)
        
        # 按分档收集每个用户的结果
        strata_results = {
            'new': [],
            'few': [],
            'frequent': []
        }
        
        n_users = topk_idx.shape[0]
        
        for user_idx in range(n_users):
            user_topk = topk_idx[user_idx]
            user_rec_items = rec_items[user_idx]
            user_pos_len = pos_len_list[user_idx]
            
            # 按分档分别收集 hit 信息
            stratum_hits = {'new': [], 'few': [], 'frequent': []}
            stratum_pos_count = {'new': 0, 'few': 0, 'frequent': 0}
            
            for k_idx in range(len(user_rec_items)):
                item_id = user_rec_items[k_idx]
                is_hit = user_topk[k_idx]
                
                stratum = self._get_item_stratum(item_id, item_counter)
                
                if stratum in stratum_hits:
                    stratum_hits[stratum].append(1 if is_hit else 0)
                    if is_hit:
                        stratum_pos_count[stratum] += 1
            
            # 为每个分档计算该用户的 NDCG
            for stratum in ['new', 'few', 'frequent']:
                if len(stratum_hits[stratum]) > 0 and stratum_pos_count[stratum] > 0:
                    # 有该档的推荐且有命中
                    dcg = self._dcg_at_k(stratum_hits[stratum], max(self.topk))
                    idcg = self._idcg_at_k(stratum_pos_count[stratum], max(self.topk))
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    strata_results[stratum].append(ndcg)
        
        # 计算每个分档的平均 NDCG@K
        metric_dict = {}
        
        for stratum in ['new', 'few', 'frequent']:
            for k in self.topk:
                if len(strata_results[stratum]) > 0:
                    avg_ndcg = np.mean(strata_results[stratum])
                else:
                    avg_ndcg = 0.0
                
                key = f"NDCG_{stratum}@{k}"
                metric_dict[key] = round(avg_ndcg, self.decimal_place)
        
        return metric_dict


class ItemPopularityStats(TopkMetric):
    r"""ItemPopularityStats 统计推荐列表中不同分档 item 的比例
    
    输出指标：
    - Coverage_new@K: top-K 中 new items 的比例
    - Coverage_few@K: top-K 中 few items 的比例
    - Coverage_frequent@K: top-K 中 frequent items 的比例
    """
    
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]
    smaller = False
    
    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        
        # 分档阈值
        self.new_threshold = (1, 3)
        self.few_threshold = (3, 10)
        self.freq_threshold = (10, float('inf'))
    
    def used_info(self, dataobject):
        """获取推荐的 item IDs 和交互次数"""
        rec_items = dataobject.get("rec.items").numpy()
        item_counter = dataobject.get("data.count_items")
        return rec_items, item_counter
    
    def _get_item_stratum(self, item_id, item_counter):
        """判断 item 属于哪个分档"""
        count = item_counter.get(item_id, 0)
        
        if self.new_threshold[0] <= count < self.new_threshold[1]:
            return 'new'
        elif self.few_threshold[0] <= count < self.few_threshold[1]:
            return 'few'
        elif count >= self.freq_threshold[0]:
            return 'frequent'
        else:
            return 'unknown'
    
    def calculate_metric(self, dataobject):
        """计算每个分档在推荐列表中的覆盖率"""
        rec_items, item_counter = self.used_info(dataobject)
        
        metric_dict = {}
        
        for k in self.topk:
            # 统计 top-k 中各分档的数量
            stratum_counts = {'new': 0, 'few': 0, 'frequent': 0, 'unknown': 0}
            total_count = 0
            
            # 遍历所有用户的 top-k 推荐
            for user_idx in range(rec_items.shape[0]):
                user_topk = rec_items[user_idx, :k]
                
                for item_id in user_topk:
                    stratum = self._get_item_stratum(item_id, item_counter)
                    stratum_counts[stratum] += 1
                    total_count += 1
            
            # 计算每个分档的比例
            for stratum in ['new', 'few', 'frequent']:
                if total_count > 0:
                    coverage = stratum_counts[stratum] / total_count
                else:
                    coverage = 0.0
                
                key = f"Coverage_{stratum}@{k}"
                metric_dict[key] = round(coverage, self.decimal_place)
        
        return metric_dict

