import math
import torch
from torch_scatter import scatter_mean, scatter_std, scatter
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm

class Initialization(object):
    DEFAULT_BIAS = 0.01
    def __init__(self, trainset, num_sample, device, args, alphas=[1.0]):
        self.loader = torch.utils.data.DataLoader(trainset, batch_size=num_sample, shuffle=True, num_workers=2)
        self.num_sample = num_sample
        self.X, self.y = next(iter(self.loader))
        print(self.X.shape)
        self.X, self.y = self.X.to(device), self.y.to(device)
        self.X = self.X.reshape(num_sample, -1)
        self.device = device
        self.alphas = alphas
        self.args = args
        
    def init_fc(self, model, max_num_candidate, use_bias = True, select_metric = "class_mean"):
        if select_metric not in {"class_mean", "class_inner_std"}:
            raise ValueError(r"selected_metric must be in {class_mean, class_inner_std}")
        
        for i, linear in enumerate(model.fc):
            out_size, in_size = linear.weight.shape
            if out_size * self.args.candidate_weights > max_num_candidate:
                num_candidate = max_num_candidate
            else:
                num_candidate = out_size * self.args.candidate_weights
            
            W = torch.empty((in_size, num_candidate), dtype = torch.float32, device = self.device)
            
            print(W.shape)
            if use_bias:
                torch.nn.init.constant_(linear.bias, Initialization.DEFAULT_BIAS)
            else:
                torch.nn.init.constant_(linear.bias, 0.0)
            with torch.no_grad():
                # self.xavier_normal_(W.data, in_size, out_size)
                torch.nn.init.kaiming_normal_(W.data, nonlinearity='relu')

                if i != 0:    
                    selected_W = self.calculate_init_metric_diversity_inner_minus_project(out_size, self.X, self.y, W, use_bias, model.ac[i], "class_mean", self.alphas[i])
                else:
                    selected_W = self.calculate_init_metric_diversity_inner_minus_project(out_size, self.X, self.y, W, use_bias, model.ac[i], "class_mean", self.alphas[i])

                linear.weight.data = selected_W.t()
                self.X.data = model.ac[i](linear(self.X))
                
                print(self.calculate_stats(selected_W))
    
    def xavier_normal_(self, tensor, fan_in, fan_out):
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        return torch.nn.init._no_grad_normal_(tensor, 0., std)
   

    def calculate_init_metric_diversity_inner_minus_project(self, num_selected, x, y, W, use_bias, ac, select_metric, alpha):
        num_samples, num_features = x.shape
        num_candidate_cell = W.shape[1]
        b = torch.zeros((num_candidate_cell, ), dtype = torch.float32, device = self.device) if use_bias else torch.ones((num_candidate_cell, ), dtype = torch.float32, device = self.device) * Initialization.DEFAULT_BIAS
        
        with torch.no_grad():
            prediction = ac(torch.addmm(b, x, W)).t()
                
            # size: num_candidates * num_class
            class_mean = scatter_mean(prediction, y)
            # size: num_candidates * num_class
            class_inner_std = scatter_std(prediction, y)
            # class_inner_std /= torch.norm(class_inner_std, dim = 0)
            # size: num_candidates
            class_inner_std_mean = class_inner_std.mean(dim = 1)
            
            # size: num_candidates
            class_outer_std = class_mean.std(dim = 1)
            
            # size: num_candidates
            if select_metric == "class_mean":
                class_belong = class_mean.argmax(dim = 1)
            else:
                class_belong = class_inner_std.argmin(dim = 1)
                
            class_inner_std_mean /= torch.norm(class_inner_std_mean)
            class_outer_std /= torch.norm(class_outer_std)
            acc_scores = class_outer_std - class_inner_std_mean
            div_scores = prediction.std(dim = 1)
            acc_scores /= torch.norm(acc_scores)
            div_scores /= torch.norm(div_scores)
            scores = alpha * acc_scores + (1 - alpha) * div_scores
                
            # size: num_candidates
            class_belong = class_mean.argmax(dim = 1)
            num_class = class_mean.shape[1]
            index_list = [torch.nonzero(class_belong.eq(i), as_tuple = True)[0] for i in range(num_class)]
                
            num_list = [len(item) for item in index_list]
            selected_num_list = self.get_split_strategy_(num_selected, num_list)
            
            scores_ind = [torch.index_select(scores, dim = 0, index = ind) for ind in index_list]
            W_ind = [torch.index_select(W, dim = 1, index = ind) for ind in index_list]
            final_index = []
            for k in range(num_class):
                Wk = W_ind[k]
                sk = scores_ind[k]
                D = Wk
                for i in range(selected_num_list[k]):
                    D_norm = torch.norm(D, dim = 0)
                    best_index = torch.argmax(sk * torch.div(D_norm, torch.norm(Wk, dim = 0)))
                    a = torch.div(D[:, best_index], D_norm[best_index])
                    D = D - torch.mm(a.reshape((-1, 1)), torch.mm(a.reshape(1, -1), Wk))
                    final_index.append(index_list[k][best_index])
            # size: num_selected
            final_index_list = torch.tensor(final_index, device = self.device)
            # size: num_features * num_selected
            selected_weights = torch.index_select(W, dim = 1, index = final_index_list)
                            
        return selected_weights


    def calculate_stats(self, weights):
        # weights: num_features * num_selected
        with torch.no_grad():
            std = weights.std(dim = 1).mean()
        return std

    def get_split_strategy_(self, selected_num, group_nums) -> list:
        """
        return the selected number for each group by following the most uniform principle with time complexity O(n log n) where n = len(group_nums)
        """
        if sum(group_nums) < selected_num:
            raise ValueError("selected_num is too large!")
    
        ans = [0 for _ in group_nums]
        n = len(group_nums)
    
        res = selected_num
        st = 0
        each_num = 0
        sorted_nums = [(item, i) for i, item in enumerate(group_nums)]
        sorted_nums.sort()
    
        while (res > 0):
            each_num = res // (n - st)
            pre_st = st
            for i in range(st, n):
                if sorted_nums[i][0] > each_num:
                    st = i
                    break
                ans[sorted_nums[i][1]] = sorted_nums[i][0]
                res -= sorted_nums[i][0]
            if pre_st == st:
                break
    
        res_num = res % (n - st)
        for i in range(n - res_num, n):
            ans[sorted_nums[i][1]] = each_num + 1
        for i in range(st, n - res_num):
            ans[sorted_nums[i][1]] = each_num
    
        return ans


        