
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils.utils import  get_classes

classes_path    = 'model_data/cls_classes.txt'
class_names, num_classes = get_classes(classes_path)

class CosineMarginProduct(nn.Module):
    def __init__(self, in_feature=448, out_feature=num_classes, s=30.0, m=0):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature)).cuda()
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output
