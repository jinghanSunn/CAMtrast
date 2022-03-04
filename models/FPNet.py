import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Function

class FPNet(nn.Module):

    def __init__(self, in_channels, num_class):
        super(FPNet, self).__init__()
        out_channels = 128
        self.FC1 = nn.Linear(in_channels, in_channels)
        self.FC2 = nn.Linear(in_channels, out_channels)
        self.classifier_fc = nn.Linear(out_channels, num_class)

        self.FP1 = nn.Linear(in_channels, in_channels)
        self.FP2 = nn.Linear(in_channels, out_channels)
        self.classifier_fp = nn.Linear(out_channels, num_class)

        self.bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        fp = self.FC1(x)
        fp = self.relu(fp)
        fp = self.FC2(fp)
        fp = self.relu(fp)
        # maxpool?

        fc = self.FC1(x)
        fc = self.relu(fc)
        fc = self.FC2(fc)
        fc = self.relu(fc)
        
        fc = GradReverse.apply(fc, 1.0)
        
        out_c = self.classifier_fc(fc)

        # fp_star = self.proj(fp, fc)
        # fp_ = self.proj(fp, fp-fp_star)
        fp_ = NB_algorithm(fp, fc)

        self.bn(fp_)
        out_p = self.classifier_fp(fp_)

        

        return out_c, out_p
    
    def proj(self, x,y):
        return (x*y/torch.norm(y,p=2))*(y/torch.norm(y,p=2))


class GradReverse(Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　其实就是传入dict{'lambd' = lambd} 
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        # 直接传入一格数
        return grad_output * -ctx.lambd, None

class Matrix(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.num_samples = coordinates.shape[0]
        self.dimension   = coordinates.shape[1]
           
    def __str__(self):
        return self.coordinates

    def plus(self,v):
        return self.coordinates + v.coordinates

    def minus(self,v):
        return self.coordinates - v.coordinates

    def magnitude(self):
        # return tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.coordinates),axis = -1)),(self.num_samples,1))
        return torch.sum((self.coordinates).pow(2), dim=-1).reshape((self.num_samples, 1))

    def normalized(self):
        magnitude = self.magnitude()
        weight = (1.0/magnitude).reshape(self.num_samples,1)
        return self.coordinates * weight

    def component_parallel_to(self,basis):
        u = basis.normalized()
        # weight = tf.reshape(tf.reduce_sum(self.coordinates * u ,axis = -1),(self.num_samples,1))
        weight = torch.sum(self.coordinates * u ,dim = -1).reshape(self.num_samples,1)
        return u * weight

    def component_orthogonal_to(self, basis):
        projection = self.component_parallel_to(basis)
        return self.coordinates - projection
       

def NB_algorithm(original_feature,trivial_featue):
    original_feature = Matrix(original_feature)
    trivial_featue = Matrix(trivial_featue)
    d = original_feature.component_orthogonal_to(trivial_featue)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f