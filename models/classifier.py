import torch 
from torch import nn

class Classifier(nn.Module):
    def __init__(self, n_way):
      super().__init__() 
      self.layer = nn.Sequential(nn.Flatten(),
                                    nn.Linear(128, n_way),# , bias=False
                                    nn.Softmax(dim=1))

      self.initilize()
 
    def forward(self,inputs): 
      x = self.layer(inputs)
      return x
    
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class LinearClassifierResNet(nn.Module):
    def __init__(self, layer=6, n_label=5, pool_type='avg', width=1):
        super(LinearClassifierResNet, self).__init__()
        if layer == 1:
            pool_size = 8
            n_channels = 128 * width
            pool = pool_type
        elif layer == 2:
            pool_size = 6
            n_channels = 256 * width
            pool = pool_type
        elif layer == 3:
            pool_size = 4
            n_channels = 512 * width
            pool = pool_type
        elif layer == 4:
            pool_size = 3
            n_channels = 1024 * width
            pool = pool_type
        elif layer == 5:
            pool_size = 7
            n_channels = 2048 * width
            pool = pool_type
        elif layer == 6:
            pool_size = 1
            n_channels = 2048 * width
            pool = pool_type
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()
        if layer < 5:
            if pool == 'max':
                self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool == 'avg':
                self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        else:
            # self.classifier.add_module('AvgPool', nn.AvgPool2d(7, stride=1))
            pass

        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LiniearClassifier', nn.Linear(n_channels * pool_size * pool_size, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        print(x.shape)
        return self.classifier(x)