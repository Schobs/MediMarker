from torch import nn
import numpy as np
import torch 


class AdaptiveWingLoss(nn.Module):
    def __init__(self,hm_lambda_scale, omega=14, theta=0.5, epsilon=1, alpha=1.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha + hm_lambda_scale

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs() # get error for each pixel
        delta_y1 = delta_y[delta_y < self.theta] #get the low activation pixels (predicted background)
        delta_y2 = delta_y[delta_y >= self.theta] #get the high activation pixels (predcited foreground)
        y1 = y[delta_y < self.theta] # get the low act target pixels (background)
        y2 = y[delta_y >= self.theta]# get the high act target pixels (foreground)
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        # print("awl: ", (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)))
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class MyWingLoss(nn.Module):
    def __init__(self, omega=30, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        # print("awl: ", (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)))
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def soft_argmax(image):
	"""
	Arguments: image patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert image.dim()==4
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W = image.shape
	soft_max = nn.functional.softmax(image.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(image.shape)
	indices_kernel = torch.arange(start=0,end=H*W).image(0)
	indices_kernel = indices_kernel.view((H,W))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	y = (indices).floor()%W
	x = (((indices).floor())/W).floor()%H
	coords = torch.stack([x,y],dim=1)
	return coords


class SoftMaxLoss(nn.Module):
    def __init__(self, loss_func=soft_argmax):
        super(SoftMaxLoss, self).__init__()

        self.loss = loss_func
        
    def forward(self, net_output, target):

        return self.loss(net_output, target)
        
# torch.nn.MSELoss(reduction='mean')
class HeatmapLoss(nn.Module):
    def __init__(self, loss_func=nn.MSELoss()):
        super(HeatmapLoss, self).__init__()

        self.loss = loss_func
        
    def forward(self, net_output, target):
        # print("in the single heatmap output loss the x and y shapes are: ", net_output.detach().cpu().numpy().shape, (target).detach().cpu().numpy().shape)
        return self.loss(net_output, target)

#i need to think of a loss that normalises between 0-1

class IntermidiateOutputLoss(nn.Module):
    """
    use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
    between them (x[0] and y[0], x[1] and y[1] etc)

    """

    def __init__(self, loss, weights):
        super(IntermidiateOutputLoss, self).__init__()
        self.weights = weights
        self.loss = loss

    def forward(self, x, y):
        # print("we have 7 outputs, 1 for each resolution,", len(x))
        # print(" each output has 20 (batchsize)", len(x[0]))
        # print("target len", len(y))
        # print("in the multiple output loss the x ", torch.stack(x,dim=1).squeeze(0).cpu().numpy().shape)
        # print(" and y shapes are:", torch.stack(y,dim=1).squeeze(0).cpu().numpy().shape)

        # for idx, inp in enumerate(x):
          
        #     print(idx, " len inp", len(inp))


        #     print(idx, ", inp shape", inp.detach().cpu().numpy().shape)
        #     print(idx, "targ shape: ", y[idx].detach().cpu().numpy().shape)
        l = self.weights[0] * self.loss(x[0], y[0])

        # print("the weights are: ", self.weights)
        # print(0, "pred shape %s and targ shape %s with weight %s  and loss %s and weighted loss %s" %(x[0].detach().cpu().numpy().shape, y[0].detach().cpu().numpy().shape, self.weights[0], self.loss(x[0], y[0]), l) )
        for i in range(1, len(x)):
            if self.weights[i] != 0:
                # print(i, "pred shape %s and targ shape %s with weight %s  and loss %s, and weighted loss: %s" %(x[i].detach().cpu().numpy().shape, y[i].detach().cpu().numpy().shape, self.weights[i], self.loss(x[i], y[i]), self.weights[i] * self.loss(x[i], y[i])) )

                l += self.weights[i] * self.loss(x[i], y[i])

        return l