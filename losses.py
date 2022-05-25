from torch import nn
import numpy as np
import torch 
import copy

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


class SigmaLoss(nn.Module):
    """Loss for regressing sigmas. It is simply the L2 loss of the squared sigma 
    i.e. make sigma as small as possible.

    """
    def __init__(self, loss_func=nn.MSELoss()):
        super(SigmaLoss, self).__init__()

        self.loss = loss_func
        
    def forward(self, sigmas):
        return self.loss([x**2 for x in sigmas])


class IntermediateOutputLoss(nn.Module):
    """
    use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
    between them (x[0] and y[0], x[1] and y[1] etc)

    """

    def __init__(self, hm_loss, ds_weights, sigma_loss=False, sigma_weight=0.005):
        super(IntermediateOutputLoss, self).__init__()
        self.ds_weights = ds_weights
        self.hm_loss = hm_loss
        self.sigma_loss = sigma_loss
        self.sigma_weight = sigma_weight

        self.loss_seperated_keys = ["hm_loss_all", "all_loss_all"] + ["hm_loss_level_"+str(i) for i in range(len(self.ds_weights))]
        if self.sigma_loss:
            self.loss_seperated_keys.append("sigma_loss")

    def forward(self, x, y, sigmas=None):

        losses_seperated = {}
        # print("we have 7 outputs, 1 for each resolution,", len(x))
        # print(" each output has 12 (batchsize)", len(x[0]))losses_seperated
        # # print("in the multiple output loss the x ", torch.stack(x,dim=1).squeeze(0).cpu().numpy().shape)
        # # print(" and y shapes are:", torch.stack(y,dim=1).squeeze(0).cpu().numpy().shape)
        # #only slice the inputs we give weights to
        # for idx, inp in enumerate(y):
        #     print(idx, " len inp", len(inp))
        #     print(idx, ", targ shape", inp.detach().cpu().numpy().shape)
        #     print(idx, "inp shape: ", x[idx].detach().cpu().numpy().shape)

        from_which_level_supervision = len(self.ds_weights)
        x = x[-from_which_level_supervision:]
        # l = self.ds_weights[0] * self.hm_loss(x[0], y[0])
        # losses_seperated["hm_loss_level_0"] = l
        l = 0
        # print("the weights are: ", self.ds_weights)
        # print(0, "pred shape %s and targ shape %s with weight %s  and loss %s and weighted loss %s" %(x[0].detach().cpu().numpy().shape, y[0].detach().cpu().numpy().shape, self.ds_weights[0], self.hm_loss(x[0], y[0]), l) )
        for i in range(0, len(y)):
            # print(i, "pred shape %s and targ shape %s with weight %s  and loss %s, and weighted loss: %s" %(x[i].detach().cpu().numpy().shape, y[i].detach().cpu().numpy().shape, self.ds_weights[i], self.hm_loss(x[i], y[i]), self.ds_weights[i] * self.hm_loss(x[i], y[i])) )
            this_lvl_loss = self.ds_weights[i] * self.hm_loss(x[i], y[i])
            l += this_lvl_loss
            losses_seperated["hm_loss_level_"+str(i)] = this_lvl_loss

        losses_seperated["hm_loss_all"] = l.detach().clone()

        if self.sigma_loss:

            sig_l = 0
            for sig in sigmas:
                sig_l += torch.square(sig)
            sig_l = self.sigma_weight *(torch.sqrt(sig_l))
            losses_seperated["sigma_loss"] = sig_l
            # print("Sigma loss: %s and HM loss %s " % (sig_l, l))
            l += (sig_l)

        losses_seperated["all_loss_all"] = l.detach().clone()

        # print("total loss: ", l)
        # print("loss dict inner", losses_seperated)

        return l, losses_seperated



class IntermediateOutputLossAndSigma(nn.Module):
    """
    use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
    between them (x[0] and y[0], x[1] and y[1] etc)

    """

    def __init__(self, hm_loss, weights):
        super(IntermediateOutputLossAndSigma, self).__init__()
        self.weights = weights
        self.hm_loss = hm_loss
        # self.sigma_loss = nn.MSELoss()
      

    def forward(self, x, y, sigmas):
        # print("we have 7 outputs, 1 for each resolution,", len(x))
        # print(" each output has 20 (batchsize)", len(x[0]))
        # print("target len", len(y))
        # print("in the multiple output loss the x ", torch.stack(x,dim=1).squeeze(0).cpu().numpy().shape)
        # print(" and y shapes are:", torch.stack(y,dim=1).squeeze(0).cpu().numpy().shape)

        # for idx, inp in enumerate(x):
          
        #     print(idx, " len inp", len(inp))


        #     print(idx, ", inp shape", inp.detach().cpu().numpy().shape)
        #     print(idx, "targ shape: ", y[idx].detach().cpu().numpy().shape)
        l = self.weights[0] * self.hm_loss(x[0], y[0])

        # print("the weights are: ", self.weights)
        # print(0, "pred shape %s and targ shape %s with weight %s  and loss %s and weighted loss %s" %(x[0].detach().cpu().numpy().shape, y[0].detach().cpu().numpy().shape, self.weights[0], self.loss(x[0], y[0]), l) )
        for i in range(1, len(x)):
            if self.weights[i] != 0:
                # print(i, "pred shape %s and targ shape %s with weight %s  and loss %s, and weighted loss: %s" %(x[i].detach().cpu().numpy().shape, y[i].detach().cpu().numpy().shape, self.weights[i], self.loss(x[i], y[i]), self.weights[i] * self.loss(x[i], y[i])) )

                l += self.weights[i] * self.hm_loss(x[i], y[i])

        sig_l = torch.mean(torch.square(sigmas))
        
        # torch.mean([x**2 for x in sigmas])

        print("HM Loss: %s, sigma Loss: %s " % (l, sig_l))


        return l + sig_l


class MultiBranchPatchLoss(nn.Module):
    def __init__(self, branch_scheme, class_loss_scheme, distance_weighted_bool):
        super(MultiBranchPatchLoss, self).__init__()
        self.branch_scheme = branch_scheme
        self.criterion_reg = nn.MSELoss(reduction='mean')
        self.distance_weighted_bool = distance_weighted_bool
        self.loss_seperated_keys = ["loss_all", "displacment_loss", "heatmap_loss"]

        if class_loss_scheme == "binary":
            self.class_criterion = nn.BCELoss(reduction='mean')
        elif class_loss_scheme == "binary_weighted": 
            self.class_criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=  torch.tensor(4096).to(device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
            raise NotImplementedError("this needs testing in new package. the hardcoded pos_weight doesn't seem right. I imagine it needs to be equal to # negative patches.")
        elif class_loss_scheme == "gaussian":
            self.class_criterion = nn.MSELoss(reduction='mean')
          #  self.class_criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=  torch.tensor(4096).to(device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        else:
            print("no recognised class loss scheme: ", class_loss_scheme )



    def weighted_mse_loss(self, input, target, weights):
      
        return torch.mean(weights * (input - target) ** 2)    


    def forward(self, predictions, labels):

        losses_seperated = {}
        total_loss = 0

        if self.branch_scheme == 'displacement' or self.branch_scheme == 'multi':
            if self.distance_weighted_bool:
                weights = labels["patch_heatmap"]
                loss_disp = self.weighted_mse_loss(predictions['patch_displacements'], labels['patch_displacements'], weights)
            else:
                loss_disp = self.criterion_reg(predictions['patch_displacements'], labels['patch_displacements'])
            
            total_loss += loss_disp
            losses_seperated["displacment_loss"] = loss_disp

        if self.branch_scheme == 'heatmap' or self.branch_scheme == 'multi':
            loss_class = self.class_criterion(predictions['patch_heatmap'], labels['patch_heatmap'])

            total_loss += loss_class
            losses_seperated["heatmap_loss"] = loss_class

        losses_seperated["loss_all"] = total_loss

        return total_loss, losses_seperated

        # if self.branch_scheme == 'multi':
        #     # print("reg loss and class loss", self.loss_regress.float(), self.loss_class)
        #     return (self.loss_class + self.loss_disp)
        # elif self.branch_scheme == 'displacement_only':
        #     # print("reg loss ", self.loss_regress.float())
        #     return (self.loss_disp)
        # else:
        #     # print("class loss ", self.loss_class)
        #     return (self.loss_class)