import mindspore.numpy as mnp
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore.train.callback import Callback
from mindspore.nn import LossBase
from mindspore import ops
from mindspore import Tensor


def Dice(predict, label, eps=1e-6):
    inter = np.sum(predict * label)
    union = np.sum(predict) + np.sum(label)
    dice = (2*inter + eps) / (union + eps)
    return dice

class WeightedMCLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(WeightedMCLoss, self).__init__()
        self.softmax = ops.Softmax(axis=0)
    def construct(self, logits, labels):
        batch_size = labels.shape[0]
        logits = self.softmax(logits)
        x = 0.0
        weight = [0.0001, 0.01, 1]

        
#         label_change = ops.ones_like(logits)
#         for i in range(label_change.shape[0]):
#             label_change[i] = ms.numpy.where(labels[0]==i, 1, 0)
        for i in range(logits.shape[0]):
            result = -label_change[i] * ops.log(logits[i])*weight[i]
            x += result.cumsum()
        return self.get_loss(x)

# class L1Loss(nn.Cell):
#     def __init__(self):
#         super(L1Loss, self).__init__()
#         self.abs = ops.Abs()
#         self.reduce_mean = ops.ReduceMean()

#     def construct(self, base, target):
#         x = self.abs(base - target)
#         return self.reduce_mean(x)
    
# class SDCE(LossBase):
#     def __init__(self, reduction='mean'):
#         super(SDCE, self).__init__(reduction)
#         self.softmax = ops.Softmax(axis=0)
#         self.dice = nn.DiceLoss()
#         self.cross = nn.BCEWithLogitsLoss()
#         self.add = ops.Add()
#         self.mul = ops.Mul()
#         self.abs = ops.Abs()
#         self.reduce_mean = ops.ReduceMean()
#         self.log = ops.Log()
#         self.pow = ops.Pow()
#     def construct(self, base, target):
#         res = 0.0
#         x = 0.0
#         for batch in range(base.shape[0]):
            
#             b_logits = base[batch]
#             b_labels = target[batch]
#             dice_total = 0.0
#             cross_total = 0.0
#             w_dice = [0.4, 0.6]
#             w_cross = [0.14, 0.35, 0.51]
            
#             c = self.cross(b_logits[0], b_labels[0])
            
#             for i in range(2):
#                 dice_total = self.add(dice_total, self.mul(self.pow(-self.log(self.dice(b_logits[i+1], b_labels[i+1])), 0.3), w_dice[i])) 
#             for i in range(3):
#                 cross_total = self.add(cross_total, self.mul(self.cross(b_logits[i], b_labels[i]), w_cross[i])) 
#             x = self.add(cross_total, dice_total)
#         res = self.add(res, x)
#         return self.reduce_mean(res)
    
class SDCE(LossBase):
    def __init__(self, reduction='mean'):
        super(SDCE, self).__init__(reduction)
        self.softmax = ops.Softmax(axis=0)
        self.dice = nn.DiceLoss()
        self.cross = nn.BCEWithLogitsLoss()
        self.add = ops.Add()
        self.mul = ops.Mul()
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()
        self.pow = ops.Pow()
    def construct(self, base, target):
        res = 0.0
        x = 0.0            
        b_logits = base
        b_labels = target
        dice_total = 0.0
        cross_total = 0.0
        w_dice = [0.4, 0.6]
        w_cross = [0.14, 0.35, 0.51]
        for i in range(2):
            dice_total = self.add(dice_total, self.mul(self.pow(-self.log(self.dice(b_logits[:, i+1], b_labels[:, i+1])), 0.3), w_dice[i])) 
        for i in range(3):
            cross_total = self.add(cross_total, self.mul(self.cross(b_logits[:, i], b_labels[:, i]), w_cross[i])) 
        x = self.add(cross_total, dice_total)
        res = self.add(res, x)
        return self.reduce_mean(res)
    
    
def Hec_dice(predict, label):
    """
    0:其他组织 1:肾脏 2:肿瘤
    hec1:肾脏+肿瘤
    hec2:肿瘤
    """
    hec1 = Dice(predict > 0, label > 0)
    hec2 = Dice(predict == 2, label == 2)
    return (hec1 + hec2) / 2

class Dice_Loss(nn.Metric):
    def __init__(self):
        super(Dice_Loss, self).__init__()
        self.clear()
    def clear(self):
        """初始化变量_abs_error_sum和_samples_num"""
        self._dice_error_sum = 0  # 保存误差和
        self._samples_num = 0    # 累计数据量

    @nn.rearrange_inputs
    def update(self, *inputs):
        """更新_abs_error_sum和_samples_num"""
#         y_pred = inputs[0].asnumpy()
#         y_pred = np.argmax(y_pred, axis=0)
#         y = inputs[1].asnumpy()
        
#         # 计算预测值与真实值的绝对误差
#         self.abs_error_sum = Hec_dice(y, y_pred)
#         print(self.abs_error_sum)

        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        print(y_pred.shape, y.shape)
        for batch in range(y_pred.shape[0]):
            y_2 = y[batch]
            y_pred_2 = y_pred[batch]
            y_2 = np.argmax(y_2, axis=0)
            y_pred_2 = np.argmax(y_pred_2, axis=0)
            dloss = Hec_dice(y_pred_2, y_2)
            print(f"实际答案1个数：{np.count_nonzero(y_2==1)},2个数：{np.count_nonzero(y_2==2)}")
            print(f"预测答案1个数：{np.count_nonzero(y_pred_2==1)},2个数：{np.count_nonzero(y_pred_2==2)}")
            print(f"Dice_Loss:{dloss}")
            self._dice_error_sum += dloss
        self._samples_num += y_pred.shape[0]
        
    def eval(self):
        """计算最终评估结果"""
        return self._dice_error_sum / self._samples_num