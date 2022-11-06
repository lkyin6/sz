import mindspore.nn as nn
import mindspore as ms
from mindspore.train.callback import Callback
import mindspore

# class EvalCallBack(Callback):
#     def __init__(self, model, eval_dataset, epochs_to_eval, per_eval, dataset_sink_mode):
#         self.model = model
#         self.eval_dataset = eval_dataset
#         # epochs_to_eval是一个int数字，代表着：每隔多少个epoch进行一次验证
#         self.epochs_to_eval = epochs_to_eval
#         self.per_eval = per_eval
#         self.dataset_sink_mode = dataset_sink_mode

#     def epoch_end(self, run_context):
#         # 获取到现在的epoch数
#         cb_param = run_context.original_args()
#         cur_epoch = cb_param.cur_epoch_num
#         # 如果达到进行验证的epoch数，则进行以下验证操作
#         if cur_epoch % self.epochs_to_eval == 0:
#             # 此处model设定的metrics是准确率Accuracy
#             acc = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
#             self.per_eval["epoch"].append(cur_epoch)
#             self.per_eval["dice"].append(acc["Dice"])
#             print("------------DICE为: {} ------------".format(acc["Dice"]))
class EvalCallBack(Callback):
    def __init__(self, model,net, eval_dataset, epochs_to_eval, per_eval, dataset_sink_mode):
        self.model = model
        self.net=net
        self.eval_dataset = eval_dataset
        # epochs_to_eval是一个int数字，代表着：每隔多少个epoch进行一次验证
        self.epochs_to_eval = epochs_to_eval
        self.per_eval = per_eval
        self.dataset_sink_mode = dataset_sink_mode
        self.best=0

    def epoch_end(self, run_context):
        # 获取到现在的epoch数
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        # 如果达到进行验证的epoch数，则进行以下验证操作
        if cur_epoch % self.epochs_to_eval == 0:
            # 此处model设定的metrics是准确率Accuracy
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
            self.per_eval["epoch"].append(cur_epoch)
            self.per_eval["dice"].append(acc["Dice"])
            if acc["Dice"]>self.best:
                self.best=acc["Dice"]
                mindspore.save_checkpoint(self.net,"./best_model/11.1_2/unet_{}_{}.ckpt".format(cur_epoch,round(acc["Dice"],2)))
            print("------------Hec_dice为: {} ------------".format(acc["Dice"]))
