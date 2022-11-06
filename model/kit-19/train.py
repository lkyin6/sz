import os
import mindspore.dataset as ds
from mindspore import Tensor
import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
from etc.mydataset import DatasetGenerator
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import context
# from nestedunet import NestedUNet
from etc.FWIoU_metric import EvalCallBack, FWIoU
from etc.loss import Dice_Loss, WeightedMCLoss, SDCE
from mindspore.nn import FocalLoss
from etc.model import AttU_Net
import moxing as mox
import etc.learning_rates as learning_rates
from etc.callback import EvalCallBack
import mindspore as ms
import argparse


def train_total(args):
    train_dataset_generator = DatasetGenerator(args.train_path, r'imaging', r'segmentation', mode='t')
    train_dataset = ds.GeneratorDataset(train_dataset_generator,
                                        ["image", "label"],
                                        shuffle=True)
    valid_dataset_generator = DatasetGenerator(args.val_path, r'imaging', r'segmentation', mode='v')
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)

    train_dataset = train_dataset.batch(8, num_parallel_workers=64, drop_remainder=True)
    valid_dataset = valid_dataset.batch(16, num_parallel_workers=64)
#     valid_dataset = valid_dataset.batch(cfg.BATCH_SIZE, num_parallel_workers=1)

#     loss = WeightedBCELoss(w0=1.39, w1=1.69)
#     loss = nn.CrossEntropyLoss()
#     loss = WeightedMCLoss()
#     loss = SDCE()
#     loss = CrossEntropyLoss()
    loss = FocalLoss(weight=Tensor([1, 1.5, 3], ms.float32,), gamma=2.0)
#     loss = FocalLoss(weight=None)
#     loss = nn.DiceLoss()
#     loss = nn.BCEWithLogitsLoss()
#     loss.add_flags_recursive(fp32=True)
    train_net = AttU_Net()
    # load pretrained model
    if args.pretrain:
        param_dict = load_checkpoint(args.premodel_path)
        load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = train_dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * 10
    
#     lr_iter = learning_rates.poly_lr(100,
#                                      total_train_steps,
#                                      total_train_steps,
#                                      end_lr=0.0,
#                                      power=0.9)

    opt = nn.Adam(params=train_net.trainable_params(), learning_rate=0.001)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(3072.0,
                                               drop_overflow_update=False)
    model = Model(train_net,
                  optimizer=opt,
                  amp_level="O3",
                  loss_fn=loss,
                  loss_scale_manager=manager_loss_scale,
                  metrics={"Dice":Dice_Loss()}
                  )
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()

    # 保存模型
    # save_checkpoint_steps表示每隔多少个step保存一次，keep_checkpoint_max表示最多保留checkpoint文件的数量
    config_ckpt = CheckpointConfig(
        save_checkpoint_steps=iters_per_epoch,
        keep_checkpoint_max=100)
    # prefix表示生成CheckPoint文件的前缀名；directory：表示存放模型的目录
    cbs_1 = ModelCheckpoint(prefix='unet-transform',
                            directory=args.save_path,
                            config=config_ckpt)
    per_eval = {"epoch": [], "dice": []}
    cbs = [time_cb, loss_cb, cbs_1, EvalCallBack(model, train_net, valid_dataset, 1, per_eval, False)]
    # 训练模型
    model.train(20, train_dataset, callbacks=cbs, dataset_sink_mode=False)
    
#     mox.file.copy_parallel(src_url=cfg.OUTPUT_DIR,
#                            dst_url='obs://image-segment/output_train')
    # mox.file.copy_parallel(src_url=cfg.SUMMARY_DIR,
    #                        dst_url='obs://image-segment/summary_log')


if __name__ == "__main__":
#     mox.file.copy_parallel(src_url='obs://mushroom-data/MD_DATA',
#                            dst_url='./MD_DATA')
  
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',default="/home/ma-user/work/project/train/", type=str,help ='train dataset path')
    parser.add_argument('--val_path',default = "/home/ma-user/work/project/val/", type=str,help='val dataset path')
    parser.add_argument('--pretrain',action="store_true", help="pretrained or not")
    parser.add_argument('--premodel_path', default=None, help="pretrained model path")
    parser.add_argument('--save_path', default='./model_para/')
    args = parser.parse_args()
    train_total(args)

    
