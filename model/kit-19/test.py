from mindspore.train.serialization import load_checkpoint, load_param_into_net
from etc.model import AttU_Net
import os
from mindspore import ops
import nibabel as nib
from mindspore import Tensor
import mindspore as ms
import numpy as np
import SimpleITK as sitk
import argparse

model_path = "/home/ma-user/work/project/best_model/11.1_2/unet_19_0.84.ckpt"
test_path = "/home/ma-user/work/project/test/"
save_path = "/home/ma-user/work/project/test_save_path/"

def test(args):
    train_net = AttU_Net()
    param_dict = load_checkpoint(args.model_path)
    load_param_into_net(train_net, param_dict)
    for sample in os.listdir(args.input_path):
        print(sample.split('.')[0])
        sample_total = nib.load(os.path.join(args.input_path, sample)).get_fdata()
        final_ans = np.zeros_like(sample_total)
        sample_total[sample_total < -79.0] = -79.0
        sample_total[sample_total > 304.0] = 304.0
        sample_total = (sample_total - 101) / float(76.9)
        print(final_ans.shape)
        for i in range(sample_total.shape[0]):
            frame = np.expand_dims(sample_total[i], axis=0)
            frame = np.expand_dims(frame, axis=0)
#             print(frame.shape)
            res = train_net(Tensor.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)))
            res2 = res.asnumpy()
            res3 = np.argmax(res2, axis=1)
            print(np.unique(res3))
            final_ans[i] =res3
        final_ans=final_ans.swapaxes(0, 2)
        out = sitk.GetImageFromArray(final_ans)
        
        sitk.WriteImage(out, os.path.join(args.output_path, sample.split('.')[0]+'.nii.gz'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',default="/home/ma-user/work/project/test/", type=str,help ='input files')
    parser.add_argument('--output_path',default = "/home/ma-user/work/project/test_save_path/", type=str,help='result dir.')
    parser.add_argument('--model_path', default = '/home/ma-user/work/project/best_model/11.1_2/unet_19_0.84.ckpt', type=str, help='model dir')
    args = parser.parse_args()
    
    
    test(args)