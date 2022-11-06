import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
import numpy as np
import os
import nibabel as nib
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import c_transforms
from mindspore.dataset.transforms import py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from etc.model import AttU_Net
import SimpleITK as sitk

def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


def resize_image_itkwithsize(itkimage, newSize, originSize, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSize:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    originSize = np.array(originSize)
    newSize = np.array(newSize)
    factor = originSize / newSize
    newSpacing = factor * originSpcaing
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


class DatasetGenerator:
    def __init__(self, root_dir, img_dir, label_dir, mode):
        self.root_dir = root_dir
        self.case_dir = [os.path.join(self.root_dir, case) for case in os.listdir(self.root_dir)]  
        self.img_dir = [os.path.join(case_dir, 'imaging.nii.gz') for case_dir in self.case_dir]
        self.seg_dir = [os.path.join(case_dir, 'segmentation.nii.gz') for case_dir in self.case_dir]
        self.fix_size = [128, 512, 512]
        self.img = []
        self.seg = []
        
        for (i, case) in enumerate(self.case_dir):
            #             imgs = nib.load(self.img_dir[i]).get_fdata()
#             segs = nib.load(self.seg_dir[i]).get_fdata()
#             for idx in range(segs.shape[0]):
#                 if len(np.unique(segs[idx]))>1:
#                     print(f"item{i}, patch{idx}")
#                     self.img.append(imgs[idx])
#                     self.seg.append(segs[idx])
            if mode=='t':
#                 src = load_itkfilewithtrucation(self.img_dir[i], 300, -200)
#                 seg = sitk.ReadImage(self.seg_dir[i], sitk.sitkUInt16)
#                 originSize = seg.GetSize()
#                 thickspacing = src.GetSpacing()[0]
#                 widthspacing = src.GetSpacing()[1]
#                 _, seg = resize_image_itkwithsize(seg, newSize=self.fix_size,
#                                               originSize=originSize,
#                                               originSpcaing=[thickspacing, widthspacing, widthspacing],
#                                               resamplemethod=sitk.sitkNearestNeighbor)
#                 _, src = resize_image_itkwithsize(src, newSize=self.fix_size,
#                                               originSize=originSize,
#                                               originSpcaing=[thickspacing, widthspacing, widthspacing],
#                                               resamplemethod=sitk.sitkLinear)
#                 segimg = sitk.GetArrayFromImage(seg)
#                 segimg = np.swapaxes(segimg, 0, 2).astype('float32')

#                 srcimg = sitk.GetArrayFromImage(src)

#                 srcimg = np.swapaxes(srcimg, 0, 2).astype('float32')
#                 print(srcimg.shape, segimg.shape)
#                 for idx in range(64):               
#                     self.img.append(srcimg[idx])
#                     self.seg.append(segimg[idx])
                srcimg = nib.load(self.img_dir[i]).get_fdata()
                segimg = nib.load(self.seg_dir[i]).get_fdata()
                srcimg[srcimg < -79.0] = -79.0
                srcimg[srcimg > 304.0] = 304.0
                srcimg = (srcimg - 101) / float(76.9)
                num1=0
                num2=0
                num3=0
                if srcimg.shape[2] == 796:
                    continue
#                 if i > 1:
#                     break
                for idx in range(srcimg.shape[0]):
                    if (len(np.unique(segimg[idx]))==3):
#                         print(idx)
#                         print(f"sample2:{i}, index2:{idx}")
                        self.img.append(srcimg[idx])
                        self.seg.append(segimg[idx])
                        num2+=1
                        num1+=1
#                         if num2>32:
#                             break
                    elif (len(np.unique(segimg[idx]))==2):
#                         print(f"sample1:{i}, index1:{idx}")
                        self.img.append(srcimg[idx])
                        self.seg.append(segimg[idx])
                        num1+=1
                        if num1>32:
                            break
#                     else:
#                         if num3 > 64:
#                             continue
#                         self.img.append(srcimg[idx])
#                         self.seg.append(segimg[idx])
#                         num3+=1
                    
                print(f"sample{i}, 1:{num1}, 2:{num2}, 0:{num3}")
                    
                
                    
            else:
#                 if i > 1:
#                     break
                srcimg = nib.load(self.img_dir[i]).get_fdata()
                segimg = nib.load(self.seg_dir[i]).get_fdata()
                srcimg[srcimg < -79.0] = -79.0
                srcimg[srcimg > 304.0] = 304.0
                srcimg = (srcimg - 101) / float(76.9)
                for idx in range(srcimg.shape[0]):
                    if(len(np.unique(segimg[idx]))>1):
                        self.img.append(srcimg[idx])
                        self.seg.append(segimg[idx])
#                 print(idx)
  
    def __getitem__(self, index):
        r_img = self.img[index]
        r_seg = self.seg[index]
        
#         r_img = (r_img - r_img.max())/(r_img.max()-r_img.min())
        
        r_img = np.expand_dims(r_img, axis=2)
        r_seg = np.expand_dims(r_seg, axis=2)
        r_img = np.transpose(r_img, (2, 0, 1)).astype('float32')
        r_seg = np.transpose(r_seg, (2, 0, 1)).astype('float32')
        r_seg_new = np.zeros([3, 512, 512])
        for i in range(3):
            r_seg_new[i] = np.where(r_seg==i, 1, 0)
#         print(r_img.shape, r_seg.shape)
    #         r_img = np.clip(r_img, 0, 255).astype('uint8')
        return r_img, r_seg_new
    def __len__(self):
        return len(self.img)

    
