from . import getParam
import numpy as np
import argparse

__all__ = ['getScanParam','getArgparse']

def getScanParam():
    ScanParam = { 'nViews' : 280,
        'nBins' : 600,
        'nWidth' : 512,
        'nHeight' : 512,
        'nStartAngle' : 40,
        'nNumAngle' : 720,
        'fSod' : 570.0,
        'fOdd' : 445.0,
        'fCellSize' : 0.127,
        'fOffSet' : 0.0,
        'fAngleOfSlope' : 0.0,
        'fRotateDir' : -1.0,
        'relax_factor':0.4,
    }
    fFovRadius = getParam.getFovRadius(ScanParam['fSod'], ScanParam['fOdd'], ScanParam['fCellSize'],
                                       ScanParam['nBins'], ScanParam['fOffSet'], ScanParam['fAngleOfSlope'])
    ScanParam['fFovRadius'] = float(fFovRadius)
    ScanParam['fPixelSize'] = float(getParam.getPixelSize(fFovRadius, ScanParam['nWidth'], ScanParam['nHeight']))
    np.random.seed(0)
    ScanParam['order'] = np.random.permutation(ScanParam['nViews'])
    return ScanParam

def getArgparse():
    args = argparse.ArgumentParser()

    # args.add_argument('--is_dataset_finsh',type=bool, default=False)
    args.add_argument('--is_noise',type=str, default='')
    args.add_argument('--input_path',type=str,default=r'D:\CT\LEARN\AEDS\mayo\train\input')
    args.add_argument('--label_path',type=str,default=r'D:\CT\LEARN\AEDS\mayo\train\label')
    args.add_argument('--proj_path',type=str,default=r'D:\CT\LEARN\AEDS\mayo\train\proj')
    args.add_argument('--test_path', type=str, default=r'D:\CT\LEARN\AEDS\mayo\train\input')
    args.add_argument('--batch_size',type=int, default=1)

    args.add_argument('--model',type=str,default='attention_unet')  #unet')  #
    args.add_argument('--k1',type=int,default=5)
    args.add_argument('--k2',type=int,default=3)
    args.add_argument('--n',type=int,default=3)
    args.add_argument('--lr',type=int,default=1e-5)
    args.add_argument('--epoch',type=int,default=100)

    return args.parse_args()
