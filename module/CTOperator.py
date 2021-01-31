import cupy
from cupy.cuda import runtime
import numpy as np
from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAarray, ResourceDescriptor, TextureDescriptor,TextureReference)
# import numba
source_texref = r'''

#define PI (3.14159265f)
extern "C"{

texture<float, 1, cudaReadModeElementType> texFP;
texture<float, 2, cudaReadModeElementType> texImage;

__global__ void fGetFp_kernel(float* d_fFPData,
	int nBins,
	double fSod,
	double fOdd,
	double fCellSize,
	double fPixelSize,
	double fFovRadius,
	double fCosLambda,
	double fSinLambda,
	int nView,
	double fOffSet,
	double fAngleOfSlope)
{
	int nB = blockIdx.x * blockDim.x + threadIdx.x;

	if (nB<nBins)
	{
		float fCosAngle = cosf(fAngleOfSlope + PI/2.0f);
		float fSinAngle = sinf(fAngleOfSlope + PI/2.0f);

		// 探测器端点坐标
		float fPointx = -fOdd + (fOffSet-float(nBins)/2.0f*fCellSize + (float(nB)+0.5f)*fCellSize)*fCosAngle;
		float fPointy =         (fOffSet-float(nBins)/2.0f*fCellSize + (float(nB)+0.5f)*fCellSize)*fSinAngle;

		//射线源坐标
		float fSx = fSod;
		float fSy = 0.0f;

		float fVecx = fPointx - fSx;
		float fVecy = fPointy - fSy;
		float fVecLength = sqrt(fVecx*fVecx + fVecy*fVecy);

		fVecx = fVecx / fVecLength;
		fVecy = fVecy / fVecLength;

		float fSamplePtx = fSx + (fSod - fFovRadius) * fVecx;
		float fSamplePty = fSy + (fSod - fFovRadius) * fVecy;

		float fProjectionValue = 0.0f;
		int nNumOfStep = ceil(2.0f*fFovRadius/(0.5f*fPixelSize));
		for(int i = 0; i < nNumOfStep; ++i)
		{
			fSamplePtx += fVecx * fPixelSize*0.5f;
			fSamplePty += fVecy * fPixelSize*0.5f;

			//采样点做旋转变换
			float fRSamplePtx = fSamplePtx*fCosLambda + fSamplePty*(-fSinLambda);
			float fRSamplePty = fSamplePtx*fSinLambda + fSamplePty*fCosLambda;

			//几何坐标转换为纹理坐标
			float fAddressx = (fRSamplePtx + fFovRadius)/fPixelSize;
			float fAddressy = (fRSamplePty + fFovRadius)/fPixelSize;

			fProjectionValue += tex2D(texImage, fAddressx, fAddressy);
		}
		fProjectionValue *= fPixelSize * 0.5f;

		float fFactor=2.0f*fFovRadius;
		d_fFPData[nB+nBins*nView] = fProjectionValue;	
	}
	__syncthreads();
}

__global__ void fGetResiduals(float* d_fResidualsData,
	float* d_fFPData,
	int nBins,
	double fSod,
	double fOdd,
	double fCellSize,
	double fPixelSize,
	double fFovRadius,
	double fCosLambda,
	double fSinLambda,
	int nView,
	double fOffSet,
	double fAngleOfSlope)
{
	int nB = blockIdx.x * blockDim.x + threadIdx.x;

	if (nB<nBins)
	{
		float fCosAngle = cosf(fAngleOfSlope + PI/2.0f);
		float fSinAngle = sinf(fAngleOfSlope + PI/2.0f);

		// 探测器端点坐标
		float fPointx = -fOdd + (fOffSet-float(nBins)/2.0f*fCellSize + (float(nB)+0.5f)*fCellSize)*fCosAngle;
		float fPointy =         (fOffSet-float(nBins)/2.0f*fCellSize + (float(nB)+0.5f)*fCellSize)*fSinAngle;

		//射线源坐标
		float fSx = fSod;
		float fSy = 0.0f;

		float fVecx = fPointx - fSx;
		float fVecy = fPointy - fSy;
		float fVecLength = sqrt(fVecx*fVecx + fVecy*fVecy);

		fVecx = fVecx / fVecLength;
		fVecy = fVecy / fVecLength;

		float fSamplePtx = fSx + (fSod - fFovRadius) * fVecx;
		float fSamplePty = fSy + (fSod - fFovRadius) * fVecy;

		float fProjectionValue = 0.0f;
		int nNumOfStep = ceil(2.0f*fFovRadius/(0.5f*fPixelSize));
		for(int i = 0; i < nNumOfStep; ++i)
		{
			fSamplePtx += fVecx * fPixelSize*0.5f;
			fSamplePty += fVecy * fPixelSize*0.5f;

			//采样点做旋转变换
			float fRSamplePtx = fSamplePtx*fCosLambda + fSamplePty*(-fSinLambda);
			float fRSamplePty = fSamplePtx*fSinLambda + fSamplePty*fCosLambda;

			//几何坐标转换为纹理坐标
			float fAddressx = (fRSamplePtx + fFovRadius)/fPixelSize;
			float fAddressy = (fRSamplePty + fFovRadius)/fPixelSize;

			fProjectionValue += tex2D(texImage, fAddressx, fAddressy);
		}
		fProjectionValue *= fPixelSize * 0.5f;

		float fFactor=2.0f*fFovRadius;
		d_fResidualsData[nB] = (d_fFPData[nB+nBins*nView]-fProjectionValue)/fFactor;	
	}
	__syncthreads();
}
__global__ void AssignResidualError_kernel(float* d_fImageData,
	int nWidth,
	int nHeight,
	int nBins,
	double fSod,
	double fOdd,
	double fCellSize,
	double fPixelSize,
	double fCosLambda,
	double fSinLambda,
	double fOffSet,
	double fAngleOfSlope,
	double relax_factor)
{
	unsigned int  w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int  h = blockIdx.y * blockDim.x + threadIdx.y;
	if (h<nHeight && w<nWidth)
	{
		//由grid中的索引转换到系统坐标系下的坐标
		float fPointx = -float(nWidth)/2.0f  *fPixelSize+0.5f *fPixelSize+float(w)*fPixelSize;
		float fPointy = -float(nHeight)/2.0f *fPixelSize+0.5f *fPixelSize+float(h)*fPixelSize;

		float fRX1 =  fPointx*fCosLambda+fPointy*fSinLambda;
		float fRX2 = -fPointx*fSinLambda+fPointy*fCosLambda;

		float fCosAngle = cosf(fAngleOfSlope + PI/2.0f);
		float fSinAngle = sinf(fAngleOfSlope + PI/2.0f);

		float fIndexx = -fOffSet + (fSod + fOdd)*fRX2/(fSinAngle*(fSod - fRX1) + fCosAngle*fRX2);  

		fIndexx = fIndexx/fCellSize + float(nBins)/2.0f - 0.5f;
		d_fImageData[h*nWidth+w] += relax_factor*tex1D(texFP, fIndexx + 0.5f);

		float fTempR = (w - float(nWidth/2.0f) - 0.5f)*(w - float(nWidth)/2.0f - 0.5f) + (h - float(nHeight)/2.0f - 0.5f)*(h - float(nHeight)/2.0f - 0.5f);
		fTempR = sqrt(fTempR);
		//将负值和视野半径之外的值设为零
		if ((d_fImageData[h*nWidth +w]<0.0f) || fTempR > float(nWidth/2.0f))
			d_fImageData[h*nWidth+w] = 0.0f;
	}
	__syncthreads();
}


}
'''

# @numba.jit
def SART2D(p, sp, order, x0):
    # x0 = xinit.copy()
    block1D = (8, 1)
    grid1D = ((sp['nBins'] + block1D[0] - 1) // block1D[0], 1)
    block2D = (8, 8)
    grid2D = ((sp['nWidth'] + block2D[0] - 1) // block2D[0],
              (sp['nHeight'] + block2D[1] - 1) // block2D[1])
    mod = cupy.RawModule(code=source_texref)
    fGetResiduals = mod.get_function('fGetResiduals')
    AssignResidualError = mod.get_function('AssignResidualError_kernel')

    channelDescImg = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayImg = CUDAarray(channelDescImg, sp['nWidth'], sp['nHeight'])
    resourceDescImg = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayImg)
    address_modeImg = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDescImg = TextureDescriptor(address_modeImg, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)

    # 1D texture
    channelDesc1D = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArray1D = CUDAarray(channelDesc1D, sp['nBins'])
    resourceDesc1D = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArray1D)
    address_mode1D = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDesc1D = TextureDescriptor(address_mode1D, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
    d_fResidualsData = cupy.zeros(sp['nBins'], cupy.float32)

    for v in range(sp['nViews']):
        # print('{}\n'.format(v))
        nView = order[v]
        fLambda = sp['fRotateDir'] * 2.0 * np.pi / float(sp['nNumAngle']) * float(nView + sp['nStartAngle'])
        fCosLambda = np.cos(fLambda)
        fSinLambda = np.sin(fLambda)
        cuArrayImg.copy_from(x0)
        TextureReference(mod.get_texref('texImage'), resourceDescImg, texDescImg)
        getErrArgs = (d_fResidualsData, p, sp['nBins'], sp['fSod'], sp['fOdd'], sp['fCellSize'], sp['fPixelSize'],
                      sp['fFovRadius'], fCosLambda, fSinLambda, nView, sp['fOffSet'], sp['fAngleOfSlope'])
        fGetResiduals(grid1D, block1D, getErrArgs)
        cuArray1D.copy_from(d_fResidualsData)
        TextureReference(mod.get_texref('texFP'), resourceDesc1D, texDesc1D)
        AssignResidualErrorArgs = (
        x0, sp['nWidth'], sp['nHeight'], sp['nBins'], sp['fSod'], sp['fOdd'], sp['fCellSize'],
        sp['fPixelSize'], fCosLambda, fSinLambda, sp['fOffSet'], sp['fAngleOfSlope'], sp['relax_factor'])
        AssignResidualError(grid2D, block2D, AssignResidualErrorArgs)
    return x0
# @numba.jit
def SART2DBackWard(grad,order,sp):
    grad_ = grad.copy()
    block1D = (8, 1)
    grid1D = ((sp['nBins'] + block1D[0] - 1) // block1D[0], 1)
    block2D = (8, 8)
    grid2D = ((sp['nWidth'] + block2D[0] - 1) // block2D[0],
              (sp['nHeight'] + block2D[1] - 1) // block2D[1])
    mod = cupy.RawModule(code=source_texref)
    AssignResidualError = mod.get_function('AssignResidualError_kernel')
    FpKernel = mod.get_function('fGetFp_kernel')
    # 2D texture
    channelDescImg = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayImg = CUDAarray(channelDescImg, sp['nWidth'], sp['nHeight'])
    resourceDescImg = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayImg)
    address_modeImg = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDescImg = TextureDescriptor(address_modeImg, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
    # 1D texture
    channelDesc1D = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArray1D = CUDAarray(channelDesc1D, sp['nBins'])
    resourceDesc1D = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArray1D)
    address_mode1D = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDesc1D = TextureDescriptor(address_mode1D, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
    d_fOneProj = cupy.zeros(sp['nBins'],cupy.float32)
    for v in range(sp['nViews']):
        nView = order[v]
        fLambda = sp['fRotateDir'] * 2.0 * np.pi / float(sp['nNumAngle']) * float(nView + sp['nStartAngle'])
        fCosLambda = np.cos(fLambda)
        fSinLambda = np.sin(fLambda)
        # A*x
        cuArrayImg.copy_from(grad)
        TextureReference(mod.get_texref('texImage'), resourceDescImg, texDescImg)
        args = (d_fOneProj, sp['nBins'], sp['fSod'], sp['fOdd'], sp['fCellSize'], sp['fPixelSize'], sp['fFovRadius'],
                fCosLambda, fSinLambda, nView, sp['fOffSet'], sp['fAngleOfSlope'])
        FpKernel(grid1D, block1D, args)
        # AT*A*x
        cuArray1D.copy_from(d_fOneProj)
        TextureReference(mod.get_texref('texFP'), resourceDesc1D, texDesc1D)
        AssignResidualErrorArgs = (grad, sp['nWidth'], sp['nHeight'], sp['nBins'], sp['fSod'], sp['fOdd'], sp['fCellSize'],
        sp['fPixelSize'], fCosLambda, fSinLambda, sp['fOffSet'], sp['fAngleOfSlope'], sp['relax_factor'])
        AssignResidualError(grid2D, block2D, AssignResidualErrorArgs)
        grad = grad_ - sp['relax_factor'] * grad
    return grad
