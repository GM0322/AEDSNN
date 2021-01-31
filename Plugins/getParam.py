import numpy as np

def getFovRadius(fSod, fOdd, fCellsize, nBins, fOffSet, fAngleOfSlope):

    fCosAngle = np.cos(fAngleOfSlope + np.pi / 2.0)
    fSinAngle = np.sin(fAngleOfSlope + np.pi / 2.0)

    fPointAx = fOdd + (fOffSet + float(nBins) / 2.0 * fCellsize) * fCosAngle
    fPointAy = (fOffSet + float(nBins) / 2.0 * fCellsize) * fSinAngle

    fPointBx = fOdd + (-fOffSet + float(nBins) / 2.0 * fCellsize) * (-fCosAngle)
    fPointBy = (-fOffSet + float(nBins) / 2.0 * fCellsize) * (-fSinAngle)

    fSourcex = -fSod
    fSourcey = 0.0

    fDisSA = np.sqrt((fPointAx - fSourcex) * (fPointAx - fSourcex) + (fPointAy - fSourcey) * (fPointAy - fSourcey))
    fDisSB = np.sqrt((fPointBx - fSourcex) * (fPointBx - fSourcex) + (fPointBy - fSourcey) * (fPointBy - fSourcey))

    fRadius = min(fSod * abs(fPointAy) / fDisSA, fSod * abs(fPointBy) / fDisSB)

    return fRadius

def getPixelSize(fFovRadius, nWidth,nHeight):
    pixelSize = 2.0 * fFovRadius / float(max(nWidth, nHeight))
    return pixelSize

