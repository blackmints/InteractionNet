#
# Copyright (C) 2006-2016 Greg Landrum
#  All Rights Reserved
#
from analysis.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.six import iteritems
import numpy as np


def MolToMPL(mol, size=(300, 300), kekulize=True, wedgeBonds=True, fitImage=False, options=None, **kwargs):
    if not mol:
        raise ValueError('Null molecule provided')
    from analysis.Draw.mplCanvas import Canvas
    canvas = Canvas(size)
    if options is None:
        options = DrawingOptions()
        options.bgColor = None
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)
    omol = mol
    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)

    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)

    drawer.AddMol(mol, **kwargs)
    omol._atomPs = drawer.atomPs[mol]
    for k, v in iteritems(omol._atomPs):
        omol._atomPs[k] = canvas.rescalePt(v)
    canvas._figure.set_size_inches(float(size[0]) / 100, float(size[1]) / 100)
    return canvas._figure


def calcAtomGaussians(mol, a=0.03, step=0.02, weights=None):
    x = np.arange(0, 1, step)
    y = np.arange(0, 1, step)
    X, Y = np.meshgrid(x, y)
    if weights is None:
        weights = [1.] * mol.GetNumAtoms()
    Z = bivariate_normal(X, Y, a, a, mol._atomPs[0][0], mol._atomPs[0][1]) * weights[0]
    for i in range(1, mol.GetNumAtoms()):
        Zp = bivariate_normal(X, Y, a, a, mol._atomPs[i][0], mol._atomPs[i][1])
        Z += Zp * weights[i]
    return X, Y, Z


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X - mux
    Ymu = Y - muy

    rho = sigmaxy / (sigmax * sigmay)
    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / denom
