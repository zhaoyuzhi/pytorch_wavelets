import torch
import torch.nn as nn
from numpy import ndarray

from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
from pytorch_wavelets.dtcwt import transform_funcs as tf


class DTCWTForward(nn.Module):
    """ Performs a 2d DTCWT Forward decomposition of an image

    Args:
        C (int): Number of channels in input
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition
        skip_hps (bool): True if the decomposition should not calculate the
            first scale's highpass coefficients (these are often noise). Will
            speed up computation significantly.

    Shape:
        - Input x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Output yh: :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.


    Attributes:
        h0o (tensor): Non learnable lowpass biorthogonal analysis filter
        h1o (tensor): Non learnable highpass biorthogonal analysis filter
        h0a (tensor): Non learnable lowpass qshift tree a analysis filter
        h1a (tensor): Non learnable highpass qshift tree a analysis filter
        h0b (tensor): Non learnable lowpass qshift tree b analysis filter
        h1b (tensor): Non learnable highpass qshift tree b analysis filter
    """
    def __init__(self, C, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=False):
        super().__init__()
        self.C = C
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.J = J
        h0o, _, h1o, _ = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'xfm{J}'.format(J=J))
        if isinstance(skip_hps, (list, tuple, ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps,] * self.J

    def forward(self, input):
        assert self.C == input.shape[1], "Input channels ({}) don't match " \
            "Initialization channels ({})".format(input.shape[1], self.C)

        coeffs = self.dtcwt_func.apply(input, self.h0o, self.h1o, self.h0a,
                                       self.h0b, self.h1a, self.h1b,
                                       self.skip_hps)
        # Return in the format: (yl, yh)
        return coeffs[0], coeffs[1:]


class DTCWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        C (int): Number of channels in input
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition
        skip_hps (bool): True if the inverse method should not look at the first
            scale's highpass coefficients (these are often noise). Will speed up
            computation significantly.

    Shape:
        - Input yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Input yh: :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.
        - Output y: :math:`(N, C_{in}, H_{in}, W_{in})`

    Attributes:
        g0o (tensor): Non learnable lowpass biorthogonal synthesis filter
        g1o (tensor): Non learnable highpass biorthogonal synthesis filter
        g0a (tensor): Non learnable lowpass qshift tree a synthesis filter
        g1a (tensor): Non learnable highpass qshift tree a synthesis filter
        g0b (tensor): Non learnable lowpass qshift tree b synthesis filter
        g1b (tensor): Non learnable highpass qshift tree b synthesis filter
    """

    def __init__(self, C, biort='near_sym_a', qshift='qshift_a', J=3):
        super().__init__()
        self.C = C
        self.biort = biort
        self.qshift = qshift
        self.J = J
        _, g0o, _, g1o = _biort(biort)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, 1), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, 1), False)
        _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, 1), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, 1), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, 1), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, 1), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'ifm{J}'.format(J=J))

    def forward(self, x):
        yl, yh = x
        assert self.C == yl.shape[1], "Input channels ({}) don't match " \
            "Initialization channels ({})".format(yl.shape[1], self.C)
        for s in yh:
            if s is not None and s.shape != torch.Size([0]):
                assert s.shape[2] == 6, "Inverse transform must have input " \
                    "with 6 orientations"
                assert s.shape[-1] == 2, "Inputs must be complex with real " \
                    "and imaginary parts in the last dimension"
                assert len(s.shape) == 6, "Bandpass inputs must have shape " \
                    "(n, c, 6, h, w, 2)"

        return self.dtcwt_func.apply(yl, *yh, self.g0o, self.g1o, self.g0a,
                                     self.g0b, self.g1a, self.g1b)
