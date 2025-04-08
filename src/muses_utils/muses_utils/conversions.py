from enum import Enum
import numpy as np


class BtConv(Enum):
    TO_BT = 'to_bt'
    TO_RAD = 'to_rad'


def bt(frequency, radiance_or_bt, to=BtConv.TO_BT):
    # IDL_LEGACY_NOTE: This function bt  is the same as bt in TOOLS/bt.pro file.

    # from John Worden
    # .com /project/SOT/idl/Archive/STABLE/src_ms-2018-02-27/TOOLS/bt.pro
    # to = 0: converts from radiance (W/cm2/cm-1/sr) to BT (erg/sec/cm2/cm-1/sr)
    # to = 1: converts from BT to radiance
    # I am pretty sure the units on BT though are K, like K/cm-1/sr

    to = BtConv(to)

    # Check frequency is scalar.
    if np.isscalar(frequency):
        save_value = frequency
        frequency = np.ndarray(shape=(1), dtype=np.float32)
        frequency[0] = save_value

    # Check if var2 is scalar.
    if np.isscalar(radiance_or_bt):
        save_value = radiance_or_bt
        radiance_or_bt = np.ndarray(shape=(1), dtype=np.float32)
        radiance_or_bt[0] = save_value

    # Convert variable(s) to array so we can work on it.
    frequency = np.asarray(frequency, dtype=np.float32)

    # note use wikipedia definition and consider frequency = c * wavenumber, 
    # and additionally, consider TES radiance is W/cm2/cm-1/sr and convert to (erg/sec)/cm2/(1/sec = freq)/sr

    PLANCK = 6.626176E-27
    CLIGHT = 2.99792458E+10
    BOLTZ = 1.380662E-16
    RADCN1 = 2. * PLANCK * CLIGHT * CLIGHT*1.E-7
    RADCN2 = PLANCK * CLIGHT / BOLTZ

    if to == BtConv.TO_BT:
        rad = radiance_or_bt
        bt = RADCN2 * frequency / np.log(1 + (RADCN1 * frequency**3 / rad))
        ret = bt
    elif to == BtConv.TO_RAD:
        temperature = radiance_or_bt
        fbeta = RADCN2 / temperature
        exp_term = np.exp(frequency * fbeta)
        planck = RADCN1 * (frequency**3) / (exp_term-1.)
        ret = planck
    else:
        raise NotImplementedError(f'to = {to}')

    return ret 