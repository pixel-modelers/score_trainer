
import os

import numpy as np
from scipy.spatial.transform import Rotation

from iotbx import pdb
from dxtbx.model.crystal import CrystalFactory
from simtbx.nanoBragg import nanoBragg_crystal
from simtbx.diffBragg.utils import get_complex_fcalc_from_pdb


def dxtbx_crystal_from_symmetry(sym):
    """sym: xtal symmetry object"""
    hall = sym.space_group_info().type().hall_symbol()
    a,b,c =  map(tuple, np.reshape(sym.unit_cell().orthogonalization_matrix(),(3,3)).T)
    cryst_descr = {'__id__': 'crystal',
                   'real_space_a': a,
                   'real_space_b': b,
                   'real_space_c': c,
                   'space_group_hall_symbol': hall}
    return CrystalFactory.from_dict(cryst_descr)


def load_crystal(pdb_id, dmin=1.5, rot=True):
    pdb_file = "%s.pdb" %pdb_id
    if not os.path.exists(pdb_file):
        os.system("iotbx.fetch_pdb %s" % pdb_id)
    C = nanoBragg_crystal.NBcrystal(init_defaults=False)
    P = pdb.input(pdb_file)
    sym = P.crystal_symmetry()
    dxtbx_crystal = dxtbx_crystal_from_symmetry(sym)
    if rot:
        Umat = dxtbx_crystal.get_U()
        rot_mat = Rotation.random(1).as_matrix()[0]
        Umat = np.dot(np.reshape(rot_mat, (3,3)), np.reshape(Umat,(3,3)))
        dxtbx_crystal.set_U(tuple(Umat.ravel()))

    C.dxtbx_crystal = dxtbx_crystal
    ma = get_complex_fcalc_from_pdb(pdb_file, dmin=dmin)
    ma = ma.as_amplitude_array()
    C.miller_array = ma
    C.symbol = ma.space_group_info().type().lookup_symbol()
    return C
