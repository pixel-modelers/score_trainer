
from simtbx.diffBragg import utils
from simtbx.nanoBragg.sim_data import SimData
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD


def run_simdata(SIM, pfs, ucell_p, ncells_p, rot_p, spectrum=None, eta_p=None, G=1,
        diffuse_gamma=None, diffuse_sigma=None):
    """
    :param SIM: sim_data.SimData instance returned by get_SIM method
    :param pfs: pfs vector, flex.size_t , specifies which pixels to simulate
        length is 3xN where N is the number of pixels . See method get_pfs
    :param ucell_p: 6-tuple of unit cell parameters (a,b,c,alpha,beta,gamma) ; Angstrom / degrees
    :param ncells_p: 3-tuple of mosaic block size params, specifies extent of
        mosaic block along unit cell vectors, e.g. 10,5,5, means mosaic block
        is 10 unit cells along unit-cell a direction, and 5 unit cells along b,c
    :param rot_p: 3-tuple of rotational offsets in radians along lab axes.
            The meaning of these numbers is illuminated in the following code
            rot_p[0] is a rotation about (-1,0,0) in radians, and so forth
            THe final perturbation is defined as M=RotX*RotY*RotZ
            >> from scitbx.matrix import sqr col
            >> x = col((-1, 0, 0))
            >> y = col((0, -1, 0))
            >> z = col((0, 0, -1))
            >> RX = x.axis_and_angle_as_r3_rotation_matrix(rot_p[0], deg=False)
            >> RY = y.axis_and_angle_as_r3_rotation_matrix(rot_p[1], deg=False)
            >> RZ = z.axis_and_angle_as_r3_rotation_matrix(rot_p[2], deg=False)
            >> M = RX * RY * RZ
            >> Umat = sqr(dxbtbx_cryst.get_U())
            >> rotated_Umat = M*Umat
            >> dxtbx_cryst.set_U(rotated_Umat)
    :param spectrum: spectrum object list of 2-tuples. each 2-tuple is (wavelength, intensity)
    :param eta_p: float value of rotational mosaicity parameter eta
    :param G: scale factor for bragg peaks (e.g. total crystal volume)
    :param diffuse_gamma: 3-tuple of diffuse scattering param gammma
    :param diffuse_sigma: 3-tuple of diffuse scattering param sigma
    :return: flex array of simulated pixel values (same length as len(lfs) / 3)
    """
    SIM.D.verbose = 0
    if spectrum is not None:
        SIM.beam.spectrum = spectrum
        SIM.D.xray_beams = SIM.beam.xray_beams

    ucell_man = utils.manager_from_params(ucell_p)
    Bmatrix = ucell_man.B_recipspace   # same as dxtbx crystal .get_B() return value
    SIM.D.Bmatrix = Bmatrix

    if eta_p is not None and SIM.crystal.n_mos_domains > 1:
        # NOTE for eta_p we need to also update how we create the SIM instance
        # see hopper_utils for examples (see methods "SimulatorFromExperiment" and "model")
        # FIXME not sure this is right yet
        SIM.update_umats_for_refinement(eta_p)
        SIM.crystal.mos_spread_deg = eta_p

    # Mosaic block
    #    diffuse_params_lookup = {}
    SIM.D.set_ncells_values(ncells_p)

    # diffuse signals
    if diffuse_gamma is not None:
        assert diffuse_sigma is not None
        SIM.D.use_diffuse = True
        SIM.D.diffuse_gamma = diffuse_gamma # gamma has Angstrom units
        SIM.D.diffuse_sigma = diffuse_sigma
    else:
        SIM.D.use_diffuse = False

    SIM.D.raw_pixels_roi *= 0

    ## update parameters:
    # TODO: if not refining Umat, assert these are 0 , and dont set them here
    rotX,rotY,rotZ = rot_p
    SIM.D.set_value(0, rotX)
    SIM.D.set_value(1, rotY)
    SIM.D.set_value(2, rotZ)

    npix = int(len(pfs)/3)
    SIM.D.verbose = 1
    SIM.D.add_diffBragg_spots(pfs)
    SIM.D.verbose = 0

    pix = G*SIM.D.raw_pixels_roi[:npix]

    return pix


def setup_sim(detector_model, crystal_model, wave=1.1, Ncells_abc=(10,15,20), shape="square",
              dev_per_node=4, thick_mm=1, beam_size_mm=0.01, flux=1e10):
    if crystal_model is None:
        S = SimData(use_default_crystal=True)
    else:
        S = SimData(use_default_crystal=False)
        S.crystal = crystal_model

    S.crystal.xtal_shape = shape
    S.crystal.thick_mm = thick_mm
    S.crystal.mos_spread_deg = 0.1
    S.crystal.Ncells_abc = Ncells_abc
    S.crystal.n_mos_domains = 200
    S.beam.spectrum = [(wave, flux)]
    S.beam.size_mm = beam_size_mm
    S.detector = detector_model
    S.instantiate_diffBragg(default_F=0)
    S.D.nopolar = True
    S.D.device_Id = COMM.rank % dev_per_node
    S.D.oversample = 4
    S.D.spot_scale = 10

    S.D.add_diffBragg_spots()
    diff_img = S.D.raw_pixels.as_numpy_array()
    return S, diff_img
