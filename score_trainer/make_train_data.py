from __future__ import print_function, division
from simtbx.nanoBragg import nanoBragg
import h5py
import numpy as np
from mpi4py import MPI
from simtbx.diffBragg import hopper_utils
from scitbx.array_family import flex
from dxtbx.model import BeamFactory
from simtbx.diffBragg import phil
from libtbx.phil import parse
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from simtbx.nanoBragg.sim_data import Amatrix_dials2nanoBragg

COMM = MPI.COMM_WORLD
from simtbx.nanoBragg import shapetype
from simtbx.diffBragg import utils
import pylab as plt
from simtbx.nanoBragg import utils as nb_utils
from simtbx.nanoBragg.sim_data import SimData
from score_trainer.crystal import load_crystal
from score_trainer.run_simdata import run_simdata, setup_sim
import os


def main():
    from argparse import ArgumentParser
    pa = ArgumentParser()
    pa.add_argument("odir", type=str, help="hdf5 files will be written here containing training data")
    pa.add_argument("--pdb", default="4bs7", type=str, help="4 letter PDB code, will be downloaded if file doesnt exist in current folder")
    pa.add_argument("--ntrial", default=1000, type=int, help="number of images to simulate for training data (default=1000)")
    pa.add_argument("--seed", default=None, type=int)
    pa.add_argument("--dets", nargs="+", default=[1,2], choices=[0,1,2], type=int, help="detector model to use (0 1 or 2). Each has slightly different pixel sizes and other properties")
    pa.add_argument("--cbfs", action="store_true", help="save CBF files (for viewing whole images for debug purposes)")
    pa.add_argument("--plot", action="store_true", help="plot the training data as its being created")
    pa.add_argument("--ndev", type=int, help="number of devices per node (default=1)", default=1)
    args = pa.parse_args()
    import time
    C = load_crystal(args.pdb, dmin=1)
    rand_seed = None
    if COMM.rank==0:
        rand_seed = int(time.time()) if args.seed is None else args.seed
    rand_seed = COMM.bcast(rand_seed)
    np.random.seed(rand_seed+COMM.rank)
    Ntrials = args.ntrial
    #COMPARE_SHAPE = shapetype.Square
    if COMM.rank==0:
        if not os.path.exists(args.odir):
            os.makedirs(args.odir)
    COMM.barrier()

    PLOT = COMM.rank == 0 and args.plot
    if PLOT:
        fix, axs = plt.subplots(nrows=3, ncols=2)
        axs = [ax for sl in axs for ax in sl]
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(np.random.random((10, 10)))

    rank_odir = os.path.join(args.odir, "rank%d" % COMM.rank)
    if not os.path.exists(rank_odir):
        os.makedirs(rank_odir)
    h5_name = os.path.join(args.odir, "training_rank%d.h5" % COMM.rank)
    with h5py.File(h5_name, 'w') as h5:
        dummie_shape = (10,2,10,10)
        training_imgs_dset = h5.create_dataset("images", dtype=np.float32, shape=dummie_shape,
                                               maxshape=(None, 2,10,10), chunks=(1,2,10,10))
        all_train_labels = []

        for i_trial in range(Ntrials):
            if i_trial % COMM.size != COMM.rank:
                continue
            distance = np.random.uniform(50,350)
            det_model = np.random.choice(args.dets)
            if det_model==0:
                pixsize=0.1
                dimension=512,512
            elif det_model==1:
                pixsize=0.172
                dimension=2527,2463
            else:
                pixsize=0.08
                dimension=4096,4096
            ydim, xdim = dimension
            wave = np.random.uniform(1, 1.4)
            thick_mm = 0.01
            beam_size_mm = 0.01
            Nabc = tuple(np.random.uniform(20,100,(3,)).astype(int).astype(np.float64))
            DET = SimData.simple_detector(distance, pixsize, (xdim,ydim))
            S, reference = setup_sim(DET,C, wave=wave, Ncells_abc=Nabc, shape="gauss",
                                     beam_size_mm=beam_size_mm, thick_mm=thick_mm, dev_per_node=args.ndev)
            G = S.determine_spot_scale()
            pfs = hopper_utils.full_img_pfs((1, ydim, xdim))
            # run ground truth simulation:
            ucell_p = C.dxtbx_crystal.get_unit_cell().parameters()
            G = S.determine_spot_scale()
            ncells_p = S.D.Ncells_abc_aniso
            gt_model = run_simdata(S, pfs=pfs, ucell_p=ucell_p, rot_p=(0,0,0), ncells_p=S.D.Ncells_abc_aniso, G=G)

            # run perturbed simulation:
            ncells_p = np.array(ncells_p)
            ncells_p = np.random.uniform(ncells_p*.5, ncells_p*2)

            S.D.xtal_shape = shapetype.Gauss
            #rot_p_scale = np.random.uniform(0.005, 0.01)*np.pi/180
            #rot_p = np.random.uniform(-rot_p_scale, rot_p_scale, 3)
            rot_p = np.random.normal(0, scale=0.01*np.pi/180., size=3)
            ucell_man = utils.manager_from_crystal(C.dxtbx_crystal)
            uc_vars = np.array(ucell_man.variables)
            uc_vars = np.random.normal(uc_vars,scale=0.03)
            ucell_man.variables = uc_vars
            ucell_p = np.array(ucell_man.unit_cell_parameters)
            S.crystal.thick_mm = np.random.uniform(thick_mm*0.5, thick_mm*2)
            G2 = S.determine_spot_scale()
            perturb_model = run_simdata(S, pfs, ucell_p=tuple(ucell_p), rot_p=tuple(rot_p), ncells_p=tuple(ncells_p), G=G2)

            air_name = os.path.join(os.path.dirname(__file__) , "air.stol")
            beam_size_mm = S.beam.size_mm
            total_flux = S.D.flux

            #beam = deepcopy(S.beam.nanoBragg_constructor_beam)
            #beam.set_wavelength(wave)
            beam = BeamFactory.simple(wave)
            water_bkgrnd = nb_utils.sim_background(
                S.detector, beam, [wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
                Fbg_vs_stol=None, sample_thick_mm=2.5, density_gcm3=1, molecular_weight=18)

            air_Fbg, air_stol = np.loadtxt(air_name).T
            air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
            air = nb_utils.sim_background(S.detector, beam, [wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
                                    molecular_weight=14,
                                    sample_thick_mm=5,
                                    Fbg_vs_stol=air_stol, density_gcm3=1.2e-3)

            water_bkgrnd = water_bkgrnd.as_numpy_array()
            air = air.as_numpy_array()
            bg = air + water_bkgrnd

            gt_img = gt_model.as_numpy_array().reshape((ydim, xdim))
            pt_img = perturb_model.as_numpy_array().reshape((ydim,xdim))

            bg_scale = np.random.uniform(0.3, 1.5)
            bg *= bg_scale
            perfect_model = gt_img + bg
            perturb_model = pt_img + bg

            # add some noise
            SIM = nanoBragg(detector=S.detector, beam=beam)
            SIM.beamsize_mm = S.beam.size_mm
            SIM.exposure_s = 1
            SIM.flux = S.D.flux
            SIM.adc_offset_adu = 10
            SIM.detector_psf_kernel_radius_pixels = 0
            SIM.detector_calibration_noice_pct = 3
            SIM.detector_psf_fwhm_mm = 0
            SIM.quantum_gain = 1
            SIM.readout_noise_adu = 1
            SIM.raw_pixels += flex.double((perfect_model).ravel())

            # SAVE CBFs?
            if args.cbfs:
                cbf_name = os.path.join(rank_odir,"best_1_%05d.cbf" % (i_trial+1))
                SIM.to_cbf(cbf_name, cbf_int=True)

            # noise step
            SIM.add_noise()

            # get the noisy data
            raw_data = SIM.raw_pixels.as_numpy_array().reshape((ydim,xdim))
            # get the regions of interest using hopper_utils.DataModeler
            # make the experiment list after adding noise
            SIM.Amatrix = Amatrix_dials2nanoBragg(C.dxtbx_crystal)
            El = SIM.as_explist()
            E = El[0]
            params = parse(phil.philz + phil.hopper_phil).extract()
            Mod = hopper_utils.DataModeler(params)
            refls = utils.refls_from_sims(gt_img[None], DET, beam, thresh=10, filter=gaussian_filter, sigma=3)
            if not Mod.GatherFromExperiment(E, refls):
                print("Failed to gather refls")
                continue

            training_imgs = []
            training_labels = []
            for i_roi, (x1,x2,y1,y2) in enumerate(Mod.rois):
                X = slice(x1, x2, 1)
                Y = slice(y1, y2, 1)
                roi_xdim = x2-x1
                roi_ydim = y2-y1
                if roi_xdim != params.roi.shoebox_size or roi_ydim != params.roi.shoebox_size:
                    print("Continuing!")
                    continue
                roi_bg = Mod.all_background[Mod.roi_id==i_roi].reshape((roi_ydim, roi_xdim))

                roi_data = raw_data[Y,X]

                roi_perf = perfect_model[Y,X]
                roi_poor = perturb_model[Y,X]

                roi_perf_tilt = gt_img[Y,X] + roi_bg
                roi_poor_tilt = pt_img[Y,X] + roi_bg

                if np.unique(roi_perf_tilt).shape[0]==1 or np.unique(roi_poor_tilt).shape[0]==1:
                    continue
                cc = pearsonr(roi_perf_tilt.ravel(), roi_poor_tilt.ravel())[0]

                if PLOT and i_roi % 40 == 0:
                    cl = roi_data.min(), roi_data.max()
                    for ax in axs:
                        ax.images[0].set_clim(cl)
                    axs[0].images[0].set_data(roi_data)
                    axs[1].images[0].set_data(roi_bg)
                    axs[2].images[0].set_data(roi_perf)
                    axs[3].images[0].set_data(roi_poor)
                    axs[4].images[0].set_data(roi_perf_tilt)
                    axs[5].images[0].set_data(roi_poor_tilt)

                    plt.suptitle("Trial %d/%d; roi %d %d; cc=%.2f" %(i_trial+1, Ntrials, i_roi+1, len(Mod.rois), cc))
                    plt.draw()
                    plt.pause(1)

                if cc > 0.9:
                    # TODO consider making this also a good example!
                    continue
                else:
                    choose_good = np.random.randint(0, 2)
                    if choose_good:
                        training_img = np.array([roi_data, roi_perf_tilt] )
                        label = 1
                    else:
                        training_img = np.array([roi_data, roi_poor_tilt] )
                        label = 0
                    training_imgs.append( training_img)
                    training_labels.append(label)

            label_start = len(all_train_labels)
            label_stop = label_start + len(training_labels)
            all_train_labels += training_labels

            t = time.time()
            training_imgs_dset.resize((len(all_train_labels), 2,10,10))
            training_imgs_dset[label_start: label_stop] = np.array(training_imgs)
            t = time.time()-t
            print("Took %.3f sec to resize and append to training images dset" %t)

            if args.cbfs:
                cbf_name = os.path.join(rank_odir, "data_1_%05d.cbf" % (i_trial + 1))
                SIM.to_cbf(cbf_name, cbf_int=True)
                SIM.raw_pixels *= 0
                SIM.raw_pixels += flex.double((perturb_model).ravel())
                cbf_name = os.path.join(rank_odir,"poor_1_%05d.cbf" % (i_trial+1))
                SIM.to_cbf(cbf_name, cbf_int=True)

            SIM.free_all()
            S.D.free_all()
            S.D.gpu_free()
            S.D.free_Fhkl2()
        h5.create_dataset("labels", data=all_train_labels, dtype=np.float32)
