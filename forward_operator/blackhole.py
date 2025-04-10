import ehtim.statistics.dataframes as ehdf
import ehtim.observing.obs_helpers as obsh
from ehtim.obsdata import Obsdata 
import pandas as pd
import torch
import numpy as np
import ehtim as eh
from piq import psnr
from ehtim.observing.pulses import trianglePulse2D
import torch.nn.functional as F
from .base import Operator, register_operator

# fix one bug in ehtim
class ObsdataWrapper(Obsdata):
    def tlist(self, conj=False, t_gather=0., scan_gather=False):
        datalist = super().tlist(conj=conj, t_gather=t_gather, scan_gather=scan_gather)
        try:
            return np.array(datalist, dtype=self.data.dtype)
        except:
            return np.array(datalist, dtype=object)


class BlackHoleObservation(object):
    def __init__(self, vis=None, sigma_vis=None, amp=None, sigma_amp=None, cphase=None, sigma_cphase=None, logcamp=None, sigma_logcamp=None, flux=None):
        self.vis = vis
        self.sigma_vis = sigma_vis
        self.amp = amp
        self.sigma_amp = sigma_amp
        self.cphase = cphase
        self.sigma_cphase = sigma_cphase
        self.logcamp = logcamp
        self.sigma_logcamp = sigma_logcamp
        self.flux = flux
    
    def __len__(self):
        return self.vis.shape[0]

    def __getitem__(self, idx):
        results = {}
        if self.vis is not None:
            results['vis'] = self.vis[idx]
            results['sigma_vis'] = self.sigma_vis[idx]
        if self.amp is not None:
            results['amp'] = self.amp[idx]
            results['sigma_amp'] = self.sigma_amp[idx]
        if self.cphase is not None:
            results['cphase'] = self.cphase[idx]
            results['sigma_cphase'] = self.sigma_cphase[idx]
        if self.logcamp is not None:
            results['logcamp'] = self.logcamp[idx]
            results['sigma_logcamp'] = self.sigma_logcamp[idx]
        if self.flux is not None:
            results['flux'] = self.flux[idx]
        return BlackHoleObservation(**results)

    def to(self, device):
        if self.vis is not None:
            vis = self.vis.to(device)
            sigma_vis = self.sigma_vis.to(device)
        if self.amp is not None:
            amp = self.amp.to(device)
            sigma_amp = self.sigma_amp.to(device)
        if self.cphase is not None:
            cphase = self.cphase.to(device)
            sigma_cphase = self.sigma_cphase.to(device)
        if self.logcamp is not None:
            logcamp = self.logcamp.to(device)
            sigma_logcamp = self.sigma_logcamp.to(device)
        if self.flux is not None:
            flux = self.flux.to(device)
        return BlackHoleObservation(vis, sigma_vis, amp, sigma_amp, cphase, sigma_cphase, logcamp, sigma_logcamp, flux)

    def detach(self):
        if self.vis is not None:
            self.vis = self.vis.detach()
            self.sigma_vis = self.sigma_vis.detach()
        if self.amp is not None:
            self.amp = self.amp.detach()
            self.sigma_amp = self.sigma_amp.detach()
        if self.cphase is not None:
            self.cphase = self.cphase.detach()
            self.sigma_cphase = self.sigma_cphase.detach()
        if self.logcamp is not None:
            self.logcamp = self.logcamp.detach()
            self.sigma_logcamp = self.sigma_logcamp.detach()
        if self.flux is not None:
            self.flux = self.flux.detach()

    def to_dict(self):
        results = {}
        if self.vis is not None:
            results['vis'] = self.vis
            results['sigma_vis'] = self.sigma_vis
        if self.amp is not None:
            results['amp'] = self.amp
            results['sigma_amp'] = self.sigma_amp
        if self.cphase is not None:
            results['cphase'] = self.cphase
            results['sigma_cphase'] = self.sigma_cphase
        if self.logcamp is not None:
            results['logcamp'] = self.logcamp
            results['sigma_logcamp'] = self.sigma_logcamp
        if self.flux is not None:
            results['flux'] = self.flux
        return results

    def to_list(self):
        return self.vis, self.sigma_vis, self.amp, self.sigma_amp, self.cphase, self.sigma_cphase, self.logcamp, self.sigma_logcamp, self.flux

    def from_dict(self, results):
        if 'vis' in results:
            self.vis = results['vis']
            self.sigma_vis = results['sigma_vis']
        if 'amp' in results:
            self.amp = results['amp']
            self.sigma_amp = results['sigma_amp']
        if 'cphase' in results:
            self.cphase = results['cphase']
            self.sigma_cphase = results['sigma_cphase']
        if 'logcamp' in results:
            self.logcamp = results['logcamp']
            self.sigma_logcamp = results['sigma_logcamp']
        if 'flux' in results:
            self.flux = results['flux']
        return self

    @classmethod
    def merge(cls, obs_list):
        results = {}
        for key in obs_list[0].__dict__.keys():
            results[key] = torch.cat([obs.__dict__[key] for obs in obs_list], dim=0)
        return BlackHoleObservation(**results)
    

@register_operator('blackhole')
class BlackHoleImagingTorch(Operator):
    """
        More Unified PyTorch Version of Black Hole Imaging Forward Operator based on eht-imaging library.
            https://github.com/achael/eht-imaging

        This class utilize a array file for observation setup (e.g. telescope u,v map)
    """

    def __init__(self, array="configs/task/EHT2017.txt", imsize=256, w_vis=0, w_amp=0,
                 w_cphase=1, w_logcamp=1, w_flux=0.5, loss_normalize=True, num_frames=64, ref_multiplier=1.0, device='cuda'):
        super().__init__(0)
        assert num_frames == 64, "num_frames should be 64"
        # load observations
        ref_obs, A_vis, sigma, time_list, cp_index, cp_conjugate, camp_index, camp_conjugate = self.process_obs(array, imsize)
        self.ref_obs = ref_obs
        self.ref_multiplier = ref_multiplier
        self.device = device
      
        # forward matrix
        self.A_vis = torch.from_numpy(A_vis).unsqueeze(0).cfloat().to(device)  # [1, T, f, HxW]
        self.sigma = torch.from_numpy(sigma).float().to(device) / self.ref_multiplier  # [Txf]
        self.time_list = time_list # [T]

        # index matrix
        self.cp_index = cp_index
        self.cp_conjugate = cp_conjugate
        self.camp_index = camp_index
        self.camp_conjugate = camp_conjugate
        
        # dimension
        self.amp_dim = self.A_vis.shape[-2] * num_frames
        self.cphase_dim = self.cp_index.shape[0]
        self.logcamp_dim = self.camp_index.shape[0]
        self.flux_dim = num_frames

        # params
        self.H = imsize
        self.W = imsize
        self.T = num_frames
        self.weight_vis = w_vis * self.amp_dim
        self.weight_amp = w_amp * self.amp_dim
        self.weight_cp = w_cphase * self.cphase_dim
        self.weight_camp = w_logcamp * self.logcamp_dim
        self.weight_flux = w_flux * self.flux_dim
        self.loss_normalize = loss_normalize

    # 0. set up forward function
    def process_obs(
            self,
            array,
            resolution
    ):
        self.uvarray     = uvarray     = eh.array.load_txt(array)

        self.psize       = psize       = 2.4240684055470684e-12
        self.pulse       = pulse       = trianglePulse2D
        self.ra          = ra          =  17.761120
        self.dec         = dec         = -29.007797
        self.rf          = rf          = 230 * 1e9
        self.bw          = bw          = 2 * 1e9 # for EHT, 16 * 1e9 for the ngEHT
        self.mjd         = mjd         = 60775   # (10/04/2025)
        self.source      = source      = 'SgrA'
        self.tint        = tint        = 102 
        self.tadv        = tadv        = 102
        self.tau         = tau         = 0.1
        self.taup        = taup        = 0.1
        self.polrep_obs  = polrep_obs  = 'stokes'

        self.elevmin     = elevmin     = 10.0 
        self.elevmax     = elevmax     = 85.0
        self.timetype    = timetype    = 'UTC'

        # SgrA best times (in UTC): 12.5 - 14.2
        self.tstart = tstart = 12.5
        self.tstop = tstop = 14.3

        ref_obs = uvarray.obsdata(ra, dec, rf, bw, tint, tadv, tstart, tstop,
                            mjd=mjd, polrep=polrep_obs, tau=tau, timetype=timetype,
                            elevmin=elevmin, elevmax=elevmax,
                            no_elevcut_space=False,
                            fix_theta_GMST=False)
        ref_obs.__class__ = ObsdataWrapper
    
        # A_vis matrix, sigma & time list
        A_vis_list = []
        sigma_list = []
        time_list = []
        for t_obs in ref_obs.tlist():
            time_list.append(t_obs[0]['time'])
            uv = np.hstack((t_obs['u'].reshape(-1, 1), t_obs['v'].reshape(-1, 1)))
            A_vis = obsh.ftmatrix(psize, resolution, resolution, uv, pulse=pulse, mask=[])
            A_vis_list.append(A_vis)
            sigma = t_obs['sigma']
            sigma_list.append(sigma)
        A_vis = np.stack(A_vis_list, axis=0) # [T, f, D]
        sigma = np.stack(sigma_list, axis=0).flatten()  # [Txf]
        time_list = np.array(time_list) # [T]

        # get map_fn and conjugate_fn
        obs_data_df = pd.DataFrame(ref_obs.data)
        map_fn, conjugate_fn = {}, {}
        for i, (time, t1, t2) in enumerate(zip(obs_data_df['time'], obs_data_df['t1'], obs_data_df['t2'])):
            map_fn[(time, t1, t2)] = i
            conjugate_fn[(time, t1, t2)] = 0
            map_fn[(time, t2, t1)] = i
            conjugate_fn[(time, t2, t1)] = 1

        # closure phase index
        cp_index, cp_conjugate = [], []
        clphasearr = ref_obs.c_phases()  
        for time, t1, t2, t3 in zip(clphasearr['time'], clphasearr['t1'], clphasearr['t2'], clphasearr['t3']):
                idx = [map_fn[(time, t1, t2)], map_fn[(time, t2, t3)], map_fn[(time, t3, t1)]]
                conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t2, t3)], conjugate_fn[(time, t3, t1)]]
                cp_index.append(idx)
                cp_conjugate.append(conj)  
        cp_index = torch.tensor(cp_index).long().cuda()
        cp_conjugate = torch.tensor(cp_conjugate).long().cuda()

        # log closure amplitude index
        camp_index, camp_conjugate = [], []
        clamparr_full = ref_obs.c_amplitudes()
        for time, t1, t2, t3, t4 in zip(clamparr_full['time'], clamparr_full['t1'], clamparr_full['t2'], clamparr_full['t3'], clamparr_full['t4']):
            idx = [map_fn[(time, t1, t2)], map_fn[(time, t3, t4)], map_fn[(time, t1, t4)], map_fn[(time, t2, t3)]]
            conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t3, t4)], conjugate_fn[(time, t1, t4)],
                    conjugate_fn[(time, t2, t3)]]
            camp_index.append(idx)
            camp_conjugate.append(conj)
        camp_index = torch.tensor(camp_index).long().cuda()
        camp_conjugate = torch.tensor(camp_conjugate).long().cuda()

        # # A_cp matrix
        # A_cp_list = []
        # clphasearr_full = ref_obs.c_phases()    
        # for t_obs in ref_obs.tlist():
        #     clphasearr = clphasearr_full[clphasearr_full['time'] == t_obs[0]['time']]
        #     uv1 = np.hstack((clphasearr['u1'].reshape(-1, 1), clphasearr['v1'].reshape(-1, 1)))
        #     uv2 = np.hstack((clphasearr['u2'].reshape(-1, 1), clphasearr['v2'].reshape(-1, 1)))
        #     uv3 = np.hstack((clphasearr['u3'].reshape(-1, 1), clphasearr['v3'].reshape(-1, 1)))
        #     A_cp = np.stack([
        #         obsh.ftmatrix(psize, resolution, resolution, uv1, pulse=pulse, mask=[]),
        #         obsh.ftmatrix(psize, resolution, resolution, uv2, pulse=pulse, mask=[]),
        #         obsh.ftmatrix(psize, resolution, resolution, uv3, pulse=pulse, mask=[])
        #     ], axis=0)
        #     A_cp_list.append(A_cp)
        # A_cp = np.stack(A_cp_list, axis=0) # [num_frames, 3, f, D]

        # # A_logcamp matrix
        # A_camp_list = []
        # clamparr_full = ref_obs.c_amplitudes()
        # for t_obs in ref_obs.tlist():
        #     clamparr = clamparr_full[clamparr_full['time'] == t_obs[0]['time']]
        #     uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
        #     uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
        #     uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
        #     uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
        #     A_camp = np.stack([
        #         obsh.ftmatrix(psize, resolution, resolution, uv1, pulse=pulse, mask=[]),
        #         obsh.ftmatrix(psize, resolution, resolution, uv2, pulse=pulse, mask=[]),
        #         obsh.ftmatrix(psize, resolution, resolution, uv3, pulse=pulse, mask=[]),
        #         obsh.ftmatrix(psize, resolution, resolution, uv4, pulse=pulse, mask=[])
        #     ], axis=0)
        #     A_camp_list.append(A_camp)
        # A_camp = np.stack(A_camp_list, axis=0) # [num_frames, 4, f, D]

        return ref_obs, A_vis, sigma, time_list, cp_index, cp_conjugate, camp_index, camp_conjugate

    @staticmethod
    def estimate_flux(obs):
        # estimate the total flux from the observation
        data = obs.unpack_bl('ALMA', 'APEX', 'amp')
        flux_per_frames = []
        for pair in data:
            amp = pair[0][1]
            flux_per_frames.append(amp)
        flux_per_frames = np.array(flux_per_frames) # [T,]
        flux = np.median(flux_per_frames)
        return flux, flux_per_frames

    # 1. visibility and flux from image x in range [0,1]
    def forward_vis(self, x):
        # x: [B, T, 1, H, W], A_vis: [1, T, f, HxW]
        x = x.to(self.A_vis)

        xvec = x.reshape(-1, self.T, 1, self.H * self.W) # [B, T, 1, HxW]
        vis = (self.A_vis * xvec).sum(-1).flatten(1)  # [B, Txf]
        return vis

    def forward_amp(self, x):
        amp = self.forward_vis(x).abs()
        sigmaamp = self.sigma + 0 * amp
        return amp, sigmaamp

    def forward_flux(self, x):
        return x.flatten(2).sum(-1) # [B, T]
    
    # 2. forward from visibilities
    def correct_vis_direction(self, vis, conj):
        vis = vis * (1 - conj) + vis.conj() * conj
        return vis

    def forward_amp_from_vis(self, vis):
        amp = vis.abs()
        sigmaamp = self.sigma + 0 * amp
        return amp, sigmaamp

    def forward_bisepectra_from_vis(self, vis):
        v1 = vis[:, self.cp_index[:, 0]] # [B, f_cp]
        v2 = vis[:, self.cp_index[:, 1]] 
        v3 = vis[:, self.cp_index[:, 2]]

        cj1 = self.cp_conjugate[None, :, 0] # [1, f_cp]
        cj2 = self.cp_conjugate[None, :, 1]
        cj3 = self.cp_conjugate[None, :, 2]

        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        return i1, i2, i3 # [B, f_cp]

    def forward_cp_from_vis(self, vis):
        i1, i2, i3 = self.forward_bisepectra_from_vis(vis)
        cphase = torch.angle(i1 * i2 * i3) # [B, f_cp]

        v1 = self.sigma[self.cp_index[:, 0]] # [f_cp]
        v2 = self.sigma[self.cp_index[:, 1]]
        v3 = self.sigma[self.cp_index[:, 2]]
        sigmacp = (v1 ** 2 / i1.abs() ** 2 + v2 ** 2 / i2.abs() ** 2 + v3 ** 2 / i3.abs() ** 2).sqrt()
        return cphase, sigmacp

    def forward_logcamp_bispectra_from_vis(self, vis):
        v1 = vis[:, self.camp_index[:, 0]].abs() # [B, f_camp]
        v2 = vis[:, self.camp_index[:, 1]].abs()
        v3 = vis[:, self.camp_index[:, 2]].abs()
        v4 = vis[:, self.camp_index[:, 3]].abs()

        cj1 = self.camp_conjugate[None, :, 0] # [1, f_camp]
        cj2 = self.camp_conjugate[None, :, 1]
        cj3 = self.camp_conjugate[None, :, 2]
        cj4 = self.camp_conjugate[None, :, 3]

        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        i4 = self.correct_vis_direction(v4, cj4)
        return i1, i2, i3, i4

    def forward_logcamp_from_vis(self, vis):
        i1, i2, i3, i4 = self.forward_logcamp_bispectra_from_vis(vis)
        logcamp = i1.log() + i2.log() - i3.log() - i4.log()

        v1 = self.sigma[self.camp_index[:, 0]] # [f_camp]
        v2 = self.sigma[self.camp_index[:, 1]]
        v3 = self.sigma[self.camp_index[:, 2]]
        v4 = self.sigma[self.camp_index[:, 3]]
        sigmaca = (v1 ** 2 / i1 ** 2 + v2 ** 2 / i2 ** 2 + v3 ** 2 / i3 ** 2 + v4 ** 2 / i4 ** 2).sqrt()
        return logcamp, sigmaca # [B, f_camp]

    def forward_from_vis(self, x):
        vis = self.forward_vis(x)

        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)
        obs = BlackHoleObservation(vis, sigmaamp, amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux)
        return obs

    # 3. forward from EHT library
    def pt2ehtim(self, pt_video):
        times = self.time_list
        frames = pt_video.detach().cpu().numpy().reshape(self.T, self.H, self.W) * self.ref_multiplier
        mov = eh.movie.Movie(frames, times, self.psize, self.ra, self.dec, rf=self.rf, polrep='stokes', pol_prim=None, pulse=eh.PULSE_DEFAULT, source=self.source, mjd=self.mjd)
        return mov

    def forward_from_eht(self, x):
        multiplier = self.ref_multiplier
        ref_obs = self.ref_obs
        pt_obs = []
        for pt_mov in x:
            eh_mov = self.pt2ehtim(pt_mov)
            # observe the image
            obs = eh_mov.observe_same_nonoise(ref_obs, ttype='direct', verbose=False)
            
            # visibilities
            vis = torch.from_numpy(obs.data['vis']).float().to(x.device) / multiplier
            sigmavis = torch.from_numpy(obs.data['sigma']).float().to(x.device) / multiplier

            # visibilities amplitude
            adf = ehdf.make_amp(obs, debias=False)
            amp = torch.from_numpy(adf['amp'].to_numpy()).float().to(x.device) / multiplier
            sigmaamp = torch.from_numpy(adf['sigma'].to_numpy()).float().to(
                x.device) / multiplier

            # closure phase
            cdf = ehdf.make_cphase_df(obs, count='min')
            cp = torch.from_numpy(cdf['cphase'].to_numpy()).float().to(x.device) * eh.DEGREE
            sigmacp = torch.from_numpy(cdf['sigmacp'].to_numpy()).float().to(x.device) * eh.DEGREE

            # log closure amplitude
            ldf = ehdf.make_camp_df(obs, count='min')
            camp = torch.from_numpy(ldf['camp'].to_numpy()).float().to(x.device)
            sigmaca = torch.from_numpy(ldf['sigmaca'].to_numpy()).float().to(x.device)

            # flux
            _, flux_per_frames = self.estimate_flux(obs)
            flux = torch.from_numpy(flux_per_frames).float().to(x.device) / multiplier

            pt_obs.append(BlackHoleObservation(vis[None], sigmavis[None], amp[None], sigmaamp[None], cp[None], sigmacp[None], camp[None], sigmaca[None], flux[None]))
        pt_obs = BlackHoleObservation.merge(pt_obs)
        return pt_obs

    # 4. chi-square evalutation
    def chi2_vis(self, x, y_vis, y_vis_sigma):
        vis = self.forward_vis(x)
        return self.chi2_vis_from_meas(vis, y_vis, y_vis_sigma)
    
    @staticmethod
    def chi2_vis_from_meas(y_vis_meas, y_vis, y_vis_sigma):
        residual = torch.abs(y_vis_meas - y_vis)
        return torch.mean(torch.square(residual / y_vis_sigma), dim=1) / 2

    def chi2_amp(self, x, y_amp, y_amp_sigma):
        amp_pred, _ = self.forward_amp(x)
        return self.chi2_amp_from_meas(amp_pred, y_amp, y_amp_sigma)

    @staticmethod
    def chi2_amp_from_meas(y_amp_meas, y_amp, y_amp_sigma):
        residual = y_amp_meas - y_amp
        return torch.mean(torch.square(residual / y_amp_sigma), dim=1)

    def chi2_cphase(self, x, y_cphase, y_cphase_sigma):
        vis = self.forward_vis(x)
        cphase_pred, _ = self.forward_cp_from_vis(vis)
        return self.chi2_cphase_from_meas(cphase_pred, y_cphase, y_cphase_sigma)

    @staticmethod
    def chi2_cphase_from_meas(y_cphase_meas, y_cphase, y_cphase_sigma):
        angle_residual = y_cphase - y_cphase_meas
        return 2. * torch.mean((1 - torch.cos(angle_residual)) / torch.square(y_cphase_sigma), dim=1)

    def chi2_logcamp(self, x, y_camp, y_logcamp_sigma):
        vis = self.forward_vis(x)
        camp_pred, _ = self.forward_logcamp_from_vis(vis)
        return self.chi2_logcamp_from_meas(camp_pred, y_camp, y_logcamp_sigma)

    @staticmethod
    def chi2_logcamp_from_meas(y_logcamp_meas, y_logcamp, y_logcamp_sigma):
        return torch.mean(torch.abs((y_logcamp_meas - y_logcamp) / y_logcamp_sigma) ** 2, dim=1)

    def chi2_flux(self, x, y_flux):
        flux_pred = self.forward_flux(x)
        return self.chi2_flux_from_meas(flux_pred, y_flux)

    @staticmethod
    def chi2_flux_from_meas(y_flux_meas, y_flux):
        return torch.mean(torch.square((y_flux_meas - y_flux) / 2), dim=1)

    # 5. noisy measurement
    def measure_eht(self, x):
        ampcal     			= False
        phasecal   			= False
        dcal        		= False
        DOFF         		= 0.0
        GAIN_OFFSET  		= 0.0
        GAINP        		= 0.0
        rlratio_std         = 0.0 
        rlphase_std         = 0.0
        seed 				= 24
        sigmat 				= 0.25
        if not ampcal:
            # These are measured from 2017 campaign. For future arrays, one thing 
            # you could do is add a realistic value for the gains of the missing antennas
            # (for example the average value from the ones you have),
            # or just pass a single value for all antennas, instead of passing
            # a dictionary.
            # GAIN_OFFSET = {'AA': 0.029,'AP': 0.028,'AZ': 0.045,'JC': 0.020,'LM': 0.147,
            # 			   'PV': 0.050,'SM': 0.019,'SP': 0.052,'SR': 0.0}
            # GAINP =       {'AA': 0.054,'AP': 0.045,'AZ': 0.056,'JC': 0.030,'LM': 0.124,
            # 			   'PV': 0.075,'SM': 0.028,'SP': 0.095,'SR': 0.0}
            GAIN_OFFSET = {
                'ALMA': 0.029,
                'APEX': 0.028,
                'SMT': 0.045,
                'JCMT': 0.020,
                'LMT': 0.147,
                'PV': 0.050,
                'SMA': 0.019,
                'SPT': 0.052,
                'SR': 0.0}  # ?
            GAINP = {
                'ALMA': 0.054,
                'APEX': 0.045,
                'SMT': 0.056,
                'JCMT': 0.030,
                'LMT': 0.124,
                'PV': 0.075,
                'SMA': 0.028,
                'SPT': 0.095,
                'SR': 0.0}
            
        if not dcal:
            # DOFF = {'AA':0.005, 'AP':0.005, 'AZ':0.01, 'LM':0.01, 'PV':0.01, 'SM':0.005,
            # 		'JC':0.01, 'SP':0.01, 'SR':0.01}
            DOFF = {
                'ALMA': 0.005,
                'APEX': 0.005,
                'SMT': 0.01,
                'LMT': 0.01,
                'PV': 0.01,
                'SMA': 0.005,
                'JCMT': 0.01,
                'SPT': 0.01,
                'SR': 0.01}

        multiplier = self.ref_multiplier
        ref_obs = self.ref_obs
        pt_obs = []
        for pt_mov in x:
            eh_mov = self.pt2ehtim(pt_mov)
            # observe the image
            obs = eh_mov.observe_same(ref_obs, ttype='direct', add_th_noise=True, taup=self.taup, jones=True, inv_jones=False, ampcal=ampcal, phasecal=phasecal, dcal=dcal,
                stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=GAIN_OFFSET, gainp=GAINP, dterm_offset=DOFF,
                rlratio_std=rlratio_std,rlphase_std=rlphase_std, seed=seed, sigmat=sigmat, verbose=False)
            
            # visibilities
            vis = torch.from_numpy(obs.data['vis']).float().to(x.device) / multiplier
            sigmavis = torch.from_numpy(obs.data['sigma']).float().to(x.device) / multiplier

            # visibilities amplitude
            adf = ehdf.make_amp(obs, debias=False)
            amp = torch.from_numpy(adf['amp'].to_numpy()).float().to(x.device) / multiplier
            sigmaamp = torch.from_numpy(adf['sigma'].to_numpy()).float().to(
                x.device) / multiplier

            # closure phase
            cdf = ehdf.make_cphase_df(obs, count='min')
            cp = torch.from_numpy(cdf['cphase'].to_numpy()).float().to(x.device) * eh.DEGREE
            sigmacp = torch.from_numpy(cdf['sigmacp'].to_numpy()).float().to(x.device) * eh.DEGREE

            # log closure amplitude
            ldf = ehdf.make_camp_df(obs, count='min')
            camp = torch.from_numpy(ldf['camp'].to_numpy()).float().to(x.device)
            sigmaca = torch.from_numpy(ldf['sigmaca'].to_numpy()).float().to(x.device)

            # flux
            _, flux_per_frames = self.estimate_flux(obs)
            flux = torch.from_numpy(flux_per_frames).float().to(x.device) / multiplier

            pt_obs.append(BlackHoleObservation(vis[None], sigmavis[None], amp[None], sigmaamp[None], cp[None], sigmacp[None], camp[None], sigmaca[None], flux[None]))
        pt_obs = BlackHoleObservation.merge(pt_obs).to(self.device)
        return pt_obs

    # 6. util functions
    # def compress(self, vis=None, sigmavis=None, amp=None, sigmaamp=None, cphase=None, sigmacp=None, logcamp=None, sigmacamp=None, flux=None):
    #     # using a dict to store observation
    #     y_dict = {
    #         'vis': vis.float(),
    #         'sigma_vis': sigmavis.float(),
    #         'amp': amp.float(),
    #         'sigma_amp': sigmaamp.float(),
    #         'cphase': cphase.float(),
    #         'sigma_cphase': sigmacp.float(),
    #         'logcamp': logcamp.float(),
    #         'sigma_logcamp': sigmacamp.float(),
    #         'flux': flux.float()
    #     }
    #     return y_dict

    # def decompress(self, y):
    #     vis, sigmavis = y['vis'], y['sigma_vis']
    #     amp, sigmaamp = y['amp'], y['sigma_amp']
    #     cphase, sigmacp = y['cphase'], y['sigma_cphase']
    #     logcamp, sigmacamp = y['logcamp'], y['sigma_logcamp']
    #     flux = y['flux']
    #     return vis, sigmavis, amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux

    # 7. sanity check
    def cosine_similarity(self, a1, a2):
        a1 = a1.flatten(1)
        a2 = a2.flatten(1)
        a1_norm = torch.norm(a1, dim=1)
        a2_norm = torch.norm(a2, dim=1)
        similarity = (a1 * a2.conj()).abs().sum(1) / (a1_norm * a2_norm)
        return similarity.min().item()

    def compare(self, y1, y2, verbose=False):
        vis1, sigmavis1, amp1, sigmaamp1, cphase1, sigmacp1, logcamp1, sigmacamp1, flux1 = y1.to_list()
        vis2, sigmavis2, amp2, sigmaamp2, cphase2, sigmacp2, logcamp2, sigmacamp2, flux2 = y2.to_list()

        vis_similarity = self.cosine_similarity(vis1, vis2)
        amp_similarity = self.cosine_similarity(amp1, amp2)
        cphase_similarity = self.cosine_similarity(cphase1, cphase2)
        logcamp_similarity = self.cosine_similarity(logcamp1, logcamp2)
        flux_similarity = self.cosine_similarity(flux1, flux2)

        sigmavis_similarity = self.cosine_similarity(sigmavis1, sigmavis2)
        sigmaamp_similarity = self.cosine_similarity(sigmaamp1, sigmaamp2)
        sigmacp_similarity = self.cosine_similarity(sigmacp1, sigmacp2)
        sigmacamp_similarity = self.cosine_similarity(sigmacamp1, sigmacamp2)

        if verbose:
            print("vis similarity: {:.3f} %".format(vis_similarity * 100))
            print("amp similarity: {:.3f} %".format(amp_similarity * 100))
            print("cphase similarity: {:.3f} %".format(cphase_similarity * 100))
            print("logcamp similarity: {:.3f} %".format(logcamp_similarity * 100))
            print("flux similarity: {:.3f} %".format(flux_similarity * 100))
            print("sigmavis similarity: {:.3f} %".format(sigmavis_similarity * 100))
            print("sigmaamp similarity: {:.3f} %".format(sigmaamp_similarity * 100))
            print("sigmacp similarity: {:.3f} %".format(sigmacp_similarity * 100))
            print("sigmacamp similarity: {:.3f} %".format(sigmacamp_similarity * 100))
        similarity = np.max(
            [vis_similarity, amp_similarity, cphase_similarity, logcamp_similarity, flux_similarity, sigmavis_similarity, sigmaamp_similarity,
             sigmacp_similarity, sigmacamp_similarity])
        return similarity

    def sanity_check(self, x):
        x = self.unnormalize(x)
        # from vis
        print('forward by visibility...')
        y_vis = self.forward_from_vis(x)

        # from EHT
        print('forward by EHT...')
        y_eht = self.forward_from_eht(x)

        # compare
        print('compare vis and EHT (cosine similarity): {:.3f} %'.format(self.compare(y_vis, y_eht) * 100))

    # 8. evaluating chi-square
    @staticmethod
    def normalize_chisq(chisq):
        overfit = chisq < 1.0
        e_chisq = chisq * (~overfit) + 1 / chisq * overfit
        return e_chisq

    def evaluate_chisq(self, x, y, normalize=False, chi_sq_list=['vis', 'amp', 'cphase', 'logcamp']):
        x = self.unnormalize(x).clip(min=0)
        y_vis, y_vis_sigma, y_amp, y_amp_sigma, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = y.to_list()
        # align flux
        x_flux = self.forward_flux(x) # [B, T]
        x_aligned = x * (y_flux / x_flux)[:, :, None, None, None]

        results = {}
        if 'vis' in chi_sq_list:
            vis_loss = self.chi2_vis(x_aligned, y_vis, y_vis_sigma)
            if normalize:
                vis_loss = self.normalize_chisq(vis_loss)
            results['vis'] = vis_loss
        if 'amp' in chi_sq_list:
            amp_loss = self.chi2_amp(x_aligned, y_amp, y_amp_sigma)
            if normalize:
                amp_loss = self.normalize_chisq(amp_loss)
            results['amp'] = amp_loss
        if 'cphase' in chi_sq_list:
            cp_loss = self.chi2_cphase(x_aligned, y_cp, y_cphase_sigma)
            if normalize:
                cp_loss = self.normalize_chisq(cp_loss)
            results['cphase'] = cp_loss
        if 'logcamp' in chi_sq_list:
            camp_loss = self.chi2_logcamp(x_aligned, y_camp, y_logcamp_sigma)
            if normalize:
                camp_loss = self.normalize_chisq(camp_loss)
            results['logcamp'] = camp_loss
        return results
    
    def evaluate_chisq_indist(self, x, y):
        _, _, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        # align flux
        x_flux = self.forward_flux(x) # [B, T]
        x_aligned = x * (y_flux / x_flux)[:, :, None, None, None]
    
        vis = self.forward_vis(x_aligned)

        # closure phase
        cphase_pred, _ = self.forward_cp_from_vis(vis)
        angle_residual = cphase_pred - y_cp
        cp_loss = 2. * (1 - torch.cos(angle_residual)) / torch.square(y_cphase_sigma)

        # log closure amplitude
        camp_pred, _ = self.forward_logcamp_from_vis(vis)
        camp_loss = torch.abs((camp_pred - y_camp) / y_logcamp_sigma) ** 2
        return cp_loss, camp_loss

    # 9. public interface
    def unnormalize(self, inputs):
        # [-1, 1] -> [0, 1]
        return inputs * 0.5 + 0.5

    def normalize(self, inputs):
        # [0, 1] -> [-1, 1]
        return inputs * 2 - 1
    
    def __call__(self, x, **kwargs):
        x = self.unnormalize(x)
        return self.forward_from_vis(x)

    def measure(self, x, **kwargs):
        x = self.unnormalize(x)
        return self.measure_eht(x)

    def loss(self, x, y):
        normalize = self.loss_normalize
        x = self.unnormalize(x)
        y_vis, y_vis_sigma, y_amp, y_amp_sigma, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = y.to_list()
        loss = 0
        report = ''
        if self.weight_vis > 0:
            vis_loss = self.chi2_vis(x, y_vis, y_vis_sigma)
            if normalize:
                vis_loss = self.normalize_chisq(vis_loss)
            loss += self.weight_vis * vis_loss
            report += 'vis_loss: {:.4f} '.format((vis_loss).sum().item())
        if self.weight_amp > 0:
            amp_loss = self.chi2_amp(x, y_amp, y_amp_sigma)
            if normalize:
                amp_loss = self.normalize_chisq(amp_loss)
            loss += self.weight_amp * amp_loss
            report += 'amp_loss: {:.4f} '.format((amp_loss).sum().item())
        if self.weight_cp > 0:
            cp_loss = self.chi2_cphase(x, y_cp, y_cphase_sigma)
            if normalize:
                cp_loss = self.normalize_chisq(cp_loss)
            loss += self.weight_cp * cp_loss
            report += 'cp_loss: {:.4f} '.format((cp_loss).sum().item())
        if self.weight_camp > 0:
            camp_loss = self.chi2_logcamp(x, y_camp, y_logcamp_sigma)
            if normalize:
                camp_loss = self.normalize_chisq(camp_loss)
            loss += self.weight_camp * camp_loss
            report += 'camp_loss: {:.4f} '.format((camp_loss).sum().item())
        if self.weight_flux > 0:
            flux_loss = self.chi2_flux(x, y_flux)
            loss += self.weight_flux * flux_loss
            report += 'flux_loss: {:.4f} '.format((flux_loss).sum().item())
        # print(report)
        return loss
