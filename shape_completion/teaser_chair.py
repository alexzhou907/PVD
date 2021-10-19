

import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal
from utils.visualize import *
from utils.file_utils import *
from utils.mitsuba_renderer2 import write_to_xml_batch, write_to_xml
from model.pvcnn_completion import PVCNN2Base

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import *
'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs



class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)[:,:,self.sv_points:]


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:,:,self.sv_points:], t=t, eps=model_output)


            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:,:,self.sv_points:], t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, partial_x, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = torch.cat([partial_x, noise_fn(size=shape, dtype=torch.float, device=device)], dim=-1)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t[:,:,self.sv_points:].shape == shape
        return img_t


    def p_sample_loop_trajectory2(self, partial_x, denoise_fn, shape, device, num_save,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        scale = np.exp(np.log(1/total_steps)/num_save)
        save_step = total_steps

        img_t = torch.cat([partial_x, noise_fn(size=shape, dtype=torch.float, device=device)], dim=-1)
        imgs = [img_t.detach().cpu()]
        for t in reversed(range(0,total_steps)):
            if (t+1) == save_step and t > 0 and len(imgs)<num_save:
                imgs.append(img_t.detach().cpu())
                save_step = int(save_step * scale)

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        imgs.append(img_t.detach().cpu())
        assert imgs[-1][:,:,self.sv_points:].shape == shape
        return imgs


    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start[:,:,self.sv_points:], x_t=data_t[:,:,self.sv_points:], t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start[:,:,self.sv_points:].shape, dtype=data_start.dtype, device=data_start.device)

        data_t = self.q_sample(x_start=data_start[:,:,self.sv_points:], t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(torch.cat([data_start[:,:,:self.sv_points], data_t], dim=-1), t)[:,:,self.sv_points:]

            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                data_t = torch.cat([x_start[:, :, :self.sv_points], self.q_sample(x_start=x_start[:, :, self.sv_points:], t=t_b)], dim=-1)
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[:, :, self.sv_points:].shape
                new_mse_b = ((pred_xstart - x_start[:, :, self.sv_points:]) ** 2).mean(dim=list(range(1, len(pred_xstart.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[:,:,self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type, args.svpoints)

        self.model = PVCNN2(num_classes=args.nc, sv_points=args.svpoints, embed_dim=args.embed_dim, use_att=args.attention,
                            dropout=args.dropout, extra_feature_channels=0)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(partial_x, self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_samples_traj2(self, partial_x, shape, device, noise_fn=torch.randn, num_save=20,
                    clip_denoised=False,
                    keep_running=False):
        return self.diffusion.p_sample_loop_trajectory2(partial_x, self._denoise, shape=shape, device=device, num_save=num_save, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)


    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas



#############################################################################
def get_svr_dataset(pc_dataroot, mesh_root, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=pc_dataroot,
        categories=category, split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet_Singleview_Points(root_pc=pc_dataroot, root_mesh=mesh_root,
                                            cache=os.path.join(mesh_root, '../cache'), split='val',
        categories=category,
        radius=3, elev=-89, azim=180, img_size=512, focal_length=1000,
        npoints=npoints, sv_samples=200,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return te_dataset

def get_mvr_dataset(pc_dataroot, views_root, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=pc_dataroot,
        categories=category, split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet_Multiview_Points(root_pc=pc_dataroot, root_views=views_root,
                                            cache=os.path.join(pc_dataroot, '../cache'), split='val',
        categories=category,
        npoints=npoints, sv_samples=200,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return te_dataset



def generate_video(netE, opt, save_dir):
    test_dataset = get_svr_dataset(opt.dataroot_pc, opt.dataroot_sv,
                                      opt.npoints, opt.classes)
    # _, test_dataset =  get_dataset(opt.dataroot, opt.npoints, opt.classes, use_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):

        # gt_all = data['test_points']
        x_all = data['sv_points']

        img_all = data['image']

        export_to_pc_batch(
            os.path.join(save_dir, 'batch_%03d_ply' % i), x_all[:, :opt.svpoints, :].numpy())
        write_to_xml_batch(os.path.join(save_dir, 'batch_%03d' % i),
                           x_all[:, :opt.svpoints, :].numpy(), cat='chair')

        for v in range(6):
            x = x_all.transpose(1, 2).contiguous()
            img = img_all
            m, s = data['mean'].float(), data['std'].float()

            gen_all = netE.gen_samples_traj2(x[:, :, :opt.svpoints].cuda(), x[:, :, opt.svpoints:].shape, 'cuda', num_save=55,
                                     clip_denoised=False)
            gen_all = torch.stack(gen_all, dim=1).detach().cpu()
            gen_all = gen_all.transpose(2, 3).contiguous()

            gen_all = gen_all * s[:, None] + m[:, None]

            x = x.transpose(1, 2).contiguous()


            for p, d in enumerate(zip(list(gen_all), list(img))):

                im = np.fliplr(np.flipud(d[-1]))
                gen = d[0]
                plt.imsave(os.path.join(save_dir, 'depth_%03d_%03d.png'%(i,p)), im, cmap='gray')
                # gen
                write_to_xml_batch(os.path.join(save_dir, 'batch_%03d'%i, 'sample_%03d/mode_%03d/xml/gen_process/' % (p,v)),
                                   gen.numpy(), cat='chair')

            for p, gen in enumerate(gen_all[:,-1]):
                Path(os.path.join(save_dir, 'batch_%03d_ply' % i, 'sample_%03d' % p,
                                  'mode_%03d' % v)).mkdir(parents=True, exist_ok=True)
                pcwrite(
                    os.path.join(save_dir, 'batch_%03d_ply' % i, 'sample_%03d/mode_%03d/partial.ply' % (p,v)), gen.numpy())

            for k, pcl in enumerate(gen_all[:, -1].cpu().numpy()):
                dir_ = os.path.join(save_dir, 'batch_%03d' % i, 'sample_%03d/mode_%03d/xml/rotate_final/' % (k, v))
                Path(dir_).mkdir(parents=True, exist_ok=True)
                for azim in np.linspace(45, 405 - (360 / 50), 50):
                    write_to_xml(
                        os.path.join(dir_, 'azim_%03d.xml' % azim),
                        pcl, cat='chair', elev=19.471, azim=azim)



def generate_video_redwood(netE, opt, save_dir):
    import open3d as o3d
    pth = "/viscam/u/alexzhou907/research/diffusion/redwood/09620_pc_partial.ply"
    pth_gt = "/viscam/u/alexzhou907/research/diffusion/redwood/09620_pc.ply"

    points = np.asarray(o3d.io.read_point_cloud(pth).points)

    gt_points = np.asarray(o3d.io.read_point_cloud(pth_gt).points)

    np.save('gt.npy', gt_points)

    test_dataset = ShapeNet15kPointClouds(root_dir=opt.dataroot_pc,
                           categories=opt.classes, split='train',
                           tr_sample_size=opt.npoints,
                           te_sample_size=opt.npoints,
                           scale=1.,
                           normalize_per_shape=False,
                           normalize_std_per_axis=False,
                           random_subsample=True)

    m, s = torch.from_numpy(test_dataset[0]['mean']).float(), torch.from_numpy(test_dataset[0]['std']).float()

    x = torch.from_numpy(points[np.random.choice(points.shape[0], size=opt.svpoints, replace=False)]).float()
    x = (x - m) / s

    x = x[None].transpose(1, 2).cuda()

    shape = list(x.shape)
    shape[-1] = opt.npoints - shape[-1]

    res = []
    for v in tqdm(range(20)):
        gen_all = netE.gen_samples_traj2(x.cuda(), torch.Size(shape), 'cuda', num_save=55,
                                 clip_denoised=False)
        gen_all = torch.stack(gen_all, dim=1).detach().cpu()
        gen_all = gen_all.transpose(2, 3).contiguous()

        gen_all = gen_all * s[:, None] + m[:, None]

        res.append(gen_all[:, -1].cpu())

        for p, gen in enumerate(gen_all):
            # gen
            write_to_xml_batch(
                os.path.join(save_dir, 'mode_%03d/xml/gen_process/' % ( v)),
                gen.numpy(), cat='chair')


        for k, pcl in enumerate(gen_all[:, -1].cpu().numpy()):
            dir_ = os.path.join(save_dir, 'mode_%03d/xml/rotate_final/' % ( v))
            Path(dir_).mkdir(parents=True, exist_ok=True)
            for azim in np.linspace(45, 405 - (360 / 50), 50):
                write_to_xml(
                    os.path.join(dir_, 'azim_%03d.xml' % azim),
                    pcl, cat='chair', elev=19.471, azim=azim)

        pcwrite(os.path.join(save_dir, 'mode_%03d.ply'%v), gen_all[:, -1].cpu().numpy()[0])


    pcwrite(os.path.join(save_dir, 'gt.ply'),
            gt_points[np.random.choice(gt_points.shape[0], size=opt.npoints, replace=False)])
def main(opt):
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    netE = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.cuda:
        netE.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    netE = netE.cuda()
    netE.multi_gpu_wrapper(_transform_)

    netE.eval()

    ckpts = [os.path.join(opt.ckpt_dir, f) for f in os.listdir(opt.ckpt_dir) if f.endswith('.pth')]

    with torch.no_grad():
        for ckpt in reversed(sorted(ckpts, key=lambda x: int(x.strip('.pth').split('_')[-1]) )):

            opt.netE = ckpt
            logger.info("Resume Path:%s" % opt.netE)

            resumed_param = torch.load(opt.netE)
            netE.load_state_dict(resumed_param['model_state'])


            generate_video_redwood( netE,opt, outf_syn)

            exit()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_pc', default='/viscam/u/alexzhou907/01DATA/shapenet/ShapeNetCore.v2.PC15k', help='input batch size')
    parser.add_argument('--dataroot_sv', default='/viscam/u/alexzhou907/01DATA/shapenet/shapenet_mit_preprocessed',
                        help='input batch size')
    parser.add_argument('--classes', default=['chair'])

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--eval_recon_mvr', default=False)
    parser.add_argument('--generate_multimodal', default=False)
    parser.add_argument('--eval_saved', default=False)
    parser.add_argument('--eval_redwood', default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--svpoints', default=200)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')


    # constrain function
    parser.add_argument('--constrain_eps', default=0.2)
    parser.add_argument('--constrain_steps', type=int, default=1)

    parser.add_argument('--ckpt_dir', default='/viscam/u/alexzhou907/research/diffusion/shape_completion/output/6_res32_pc_chair_mse_fs_dattng_2e2-1e4_linbeta_0wd_0.1do/2020-11-01-19-21-18', help="path to netE (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='/viscam/u/alexzhou907/research/diffusion/shapenet/output/test/2020-10-10-20-11-46/syn/epoch_2799_samples.pth')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()

    main(opt)
