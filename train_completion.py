import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_completion import PVCNN2Base
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import ShapeNet_Multiview_Points
'''
some utils
'''
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

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

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
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

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
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


def get_dataset(dataroot_pc, dataroot_sv, npoints, svpoints, category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot_pc,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    tr_dataset = ShapeNet_Multiview_Points(root_pc=dataroot_pc, root_views=dataroot_sv,
                                            cache=os.path.join(dataroot_pc, '../cache'), split='train',
        categories=[category],
        npoints=npoints, sv_samples=svpoints,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(gpu, opt, output_dir, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)


    ''' data '''
    train_dataset = get_dataset(opt.dataroot_pc, opt.dataroot_sv, opt.npoints, opt.svpoints,opt.category)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)


    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer= optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0


    for epoch in range(start_epoch, opt.niter):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):
            randind = np.random.choice(20) #20 views
            x = data['train_points'].transpose(1,2)
            sv_x = data['sv_points'][:,randind].transpose(1,2)

            sv_x[:,:,opt.svpoints:] = x[:,:,opt.svpoints:]
            noises_batch = noises_init[data['idx']].transpose(1,2)

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                sv_x = sv_x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
            elif opt.distribution_type == 'single':
                sv_x = sv_x.cuda()
                noises_batch = noises_batch.cuda()

            loss = model.get_loss_iter(sv_x, noises_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()


            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm: {:>10.2f},   netgradNorm: {:>10.2f}     '
                             .format(
                        epoch, opt.niter, i, len(dataloader),loss.item(),
                    netpNorm, netgradNorm,
                        ))


        if (epoch + 1) % opt.diagIter == 0 and should_diag:

            logger.info('Diagnosis:')

            x_range = [x.min().item(), x.max().item()]
            kl_stats = model.all_kl(sv_x)
            logger.info('      [{:>3d}/{:>3d}]    '
                         'x_range: [{:>10.4f}, {:>10.4f}],   '
                         'total_bpd_b: {:>10.4f},    '
                         'terms_bpd: {:>10.4f},  '
                         'prior_bpd_b: {:>10.4f}    '
                         'mse_bt: {:>10.4f}  '
                .format(
                epoch, opt.niter,
                *x_range,
                kl_stats['total_bpd_b'].item(),
                kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(), kl_stats['mse_bt'].item()
            ))



        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval')

            model.eval()
            m, s = train_dataset.all_points_mean.reshape(1, -1), train_dataset.all_points_std.reshape(1, -1)

            with torch.no_grad():

                x_gen_eval = model.gen_samples(sv_x[:,:,:opt.svpoints], sv_x[:,:,opt.svpoints:].shape, sv_x.device, clip_denoised=False).detach().cpu()


                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info('      [{:>3d}/{:>3d}]  '
                             'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                             'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.niter,
                    *gen_eval_range, *gen_stats,
                ))

            export_to_pc_batch('%s/epoch_%03d_samples_eval' % (outf_syn, epoch),
                                    (x_gen_eval.transpose(1, 2)*s+m).numpy()*3)

            export_to_pc_batch('%s/epoch_%03d_ground_truth' % (outf_syn, epoch),
                               (sv_x.transpose(1, 2).detach().cpu()*s+m).numpy()*3)

            export_to_pc_batch('%s/epoch_%03d_partial' % (outf_syn, epoch),
                               (sv_x[:,:,:opt.svpoints].transpose(1, 2).detach().cpu()*s+m).numpy()*3)


            model.train()







        if (epoch + 1) % opt.saveIter == 0:

            if should_diag:


                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])

    dist.destroy_process_group()

def main():
    opt = parse_args()

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''

    train_dataset = get_dataset(opt.dataroot_pc, opt.dataroot_sv, opt.npoints, opt.svpoints,opt.category)
    noises_init = torch.randn(len(train_dataset), opt.npoints-opt.svpoints, opt.nc)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_pc', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--dataroot_sv', default='GenReData/')
    parser.add_argument('--category', default='chair')

    parser.add_argument('--bs', type=int, default=48, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

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

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=50, help='unit: epoch')
    parser.add_argument('--vizIter', default=50, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
