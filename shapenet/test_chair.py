import torch
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from model.unet import get_model
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from utils.mitsuba_renderer import write_to_xml_batch
from model.pvcnn_generation import PVCNN2Base

from tqdm import tqdm

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
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

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

        model_output = denoise_fn(data, t)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
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

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False, use_var=True):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean
        if use_var:
            sample = sample + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                      clip_denoised=True, max_timestep=None, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """
        if max_timestep is None:
            final_time = self.num_timesteps
        else:
            final_time = max_timestep

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, final_time if not keep_running else len(self.betas))):
            img_t = constrain_fn(img_t, t)
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False).detach()


        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t

    def interpolate(self, x0, x1, t, lamb, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding0 = self.q_sample(x0, t_vec)
        encoding1 = self.q_sample(x1, t_vec)

        enc = encoding0 * lamb + (1-lamb) * encoding1

        img_t = enc

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t

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

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim, use_att=args.attention,
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

        assert out.shape == torch.Size([B, D, N])
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

    def gen_samples(self, shape, device, noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                    clip_denoised=False, max_timestep=None,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            constrain_fn=constrain_fn,
                                            clip_denoised=clip_denoised, max_timestep=max_timestep,
                                            keep_running=keep_running)

    def reconstruct(self, x0, t, constrain_fn=lambda x, t:x):

        return self.diffusion.reconstruct(x0, t, self._denoise, constrain_fn=constrain_fn)

    def interpolate(self, x0, x1, t, lamb, constrain_fn=lambda x, t:x):

        return self.diffusion.interpolate(x0, x1, t, lamb, self._denoise, constrain_fn=constrain_fn)

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

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category, split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category, split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset

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

def get_mvr_dataset(pc_dataroot, mesh_root, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=pc_dataroot,
        categories=category, split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet_Multiview_Points(root_pc=pc_dataroot, root_mesh=mesh_root,
                                            cache=os.path.join(mesh_root, '../cache'), split='val',
        categories=category,
        npoints=npoints, sv_samples=200,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )


    return te_dataset

def get_mvr_dataset_v2(pc_dataroot, views_root, npoints,category):
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


def evaluate_gen(opt, ref_pcs, logger):

    if ref_pcs is None:
        _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.classes, use_mask=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
        ref = []
        for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
            x = data['test_points']
            m, s = data['mean'].float(), data['std'].float()

            ref.append(x*s + m)

        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
    # results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    # results = {k: (v.cpu().detach().item()
    #                if not isinstance(v, float) else v) for k, v in results.items()}
    #
    # pprint(results)
    # logger.info(results)

    jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
    pprint('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))

def evaluate_recon(opt, netE, save_dir, logger):
    test_dataset = get_mvr_dataset(opt.dataroot, '/viscam/u/alexzhou907/01DATA/shapenet/ShapeNetCore.v2',
                                      opt.npoints, opt.classes)
    # _, test_dataset =  get_dataset(opt.dataroot, opt.npoints, opt.classes, use_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)
    ref = []
    samples = []
    images = []
    masked = []
    k = 0
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):

        randind = i%24
        gt_all = data['test_points'][:,randind:randind+1]
        x_all = data['sv_points'][:,randind:randind+1]
        mask_all= data['masks'][:,randind:randind+1]
        img_all = data['image'][:,randind:randind+1]


        B,V,N,C = x_all.shape

        gt = gt_all.reshape(B*V,N,C).transpose(1,2).contiguous()
        x = x_all.reshape(B*V,N,C).transpose(1,2).contiguous()
        mask = mask_all.reshape(B*V,N,C).transpose(1,2).contiguous()
        img = img_all.reshape(B*V, *img_all.shape[2:])

        m, s = data['mean'].float(), data['std'].float()

        # visualize_pointcloud_batch(os.path.join(save_dir, 'y_%03d.png'%i), x.transpose(1,2) * s + m, None, None, None)
        # for t in [10]:
        # t_vec = torch.empty(gt.shape[0], dtype=torch.int64, device='cuda').fill_(80)
        # recon = netE.diffusion.q_sample(gt.cuda(), t_vec).detach().cpu()
        recon = netE.gen_samples(x.shape, 'cuda',
                                      constrain_fn=get_constrain_function(x.cuda(), mask.cuda(), opt.constrain_eps,
                                                                          opt.constrain_steps),
                                      clip_denoised=False).detach().cpu()

        # recon = recon.transpose(1, 2).contiguous()
        # x = x.transpose(1, 2).contiguous()
        # gt = gt.transpose(1, 2).contiguous()
        # write_to_xml_batch(os.path.join(save_dir, 'intermediate_%03d' % i),
        #                    (recon.detach().cpu() * s[0].squeeze() + m[0].squeeze()).numpy())
        # write_to_xml_batch(os.path.join(save_dir, 'x_%03d' % i),
        #                    (gt.detach().cpu() * s[0].squeeze() + m[0].squeeze()).numpy())
        # write_to_xml_batch(os.path.join(save_dir, 'noise_%03d' % i),
        #                    (torch.randn_like(gt).detach().cpu() * s[0].squeeze() + m[0].squeeze()).numpy())
        # for d in zip(list(data['test_points'].reshape(B*V,N,C)), list(recon), list(x), list(torch.zeros_like(x))):
        #     write_to_xml_batch(os.path.join(save_dir, 'x_%03d'%k), (torch.stack(d[:-1], dim=0)* s[0:1] + m[0:1]).numpy())
        #     visualize_pointcloud_batch(os.path.join(save_dir, 'x_%03d.png'%k), torch.stack(d, dim=0)* s[0:1] + m[0:1], None, None, None)
        #
        #     k+=1

        x_adj = x.reshape(B,V,N,C)* s + m
        recon_adj = recon.reshape(B,V,N,C)* s + m
        img = img.reshape(B,V,*img.shape[1:])

        ref.append( gt_all * s + m)
        masked.append(x_adj[:,:,:test_dataloader.dataset.sv_samples,:])
        samples.append(recon_adj)
        images.append(img)


    ref_pcs = torch.cat(ref, dim=0)
    sample_pcs = torch.cat(samples, dim=0)
    images = torch.cat(images, dim=0)
    masked = torch.cat(masked, dim=0)

    B, V, N, C = ref_pcs.shape


    torch.save(ref_pcs.reshape(B*V, N, C), os.path.join(save_dir, 'recon_gt.pth'))
    torch.save(images.reshape(B*V, *images.shape[2:]), os.path.join(save_dir, 'recon_depth.pth'))
    torch.save(masked.reshape(B*V, *masked.shape[2:]), os.path.join(save_dir, 'recon_masked.pth'))
    # Compute metrics
    results = EMD_CD(sample_pcs.reshape(B*V, N, C),
                     ref_pcs.reshape(B*V, N, C), opt.batch_size, reduced=False)

    results = {ky: val.reshape(B,V) if val.shape == torch.Size([B*V,]) else val for ky, val in results.items()}

    pprint({key: val.mean().item() for key, val in results.items()})
    logger.info({key: val.mean().item() for key, val in results.items()})

    results['pc'] = sample_pcs
    torch.save(results, os.path.join(save_dir, 'ours_results.pth'))

    #
    # results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    #
    # results = {k: (v.cpu().detach().item()
    #                if not isinstance(v, float) else v) for k, v in results.items()}
    # pprint(results)
    # logger.info(results)

def evaluate_recon_mvr(opt, netE, save_dir, logger):
    test_dataset = get_mvr_dataset_v2(opt.dataroot, '/viscam/u/alexzhou907/01DATA/shapenet/shapenet_mit_preprocessed',
                                      opt.npoints, opt.classes)
    # _, test_dataset =  get_dataset(opt.dataroot, opt.npoints, opt.classes, use_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)
    ref = []
    samples = []
    images = []
    masked = []
    k = 0
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):

        gt_all = data['test_points']
        x_all = data['sv_points']
        mask_all= data['masks']
        # img_all = data['image']


        B,V,N,C = x_all.shape
        gt_all = gt_all[:,None,:,:].expand(-1, V, -1,-1)

        x = x_all.reshape(B * V, N, C).transpose(1, 2).contiguous()
        mask = mask_all.reshape(B * V, N, C).transpose(1, 2).contiguous()
        # img = img_all.reshape(B * V, *img_all.shape[2:])

        m, s = data['mean'].float(), data['std'].float()

        recon = netE.gen_samples(x.shape, 'cuda',
                                      constrain_fn=get_constrain_function(x.cuda(), mask.cuda(), opt.constrain_eps,
                                                                          opt.constrain_steps),
                                      clip_denoised=False).detach().cpu()

        recon = recon.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).contiguous()

        x_adj = x.reshape(B,V,N,C)* s + m
        recon_adj = recon.reshape(B,V,N,C)* s + m
        # img = img.reshape(B,V,*img.shape[1:])

        ref.append( gt_all * s + m)
        masked.append(x_adj[:,:,:test_dataloader.dataset.sv_samples,:])
        samples.append(recon_adj)
        # images.append(img)

    ref_pcs = torch.cat(ref, dim=0)
    sample_pcs = torch.cat(samples, dim=0)
    # images = torch.cat(images, dim=0)
    masked = torch.cat(masked, dim=0)

    B, V, N, C = ref_pcs.shape


    torch.save(ref_pcs.reshape(B,V, N, C), os.path.join(save_dir, 'recon_gt.pth'))
    # torch.save(images.reshape(B,V, *images.shape[2:]), os.path.join(save_dir, 'recon_depth.pth'))
    torch.save(masked.reshape(B,V, *masked.shape[2:]), os.path.join(save_dir, 'recon_masked.pth'))
    # Compute metrics
    results = EMD_CD(sample_pcs.reshape(B*V, N, C),
                     ref_pcs.reshape(B*V, N, C), opt.batch_size, reduced=False)

    results = {ky: val.reshape(B,V) if val.shape == torch.Size([B*V,]) else val for ky, val in results.items()}

    pprint({key: val.mean().item() for key, val in results.items()})
    logger.info({key: val.mean().item() for key, val in results.items()})

    results['pc'] = sample_pcs
    torch.save(results, os.path.join(save_dir, 'ours_results.pth'))

    #
    # results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    #
    # results = {k: (v.cpu().detach().item()
    #                if not isinstance(v, float) else v) for k, v in results.items()}
    # pprint(results)
    # logger.info(results)

    del ref_pcs, masked


def generate(netE, opt, logger):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.classes)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    with torch.no_grad():

        samples = []
        ref = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['test_points'].transpose(1,2)
            m, s = data['mean'].float(), data['std'].float()

            gen = netE.gen_samples(x.shape,
                                       'cuda', clip_denoised=False).detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()



            gen = gen * s + m
            x = x * s + m
            samples.append(gen)
            ref.append(x)

            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'x.png'), gen[:64], None,
                                       None, None)

            write_to_xml_batch(os.path.join(str(Path(opt.eval_path).parent), 'xml_samples_%03d'%i), gen[:min(gen.shape[0], 40)].numpy(), cat='chair')

        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)

        torch.save(samples, opt.eval_path)



    return ref



def generate_multimodal(opt, netE, save_dir, logger):
    test_dataset = get_svr_dataset(opt.dataroot, '/viscam/u/alexzhou907/01DATA/shapenet/ShapeNetCore.v2',
                                      opt.npoints, opt.classes)
    # _, test_dataset =  get_dataset(opt.dataroot, opt.npoints, opt.classes, use_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):

        gt_all = data['test_points']
        x_all = data['sv_points']
        mask_all= data['masks']
        img_all = data['image']


        # visualize_pointcloud_batch(os.path.join(save_dir, 'y_%03d.png'%i), x.transpose(1,2) * s + m, None, None, None)
        # for t in [10]:
        # recon = netE.reconstruct2(x.cuda(), mask.cuda(), get_constrain_function(x.cuda(), mask.cuda(), opt.constrain_eps,
        #                                                               opt.constrain_steps)).detach().cpu()

        for v in range(10):
            x = x_all.transpose(1, 2).contiguous()
            mask = mask_all.transpose(1, 2).contiguous()
            img = img_all

            m, s = data['mean'].float(), data['std'].float()

            recon = netE.gen_samples(x.shape, 'cuda',
                                          constrain_fn=get_constrain_function(x.cuda(), mask.cuda(), opt.constrain_eps,
                                                                              opt.constrain_steps),
                                          clip_denoised=False).detach().cpu()

            recon = recon.transpose(1, 2).contiguous()
            x = x.transpose(1, 2).contiguous()

            for p, d in enumerate(zip(list(gt_all), list(recon), list(x), list(img))):

                im = np.fliplr(np.flipud(d[-1]))
                plt.imsave(os.path.join(save_dir, 'depth_%03d_%03d.png'%(i,p)), im, cmap='gray')
                write_to_xml_batch(os.path.join(save_dir, 'x_%03d_%03d'%(i,p), 'mode_%03d'%v), (torch.stack(d[:-1], dim=0)* s[0:1] + m[0:1]).numpy())

                export_to_pc_batch(os.path.join(save_dir, 'x_ply_%03d_%03d'%(i,p), 'mode_%03d'%v),(torch.stack(d[:-1], dim=0)* s[0:1] + m[0:1]).numpy())






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
            #'/viscam/u/alexzhou907/research/diffusion/shapenet/output/66_res32_pc_chair_mse_fs_dattng_2e2-1e4_linbeta_0wd_0.1do/2020-10-04-01-02-45/epoch_1399.pth'
            #'/viscam/u/alexzhou907/research/diffusion/shapenet/output/67_res32_pc_car_mse_fs_dattng_2e2-1e4_linbeta_0wd_0.1do/2020-10-04-01-03-38/epoch_2799.pth'
            opt.netE = '/viscam/u/alexzhou907/research/diffusion/shapenet/output/66_res32_pc_chair_mse_fs_dattng_2e2-1e4_linbeta_0wd_0.1do_best/2020-10-16-12-23-44/epoch_1799.pth'#ckpt
            logger.info("Resume Path:%s" % opt.netE)

            resumed_param = torch.load(opt.netE)
            netE.load_state_dict(resumed_param['model_state'])


            ref = None
            if opt.generate:
                epoch = int(os.path.basename(ckpt).split('.')[0].split('_')[-1])
                opt.eval_path = os.path.join(outf_syn, 'epoch_{}_samples.pth'.format(epoch))
                Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
                ref=generate(netE, opt, logger)
            if opt.eval_gen:
                # Evaluate generation
                evaluate_gen(opt, ref, logger)

            if opt.eval_recon:
                # Evaluate generation
                evaluate_recon(opt, netE, outf_syn, logger)

            if opt.eval_recon_mvr:
                # Evaluate generation
                evaluate_recon_mvr(opt, netE, outf_syn, logger)

            if opt.generate_multimodal:

                generate_multimodal(opt, netE, outf_syn, logger)



            exit()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/viscam/u/alexzhou907/01DATA/shapenet/ShapeNetCore.v2.PC15k', help='input batch size')
    parser.add_argument('--classes', default=['chair'])

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--generate',default=True)
    parser.add_argument('--eval_gen', default=False)
    parser.add_argument('--eval_recon', default=False)
    parser.add_argument('--eval_recon_mvr', default=False)
    parser.add_argument('--generate_multimodal', default=False)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
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
    parser.add_argument('--constrain_eps', default=.051)
    parser.add_argument('--constrain_steps', type=int, default=1)

    parser.add_argument('--ckpt_dir', default='/viscam/u/alexzhou907/research/diffusion/shapenet/output/66_res32_pc_chair_mse_fs_dattng_2e2-1e4_linbeta_0wd_0.1do_best/2020-10-16-12-23-44/', help="path to netE (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='/viscam/u/alexzhou907/research/diffusion/shapenet/output/test_chair/2020-10-18-13-46-21/syn/epoch_1699_samples.pth')

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
    set_seed(opt)

    main(opt)

    # results in /viscam/u/alexzhou907/research/diffusion/shapenet/output/test_chair/2020-10-18-13-46-21