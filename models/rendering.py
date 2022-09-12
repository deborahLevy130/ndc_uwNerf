import torch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                ts,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                uw_nerf = False,
                no_atten = False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        typ = model.typ
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time:
            if model.uw_model_trans:
                b_embedded_ = repeat(b_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                if model.uw_model_trans:
                    inputs = [xyz_embedded,b_embedded_[i:i + chunk]]

                    out_chunks += [model(inputs, sigma_only=True)]
                else:
                    out_chunks += [model(xyz_embedded, sigma_only=True)]
            out = torch.cat(out_chunks, 0)
            if model.uw_model_trans:
                # out = rearrange(out, '(n1 n2) 2 -> n1 n2 2', n1=N_rays, n2=N_samples_)
                static_sigmas = torch.unsqueeze(out[..., 0],dim = 1)
                static_sigmas = rearrange(static_sigmas, '(n1 n2) 1 -> n1 n2 ', n1=N_rays, n2=N_samples_)
                backscatter_sigmas = torch.unsqueeze(out[...,1],dim=1)
                backscatter_sigmas = rearrange(backscatter_sigmas, '(n1 n2) 1 -> n1 n2 ', n1=N_rays, n2=N_samples_)
            else:
                static_sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            # create other necessary inputs
            if model.encode_appearance:
                a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if output_transient:
                t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if model.uw_model_trans:
                b_embedded_ = repeat(b_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                # inputs for original NeRF
                inputs = [embedding_xyz(xyz_[i:i+chunk]), dir_embedded_[i:i+chunk]]
                # additional inputs for NeRF-W
                if model.encode_appearance and not(model.uw_model): #uncomment in order to input embedding
                    inputs += [a_embedded_[i:i+chunk]]
                if output_transient:
                    inputs += [t_embedded_[i:i+chunk]]
                if model.uw_model_trans:
                    inputs += [b_embedded_[i:i + chunk]]
                out_chunks += [model(torch.cat(inputs, 1), output_transient=output_transient)]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            static_rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3] # (N_rays, N_samples_)
            if output_transient:
                transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                transient_betas = out[..., 8]
            if model.uw_model_trans and output_transient:
                backscatter_rgbs = out[..., 9:12]
                backscatter_sigmas = out[..., 12]
                backscatter_betas = out[..., 13]
            elif model.uw_model_trans:
                backscatter_rgbs = out[..., 4:7]
                backscatter_sigmas = out[..., 7]
                backscatter_betas = out[..., 8]

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        if model.ndc:
            deltas = torch.cat([deltas, delta_inf], -1) *torch.norm(rays_d.squeeze(dim=1), dim=-1, keepdim=True) # (N_rays, N_samples_)
        else:
            deltas = torch.cat([deltas, delta_inf], -1)
 # (N_rays, N_samples_)

        if output_transient:
            static_alphas = 1-torch.exp(-deltas*static_sigmas)
            transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
            alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
        elif model.uw_model_trans:
            static_alphas = 1 - torch.exp(-deltas * static_sigmas)
            backscatter_alphas = 1 - torch.exp(-deltas * backscatter_sigmas)
            alphas = 1 - torch.exp(-deltas * (static_sigmas + backscatter_sigmas))
        else:
#             noise = torch.randn_like(static_sigmas) * noise_std
            alphas = 1-torch.exp(-deltas*static_sigmas)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        if output_transient:
            static_weights = static_alphas * transmittance
            transient_weights = transient_alphas * transmittance
        elif model.uw_model_trans:
            static_weights = static_alphas * transmittance
            backscatter_weights = backscatter_alphas * transmittance

        weights = alphas * transmittance
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        if output_transient:
            results['transient_sigmas'] = transient_sigmas
        if test_time and typ == 'coarse':
            return


        if output_transient:
            if model.uw_model:
                if model.uw_model_trans:
                    a = repeat(a_embedded[:,0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    b = repeat(a_embedded[:,3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    c = repeat(a_embedded[:,6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    B_inf = repeat(b_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    beta_b = repeat(b_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                            torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                    attenuation = (torch.exp(-5*torch.sigmoid(betaD)*rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                    backscatter = rearrange(torch.softmax(transmittance,dim=1), 'n1 n2 -> n1 n2 1')*backscatter_rgbs
                    static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(static_rgbs)*attenuation
                                            +backscatter ,
                                     'n1 n2 c -> n1 c', 'sum')
                else:
                    a = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    b = repeat(a_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    c = repeat(a_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    B_inf = repeat(a_embedded[:, 12:15], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    beta_b = repeat(a_embedded[:, 15:], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                    betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                            torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                    attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                    direct = reduce(
                        rearrange(static_weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(static_rgbs) * attenuation, 'n1 n2 c -> n1 c', 'sum')
                    BS =  reduce(rearrange(torch.softmax(transmittance, dim=1),
                                                               'n1 n2 -> n1 n2 1') *B_inf * (torch.exp(
                        -5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))), 'n1 n2 c -> n1 c', 'sum')
                    # static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(
                    #     static_rgbs) * attenuation + rearrange(torch.softmax(transmittance, dim=1),
                    #                                            'n1 n2 -> n1 n2 1') *B_inf * (torch.exp(
                    #     -5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))),
                    #                         'n1 n2 c -> n1 c', 'sum')
                    static_rgb_map = direct+BS
                    results[f'rgbBS_{typ}'] = BS
                    results[f'direct_{typ}'] = direct
            elif model.transient_uw:

                B_inf = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                beta_b = repeat(a_embedded[:, 3:], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                analiticBS = reduce(rearrange(torch.softmax(transmittance
                                                     , dim=1),
                                       'n1 n2 -> n1 n2 1') * B_inf * (
                                 torch.exp(-5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))), 'n1 n2 c -> n1 c','sum')
                # BS = reduce(rearrange(torch.softmax(transmittance, dim=1),
                #                       'n1 n2 -> n1 n2 1') * static_rgbs, 'n1 n2 c -> n1 c', 'sum')

                # static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(
                #     static_rgbs) * attenuation + rearrange(torch.softmax(transmittance, dim=1),
                #                                            'n1 n2 -> n1 n2 1') *B_inf * (torch.exp(
                #     -5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))),
                #                         'n1 n2 c -> n1 c', 'sum')
                BS = reduce(rearrange(torch.softmax(transmittance
                                                     , dim=1),
                                       'n1 n2 -> n1 n2 1') * B_inf * (
                                 torch.exp(-5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))), 'n1 n2 c -> n1 c','sum')
                static_rgb_map =  BS
                results[f'rgbBS_{typ}'] = BS
                results[f'analiticBS_{typ}'] = analiticBS
                results[f'BS_coeff_{typ}'] = a_embedded
            else:
                static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                    'n1 n2 c -> n1 c', 'sum')
            if white_back:
                static_rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            if model.transient_uw:
                a = repeat(t_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                b = repeat(t_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                c = repeat(t_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                d = repeat(t_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                # B_inf = repeat(t_embedded[:, 12:15], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                # beta_b = repeat(t_embedded[:, 15:], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                        torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                direct = reduce(
                    rearrange(transient_weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(transient_rgbs) * attenuation,
                    'n1 n2 c -> n1 c', 'sum')
                # BS = reduce(rearrange(torch.softmax(transmittance, dim=1),
                #                       'n1 n2 -> n1 n2 1') * B_inf * (torch.exp(
                #     -5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))), 'n1 n2 c -> n1 c', 'sum')
                transient_rgb_map = direct
                # results[f'rgbBS_{typ}'] = BS
                results[f'direct_{typ}'] = direct
                results[f'D_coeff_{typ}'] = t_embedded
            else:
                transient_rgb_map = \
                    reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
            results['beta'] = reduce(transient_weights*transient_betas, 'n1 n2 -> n1', 'sum')
            # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
            # See "Notes on differences with the paper" in README.
            results['beta'] += model.beta_min
            
            # the rgb maps here are when both fields exist
            results['_rgb_fine_static'] = static_rgb_map
            results['_rgb_fine_transient'] = transient_rgb_map
            results['rgb_fine'] = static_rgb_map if model.uw_model_trans else static_rgb_map + transient_rgb_map
            results['transient_rgb_map'] = transient_rgb_map
            results['depth_fine_transient'] = \
                reduce(transient_weights * z_vals, 'n1 n2 -> n1', 'sum')
            if typ == 'coarse':
                results[f'rgb_{typ}'] = static_rgb_map
            if test_time:
                # Compute also static and transient rgbs when only one field exists.
                # The result is different from when both fields exist, since the transimttance
                # will change.
                static_alphas_shifted = \
                    torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
                static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
                static_weights_ = static_alphas * static_transmittance
                transient_alphas_shifted = \
                    torch.cat([torch.ones_like(transient_alphas[:, :1]), 1 - transient_alphas], -1)
                transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1)
                transient_weights_ = transient_alphas * transient_transmittance

                if model.uw_model:
                    if model.uw_model_trans:
                        a = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        b = repeat(a_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        c = repeat(a_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        B_inf = repeat(b_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        beta_b = repeat(b_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                                torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                        attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                        backscatter = rearrange(torch.softmax(transmittance, dim=1),
                                                'n1 n2 -> n1 n2 1') * backscatter_rgbs

                        static_rgb_map_ = reduce(
                            rearrange(static_weights_, 'n1 n2 -> n1 n2 1') * torch.sigmoid(static_rgbs) * attenuation
                            + backscatter,
                            'n1 n2 c -> n1 c', 'sum')
                    else:
                        a = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        b = repeat(a_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        c = repeat(a_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        B_inf = repeat(a_embedded[:, 12:15], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        beta_b = repeat(a_embedded[:, 15:], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                        betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                                torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                        attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))) #TODO: check difference between weight and weights_
                        static_rgb_map_ = reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1') * torch.sigmoid(
                            static_rgbs) * attenuation + rearrange(torch.softmax(static_transmittance, dim=1),
                                                                   'n1 n2 -> n1 n2 1') * B_inf * (torch.exp(
                            -5 * torch.sigmoid(beta_b) * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))),
                                                'n1 n2 c -> n1 c', 'sum')

                else:
                    static_rgb_map_ = \
                        reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1')*static_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                if white_back:
                    static_rgb_map_ += 1-rearrange(weights_sum, 'n -> n 1')
                results['rgb_fine_static'] = static_rgb_map_
                results['depth_fine_static'] = \
                    reduce(static_weights_*z_vals, 'n1 n2 -> n1', 'sum')

                transient_alphas_shifted = \
                    torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1)
                transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1)
                transient_weights_ = transient_alphas * transient_transmittance
                results['rgb_fine_transient'] = \
                    reduce(rearrange(transient_weights_, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                results['depth_fine_transient'] = \
                    reduce(transient_weights_*z_vals, 'n1 n2 -> n1', 'sum')

        else: # no transient field
            if model.uw_model and not(model.uw_model_trans):
                a = repeat(a_embedded[:,0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                b = repeat(a_embedded[:,3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                c = repeat(a_embedded[:,6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                B_inf = repeat(a_embedded[:, 12:15], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                beta_b = repeat(a_embedded[:, 15:], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                        torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                attenuation = (torch.exp(-5*torch.sigmoid(betaD)*rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                # rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * torch.sigmoid(static_rgbs)*attenuation+rearrange(torch.softmax(transmittance,dim=1), 'n1 n2 -> n1 n2 1')*B_inf*(torch.exp(-5*torch.sigmoid(beta_b)*rearrange(z_vals, 'n1 n2 -> n1 n2 1'))) ,
                #                  'n1 n2 c -> n1 c', 'sum')
                rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * (static_rgbs)*attenuation+rearrange(torch.softmax(transmittance,dim=1), 'n1 n2 -> n1 n2 1')*B_inf*(torch.exp(-5*torch.sigmoid(beta_b)*rearrange(z_vals, 'n1 n2 -> n1 n2 1'))) ,
                                 'n1 n2 c -> n1 c', 'sum')

                #+rearrange(torch.softmax(transmittance,dim=1), 'n1 n2 -> n1 n2 1')*B_inf*torch.sigmoid(torch.exp(-beta_b*rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
            elif model.uw_model_trans:
                a = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                b = repeat(a_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                c = repeat(a_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                d = repeat(a_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                B_inf = repeat(b_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                beta_b = repeat(b_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
                betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                        torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
                attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
                backscatter =  rearrange(torch.softmax(transmittance
                                                       , dim=1),
                                        'n1 n2 -> n1 n2 1') * backscatter_rgbs #rearrange(torch.softmax(backscatter_weights, dim=1),
                                                                                #'n1 n2 -> n1 n2 1') * backscatter_rgbs
                analiticBS = rearrange(torch.softmax(transmittance
                                                       , dim=1),
                                        'n1 n2 -> n1 n2 1') * B_inf*(torch.exp(-5*torch.sigmoid(beta_b)*rearrange(z_vals, 'n1 n2 -> n1 n2 1')))

                rgb_map = reduce(
                    rearrange(static_weights, 'n1 n2 -> n1 n2 1') * static_rgbs * attenuation
                    + backscatter,
                    'n1 n2 c -> n1 c', 'sum')
                results[f'analiticBS_{typ}'] =  reduce(analiticBS, 'n1 n2 c -> n1 c', 'sum')
                results[f'rgbBS_{typ}'] =  reduce(backscatter, 'n1 n2 c -> n1 c', 'sum')
                results[f'direct_{typ}'] = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * static_rgbs * attenuation,'n1 n2 c -> n1 c', 'sum')
                results[f'backscatter_sigmas'] = backscatter_sigmas
                results[f'betaBS_{typ}'] = reduce(backscatter_weights * backscatter_betas, 'n1 n2 -> n1', 'sum')
                results[f'betaBS_{typ}'] += model.beta_min
            else:
                rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
            if white_back:
                rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            results[f'rgb_{typ}'] = rgb_map
            results[f'sigma_var_{typ}']=static_sigmas.var(dim=1)

        results[f'depth_{typ}'] = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')
        results[f'sigma_var_{typ}'] = static_sigmas.var(dim=1)
        results[f'sigma_{typ}'] = static_sigmas
        results[f'alphas_{typ}'] = alphas
        results[f'z_vals_{typ}'] = z_vals
        results[f'transmittance_{typ}'] = transmittance
        return

    def uw_inference(results, model, xyz, z_vals, **kwargs):
            """
            Helper function that performs model inference.
            Inputs:
                results: a dict storing all results
                model: NeRF model (coarse or fine)
                xyz: (N_rays, N_samples_, 3) sampled positions
                      N_samples_ is the number of sampled points on each ray;
                                 = N_samples for coarse model
                                 = N_samples+N_importance for fine model
                z_vals: (N_rays, N_samples_) depths of the sampled positions
                test_time: test time or not
            """
            typ = model.typ
            N_samples_ = xyz.shape[1]
            xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)

            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []

             # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            # create other necessary inputs

            a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)

            t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)

            if model.input_z:
                origins = repeat(rays_o.squeeze(dim=1),'n1 c -> (n1 n2) c',n2=N_samples_)

            for i in range(0, B, chunk):
                # inputs for original NeRF
                if model.input_z:
                    distance_from_cam = torch.linalg.vector_norm((xyz_[i:i + chunk]-origins[i:i+chunk]),dim=-1,keepdim = True)
                    z_vals_input = distance_from_cam
                    inputs = [embedding_xyz(xyz_[i:i + chunk]), dir_embedded_[i:i + chunk],a_embedded_[i:i + chunk],t_embedded_[i:i + chunk],z_vals_input ]
                else:
                    inputs = [embedding_xyz(xyz_[i:i + chunk]), dir_embedded_[i:i + chunk],a_embedded_[i:i + chunk],t_embedded_[i:i + chunk] ]


                out_chunks += [model(torch.cat(inputs, 1))]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)

            object_sigmas = out[..., :1].squeeze()  # (N_rays, N_samples_)
            if not(model.analitic_bs):

                bs_rgbs = out[..., 1:4]  # (N_rays, N_samples_, 3)

                direct_rgbs = out[..., 4:7]
                if no_atten:
                    direct_attenuation = torch.ones_like(out[..., 7:10])
                else:
                    direct_attenuation = (out[..., 7:10])
                bs_beta = out[...,10:]
            else:
                direct_rgbs = out[..., 1:4]
                if no_atten:
                    direct_attenuation = torch.ones_like(out[..., 4:])
                else:
                    direct_attenuation = (out[..., 4:])

            # Convert these values using volume rendering
            deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
            if model.ndc:
                deltas = torch.cat([deltas, delta_inf], -1) * torch.norm(rays_d.squeeze(dim=1), dim=-1,
                                                                         keepdim=True)  # (N_rays, N_samples_)
            else:
                deltas = torch.cat([deltas, delta_inf], -1)

            alphas = 1 - torch.exp(-deltas * object_sigmas)


            alphas_shifted = \
                torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas], -1)  # [1, 1-a1, 1-a2, ...]
            transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)  # [1, 1-a1, (1-a1)(1-a2), ...]

            weights = alphas * transmittance
            weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

            results[f'weights_{typ}'] = weights
            results[f'opacity_{typ}'] = weights_sum

            a = repeat(t_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            b = repeat(t_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            c = repeat(t_embedded[:, 6:9], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            d = repeat(t_embedded[:, 9:12], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            B_inf = repeat(a_embedded[:, 0:3], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            beta_b = repeat(a_embedded[:, 3:6], 'n1 n2 -> n1 N n2 ', N=N_samples_)
            betaD = a * torch.exp(b * rearrange(z_vals, 'n1 n2 -> n1 n2 1')) + c * \
                    torch.exp(d * rearrange(z_vals, 'n1 n2 -> n1 n2 1'))
            analitic_attenuation = (torch.exp(-5 * torch.sigmoid(betaD) * rearrange(z_vals, 'n1 n2 -> n1 n2 1')))
            dist_from_cam = torch.linalg.vector_norm((xyz - repeat(rays_o.squeeze(dim=1), 'n1 c -> n1 n2 c', n2=N_samples_)), dim=-1,
                                     keepdim=True)
            analitic_bs = B_inf * (torch.exp(
                            -5 * torch.sigmoid(beta_b) * (xyz[:,:,-1].unsqueeze(dim=2))))
            # backscatter = reduce(rearrange(torch.softmax(transmittance, dim=1),
            #                         'n1 n2 -> n1 n2 1') * bs_rgbs,'n1 n2 c -> n1 c', 'sum')
            if not(model.analitic_bs):
                backscatter = reduce(rearrange(transmittance,
                                           'n1 n2 -> n1 n2 1') * bs_rgbs, 'n1 n2 c -> n1 c', 'sum')
            # backscatter = reduce((rearrange((weights),
            #                                'n1 n2 -> n1 n2 1') * bs_rgbs), 'n1 n2 c -> n1 c', 'sum')
            # bs_betas = reduce(torch.softmax(transmittance, dim=1)*bs_beta.squeeze(), 'n1 n2 -> n1', 'sum')
                bs_betas = reduce(transmittance*bs_beta.squeeze(), 'n1 n2 -> n1', 'sum')

            # bs_betas = reduce((weights)*bs_beta.squeeze(), 'n1 n2 -> n1', 'sum')

            # bs_betas = reduce(weights*bs_beta.squeeze(), 'n1 n2 -> n1', 'sum')
                bs_betas += model.beta_min
            else:
                backscatter = (1/N_samples_)*reduce(rearrange(transmittance,
                                                        'n1 n2 -> n1 n2 1') * analitic_bs, 'n1 n2 c -> n1 c','sum')

            # direct =  reduce(
            #     rearrange(weights, 'n1 n2 -> n1 n2 1') * direct_rgbs * direct_attenuation,'n1 n2 c -> n1 c', 'sum')
            direct = reduce(
                rearrange(weights, 'n1 n2 -> n1 n2 1') * direct_rgbs *direct_attenuation, 'n1 n2 c -> n1 c', 'sum')
            static_rgb_map = direct+backscatter





                # results[f'bs_raw_{typ}'] = reduce(bs_rgbs,'n1 n2 c -> n1 c', 'sum')

                # results[f'attenuation_analitic_{typ}'] = reduce(analitic_attenuation,'n1 n2 c -> n1 c', 'sum')
                # results[f'attenuation_raw_{typ}'] = reduce(direct_attenuation,'n1 n2 c -> n1 c', 'sum')

                # the rgb maps here are when both fields exist
            results[f'rgb_{typ}'] = static_rgb_map
            if typ == 'fine':
                results[f'rgbBS_{typ}'] = backscatter
                results[f'bs_analitic_{typ}'] = (1/N_samples_)*reduce(rearrange(transmittance,
                                                        'n1 n2 -> n1 n2 1') * analitic_bs, 'n1 n2 c -> n1 c','sum')
                # results[f'bs_analitic_{typ}'] = analitic_bs

                results[f'direct_{typ}'] = direct
                results[f'sigma_var_{typ}'] = object_sigmas.var(dim=1)
                results[f'sigma_{typ}'] = object_sigmas
                if not(model.analitic_bs):
                    results['BS'] = bs_rgbs
                # if model.input_z:
                # results['dist_from_cam'] = torch.linalg.vector_norm((xyz - repeat(rays_o.squeeze(dim=1),'n1 c -> n1 n2 c',n2=N_samples_)),dim=-1,keepdim = True)
                results[f'alphas_{typ}'] = alphas
                results[f'z_vals_{typ}'] = z_vals
                results[f'beta_b'] = beta_b
                results[f'B_inf'] = B_inf
                results[f'transmittance_{typ}'] = transmittance
                results['depth_fine'] = \
                    reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')
                # results['atten_fine'] = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') *direct_attenuation, 'n1 n2 c -> n1 c', 'sum')
            return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    # near, far = rays[:, 6:7], rays[:, 7:8] # both
    model = models['coarse']
    if model.ndc:
        near, far = torch.zeros_like(rays[:, 6:7]), torch.ones_like(rays[:, 7:8])
        dir_embedded = embedding_dir(
            kwargs.get('view_dir', rays_d) / torch.norm(rays_d.squeeze(dim=1), dim=-1, keepdim=True))

    else:
        near, far = rays[:, 6:7], rays[:, 7:8]# both (N_rays, 1)
        dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))
        # print(near)
    # print(far)
    # Embed direction


    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    output_transient = False
    model = models['coarse']
    if model.encode_appearance:
        if 'a_embedded' in kwargs:
            a_embedded = kwargs['a_embedded']
        else:
            a_embedded = embeddings['a'](ts)  # TODO: check ts value
    if not(uw_nerf):
        if model.uw_model_trans:
            b_embedded = embeddings['b'](ts)
    if uw_nerf:
        if model.encode_transient:
            if 't_embedded' in kwargs:
                t_embedded = kwargs['t_embedded']
            else:
                t_embedded = embeddings['t'](ts)
        uw_inference(results, models['coarse'], xyz_coarse, z_vals, **kwargs)
    else:
        inference(results, models['coarse'], xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        model = models['fine']
        if model.encode_appearance:
            if 'a_embedded' in kwargs:
                a_embedded = kwargs['a_embedded']
            else:
                a_embedded = embeddings['a'](ts) #TODO: check ts value
        output_transient = kwargs.get('output_transient', True) and model.encode_transient
        if output_transient:
            if 't_embedded' in kwargs:
                t_embedded = kwargs['t_embedded']
            else:
                t_embedded = embeddings['t'](ts)
        if not(uw_nerf):
            if model.uw_model_trans:
                b_embedded = embeddings['b'](ts)
        if uw_nerf:
            if model.encode_transient:
                if 't_embedded' in kwargs:
                    t_embedded = kwargs['t_embedded']
                else:
                    t_embedded = embeddings['t'](ts)
            uw_inference(results, model, xyz_fine, z_vals, **kwargs)
        else:
            inference(results, model, xyz_fine, z_vals, test_time, **kwargs)

    return results
