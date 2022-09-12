import torch
from torch import nn
from einops import rearrange, reduce, repeat

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        self.cos = nn.CosineSimilarity(dim=0)
        self.cos_i = nn.CosineSimilarity(dim=1)

    def forward(self, inputs, targets):
        ret = {}
        ret['c_l'] = 0.5 * (((inputs['rgb_coarse']-targets)**2)).mean()
        # if 'rgbBS_fine' in inputs:
        #     ret['bs_f'] = 0.5 * ((inputs['rgbBS_fine']-inputs['analiticBS_fine'])**2).mean()
        #     ret['bs_c'] = 0.5 * ((inputs['rgbBS_coarse'] - inputs['analiticBS_coarse']) ** 2).mean()
        #     ret['b_l'] = 3 + torch.log(inputs['betaBS_fine']).mean()
        #     ret['s_l'] = self.lambda_u * inputs['backscatter_sigmas'].mean()
        #     ret['f_l'] = \
        #         ((inputs['rgb_fine'] - targets) ** 2 / (2 * inputs['betaBS_fine'].unsqueeze(1) ** 2)).mean()
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()
            if 'direct_fine' in inputs:
                # ret['f_l'] = 0.5 * (((inputs['rgb_fine'] - targets) ** 2)/(2*repeat((1/inputs['sigma_var_fine'])**2,'n1->n1 3'))).mean()
                # ret['f_l'] = \
                #     ((inputs['rgb_fine']-targets)**2/(2*inputs['bs_betas_fine'].unsqueeze(1)**2)).mean()
                ret['f_l'] = 0.5 * (((inputs['rgb_fine'] - targets) ** 2)).mean()
                # ret['b_l'] = 3 + torch.log(inputs['bs_betas_fine']).mean()
                # ret['b_inf'] = 0.5 * (torch.var(inputs['weights_fine'], dim=1).unsqueeze(dim=1)*(((inputs['rgbBS_fine'] - inputs['B_inf']) ** 2))).mean()
                # ret['b_l'] = 10*(3 + torch.log(1/inputs['sigma_var_fine'])).mean()
                # ret['bs_f'] = 0.5 * ((inputs['rgbBS_fine']-inputs['bs_analitic_fine'])**2).mean()
                ret['beta']=-0.5*torch.max(torch.zeros_like(inputs['beta_b']),inputs['beta_b']).mean()
                ret['B_inf'] = -0.5*torch.max(torch.zeros_like(inputs['B_inf']),inputs['B_inf']).mean()
                # ret['a_fine'] = 0.5 * ((inputs['attenuation_analitic_fine']-inputs['attenuation_raw_fine'])**2).mean()
                # ret['bs_coarse'] = 0.5 * ((inputs['bs_analitic_coarse']-inputs['bs_raw_coarse'])**2).mean()
                # ret['a_coarse'] = 0.5 * ((inputs['attenuation_analitic_coarse']-inputs['attenuation_raw_coarse'])**2).mean()
                ret['s_f'] = self.lambda_u *inputs['sigma_fine'].mean()
            # if 'atten_fine' in inputs:
                # ret['cor_d_b'] = self.lambda_u*0.1*((1-self.cos(inputs['rgbBS_fine']-inputs['rgbBS_fine'].mean(dim=0, keepdim=True),repeat(inputs['depth_fine'].unsqueeze(dim=1),'n1 1 -> n1 3')-repeat(inputs['depth_fine'].unsqueeze(dim=1),'n1 1 -> n1 3').mean(dim=0, keepdim=True)))**2).mean()
                # ret['cor_b_a'] = self.lambda_u*0.1*((-1-self.cos(inputs['rgbBS_fine']-inputs['rgbBS_fine'].mean(dim=0, keepdim=True),inputs['atten_fine']-inputs['atten_fine'].mean(dim=0, keepdim=True)))**2).mean()
                # ret['cor_bi_zi'] = self.lambda_u*0.1*((-1-self.cos_i(inputs['BS']-inputs['BS'].mean(dim=1,keepdim=True),inputs['dist_from_cam']-inputs['dist_from_cam'].mean(dim=1,keepdim=True)))**2).mean()
                # ret['s_c'] = self.lambda_u *10* inputs['sigma_coarse'].mean()
                # ret['s_c'] = self.lambda_u * inputs['sigma_coarse'].mean()
            else:
                ret['f_l'] = \
                    ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                # ret['f_l'] = \
                #     ((inputs['rgb_fine'] - targets) ** 2 / (2 * inputs['depth_fine'].unsqueeze(1) ** 2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean()
                # ret['b_l'] = 3 + torch.log(inputs['depth_fine']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()
                # ret['bs_f'] = 0.5 * ((inputs['rgbBS_fine']-inputs['analiticBS_fine'])**2).mean()
                #ret['beta_d'] = self.lambda_u*torch.sum(torch.relu(-inputs['D_coeff_fine']))
                #ret['beta_b'] = self.lambda_u * torch.sum(torch.relu(-inputs['BS_coeff_fine']))
                # ret['bs_c'] = 0.5 * ((inputs['rgbBS_coarse'] - inputs['analiticBS_coarse']) ** 2).mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss}