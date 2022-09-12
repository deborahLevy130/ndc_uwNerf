import torch
from torch import nn

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03,uw_model = False, uw_model_trans = False,transient_uw = False,ndc = False):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.uw_model = uw_model
        self.uw_model_trans = uw_model_trans
        self.transient_uw = transient_uw
        self.ndc = ndc
        # self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.encode_appearance = encode_appearance
        if self.uw_model and self.uw_model_trans:
            in_channels_a = 12

        elif self.uw_model:
            in_channels_a = 18
        elif self.transient_uw:
            in_channels_a = 6

        else:
            in_channels_a = in_channels_a


        self.in_channels_a = in_channels_a if encode_appearance else 0
        # self.encode_transient = False if typ=='coarse' else encode_transient
        self.encode_transient = encode_transient
        self.in_channels_t = 12 if self.transient_uw else in_channels_t
        self.in_channels_backscatter = 6
        self.beta_min = beta_min

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        if uw_model:
            self.dir_encoding = nn.Sequential(
                 nn.Linear(W + in_channels_dir, W // 2), nn.ReLU(True)) #uncoment in order to input embbeding inside network
                # nn.Linear(W + in_channels_dir , W // 2), nn.ReLU(True)) #uncoment in order to  not input embbeding inside network
        else:
            self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

        if self.uw_model_trans:
            # transient encoding layers
            self.backscatter_encoding = nn.Sequential(
                nn.Linear(W + self.in_channels_backscatter, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
            # transient output layers
            self.backscatter_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.backscatter_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.backscatter_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            if self.uw_model_trans:
                input_xyz = x[0]
                input_b = x[1]
                    # , input_b = \
                    # torch.split(x, [self.in_channels_xyz,
                    #                 self.in_channels_backscatter], dim=-1)
            else:
                input_xyz = x
        elif output_transient:
            if self.uw_model:
                input_xyz, input_dir_a, input_t  = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_dir,
                                    self.in_channels_t], dim=-1)  # uncomment if don't want to insert learnable params
            elif self.uw_model_trans:
                input_xyz, input_dir_a, input_t, input_b = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_dir,
                                    self.in_channels_t,
                                    self.in_channels_backscatter], dim=-1)
            else:
                input_xyz, input_dir_a, input_t = \
                    torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            if self.uw_model and not(self.uw_model_trans):
                # input_xyz, input_dir_a = \  #uncomment in order to input embedding into network
                # torch.split(x, [self.in_channels_xyz,
                #                 self.in_channels_dir+self.in_channels_a], dim=-1)
                input_xyz, input_dir_a = \
                    torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir], dim=-1) # uncomment if don't want to insert learnable params
            elif self.uw_model_trans:
                input_xyz, input_dir_a, input_b  = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_dir,
                                    self.in_channels_backscatter], dim=-1)
            else:
                input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if self.uw_model_trans:
            backscatter_encoding_input = torch.cat([xyz_encoding_final, input_b], 1)
            backscatter_encoding = self. backscatter_encoding(backscatter_encoding_input)
            backscatter_sigma = self.backscatter_sigma(backscatter_encoding) # (B, 1)
        if sigma_only:
            if self.uw_model_trans:
                return torch.cat([static_sigma, backscatter_sigma], 1) # (B, 2)
            else:
                return static_sigma


        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)
        if self.uw_model_trans:
            backscatter_rgb = self.backscatter_rgb(backscatter_encoding)  # (B, 3)
            backscatter_beta = self.backscatter_beta(backscatter_encoding)  # (B, 1)
            backscatter = torch.cat([backscatter_rgb, backscatter_sigma,
                                   backscatter_beta], 1)  # (B, 5)
        if not output_transient:
            if self.uw_model_trans:
                return torch.cat([static, backscatter], 1) # (B, 9)
            else:
                return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)
        if self.uw_model_trans:
            return torch.cat([static,backscatter, transient], 1) # (B, 9)
        else:
            return torch.cat([static, transient], 1)  # (B, 9)


class NeRFUw(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=True, in_channels_a=6,
                 encode_transient=True, in_channels_t=12,
                 beta_min=0.03,input_z = False,N_samples=64,ndc = False,analitic_bs=False):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.input_z = input_z
        self.N_samples = N_samples
        # self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.encode_appearance = encode_appearance
        self.analitic_bs = analitic_bs
        self.in_channels_a = in_channels_a if encode_appearance else 0
        # self.encode_transient = False if typ=='coarse' else encode_transient
        self.encode_transient = encode_transient
        self.in_channels_t = in_channels_t if encode_appearance else 0
        self.ndc = ndc
        self.beta_min = beta_min

        # xyz encoding layers - MLP G
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # Backscatter encoding layers
        if self.input_z:
            self.backscatter_encoding = nn.Sequential(
                nn.Linear(in_channels_dir+1 + self.in_channels_a, W ), nn.ReLU(True), nn.Linear(W , W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
        elif self.analitic_bs:
            # static output layers
            self.objects_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

            # Direct encoding layers

            self.direct_encoding = nn.Sequential(
                nn.Linear(W + in_channels_dir, W // 2), nn.ReLU(True), nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
            self.direct_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            # self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t, W // 2),nn.ReLU(True),
            #                                         nn.Linear(W // 2, W // 2), nn.ReLU(True),
            #                                         nn.Linear(W // 2, 3), nn.Sigmoid())
            if self.input_z:
                self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t + 1, W // 2),
                                                        nn.ReLU(True),
                                                        nn.Linear(W // 2, 3), nn.Sigmoid())


            else:

                self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t, 3), nn.Sigmoid())
            return

        else:
            self.backscatter_encoding = nn.Sequential(
                nn.Linear(W + in_channels_dir + self.in_channels_a, W // 2), nn.ReLU(True), nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
        self.backscatter_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
        self.backscatter_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

        # static output layers
        self.objects_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())


        # Direct encoding layers

        self.direct_encoding = nn.Sequential(
                nn.Linear(W + in_channels_dir , W // 2), nn.ReLU(True), nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
        self.direct_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
        # self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t, W // 2),nn.ReLU(True),
        #                                         nn.Linear(W // 2, W // 2), nn.ReLU(True),
        #                                         nn.Linear(W // 2, 3), nn.Sigmoid())
        if self.input_z:
            self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t+1,  W // 2), nn.ReLU(True),
                                                    nn.Linear(W // 2, 3), nn.Sigmoid())


        else:

            self.direct_attenuation = nn.Sequential(nn.Linear(W // 2 + self.in_channels_t, 3),nn.Sigmoid())



    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        else:
            if self.input_z:
                input_xyz, input_dir, input_a, input_t,z_vals = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_dir,
                                    self.in_channels_a,
                                    self.in_channels_t,1], dim=-1)
            else:

                input_xyz, input_dir, input_a,input_t = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_dir,
                                    self.in_channels_a,
                                    self.in_channels_t], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        object_sigma = self.objects_sigma(xyz_)  # (B, 1)
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if sigma_only:
            return object_sigma
        if not(self.analitic_bs):
            if self.input_z:
                bs_encoding_input = torch.cat([input_dir,input_a ,z_vals], 1)
            else:
                bs_encoding_input = torch.cat([xyz_encoding_final, input_dir,input_a], 1)
            bs_encoding = self.backscatter_encoding(bs_encoding_input)
            bs_rgb = self.backscatter_rgb(bs_encoding)  # (B, 3)
            bs_betta = self.backscatter_beta(bs_encoding)
        d_encoding_input = torch.cat([xyz_encoding_final, input_dir], 1)
        d_encoding = self.direct_encoding(d_encoding_input)
        d_rgb = self.direct_rgb(d_encoding)
        if self.input_z:
            d_attenuation_input = torch.cat([d_encoding, input_t,z_vals], 1)
        else:
            d_attenuation_input = torch.cat([d_encoding,input_t],1)
        d_attenuation = self.direct_attenuation(d_attenuation_input)
        if self.analitic_bs:
            return torch.cat([object_sigma, d_rgb, d_attenuation], 1)
        else:
            return torch.cat([object_sigma,bs_rgb,d_rgb,d_attenuation,bs_betta],1)

