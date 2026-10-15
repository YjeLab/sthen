import torch

from torch.nn import Linear, Identity, Module

from torch import nn, einsum  # neural-network module and einsum notation

import torch.nn.functional as F  # functional interface

from einops.layers.torch import Rearrange  # Rearrange layer

from einops import rearrange, repeat  # rearrange/repeat helpers

from timm.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np

import os

import psutil

def exists(v):

    return v is not None



def default(v, d):

    return v if exists(v) else d



# appendix B

# https://github.com/glassroom/heinsen_sequence



def heinsen_associative_scan_log(log_coeffs, log_values):

    a_star = log_coeffs.cumsum(dim = 1)

    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)

    log_h = a_star + log_h0_plus_b_star

    return log_h.exp()



# appendix B.3



def g(x):

    return torch.where(x >= 0, x + 0.5, x.sigmoid())



def log_g(x):

    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))





class minGRU(Module):

    def __init__(self, dim, expansion_factor = 1., proj_out = None):

        super().__init__()



        dim_inner = int(dim * expansion_factor)

        proj_out = default(proj_out, expansion_factor != 1.)



        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)

        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()



    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):

        seq_len = x.shape[1]

        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)



        if seq_len == 1:

            # handle sequential



            hidden = g(hidden)

            gate = gate.sigmoid()

            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)

        else:

            # parallel



            log_coeffs = -F.softplus(gate)



            log_z = -F.softplus(-gate)

            log_tilde_h = log_g(hidden)

            log_values = log_z + log_tilde_h



            if exists(prev_hidden):

                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)

                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))



            out = heinsen_associative_scan_log(log_coeffs, log_values)

            out = out[:, -seq_len:]



        next_prev_hidden = out[:, -1:]



        out = self.to_out(out)



        if not return_next_prev_hidden:

            return out



        return out, next_prev_hidden

    

class PreNorm(nn.Module):

    """Pre-normalization module that applies layer normalization before the function."""

    def __init__(self, dim, fn):

        super().__init__()

        self.norm = nn.LayerNorm(dim)  # create layer normalization

        self.fn = fn  # the function to apply



    def forward(self, x,**kwargs):

        return self.fn(self.norm(x), **kwargs)

        

     

import matplotlib.pyplot as plt  # matplotlib for visualization



class Attention(nn.Module):

    """Multi-head self-attention module"""

    def __init__(self, dim, N, axis, heads=8, dim_head=64, dropout=0.):

        super().__init__()

        

        self.axis = axis  # axis direction

        self.heads = heads  # number of attention heads

        self.scale = dim_head ** -0.5  # scaling factor

        self.dim_head = dim_head  # dimension per head

        self.token_num = N  # number of tokens

        

        # create a separate projection matrix per head

        self.to_q = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)])

        self.to_k = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)])

        self.to_v = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)])

        

        self.to_out = nn.Sequential(  # output projection

            nn.Linear(dim*self.heads, dim),  # project the inner dimension back to d

            nn.Dropout(dropout)  # dropout layer

        )



    def forward(self, x, visualize=False):

        b, n, d = x.shape  # batch size, sequence length, feature dim

        h = self.heads  # number of attention heads

        

        # each head processes the full embedding

        head_outputs = []

        attentions = []  # storage for attention matrices

        for i in range(h):

            # project the full embedding to avoid splitting

            q_i = self.to_q[i](x)  # [b, n, d]

            k_i = self.to_k[i](x)  # [b, n, d]

            v_i = self.to_v[i](x)  # [b, n, d]

            # compute attention scores

            dots = einsum('b i d, b j d -> b i j', q_i, k_i) * self.scale

            # apply softmax to obtain attention weights

            attn = dots.softmax(dim=-1) # [b, n, n]

            if visualize:

                attentions.append(attn[0])  # store the attention matrix

            # apply attention weights to the values

            out = einsum('b i j, b j d -> b i d', attn, v_i) # [b,n,d]

            head_outputs.append(out)

                

        # concatenate the outputs of all heads

        out = torch.cat(head_outputs, dim=-1)  # [b, n, h*d]



        # apply the output projection

        out = self.to_out(out)

        

        if visualize:

            # save the attention matrices as .npy files

            for i, attn in enumerate(attentions):

                attn_np = attn.cpu().detach().numpy()

                np.save(f'Visualization\Resources\\attention_matrix_head_{i+1}.npy', attn_np)



        return out  # return the attention output

    

class GateFFN(nn.Module):

    """Implement the gated feed-forward network"""

    def __init__(self, d_model, enc_in):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)



        self.ff1 = nn.Sequential(

            nn.Linear(d_model, d_model),

            nn.GELU(),

            nn.Dropout(0.1)

        )

        self.ff2 = nn.Sequential(

            nn.Linear(enc_in, enc_in),

            nn.GELU(),

            nn.Dropout(0.1)

        )

    def forward(self, x):

        

        y_0 = self.ff1(x)

        y_0 = y_0 + x

        y_0 = self.norm1(y_0)

        y_1 = y_0.permute(0, 2, 1)

        y_1 = self.ff2(y_1)

        y_1 = y_1.permute(0, 2, 1)

        y_2 = y_1 * y_0 + x

        y_2 = self.norm1(y_2)



        return y_2

       

# define the TemporalConvNet class

class TemporalConvNet(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.1):

        super(TemporalConvNet, self).__init__()

        self.GRU = minGRU(input_dim)

        self.norm = nn.LayerNorm(input_dim)

        

    def forward(self, x):

        # input x: [B, T, D]

        x = self.GRU(x) 

        x = self.norm(x )

        return x

       

class ST_ConvEncoder(nn.Module):

    def __init__(self, dim, seq, height, width, dropout=0., num_clusters=4, 

                 initial_temp=1.0, min_temp=0.01, decay_rate=0.99):

        super().__init__()

        self.dim = dim

        self.seq = seq

        self.height = height

        self.width = width

        self.patch_num = height * width

        self.num_clusters = num_clusters

        self.dropout = dropout

        

        # temperature-related parameters

        self.initial_temp = initial_temp

        self.min_temp = min_temp

        self.decay_rate = decay_rate

        

        # register temperature and step as buffers so they are saved in the model state

        self.register_buffer('temperature', torch.tensor(initial_temp, dtype=torch.float32))

        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

        # clustering layer with learnable cluster centers

        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim * seq))

        

        # create an independent TCN per cluster

        self.tcn_layers = nn.ModuleList([

            TemporalConvNet(dim, dim, kernel_size=3, dropout=dropout) 

            for _ in range(num_clusters)

        ])

        

        # cache the latest cluster assignment for visualization

        self.register_buffer('last_cluster_weights', None, persistent=False)

        self.register_buffer('last_cluster_indices', None, persistent=False)

    

    def update_temperature(self):

        """

        update the temperature parameter

        """

        # use torch.tensor ops to ensure correct gradient computation

        self.temperature = torch.max(

            torch.tensor(self.min_temp, device=self.temperature.device, dtype=self.temperature.dtype),

            torch.tensor(self.initial_temp, device=self.temperature.device, dtype=self.temperature.dtype) * 

            (torch.tensor(self.decay_rate, device=self.temperature.device, dtype=self.temperature.dtype) ** self.step)

        )

    

    def forward(self, x):

        B, T, N, D = x.shape



        # step 1: compute the similarity between each node and the cluster centers

        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * D)  # [B, N, T*D]

        x_norm = F.normalize(x_flat, p=2, dim=-1)  # normalized features [B, N, T*D]

        centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)  # normalized cluster centers [K, T*D]

        sim = torch.einsum('bnd,kd->bnk', x_norm, centers_norm)  # cosine similarity [B, N, K]

        

        # use softmax to produce soft assignment weights [B, N, K]

        cluster_weights = F.softmax(sim / self.temperature, dim=-1)

        

        # save the cluster weights and the argmax cluster index for visualization

        self.last_cluster_weights = cluster_weights.detach()

        self.last_cluster_indices = torch.argmax(cluster_weights, dim=-1).detach()

        

        # initialize the output tensor

        x_out = torch.zeros_like(x)  # [B, T, N, D]

        

        # process each cluster separately

        for k in range(self.num_clusters):

            cluster_weight = cluster_weights[:, :, k]  # weight of the current cluster [B, N]

            if not cluster_weight.sum().item() > 0:

                continue

            

            # weighted feature aggregation

            weighted_features = x * cluster_weight.unsqueeze(1).unsqueeze(-1)

            aggregated_features = rearrange(weighted_features, 'b t n d -> (b n) t d')  # [B*N, T, D]

            

            # apply the TCN to the time series

            tcn_out = self.tcn_layers[k](aggregated_features)

            tcn_out = rearrange(tcn_out, '(b n) t d -> b t n d', b=B, n=N)  # [B, T, N, D]

            

            # features processed with broadcasting and weight assignment

            weight_expanded = cluster_weight.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]

            x_out += tcn_out * weight_expanded  # [B, T, N, D]

        # update the temperature parameter (step +1 per forward pass)

        self.step += 1

        self.update_temperature()

        

        return x_out

        

 



class GatedTransformer(nn.Module):

    """Gated Transformer block combining self-attention and a feed-forward network."""

    def __init__(self, dim, heads, dim_head, axis,seq, height, width, dropout=0., attn_dropout=0., drop_path=0.1,adj_matrix=None):

        """

        initialize the gated Transformer

        dim: input dimension

        heads: number of attention heads

        dim_head: dimension per head

        mlp_dim: MLP hidden dim

        axis: axis direction

        dropout: dropout ratio

        attn_dropout: attention dropout ratio

        drop_path: drop-path ratio

        """

        super().__init__()

        self.norm = nn.LayerNorm(dim)  # final layer normalization

        self.axis = axis  # axis direction

        if self.axis == 'T':

            N = seq

        elif self.axis == 'S':

            N = height*width

        self.attn_norm = PreNorm(dim, Attention(dim, N = N,axis=self.axis, heads=heads, dim_head=dim_head, dropout=attn_dropout))

        self.ff_norm = PreNorm(dim, GateFFN(d_model = dim, enc_in=N))

        self.attn_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ff_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        self.apply(self._init_weights)  # apply weight initialization



    def _init_weights(self, m):

        """Initialize model weights"""

        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=.02)  # linear-layer weights initialized with truncated normal

            if isinstance(m, nn.Linear) and m.bias is not None:

                nn.init.constant_(m.bias, 0)  # bias initialized to 0

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)  # LayerNorm bias initialized to 0

            nn.init.constant_(m.weight, 1.0)  # LayerNorm weight initialized to 1

            

    def forward(self, x):

        """Forward pass"""

        x = x + self.attn_drop_path(self.attn_norm(x))  # residual connection: input + attention output (with drop path)

        x = x + self.ff_drop_path(self.ff_norm(x))  # residual connection: previous output + FFN output (with drop path)

        

        return self.norm(x)  # return the final normalized output and the similarity loss

      

class HASTFormerLayer(nn.Module):

    """HASTFormer layer implementing spatio-temporal attention"""

    def __init__(self, dim, heads, dim_head, seq, height, width, dropout=0., attn_dropout=0., drop_path=0.1,adj_matrix=None):

        """

        initialize the PredFormer layer

        dim: input dimension

        heads: number of attention heads

        dim_head: dimension per head

        mlp_dim: MLP hidden dim

        dropout: dropout ratio

        attn_dropout: attention dropout ratio

        drop_path: drop-path ratio

        """

        super(HASTFormerLayer, self).__init__()

        self.temporal_transformer = GatedTransformer(dim, heads, dim_head, 

                                                'T',seq, height, width,dropout, attn_dropout, drop_path)  # temporal Transformer

        self.st_joint_layer = ST_ConvEncoder(dim, seq, height, width,dropout=dropout)  # Mamba layer



    def forward(self, x):

        """Forward pass"""

        b, t, n, _ = x.shape  # input shape: batch size, time steps, nodes, feature dim

        x_s, x_ori = x, x  # keep the original input

        # temporal branch

        x_st = x_s.permute(0, 2, 1, 3)  # dimension permutation: b n t d

        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  # merge the batch and node dimensions

        x_st = self.temporal_transformer(x_st)  # through the temporal Transformer

        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)  # restore the batch and node dimensions

        x_st = rearrange(x_st, 'b n t d -> b t n d')  # permute back to the original order

        

        x_st = self.st_joint_layer(x_st) + x_st  # through the Mamba layer



        return x_st

    

def sinusoidal_embedding(n_channels, dim):

    """

    generate the sinusoidal positional encoding

    n_channels: number of channels

    dim: embedding dimension

    """

    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]

                            for p in range(n_channels)])  # build the positional-encoding matrix

    pe[:, 0::2] = torch.sin(pe[:, 0::2])  # sine on even indices

    pe[:, 1::2] = torch.cos(pe[:, 1::2])  # cosine on odd indices

    return rearrange(pe, '... -> 1 ...')  # add the batch dimension and return

      

class HASTFormer(nn.Module):

    """HASTFormer model for spatio-temporal sequence forecasting"""

    def __init__(self, in_channel, d_model, his_len, pre_len, num_heads, 

                 num_layers, in_h=None, in_w=None, nodes=None, data_type='Grid', patch_size=2, dropout=0.1):

        super().__init__()

        self.data_type = data_type  # data type

        self.patch_size = patch_size  # patch size

        self.num_frames_in = his_len  # number of input frames

        self.num_frames_out = pre_len  # number of output frames (prediction length)

        self.dim = d_model  # model dimension

        self.num_channels = in_channel  # number of channels

        self.heads = num_heads  # number of attention heads

        self.dim_head = self.dim   # dimension per head

        self.dropout = dropout  # dropout ratio

        self.attn_dropout = dropout  # attention dropout ratio

        self.drop_path = dropout  # drop-path ratio

        self.Ndepth = num_layers  # number of HASTFormer layers

        

        

        if self.data_type == 'Grid':  # ensure grid input

            # ... keep the original grid data processing unchanged ...

            self.image_height = in_h  # image height

            self.image_width = in_w  # image width

            # compute the padding so that height and width are divisible by patch_size

            self.pad_h = (self.patch_size - self.image_height % self.patch_size) % self.patch_size

            self.pad_w = (self.patch_size - self.image_width % self.patch_size) % self.patch_size

            # compute the padded height and width

            self.padded_height = self.image_height + self.pad_h

            self.padded_width = self.image_width + self.pad_w

            # update the patch count computation

            self.num_patches_h = self.padded_height // self.patch_size  # number of patches along the height

            self.num_patches_w = self.padded_width // self.patch_size  # number of patches along the width

            self.num_patches = self.num_patches_h * self.num_patches_w  # total number of patches`

            self.patch_dim = self.num_channels * self.patch_size ** 2  # dimension of each patch

            # embedding layer for image inputs

            self.to_patch_embedding_image = nn.Sequential(

                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),  # split the image into patches

                nn.Linear(self.patch_dim, self.dim),  # project to the model dimension

            )

            # MLP head for image outputs

            self.mlp_head_image = nn.Sequential(

                nn.LayerNorm(self.dim),  # layer normalization

                nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)  # project back to the patch dimension

            )

         

        if self.data_type == 'Topology':

            # topology data processing - optimized version

            self.nodes = nodes  # number of nodes

            # compute the padding so that the node count is divisible by patch_size

            self.pad_n = (self.patch_size - self.nodes % self.patch_size) % self.patch_size

            # compute the padded node count

            self.padded_nodes = self.nodes + self.pad_n

            # compute the number of node groups

            self.num_node_groups = self.padded_nodes // self.patch_size

            # feature dimension per node group

            self.node_group_dim = self.num_channels * self.patch_size

            

            # embedding layer for sequence inputs - works directly on node groups

            self.to_patch_embedding_seq = nn.Sequential(

                Rearrange('b t (g p) c -> b t g (p c)', p=self.patch_size),  # group nodes

                nn.Linear(self.node_group_dim, self.dim),  # project to the model dimension

            )

            

            # MLP head for sequence outputs

            self.mlp_head_seq = nn.Sequential(

                nn.LayerNorm(self.dim),  # layer normalization

                nn.Linear(self.dim, self.node_group_dim),  # project back to the node-group feature dim

            )

            

            # update the patch count computation for positional encoding

            self.num_patches_h = 1

            self.num_patches_w = self.num_node_groups

            self.num_patches = self.num_node_groups

            

        # positional encoding

        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),

                                          requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)  # positional encoding

        

        self.blocks = nn.ModuleList([

            HASTFormerLayer(self.dim, self.heads, self.dim_head, his_len, self.num_patches_h, self.num_patches_w, 

                           self.dropout, self.attn_dropout, self.drop_path, None)

            for i in range(self.Ndepth)  # create HASTFormer layers

        ])

        if self.num_frames_in > self.num_frames_out:

            self.projection = nn.Linear(self.num_frames_in, self.num_frames_out)  # projection layer for more input frames than output frames

        

    def forward(self, x):

        """Forward pass"""

        # dispatch on the input type

        if len(x.shape) == 5:  # x: [B, T, C, H, W]

            is_image_input = True

            B, T, C, H, W = x.shape  # input shape

            

            # keep the original size for later cropping

            orig_H, orig_W = H, W

            

            # apply padding when required

            if self.pad_h > 0 or self.pad_w > 0:

                # create the padded tensor

                x_padded = torch.zeros(B, T, C, H + self.pad_h, W + self.pad_w, device=x.device)

                x_padded[:, :, :, :H, :W] = x

                x = x_padded

                H, W = H + self.pad_h, W + self.pad_w

            

            # patch embedding

            x = self.to_patch_embedding_image(x) # [B, T, N, D]

            

        else:  # x: [B, T, N, C]

            B, T, N, C = x.shape

            orig_N = N



            # pad with F.pad

            if self.pad_n > 0:

                x = F.pad(x, (0, 0, 0, self.pad_n), mode='constant', value=0)

                N += self.pad_n

            

            # apply rearrange and linear projection directly

            x = self.to_patch_embedding_seq(x)

            is_image_input = False

            

        # positional encoding

        if T <= self.num_frames_in:  # keep the time steps within the positional-encoding length

            x += self.pos_embedding[:, :T, :, :].to(x.device)

        

        # through the HASTFormer layer

        for blk in self.blocks:

            x = blk(x)  # through each HASTFormer layer [B, T, N, D]

        

        # choose the output processing according to the input type

        if is_image_input:

            # MLP head - image output

            x = self.mlp_head_image(x.reshape(-1, self.dim))  # flatten the features and pass the MLP head

            x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)  # reshape

            x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)  # permute and reshape into the final output shape

            

            # crop back to the original size if padding was applied

            if self.pad_h > 0 or self.pad_w > 0:

                x = x[:, :, :, :orig_H, :orig_W]

            if T > self.num_frames_out:

                x = self.projection(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)  # projection layer for more input frames than output frames

        else:

            # topology data output processing - simplified version

            x = self.mlp_head_seq(x)

            x = x.view(B, T, -1, C)

            

            # crop back to the original node count

            x = x[:, :, :orig_N, :]

            if T > self.num_frames_out:

                x = self.projection(x.permute(0,2,3,1)).permute(0,3,1,2)  # projection layer for more input frames than output frames

        

        

        return x

    

if __name__ == '__main__':

    model = HASTFormer(in_channel=2, d_model=128, his_len=6, pre_len=1, num_heads=4, 

                 num_layers=4, in_h=32, in_w=32, nodes=None, data_type='Grid', patch_size=2, dropout=0.1)  # create the HASTFormer instance

    # print the parameter count

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameter count: {total_params}")  # count and print the trainable parameters

    # print model memory usage

    model = model.cuda()  # move the model to the GPU

    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])

    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    mem_total = mem_params + mem_buffers  # total model memory

    print(f"Model memory usage: {mem_total / 1024 ** 2:.2f} MB")  # print the memory usage in MB

    # print current GPU memory usage

    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    B,T,C,H,W = 16, 6, 2, 32, 32  # batch size, channels, time steps, height and width

    x = torch.randn(B,T,C,H,W).cuda()  # create a random input tensor and move it to the GPU

    print(f"Input shape: {x.shape}")  # print the input shape

    # forward pass and memory footprint

    torch.cuda.reset_peak_memory_stats()  # reset the memory statistics

    out = model(x)  # forward pass

    peak_memory = torch.cuda.max_memory_allocated()  # peak memory usage

    print(f"Peak GPU memory after forward pass: {peak_memory / 1024 ** 2:.2f} MB")

    # print GPU memory usage after the forward pass

    print(f"GPU memory after forward pass: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    print(f"Output shape: {out.shape}")  # print the output shape

    # print CPU memory usage

    process = psutil.Process(os.getpid())

    cpu_mem = process.memory_info().rss / 1024 ** 2  # current process memory usage in MB

    print(f"CPU memory usage: {cpu_mem:.2f} MB")
