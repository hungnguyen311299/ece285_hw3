import math
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * x.shape[1] ** 0.5


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        angles = t[:, None] * freqs[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return F.silu(x)


class ResnetBlock(nn.Module):
    """Two conv blocks; the time embedding scales and shifts the first one."""

    def __init__(self, dim, dim_out, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t):
        scale_shift = self.mlp(t)[:, :, None, None].chunk(2, dim=1)
        return self.block2(self.block1(x, scale_shift)) + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.heads, self.scale = heads, dim_head ** -0.5
        hidden = heads * dim_head
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = (t.reshape(b, self.heads, -1, h * w) for t in self.to_qkv(self.norm(x)).chunk(3, dim=1))
        mk, mv = (m.expand(b, -1, -1, -1) for m in self.mem_kv)
        k, v = torch.cat((mk, k), dim=-1), torch.cat((mv, v), dim=-1)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        return self.to_out(out.reshape(b, -1, h, w))


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = (t.reshape(b, self.heads, -1, h * w).transpose(-1, -2)
                   for t in self.to_qkv(self.norm(x)).chunk(3, dim=1))
        mk, mv = (m.expand(b, -1, -1, -1) for m in self.mem_kv)
        k, v = torch.cat((mk, k), dim=-2), torch.cat((mv, v), dim=-2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(-1, -2).reshape(b, -1, h, w))


def Downsample(dim, dim_out):
    return nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(dim * 4, dim_out, 1))


def Upsample(dim, dim_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, dim_out, 3, padding=1))


class Unet(nn.Module):

    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()
        self.channels = channels
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in, d_in, time_dim),
                ResnetBlock(d_in, d_in, time_dim),
                Attention(d_in) if last else LinearAttention(d_in),
                nn.Conv2d(d_in, d_out, 3, padding=1) if last else Downsample(d_in, d_out),
            ]))

        self.mid_block1 = ResnetBlock(dims[-1], dims[-1], time_dim)
        self.mid_attn = Attention(dims[-1])
        self.mid_block2 = ResnetBlock(dims[-1], dims[-1], time_dim)

        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            first, last = i == 0, i == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_out + d_in, d_out, time_dim),
                ResnetBlock(d_out + d_in, d_out, time_dim),
                Attention(d_out) if first else LinearAttention(d_out),
                nn.Conv2d(d_out, d_in, 3, padding=1) if last else Upsample(d_out, d_in),
            ]))

        self.final_res_block = ResnetBlock(dim * 2, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        skips = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, t)
            skips.append(x)
            x = block2(x, t)
            x = attn(x) + x
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, up in self.ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), t)
            x = block2(torch.cat((x, skips.pop()), dim=1), t)
            x = attn(x) + x
            x = up(x)

        return self.final_conv(self.final_res_block(torch.cat((x, r), dim=1), t))


def cosine_alphas_cumprod(timesteps, s=0.008):
    t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clip(0, 0.999)
    return torch.cumprod(1 - betas, dim=0)


class GaussianDiffusion(nn.Module):
    """Images go in and come out in [0, 1]; internally they are scaled to [-1, 1].

    """

    def __init__(self, model, image_size, timesteps=1000, sampling_timesteps=100):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        alphas_cumprod = cosine_alphas_cumprod(timesteps)
        for name, value in (("alphas_cumprod", alphas_cumprod),
                            ("sqrt_alphas_cumprod", alphas_cumprod.sqrt()),
                            ("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt()),
                            ("loss_weight", alphas_cumprod)):  # SNR / (SNR + 1)
            self.register_buffer(name, value.float(), persistent=False)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict({k: v for k, v in state_dict.items() if k.startswith("model.")},
                                       strict)

    def coef(self, buffer, t):
        return buffer[t][:, None, None, None]

    def forward(self, images):
        """Training loss for a batch of [0, 1] images."""
        x0 = images * 2 - 1
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        a, s = self.coef(self.sqrt_alphas_cumprod, t), self.coef(self.sqrt_one_minus_alphas_cumprod, t)
        with torch.autocast("cuda", enabled=False):
            x_t = a * x0 + s * noise
        v = a * noise - s * x0
        loss = F.mse_loss(self.model(x_t, t), v, reduction="none").mean(dim=(1, 2, 3))
        return (loss * self.loss_weight[t]).mean()

    @torch.inference_mode()
    def sample(self, batch_size=16):
        """Deterministic DDIM (eta = 0) with `sampling_timesteps` steps; returns [0, 1] images."""
        device = self.alphas_cumprod.device
        times = torch.linspace(-1, self.num_timesteps - 1, self.sampling_timesteps + 1).int().tolist()[::-1]
        x = torch.randn(batch_size, self.model.channels, self.image_size, self.image_size, device=device)
        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((batch_size,), time, device=device, dtype=torch.long)
            a, s = self.coef(self.sqrt_alphas_cumprod, t), self.coef(self.sqrt_one_minus_alphas_cumprod, t)
            x0 = (a * x - s * self.model(x, t)).clamp(-1, 1)
            if time_next < 0:
                x = x0
                break
            noise = (x - a * x0) / s
            alpha_next = self.alphas_cumprod[time_next]
            x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * noise
        return (x + 1) / 2
