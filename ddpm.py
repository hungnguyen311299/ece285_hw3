import math
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization for image feature maps."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * x.shape[1] ** 0.5


class SinusoidalPosEmb(nn.Module):
    """Convert diffusion timesteps into sinusoidal feature vectors."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        frequency_scale = -math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device) * frequency_scale)
        phases = t[:, None] * frequencies[None]
        return torch.cat((phases.sin(), phases.cos()), dim=-1)


class Block(nn.Module):
    """Convolution, normalization, optional conditioning, and SiLU."""

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
    """Two convolution blocks conditioned on the diffusion timestep."""

    def __init__(self, dim, dim_out, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t):
        time_scale_shift = self.mlp(t)[:, :, None, None].chunk(2, dim=1)
        residual = self.res_conv(x)
        x = self.block1(x, time_scale_shift)
        return self.block2(x) + residual


class LinearAttention(nn.Module):
    """Memory-efficient attention over spatial positions."""

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.heads, self.scale = heads, dim_head ** -0.5
        hidden = heads * dim_head
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        spatial_size = height * width
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        query, key, value = (
            feature_map.reshape(batch_size, self.heads, -1, spatial_size)
            for feature_map in qkv
        )
        memory_key, memory_value = (
            memory.expand(batch_size, -1, -1, -1) for memory in self.mem_kv
        )
        key = torch.cat((memory_key, key), dim=-1)
        value = torch.cat((memory_value, value), dim=-1)

        query = query.softmax(dim=-2) * self.scale
        key = key.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", key, value)
        output = torch.einsum("bhde,bhdn->bhen", context, query)
        output = output.reshape(batch_size, -1, height, width)
        return self.to_out(output)


class Attention(nn.Module):
    """Full attention over spatial positions."""

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        spatial_size = height * width
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        query, key, value = (
            feature_map.reshape(batch_size, self.heads, -1, spatial_size).transpose(-1, -2)
            for feature_map in qkv
        )
        memory_key, memory_value = (
            memory.expand(batch_size, -1, -1, -1) for memory in self.mem_kv
        )
        key = torch.cat((memory_key, key), dim=-2)
        value = torch.cat((memory_value, value), dim=-2)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(-1, -2).reshape(batch_size, -1, height, width)
        return self.to_out(output)


def Downsample(dim, dim_out):
    return nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(dim * 4, dim_out, 1))


def Upsample(dim, dim_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, dim_out, 3, padding=1))


class Unet(nn.Module):
    """U-Net that predicts the diffusion target for a noisy image."""

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
        for stage_index, (input_dim, output_dim) in enumerate(in_out):
            is_last_stage = stage_index == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(input_dim, input_dim, time_dim),
                ResnetBlock(input_dim, input_dim, time_dim),
                Attention(input_dim) if is_last_stage else LinearAttention(input_dim),
                nn.Conv2d(input_dim, output_dim, 3, padding=1)
                if is_last_stage else Downsample(input_dim, output_dim),
            ]))

        self.mid_block1 = ResnetBlock(dims[-1], dims[-1], time_dim)
        self.mid_attn = Attention(dims[-1])
        self.mid_block2 = ResnetBlock(dims[-1], dims[-1], time_dim)

        for stage_index, (input_dim, output_dim) in enumerate(reversed(in_out)):
            is_first_stage = stage_index == 0
            is_last_stage = stage_index == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(output_dim + input_dim, output_dim, time_dim),
                ResnetBlock(output_dim + input_dim, output_dim, time_dim),
                Attention(output_dim) if is_first_stage else LinearAttention(output_dim),
                nn.Conv2d(output_dim, input_dim, 3, padding=1)
                if is_last_stage else Upsample(output_dim, input_dim),
            ]))

        self.final_res_block = ResnetBlock(dim * 2, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        initial_features = x.clone()
        time_embedding = self.time_mlp(time)
        skip_connections = []

        for first_block, second_block, attention, downsample in self.downs:
            x = first_block(x, time_embedding)
            skip_connections.append(x)
            x = second_block(x, time_embedding)
            x = attention(x) + x
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_embedding)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_embedding)

        for first_block, second_block, attention, upsample in self.ups:
            x = first_block(torch.cat((x, skip_connections.pop()), dim=1), time_embedding)
            x = second_block(torch.cat((x, skip_connections.pop()), dim=1), time_embedding)
            x = attention(x) + x
            x = upsample(x)

        x = torch.cat((x, initial_features), dim=1)
        x = self.final_res_block(x, time_embedding)
        return self.final_conv(x)


def cosine_alphas_cumprod(timesteps, s=0.008):
    normalized_time = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((normalized_time + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clip(0, 0.999)
    return torch.cumprod(1 - betas, dim=0)


class GaussianDiffusion(nn.Module):
    """Train and sample a diffusion model.

    Inputs and outputs use the [0, 1] range. Calculations inside the model use
    [-1, 1].
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
                            ("loss_weight", alphas_cumprod)):  # Equivalent to SNR / (SNR + 1).
            self.register_buffer(name, value.float(), persistent=False)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict({k: v for k, v in state_dict.items() if k.startswith("model.")},
                                       strict)

    def coef(self, buffer, t):
        return buffer[t][:, None, None, None]

    def forward(self, images):
        """Return the training loss for a batch of images in [0, 1]."""
        clean_images = images * 2 - 1
        timesteps = torch.randint(
            0, self.num_timesteps, (clean_images.shape[0],), device=clean_images.device
        )
        noise = torch.randn_like(clean_images)
        signal_scale = self.coef(self.sqrt_alphas_cumprod, timesteps)
        noise_scale = self.coef(self.sqrt_one_minus_alphas_cumprod, timesteps)
        with torch.autocast("cuda", enabled=False):
            noisy_images = signal_scale * clean_images + noise_scale * noise
        velocity = signal_scale * noise - noise_scale * clean_images
        prediction = self.model(noisy_images, timesteps)
        per_image_loss = F.mse_loss(prediction, velocity, reduction="none").mean(dim=(1, 2, 3))
        return (per_image_loss * self.loss_weight[timesteps]).mean()

    @torch.inference_mode()
    def sample(self, batch_size=16):
        """Generate [0, 1] images with deterministic DDIM sampling."""
        device = self.alphas_cumprod.device
        timesteps = torch.linspace(
            -1, self.num_timesteps - 1, self.sampling_timesteps + 1
        ).int().tolist()[::-1]
        images = torch.randn(
            batch_size, self.model.channels, self.image_size, self.image_size, device=device
        )
        for current_time, next_time in zip(timesteps[:-1], timesteps[1:]):
            time_batch = torch.full((batch_size,), current_time, device=device, dtype=torch.long)
            signal_scale = self.coef(self.sqrt_alphas_cumprod, time_batch)
            noise_scale = self.coef(self.sqrt_one_minus_alphas_cumprod, time_batch)
            velocity = self.model(images, time_batch)
            predicted_clean = (signal_scale * images - noise_scale * velocity).clamp(-1, 1)
            if next_time < 0:
                images = predicted_clean
                break
            predicted_noise = (images - signal_scale * predicted_clean) / noise_scale
            next_alpha = self.alphas_cumprod[next_time]
            images = next_alpha.sqrt() * predicted_clean + (1 - next_alpha).sqrt() * predicted_noise
        return (images + 1) / 2
