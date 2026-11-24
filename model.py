from ddpm import GaussianDiffusion, Unet


def build_diffusion(image_size: int) -> GaussianDiffusion:
    return GaussianDiffusion(Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=3), image_size,
                             timesteps=1000, sampling_timesteps=100)
