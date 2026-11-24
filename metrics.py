import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

class InceptionMetrics:
    """FID on 2,048 pooled features and IS on 1,000 ImageNet probabilities."""

    def __init__(self, device, real_loader):
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.classifier = inception.fc.to(device).eval()
        inception.fc = torch.nn.Identity()
        self.inception = inception.to(device).eval()
        self.device = device
        real = torch.cat([self.features(images) for images in real_loader]).cpu().numpy()
        self.count = len(real)
        self.real_mean, self.real_cov = real.mean(0), np.cov(real, rowvar=False)
        values, vectors = np.linalg.eigh(self.real_cov)
        self.real_cov_sqrt = (vectors * np.sqrt(values.clip(min=0))) @ vectors.T

    @torch.inference_mode()
    def features(self, images):
        images = images.to(self.device).clamp(0, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear",
                               align_corners=False, antialias=True)
        return self.inception(images)

    @torch.inference_mode()
    def score(self, diffusion):
        """Generate one sample per held-out image and calculate FID and IS."""
        was_training = diffusion.training
        diffusion.eval()
        generated = []
        remaining = self.count
        while remaining:
            batch_size = min(64, remaining)
            generated.append(self.features(diffusion.sample(batch_size=batch_size)))
            remaining -= batch_size
        diffusion.train(was_training)

        fake_features = torch.cat(generated)
        log_probs = F.log_softmax(self.classifier(fake_features), dim=1).double()
        marginal = log_probs.exp().mean(0).log()
        inception_score = (log_probs.exp() * (log_probs - marginal)).sum(1).mean().exp().item()

        fake = fake_features.double().cpu().numpy()
        mean, cov = fake.mean(0), np.cov(fake, rowvar=False)
        middle = self.real_cov_sqrt @ cov @ self.real_cov_sqrt
        trace_sqrt = np.sqrt(np.linalg.eigvalsh((middle + middle.T) / 2).clip(min=0)).sum()
        fid = ((self.real_mean - mean) ** 2).sum() + np.trace(self.real_cov + cov) - 2 * trace_sqrt
        return float(fid), inception_score
