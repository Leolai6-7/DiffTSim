import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def diffusion_augment(model, x, t, noise=None):
    """
    對輸入資料 x 加上 noise，並從 timestep t 開始反向生成
    """
    if noise is None:
        noise = torch.randn_like(x)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(t + 1)
    noisy = scheduler.add_noise(x, noise, timesteps=torch.tensor([t]).to(x.device))
    model.eval()
    with torch.no_grad():
        for step in scheduler.timesteps:
            model_input = noisy
            noise_pred = model(model_input, torch.tensor([step / 1000.]).to(x.device))
            noisy = scheduler.step(noise_pred, step, noisy).prev_sample
    return noisy

def denoise(model, x_noisy, t):
    """
    對 noisy 資料從 timestep t 開始去雜訊
    """
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(t + 1)
    sample = x_noisy
    model.eval()
    with torch.no_grad():
        for step in scheduler.timesteps:
            model_input = sample
            noise_pred = model(model_input, torch.tensor([step / 1000.]).to(x_noisy.device))
            sample = scheduler.step(noise_pred, step, sample).prev_sample
    return sample
