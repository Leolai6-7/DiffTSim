import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.simsiam import SimSiam, simsiam_loss
from augmentations.diffusion import diffusion_augment

def pretrain_diffusion(model, diffusion_model, dataset, optimizer, device, epochs=20, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    diffusion_model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            t = torch.randint(50, 100, (1,)).item()
            x1 = diffusion_augment(diffusion_model, x, t)
            x2 = diffusion_augment(diffusion_model, x, t)

            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, z2, p2, z1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")
