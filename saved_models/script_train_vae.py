""" Train variational auto-encoder. """

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.data import get_mnist_dataset
from reference_models.vae import VariationalAutoEncoder

torch.manual_seed(42)
rng = np.random.Generator(np.random.PCG64(42))

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

for dataset_name, raw_dataset in zip(
    ["mnist", "fmnist", "kmnist", "notmnist"],
    [
        get_mnist_dataset("mnist", rng), get_mnist_dataset("fmnist", rng),
        get_mnist_dataset("kmnist", rng), get_mnist_dataset("notmnist", rng),
    ]
):
    # Prepare data
    trainloader = DataLoader(
        TensorDataset(torch.tensor(raw_dataset.x_train, dtype=torch.float32)),
        batch_size=256, shuffle=True, num_workers=8,
    )

    # Create model
    model = VariationalAutoEncoder()
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    for epoch in range(1000):
        running_loss = 0.0
        loss_count = 0
        for data in trainloader:
            # Forward pass
            inputs = data[0]
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = inputs.to(device)
            loss, _, _ = model(inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_count += 1

        print(f"[{epoch + 1: >4}] {running_loss / loss_count:.6f}")
        running_loss = 0.0
        loss_count = 0

    # Save model
    model.eval()
    torch.save(model.state_dict(), f"./saved_models/vae-{dataset_name}.pt")
