""" Module to generate adversarial examples. """

from typing import Callable

import torch


def generate_adversarial(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor, eps: float,
) -> torch.Tensor:
    """ Use the PGD attack to generate adversarial examples. """

    if eps > 0:
        with torch.enable_grad():
            orig_inputs = inputs.clone().detach()
            inputs = inputs.clone().detach()
            inputs += 2 * eps * torch.rand_like(inputs) - eps  # Random init
            inputs = torch.clip(inputs, 0, 1)

            for _ in range(10):
                # Compute loss
                inputs = inputs.clone().detach().requires_grad_()
                pred = torch.argmax(model_fn(inputs), dim=1)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(model_fn(inputs), pred)
                loss.backward()

                with torch.no_grad():
                    # Add adversarial noise
                    assert inputs.grad is not None
                    grad_sign = torch.sign(inputs.grad)
                    inputs += (eps / 10) * grad_sign
                    inputs = torch.max(
                        torch.min(inputs, orig_inputs + eps),
                        orig_inputs - eps,
                    )
                    inputs = torch.clip(inputs, 0, 1)

    return inputs
