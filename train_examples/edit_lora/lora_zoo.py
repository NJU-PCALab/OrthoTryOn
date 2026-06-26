import os

import torch
import torch.nn as nn


class RotatedSharedLoRA(nn.Module):
    def __init__(self, original_layer, rank=64, num_tasks=3, alpha=64):
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("RotatedSharedLoRA requires nn.Linear.")
        if rank < 1 or num_tasks < 1:
            raise ValueError("rank and num_tasks must be positive.")

        self.original_layer = original_layer
        self.rank = rank
        self.num_tasks = num_tasks
        self.scaling = alpha / rank
        self.current_task_id = 0

        self.original_layer.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, original_layer.out_features))

        rotations = []
        for _ in range(num_tasks):
            rotation, _ = torch.linalg.qr(torch.randn(rank, rank))
            rotations.append(rotation)
        self.register_buffer("task_rotations", torch.stack(rotations), persistent=True)

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def set_task(self, task_id):
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id={task_id} is outside [0, {self.num_tasks}).")
        self.current_task_id = task_id

    def forward(self, x):
        base = self.original_layer(x)
        dtype = x.dtype
        rotation = self.task_rotations[self.current_task_id].to(dtype=dtype)
        latent = x @ self.lora_A.to(dtype=dtype)
        update = (latent @ rotation.t()) @ self.lora_B.to(dtype=dtype)
        return base + update * self.scaling


def inject_rotated_shared_lora(model, rank=64, num_tasks=3, alpha=64, target_modules=None):
    if not target_modules:
        return model

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, RotatedSharedLoRA)
        and any(name.endswith(target) for target in target_modules)
    ]

    for name in names:
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = getattr(parent, child_name)
        setattr(
            parent,
            child_name,
            RotatedSharedLoRA(layer, rank=rank, num_tasks=num_tasks, alpha=alpha),
        )

    return model


def save_rotated_lora_weights(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_" in name or "task_rotations" in name
    }
    if not state_dict:
        raise RuntimeError("No Rotated Shared LoRA weights were found.")
    torch.save(state_dict, os.path.join(save_directory, "adapter_model.bin"))


def load_rotated_lora_weights(model, load_directory):
    weights_path = os.path.join(load_directory, "adapter_model.bin")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")
    if not any("task_rotations" in key for key in state_dict):
        raise RuntimeError("Checkpoint is missing task_rotations.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected
