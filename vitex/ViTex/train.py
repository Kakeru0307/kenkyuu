import os
import torch
from dataloader import get_train_dataloader
from accelerate import Accelerator
from torch import optim
from model.dual_unet import DualUNet
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from config import UNetConfig, TrainingConfig
from d3pm import D3PM
import math
from torch.optim.lr_scheduler import LambdaLR

unet_config = UNetConfig(
    x1_shape=(11, 128, 128),
    x2_shape=(1, 128, 128),
    d_temb=256,
    hyperscore1_shape=(33, 8, 8),
    chord1_shape=(3, 32, 12),
    hyperscore2_shape=(2, 8, 3),
    base_channel=32,
    channel_mult=[1, 2, 4, 4],
    num_blocks=2,
    hyperscore1_level=[2, 3],
    hyperscore2_level=[2, 3],
    chord_level=[1, 2],
    fuse_rhythm_level=[],
    self_attn_level=[2, 3],
    num_heads=4,
    N=4,
    use_d3pm=True
)


training_config = TrainingConfig(
    batch_size=100,
    lr=3e-4,
    num_epochs=50,
    num_workers=10,
    pin_memory=True,
    accumulation_step=1,
    output_path="output",
    data_path="/workspace/midi_pkl",
    save_interval=20000,
    log_interval=100,
    null_condition_prob=0.5,


)


def main():
    # make dirs
    import os
    output_folder = training_config.output_path
    os.makedirs(output_folder, exist_ok=True)
    # add timestamp here
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join(output_folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)
    ckpt_folder = os.path.join(output_folder, "ckpt")
    os.makedirs(ckpt_folder, exist_ok=True)
    log_folder = os.path.join(output_folder, "log")
    os.makedirs(log_folder, exist_ok=True)
    # save config at output_folder/config.yaml
    import yaml
    with open(os.path.join(output_folder, "config.yaml"), "w") as f:
        yaml.dump(vars(unet_config), f)
        yaml.dump(vars(training_config), f)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.accumulation_step, mixed_precision="bf16")
    writer = SummaryWriter(log_dir=log_folder)

    # 1. Load dataset and dataloader
    dataloader = get_train_dataloader(training_config)

    # 2. Load model
    model = DualUNet(**vars(unet_config))

    optimizer = optim.AdamW(model.parameters(), lr=training_config.lr)


    total_steps = len(dataloader) * training_config.num_epochs
    warmup_steps = int(0.02 * total_steps)  # 前 2% 步长线性升高

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

    scheduler = LambdaLR(optimizer, lr_lambda)


    d3pm = D3PM(model, 1000, num_classes=4, hybrid_loss_coeff=0.0)

    # Prepare everything with accelerator
    model, d3pm, optimizer, scheduler,  dataloader = accelerator.prepare(
        model, d3pm, optimizer, scheduler, dataloader
    )


    resume_ckpt = "output/20250905-080132/ckpt"
    if os.path.exists(resume_ckpt):
        accelerator.load_state(resume_ckpt)
        accelerator.print(f"Resumed full state from {resume_ckpt}")

        
        optimizer = optim.AdamW(model.parameters(), lr=training_config.lr)
        scheduler = LambdaLR(optimizer, lr_lambda)
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        accelerator.print(f"Optimizer & scheduler re-initialized with lr={training_config.lr}")

    # 4. Training loop
    global_step = 0
    model.train()
    for epoch in range(training_config.num_epochs):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for batch_segment in pbar:
            with accelerator.accumulate(model):

                loss, loss_dict = d3pm(batch_segment, training_config.null_condition_prob)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                if accelerator.is_main_process:
                    if global_step % training_config.log_interval == 0:
                        for k, v in loss_dict.items():
                            writer.add_scalar(
                                f"loss/{k}", v, global_step)
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar("lr", current_lr, global_step)
                    pbar.set_description(
                        f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | vb_loss_1: {loss_dict['vb_loss_1']:.4f}, ce_loss_1: {loss_dict['ce_loss_1']:.4f} | vb_loss_2: {loss_dict['vb_loss_2']:.4f}, ce_loss_2: {loss_dict['ce_loss_2']:.4f}")

                global_step += 1

            # Save checkpoint
            if accelerator.is_main_process:
                if global_step % training_config.save_interval == 0:
                    accelerator.save_state(ckpt_folder)

    writer.close()


if __name__ == "__main__":
    main()
