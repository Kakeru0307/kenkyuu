from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class UNetConfig:
    x1_shape: tuple[int, int, int]
    x2_shape: tuple[int, int, int]
    d_temb: int
    hyperscore1_shape: tuple[int, int, int]
    chord1_shape: tuple[int, int, int]
    hyperscore2_shape: tuple[int, int, int]
    base_channel: int
    channel_mult: list[int]
    num_blocks: int

    hyperscore1_level: list[int]
    hyperscore2_level: list[int]
    chord_level: list[int]
    fuse_rhythm_level: list[int]
    self_attn_level: list[int]
    num_heads: int
    N: int
    use_d3pm: bool


@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    num_epochs: int
    num_workers: int
    pin_memory: bool

    accumulation_step: int
    output_path: str
    data_path: str

    save_interval: int
    log_interval: int

    null_condition_prob: float

@dataclass
class GANConfig:
    
    C: int
    H: int
    W: int
    
    noise_channel:int
    inpainting_prob: float

