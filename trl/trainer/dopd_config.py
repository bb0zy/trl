from .grpo_config import GRPOConfig
from dataclasses import dataclass, field

@dataclass
class DOPDConfig(GRPOConfig):
    topk_logprobs_num: int = 16
    teacher_rl_url: str = ""
    teacher_ref_url: str = ""