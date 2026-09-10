from .grpo_config import GRPOConfig
from dataclasses import dataclass, field

@dataclass
class DOPDConfig(GRPOConfig):
    topk_logprobs_num: int = 0
    teacher_url: str = ""