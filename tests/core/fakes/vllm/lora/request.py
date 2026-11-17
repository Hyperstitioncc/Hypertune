import dataclasses


@dataclasses.dataclass
class LoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str = ""

    def __post_init__(self):
        assert self.lora_int_id >= 1, "lora_int_id must be >= 1"
