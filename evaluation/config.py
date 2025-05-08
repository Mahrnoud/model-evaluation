# 3. Create a config class for better parameter management
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class EvaluationConfig:
    model_path: str
    tokenizer_path: str
    dataset_path: str
    question_lang: str = "en"
    sample_size: Optional[int] = None
    output_dir: str = "qa_evaluation_results"
    max_length: int = 256
    model_config: Optional[Dict[str, Any]] = None
    prepare_human_eval: bool = True
    debug_mode: bool = False

    def __post_init__(self):
        # Validate configuration
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
