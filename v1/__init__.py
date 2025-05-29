from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoImageProcessor,
    AutoConfig,
)

from .processor import (
    Qwen2VLImagePointerProcessor,
    get_processor,
    V1Processor,
    collate_fn,
)
from .modeling_v1 import V1ForConditionalGeneration
from .configuration_v1 import V1Config

print("Registering V1 model and processor")
AutoConfig.register("v1", V1Config)
AutoModelForCausalLM.register(
    V1Config, V1ForConditionalGeneration
)
AutoProcessor.register(V1Config, V1Processor)
AutoImageProcessor.register(V1Config, Qwen2VLImagePointerProcessor)
