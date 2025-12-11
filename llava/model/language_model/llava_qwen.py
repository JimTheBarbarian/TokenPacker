from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Config, Qwen3Model

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaQwenConfig(Qwen3Config):
    model_type = "llava_qwen3"


class LlavaQwen3model(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        super(LlavaQwen3model, self).__init__(config)


class LlavaQwen3ForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig # Added this line to explicitly set the config_class

    def __init__(self, config):
        super(LlavaQwen3ForCausalLM, self).__init__(config)

        self.model = LlavaQwen3model(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        mode: Optional[str] = "",
        h_block: Optional[int] = None,
        w_block: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
      

      if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                mode,
                h_block,
                w_block

            )
      if position_ids is None and inputs_embeds is not None: 
        # Create position_ids based on the new sequence length after image token insertion
            seq_length = inputs_embeds.shape[1]
            position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids. unsqueeze(0).expand(inputs_embeds.shape[0], -1)

      return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_qwen3", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwen3ForCausalLM)
