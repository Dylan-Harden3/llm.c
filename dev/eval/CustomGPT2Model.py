import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model


class CustomSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner // 2, config.n_embd, bias=True)
        self.act = nn.SiLU()

        self.c_fc.weight = nn.Parameter(self.c_fc.weight.t())
        self.c_proj.weight = nn.Parameter(self.c_proj.weight.t())

    def forward(self, x):
        x = nn.functional.linear(x, self.c_fc.weight.t(), self.c_fc.bias)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        return nn.functional.linear(x, self.c_proj.weight.t(), self.c_proj.bias)


class CustomGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Model(config).h[0].attn
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CustomSwiGLU(config)

    def forward(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = x
        x = self.ln_1(x)
        attn_outputs = self.attn(
            x,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        x = attn_outputs[0]
        x = residual + x
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x
        outputs = (x,) + attn_outputs[1:]
        return outputs


class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)


class CustomGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
