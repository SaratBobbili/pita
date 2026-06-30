from transformers import LlamaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position


class CustomLlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config, loss_type, use_bias, classifier_type, *, num_atoms: int = 11, V_min: float = 0.0, V_max: float = 1.0):
        assert classifier_type in ["Q", "V"]
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier_type = classifier_type
        self.model = LlamaModel(config)
        if loss_type == "mse":
            self.loss_fct = MSELoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "bce":
            self.loss_fct = BCEWithLogitsLoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "mle":
            self.num_atoms = num_atoms
            self.V_min = V_min
            self.V_max = V_max
            self.atoms = torch.linspace(self.V_min, self.V_max, self.num_atoms).float()
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels * self.num_atoms, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, self.num_atoms, bias=use_bias)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}.")

        self.loss_type = loss_type
        self.use_bias = use_bias
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def zero_init_classifier(self):
        nn.init.zeros_(self.score.weight)
        if self.use_bias:
            nn.init.zeros_(self.score.bias)

    def calculate_loss(self, logits, labels, loss_weights, loss_mask):
        assert len(logits.shape) == 3
        bs, seqlen, _ = logits.shape
        assert loss_mask.shape == (bs, seqlen)
        assert labels.shape == (bs,)
        assert loss_weights.shape == (bs,)

        if self.loss_type == "mse":
            relevant_logits = torch.sigmoid(logits).squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(relevant_logits, labels_expanded.to(relevant_logits.dtype))
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        elif self.loss_type == "bce":
            assert logits.shape[2] == 1
            logits = logits.squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(logits, labels_expanded)
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        elif self.loss_type == "mle":
            log_pmfs = F.log_softmax(logits, dim=-1)
            label_indices = torch.round(labels * (self.num_atoms - 1)).long()
            label_indices = torch.clamp(label_indices, 0, self.num_atoms - 1)
            loss = -log_pmfs[torch.arange(bs), :, label_indices]
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        else:
            raise RuntimeError("Impossible to reach.")
        assert loss.shape == loss_weights.shape
        loss = loss * loss_weights
        loss = loss.mean()
        return loss

    def calculate_predictions(self, logits):
        bs, seqlen, num_labels = logits.shape
        if self.loss_type in ["mse", "bce"]:
            return torch.sigmoid(logits).squeeze(-1)
        elif self.loss_type == "mle":
            pmfs = torch.softmax(logits, dim=-1)
            if self.atoms.device != pmfs.device:
                self.atoms = self.atoms.to(pmfs.device)
            return (pmfs * self.atoms).sum(dim=-1)
        else:
            raise NotImplementedError()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_weights: Optional[torch.FloatTensor] = None,
            logit_indices: Optional[torch.LongTensor] = None,
            loss_mask: Optional[torch.BoolTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs = input_ids.size(0)

        if self.classifier_type == "Q":
            if labels is not None:
                bs, seqlen = input_ids.shape
                transformer_outputs = self.model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0]
                logits = self.score(hidden_states)
                if self.loss_type in ["mse", "bce"]:
                    logits = logits.unsqueeze(-1)
                elif self.loss_type == "mle":
                    logits = logits.view(bs, seqlen, self.num_labels, self.num_atoms)
                indexed_logits = logits[:, :-1][torch.arange(bs)[:, None], torch.arange(seqlen-1), input_ids[:, 1:]]
                indexed_logits = indexed_logits.float()
                loss = self.calculate_loss(indexed_logits, labels, loss_weights, loss_mask[:, 1:])
                return SequenceClassifierOutputWithPast(loss=loss, logits=indexed_logits)
            else:
                bs, _ = input_ids.shape
                top_k = self.num_labels
                if logit_indices is not None:
                    top_k = logit_indices.size(1)
                transformer_outputs = self.model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0]
                hidden_states = hidden_states[:, -1]
                logits = self.score(hidden_states)

                if self.loss_type in ["mse", "bce"]:
                    if logit_indices is not None:
                        logits = logits[torch.arange(bs)[:, None], logit_indices]
                elif self.loss_type == "mle":
                    if logit_indices is not None:
                        offsets = torch.arange(self.num_atoms, device=logit_indices.device)
                        expanded = logit_indices.unsqueeze(-1) * self.num_atoms + offsets
                        expanded = expanded.view(bs, -1)
                        logits = logits[torch.arange(bs)[:, None], expanded]
                    logits = logits.float().view(bs, top_k, self.num_atoms)

                return SequenceClassifierOutputWithPast(
                    logits=logits,
                    past_key_values=transformer_outputs.past_key_values,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )

        elif self.classifier_type == "V":
            assert return_dict, "V must return dict"
            if labels is not None:
                assert logit_indices is None
                assert loss_mask is not None
                transformer_outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = transformer_outputs[0]
                logits = self.score(hidden_states).float()
                loss = self.calculate_loss(logits, labels, loss_weights, loss_mask)
                return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
            else:
                top_k = logit_indices.size(1)
                transformer_outputs = self.model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
                )
                output_past_key_values = transformer_outputs.past_key_values
                dtype, device = output_past_key_values[0][0].dtype, output_past_key_values[0][0].device
                min_dtype = torch.finfo(dtype).min
                next_input_ids = logit_indices.to(input_ids.device)
                expanded_attention_mask = torch.cat([attention_mask, torch.ones((bs, top_k), dtype=torch.long, device=attention_mask.device)], dim=1)
                cache_position = torch.arange(attention_mask.shape[1], expanded_attention_mask.shape[1], device=device)
                actual_position_ids = (torch.ones((1, top_k)) * attention_mask.shape[1]).to(dtype=attention_mask.dtype, device=device)
                actual_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    expanded_attention_mask, top_k, expanded_attention_mask.shape[1],
                    dtype=dtype, device=device, min_dtype=min_dtype,
                    cache_position=cache_position, batch_size=input_ids.shape[0])
                diagonal_mask = torch.full((top_k, top_k), min_dtype)
                diagonal_mask.fill_diagonal_(0)
                diagonal_mask = diagonal_mask.to(dtype=actual_attention_mask.dtype, device=device)
                actual_attention_mask[:, :, :, -top_k:] = diagonal_mask
                transformer_outputs = self.model(
                    next_input_ids, attention_mask=actual_attention_mask, position_ids=actual_position_ids,
                    past_key_values=output_past_key_values, use_cache=True, cache_position=cache_position)
                hidden_states = transformer_outputs[0]
                hidden_states = hidden_states[:, -top_k:]
                logits = self.score(hidden_states)
                if self.loss_type == "mle":
                    assert logits.shape == (bs, top_k, self.num_atoms)
                else:
                    logits = logits.squeeze(-1)
                return SequenceClassifierOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=output_past_key_values,
                )
