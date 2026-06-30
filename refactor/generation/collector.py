import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, set_seed, DataCollatorForLanguageModeling
from models.classifier import CustomLlamaForSequenceClassification
from models.guidance import CustomValueGuidedLogitProcessor, generate_with_classifier_guidance
from training.dataset import tokenize_with_chat_template, get_output_indices
from scoring.arithmetic import quick_evaluate_single, evaluate_preference, numeric_or_symbolic_correctness, sample_match_strict


def load_models(cfg, device):
    model_loading_kwargs = {}
    if cfg.models.dtype == 'bfloat16':
        model_loading_kwargs['torch_dtype'] = torch.bfloat16

    ref_model = AutoModelForCausalLM.from_pretrained(cfg.models.ref_model_id, **model_loading_kwargs, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.ref_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab_size = len(tokenizer)
    classifier_ckpt = cfg.models.classifier_ckpt or cfg.models.classifier_model_id
    classifier_model = CustomLlamaForSequenceClassification.from_pretrained(
        classifier_ckpt, **model_loading_kwargs,
        num_labels=vocab_size, loss_type='bce', use_bias=False,
        classifier_type=cfg.models.classifier_type, device_map=device)

    if cfg.models.classifier_ckpt is None:
        classifier_model.zero_init_classifier()

    ref_model.eval()
    classifier_model.eval()

    logit_processor = CustomValueGuidedLogitProcessor(
        eta=cfg.models.eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
        value_classifier=classifier_model, inference_mode=cfg.models.inference_mode,
        top_k=cfg.models.top_k, cd_baseline=cfg.models.cd_baseline, use_cache=True)

    logit_processor_disabled = CustomValueGuidedLogitProcessor(
        eta=cfg.models.eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
        value_classifier=classifier_model, inference_mode='disabled',
        top_k=cfg.models.top_k, cd_baseline=cfg.models.cd_baseline, use_cache=True)

    return ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled


def get_match_fn(cfg):
    match_fn_type = getattr(cfg.dataset, 'match_fn', 'symbolic')
    if match_fn_type == 'strict':
        return sample_match_strict
    return numeric_or_symbolic_correctness


def get_generate_kwargs(cfg):
    temperature = cfg.generation.temperature
    do_sample = temperature != 0
    if not do_sample:
        temperature = 1.0
    return {
        'temperature': temperature,
        'top_p': cfg.generation.top_p,
        'do_sample': do_sample,
        'max_new_tokens': cfg.generation.max_new_tokens,
        'top_k': 0,
    }
