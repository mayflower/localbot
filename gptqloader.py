"""Simple interactive bot using a local llm and vectorstore"""
import os
import torch
import transformers
import inspect
from transformers import pipeline, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as safe_load
from quant import make_quant # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/cuda/quant.py
from embeddings import HuggingFaceEmbeddingsGPU
from modelutils import find_layers # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/cuda/modelutils.py


class GPTQLoader:

# This function is a replacement for the load_quant function in the
# GPTQ-for_LLaMa repository. It supports more models and branches.
    @staticmethod 
    def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128, eval=True):

        def noop(*args, **kwargs):
            pass

        config = AutoConfig.from_pretrained(model)
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config)
        torch.set_default_dtype(torch.float)
        if eval:
            model = model.eval()
        layers = find_layers(model)
        for name in exclude_layers:
            if name in layers:
                del layers[name]

        gptq_args = inspect.getfullargspec(make_quant).args

        make_quant_kwargs = {
            'module': model,
            'names': layers,
            'bits': wbits,
        }
        if 'groupsize' in gptq_args:
            make_quant_kwargs['groupsize'] = groupsize
        if 'faster' in gptq_args:
            make_quant_kwargs['faster'] = faster_kernel
        if 'kernel_switch_threshold' in gptq_args:
            make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

        make_quant(**make_quant_kwargs)

        print('Loading model ...')
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            model.load_state_dict(safe_load(checkpoint), strict=False)
        else:
            model.load_state_dict(torch.load(checkpoint), strict=False)

        model.seqlen = 2048
        print('Done.')

        return model


    # The function that loads the model in modules/models.py
    @staticmethod
    def load_quantized(model_name, model_file):
        # Find the model type
        name = model_name.lower()
        if any((k in name for k in ['llama', 'alpaca', 'vicuna', 'llava'])):
            model_type = 'llama'
        elif any((k in name for k in ['opt-', 'galactica'])):
            model_type = 'opt'
        elif any((k in name for k in ['gpt-j', 'pygmalion-6b'])):
            model_type = 'gptj'
        else:
            print("Can't determine model type from model name. Please specify it manually using --model_type "
                    "argument")
            exit()


        # Find the quantized model weights file (.pt/.safetensors)
        path_to_model = model_name
        pt_path = model_file

        threshold = False if model_type == 'gptj' else 128
        model = GPTQLoader._load_quant(str(path_to_model), str(pt_path), 4, 128, kernel_switch_threshold=threshold)

        model = model.to(torch.device('cuda:0'))

        return model

