import os
from typing import Dict
import argparse
import timeit
import logging
# from common.fast_inference import FastInferenceInterface
# from common.together_web3.computer import RequestTypeLanguageModelInference
# from common.together_web3.together import TogetherWeb3, TogetherClientOptions
# from utils.fast_inference import FastInferenceInterface
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

from src.models.ssm_seq import SSMLMHeadModel


class H3Inference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        #for key in args.keys():
        #    print("{}: {}".format(arg, getattr(args, arg)))
        print("=========================================\n")
        
        self.task_info={
            "prompt_seqs": None,
            "output_len":16,
            "top_k": 50,
            "top_p": 0,
        }

        model_name_to_args = {
            'H3-125M': {
                'dmodel': 768,
                'nlayer': 12,
                'nheads': 12,
                'attn-layer-idx': [6]
            },
            'H3-355M': {
                'dmodel': 1024,
                'nlayer': 24,
                'nheads': 16,
                'attn-layer-idx': [8, 16]
            },
            'H3-1.3B': {
                'dmodel': 2048,
                'nlayer': 24,
                'nheads': 16,
                'attn-layer-idx': [8, 16]
            },
            'H3-2.7B': {
                'dmodel': 2560,
                'nlayer': 32,
                'nheads': 20,
                'attn-layer-idx': [8, 16, 24]
            },
        }

        device = 'cuda'
        self.device = device
        dtype = torch.float16

        h3_model_name = args['h3_model_name']
        d_model = model_name_to_args[h3_model_name]['dmodel']
        n_layer = model_name_to_args[h3_model_name]['nlayer']
        nheads = model_name_to_args[h3_model_name]['nheads']
        attn_layer_idx = model_name_to_args[h3_model_name]['attn-layer-idx']
        ssm_cfg = dict(mode='diag', measure='diag-lin')
        attn_cfg = dict(num_heads=nheads)
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        ckpt_path = args['ckpt_path']

        torch.manual_seed(0)
        with torch.no_grad():
            # Prepare model.
            model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)

            if ckpt_path is not None:
                state_dict = torch.load(ckpt_path, map_location=device)
                if 'pytorch-lightning_version' in state_dict:
                    state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                                if k.startswith('model.')}
                model.load_state_dict(state_dict)
                model.eval()

                # Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
                # Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
                        module.to(dtype=dtype)

            self.model = model
            self.tokenizer = tokenizer   
        print(f"<H3Inference.__init__> initialization done")
    
    def dispatch_request(self, args, env) -> Dict:
        print(f"dispatch_request get {args}")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [str(args['prompt'])]
        self.task_info["output_len"] = int(args.get("max_tokens", 16))
        self.task_info["top_k"] = int(args.get("top_k", 50))
        self.task_info["top_p"] = float(args.get("top_p", 0.9))
          
        result = self._run_inference()
        print(f"<H3Inference.dispatch_request> return: {result}")
        return result

    def _run_inference(self):
        print(f"<H3Inference._run_inference> enter rank-<{0}>")
        
        with torch.no_grad():
            prompt = self.task_info["prompt_seqs"][0]
            input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(device=self.device)
            max_length = input_ids.shape[1] + self.task_info['output_len']

            time = timeit.default_timer()
            output_ids = self.model.generate(input_ids=input_ids, max_length=max_length,
                       return_dict_in_generate=False, output_scores=False, 
                       timing=False, top_p=self.task_info["top_p"], top_k=self.task_info["top_k"], 
                       eos_token_id=self.tokenizer.eos_token_id)[:, input_ids.shape[1]:] # do not include input in the result
            time_elapsed = timeit.default_timer() - time
            
        print("[INFO] H3 time costs: {:.2f} ms. <rank-{}>".format(time_elapsed * 1000, 0))
        
        assert output_ids is not None

        inference_result = []
        outputs = self.tokenizer.batch_decode(output_ids)
        
        for i, (context, output) in enumerate(zip(self.task_info["prompt_seqs"], outputs)):
            item = {'choices': [{"text":output, "finish_reason":"length", "index":0}], }
            inference_result.append(item)
        #  So far coordinator does not support batch. 
        return {
            "result_type": RequestTypeLanguageModelInference,
            "choices": inference_result[0]['choices'],
            "raw_compute_time": time_elapsed
        }
        

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default=os.environ.get('SERVICE', 'Together-h3-125m'),
                        help='worker name for together coordinator.')
    parser.add_argument('--model_name', type=str, default='H3-125M',
                        help='model name.')
    parser.add_argument('--ckpt_path', type=str, default='/home/user/.together/models/H3-125M/model.pt',
                        help='path to the checkpoint file.')
    parser.add_argument('--worker_name', type=str, default=os.environ.get('WORKER', 'worker1'),
                        help='worker name for together coordinator.')
    parser.add_argument('--group_name', type=str, default=os.environ.get('GROUP', 'group1'),
                        help='group name for together coordinator.')
    
    args = parser.parse_args()
    
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")

    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket"
    )
    fip = H3Inference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "h3_model_name": args.model_name,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "ckpt_path": args.ckpt_path,
    })
    fip.start()