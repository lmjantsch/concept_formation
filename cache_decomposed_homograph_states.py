import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import json
import sys
from typing import Dict, List
from argparse import ArgumentParser, Namespace

project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoTokenizer, PretrainedConfig
import torch
from torch.utils.data import DataLoader
from nnsight import LanguageModel

from data.utils import get_entity_idx, get_prompts
from mi_toolbox.utils.data_types import DataDict
from mi_toolbox.utils.collate import TokenizeCollator
from mi_toolbox.transformer_caching import caching_wrapper, decompose_attention_to_neuron, decompose_glu_to_neuron

def get_caching_function(layer_slice: slice):
    """
    Returns the caching function closure with the specific layer target.
    """
    def caching_function(llm: LanguageModel, config: PretrainedConfig, batch: Dict[str, List]) -> Dict:
        try:
            batch_cache = {}
            batch_ent_pos_idx = batch['batch_ent_pos_idx']
            batch_cache['attention_mask'] = batch['attention_mask']
            batch_cache['ent_pos_idx'] = batch_ent_pos_idx[1]

            with llm.trace(batch) as tracer:         
                emb = llm.model.embed_tokens.output
                
                batch_cache['emb'] = emb[batch_ent_pos_idx].cpu().save()
                batch_cache['full_emb'] = emb.cpu().save()
                
                # Iterate over selected layers
                for i, layer in enumerate(llm.model.layers[layer_slice]):
                    attn_norm_std = torch.std(layer.input, dim=-1)
                    batch_cache[f'{i}.attn_norm_std'] = attn_norm_std.cpu().save()
                    
                    # Attention Decomposition
                    v_proj = layer.self_attn.v_proj.output
                    _, attn_weight = layer.self_attn.output
                    o_proj_WT = layer.self_attn.o_proj.weight.T
                    
                    d_attn = decompose_attention_to_neuron(
                        attn_weight, 
                        v_proj, 
                        o_proj_WT,
                        config.num_attention_heads,
                        config.num_key_value_heads,
                        config.head_dim
                    ) 
                    batch_cache[f'{i}.v_proj'] = v_proj.cpu().save()
                    batch_cache[f'{i}.d_attn'] = d_attn.cpu().save()
                    
                    mid = layer.post_attention_layernorm.input[batch_ent_pos_idx]
                    mlp_norm_std = torch.std(layer.post_attention_layernorm.input, dim=-1)
                    batch_cache[f'{i}.mid'] = mid.cpu().save()
                    batch_cache[f'{i}.mlp_norm_std'] = mlp_norm_std.cpu().save()

                    # MLP Decompositon
                    up_proj = layer.mlp.up_proj.output
                    activation_product = layer.mlp.down_proj.input 
                    down_proj_WT = layer.mlp.down_proj.weight.T
                    
                    d_mlp = decompose_glu_to_neuron(
                        act_prod=activation_product, 
                        down_proj_WT=down_proj_WT
                    )
                    batch_cache[f'{i}.d_mlp'] = d_mlp.cpu().save()
                    batch_cache[f'{i}.up_proj'] = up_proj.cpu().save()

                    post = layer.output[batch_ent_pos_idx]
                    batch_cache[f'{i}.post'] = post.cpu().save()
                    
                    
        except Exception as e:
            print(f"Error during tracing: {e}")
            raise e
        finally:
            if 'tracer' in locals():
                del tracer
                
        return batch_cache

    return caching_function

def parse_arguments():
    parser = ArgumentParser(description="Cache Transformer States for Homographs")
    
    # Required arguments
    parser.add_argument("homograph_id", type=int, help="Index of the homograph in the dataset")
    
    # Optional arguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B", help="HuggingFace Model ID")
    parser.add_argument("--data_path", type=str, default="data/homograph_data/homograph_small.json", help="Path to JSON data file")
    parser.add_argument("--output_base_dir", type=str, default="/raid/dacslab/CONCEPT_FORMATION/homograph_small", help="Base output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for DataLoader")
    parser.add_argument("--layer_start", type=int, default=0, help="Start layer index")
    parser.add_argument("--layer_end", type=int, default=10, help="End layer index")
    
    args = parser.parse_args()

    # Set device environment variable if needed, though usually handled externally
    if 'cuda' in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]

    print(f"Processing Homograph ID: {args.homograph_id}")
    print(f"Model: {args.model_id}")

    return args

def main(args: Namespace):

    # Load Data
    full_data_path = os.path.join(project_root, args.data_path)
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"Data file not found at {full_data_path}")
        
    with open(full_data_path) as f:
        data = json.load(f)

    if args.homograph_id >= len(data):
        print(f"Error: homograph_id {args.homograph_id} is out of bounds (size: {len(data)})")
        sys.exit(1)

    sample_data = [data[args.homograph_id]]
    prompts = get_prompts(sample_data, context_type='minimal_context')
    
    # Prepare Data
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ent_idx = get_entity_idx(tokenizer, prompts)
    extract_collate_fn = TokenizeCollator(tokenizer, collate_fn={
        'ent_idx': lambda key, value: {'batch_ent_pos_idx': (list(range(len(value))), value)}
    })

    extract_dd = DataDict.from_dict({'prompts': prompts, "ent_idx": ent_idx})
    extract_dl = DataLoader(
        extract_dd, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=extract_collate_fn
    )

    # Prepare Extraction
    layer_slice = slice(args.layer_start, args.layer_end)
    cache_fn = get_caching_function(layer_slice)
    
    # Run Caching
    big_cache = caching_wrapper(
        args.model_id, 
        extract_dl, 
        cache_fn, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Save Results
    safe_model_name = args.model_id.replace('/', '_')
    output_dir = os.path.join(args.output_base_dir, safe_model_name, str(args.homograph_id))
    
    if args.model_id in big_cache:
        model_cache = big_cache[args.model_id]
        print(f"Saving cache to {output_dir}...")
        model_cache.save(output_dir)
    else:
        print("Error: No cache generated for model.")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)