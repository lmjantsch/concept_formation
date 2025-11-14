import re

def get_prompts(homograph_data, context_type='context'):
    prompts = []
    for homograph in homograph_data:
        for meaning in homograph['meanings']:
            prompts.extend(meaning[context_type])
    
    return prompts

def get_entity_idx(tokenizer, prompts, entities=None):
    ent_idx = []
    
    if entities:
        if len(entities) != len(prompts):
            raise ValueError(f"Found {len(entities)} entities but {len(prompts)} prompts.")
        for prompt, entity in zip(prompts, entities):
            tokens = tokenizer.tokenize(prompt)
            idx = [i for i, tok in enumerate(tokens) if tok == entity]
            if not idx:
                raise ValueError(f"{entity} could not be found in {tokens}")
            ent_idx.append(idx[-1])
        return ent_idx

    for prompt in prompts:
        tokens = tokenizer.tokenize(prompt)
        ent_idx.append(len(tokens) - 1)
    
    return ent_idx


