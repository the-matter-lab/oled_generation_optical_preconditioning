import os
import torch
import argparse
import yaml
from rdkit import Chem
from rdkit import RDLogger
from datetime import datetime
from transformers import AutoTokenizer, GenerationConfig
from model import OledModel
from mixed_config import get_config

"""
SMILES Generator Pipeline
=========================

This script uses a fine-tuned transformer model to generate molecular SMILES strings
conditioned on property control tokens such as strength, absorption, splitting, and solvent.

Usage:
------
Run from the command line:

    python generate_smiles.py --model_path finetuned_coldstart_v1.ckpt

Optional flags:
---------------
    --max_new_tokens        (default: 150)
    --num_return_sequences  (default: 180)
    --num_beams             (default: 200)
    --temperature           (default: 0.8)
    --do_sample             (default: True)
    --prompt                (default: "<bos><strength4><absorption0><splitting0><solvent0>")
    --generated_data        (default: "generated_data")

Prompt control tokens (examples):
---------------------------------
    - <strengthX>    where X ∈ [0–6]
    - <absorptionX>  where X ∈ [0–6]
    - <splittingX>   where X ∈ [0–6]
    - <solventX>     where X ∈ [0–9]

Solvent mapping:
----------------
    'ClCCl'         -> <solvent0>  # Dichloromethane  
    'CC#N'          -> <solvent1>  # Acetonitrile  
    'Cc1ccccc1'     -> <solvent2>  # Toluene  
    'ClC(Cl)Cl'     -> <solvent3>  # Chloroform  
    'C1COC1'        -> <solvent4>  # Oxetane  
    'CO'            -> <solvent5>  # Methanol  
    'CCO'           -> <solvent6>  # Ethanol  
    'CS(C)=O'       -> <solvent7>  # DMSO  
    'C1CCCCC1'      -> <solvent8>  # Cyclohexane  
    'CN(C)C=O'      -> <solvent9>  # DMF  

Output folder structure:
------------------------
A folder named by timestamp (e.g. `generated_data/20250605-103015/`) will be created.
Inside you will find:

    - smiles.txt               : all valid generated SMILES
    - generation_config.yaml   : parameters used in this run (model path, prompt, generation settings)


"""


# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def generate_text(tokenizer, model, start_str, max_new_tokens, num_return_sequences, num_beams, temperature, do_sample):
    """
    Generate molecular SMILES strings from a pretrained or fine-tuned language model.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for encoding the prompt and decoding the output tokens.
    model : torch.nn.Module
        Language model used to generate the output sequences.
    start_str : str
        Prompt string to condition the generation. Must include property control tokens such as:
        - <strengthX>
        - <absorptionX>
        - <splittingX>
        - <solventX>
        Example: '<bos><strength4><absorption0><splitting0><solvent0>'
    max_new_tokens : int
        Maximum number of new tokens to generate per sequence.
    num_return_sequences : int
        Total number of sequences to generate per input.
    num_beams : int
        Beam width for beam search. Higher values explore more candidate sequences.
    temperature : float
        Sampling temperature. Lower values yield more deterministic outputs; higher values increase randomness.
    do_sample : bool
        If True, use sampling instead of greedy decoding. Must be True if using temperature.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_return_sequences, sequence_length) containing generated token IDs.

    Token Dictionary for Property Control
    -------------------------------------
    Property tokens used in `start_str`:

    - Absorption:       <absorption0> (ID: 583) to <absorption6> (ID: 589)
    - Rate:             <rate0> (590) to <rate3> (593)
    - Splitting:        <splitting0> (594) to <splitting6> (600)
    - Strength:         <strength0> (601) to <strength6> (607)
    - Solvent:          <solvent0> (608) to <solvent9> (617)

    Solvent Label Map (for reference)
    ---------------------------------
    {
        'ClCCl':         '<solvent0>',  # Dichloromethane  
        'CC#N':          '<solvent1>',  # Acetonitrile  
        'Cc1ccccc1':     '<solvent2>',  # Toluene  
        'ClC(Cl)Cl':     '<solvent3>',  # Chloroform  
        'C1COC1':        '<solvent4>',  # Oxetane  
        'CO':            '<solvent5>',  # Methanol  
        'CCO':           '<solvent6>',  # Ethanol  
        'CS(C)=O':       '<solvent7>',  # DMSO  
        'C1CCCCC1':      '<solvent8>',  # Cyclohexane  
        'CN(C)C=O':      '<solvent9>',  # DMF  
    }
    """
    input_tokens = tokenizer.encode(start_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=do_sample,
    )
    return model.generate(input_tokens, generation_config=generation_config)


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained('./smiles_tokenizer')

    cfg = get_config()
    cfg.gpt2_config['vocab_size'] = tokenizer.vocab_size
    cfg.gpt2_config['bos_token_id'] = tokenizer.bos_token_id
    cfg.gpt2_config['eos_token_id'] = tokenizer.eos_token_id
    cfg.gpt2_config['pad_token_id'] = tokenizer.pad_token_id

    model_wrapper = OledModel(cfg, tokenizer)
    model_wrapper.to_lora()

    # Load checkpoint to CPU initially
    model_wrapper.load_state_dict(
        torch.load(args.model_path, weights_only=True, map_location=torch.device('cpu'))
    )

    # Attempt to move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_wrapper.model = model_wrapper.model.to(device)
        model_wrapper.model.eval()
        print(f"Using device: {device}")
    except Exception as e:
        print(f"⚠️  Failed to use GPU due to: {e}\nSwitching to CPU.")
        device = torch.device("cpu")
        model_wrapper.model = model_wrapper.model.to(device)
        model_wrapper.model.eval()

    model = model_wrapper.model

    try:
        output_tokens = generate_text(
            tokenizer, model, args.prompt,
            args.max_new_tokens, args.num_return_sequences,
            args.num_beams, args.temperature, args.do_sample
        )
    except torch.cuda.OutOfMemoryError as e:
        print("⚠️  CUDA out of memory during generation. Switching to CPU and retrying...")
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        output_tokens = generate_text(
            tokenizer, model, args.prompt,
            args.max_new_tokens, args.num_return_sequences,
            args.num_beams, args.temperature, args.do_sample
        )

    valid_smiles = []
    for e in output_tokens:
        output_token = [t for t in e.cpu().tolist() if t < 583]
        smiles = tokenizer.decode(output_token, skip_special_tokens=True)
        smiles = ''.join(smiles.split())
        try:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            valid_smiles.append(smiles)
        except:
            continue

    print(f"{len(valid_smiles)} valid SMILES generated.")

    # Output directory setup
    os.makedirs(args.generated_data, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.generated_data, run_id)
    os.makedirs(run_dir)

    # Save SMILES
    smiles_path = os.path.join(run_dir, "smiles.txt")
    with open(smiles_path, "w") as f:
        for s in valid_smiles:
            f.write(s + "\n")

    # Save config
    config = {
        'model_path': args.model_path,
        'prompt': args.prompt,
        'max_new_tokens': args.max_new_tokens,
        'num_return_sequences': args.num_return_sequences,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'do_sample': args.do_sample,
        'n_valid_smiles': len(valid_smiles),
        'device_used': str(device),
    }
    with open(os.path.join(run_dir, "generation_config.yaml"), "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate molecules using fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="<bos><strength4><absorption0><splitting0><solvent0>", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_return_sequences", type=int, default=180)
    parser.add_argument("--num_beams", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--generated_data", type=str, default="generated_data", help="Directory to store generated results")
    args = parser.parse_args()
    main(args)
