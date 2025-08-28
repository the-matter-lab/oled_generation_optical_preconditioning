import os
import json
import csv
import torch
import argparse
from datetime import datetime
from rdkit import RDLogger
from transformers import AutoTokenizer, GenerationConfig
from model import OledModel
from mixed_config import get_config

RDLogger.DisableLog('rdApp.*')

SOLVENTS = [
    ("ClCCl",        "dichloromethane"),
    ("CC#N",         "acetonitrile"),
    ("Cc1ccccc1",    "toluene"),
    ("ClC(Cl)Cl",    "chloroform"),
    ("C1COC1",       "oxetane"),
    ("CO",           "methanol"),
    ("CCO",          "ethanol"),
    ("CS(C)=O",      "dmso"),
    ("C1CCCCC1",     "cyclohexane"),
    ("CN(C)C=O",     "dmf"),
]
SOLVENT_SMILES_TO_IDX = {smiles: i for i, (smiles, _) in enumerate(SOLVENTS)}
SOLVENT_NAME_TO_IDX   = {name: i for i, (_, name) in enumerate(SOLVENTS)}

CHEM_TOKEN_MAX_ID = 583

def int_in_range(lo, hi):
    def _check(v):
        try:
            iv = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be an integer in [{lo},{hi}]")
        if not (lo <= iv <= hi):
            raise argparse.ArgumentTypeError(f"must be in [{lo},{hi}]")
        return iv
    return _check

def parse_solvent(v: str) -> int:
    v = v.strip()
    if v.isdigit():
        idx = int(v)
        if 0 <= idx < len(SOLVENTS):
            return idx
        raise argparse.ArgumentTypeError("solvent index must be in [0,9]")
    if v in SOLVENT_SMILES_TO_IDX:
        return SOLVENT_SMILES_TO_IDX[v]
    key = v.lower()
    if key in SOLVENT_NAME_TO_IDX:
        return SOLVENT_NAME_TO_IDX[key]
    raise argparse.ArgumentTypeError(
        "unrecognized solvent. pass index [0–9], one of SMILES "
        f"{list(SOLVENT_SMILES_TO_IDX.keys())}, or a name "
        f"{list(SOLVENT_NAME_TO_IDX.keys())}"
    )

def generate_text(tokenizer, model, start_str, max_new_tokens, num_return_sequences, num_beams, temperature, do_sample):
    input_tokens = tokenizer.encode(start_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token to fall back on.")
    gen_cfg = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=do_sample,
    )
    return model.generate(input_tokens, generation_config=gen_cfg)

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(args.datapath, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('./smiles_tokenizer')

    cfg = get_config()
    cfg.gpt2_config['vocab_size']   = tokenizer.vocab_size
    cfg.gpt2_config['bos_token_id'] = tokenizer.bos_token_id
    cfg.gpt2_config['eos_token_id'] = tokenizer.eos_token_id
    cfg.gpt2_config['pad_token_id'] = tokenizer.pad_token_id

    model_wrapper = OledModel(cfg, tokenizer)
    model_wrapper.to_lora()
    try:
        state = torch.load(args.model_path, weights_only=True, map_location=torch.device('cpu'))
    except TypeError:
        state = torch.load(args.model_path, map_location=torch.device('cpu'))
    model_wrapper.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_wrapper.model = model_wrapper.model.to(device)
        model_wrapper.model.eval()
    except Exception:
        device = torch.device("cpu")
        model_wrapper.model = model_wrapper.model.to(device)
        model_wrapper.model.eval()

    prompt = (
        f"<bos>"
        f"<strength{args.strength}>"
        f"<absorption{args.absorption}>"
        f"<splitting{args.splitting}>"
        f"<rate{args.rate}>"
        f"<solvent{args.solvent}>"
    )

    model = model_wrapper.model
    try:
        output_tokens = generate_text(
            tokenizer, model, prompt,
            args.max_new_tokens, args.num_return_sequences,
            args.num_beams, args.temperature, args.do_sample
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        output_tokens = generate_text(
            tokenizer, model, prompt,
            args.max_new_tokens, args.num_return_sequences,
            args.num_beams, args.temperature, args.do_sample
        )

    smiles_csv = os.path.join(args.datapath, "smiles.csv")
    n_written = 0
    with open(smiles_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for e in output_tokens:
            toks_list = e.cpu().tolist()
            if CHEM_TOKEN_MAX_ID is not None:
                toks_list = [t for t in toks_list if t < CHEM_TOKEN_MAX_ID]
            s = tokenizer.decode(toks_list, skip_special_tokens=True)
            s = ''.join(s.split())
            writer.writerow([s])
            n_written += 1
            f.flush()

    solvent_smiles, solvent_name = SOLVENTS[args.solvent]

    metadata = {
        "model_path": args.model_path,
        "prompt": prompt,
        "controls": {
            "strength": args.strength,
            "absorption": args.absorption,
            "splitting": args.splitting,
            "rate": args.rate,
            "solvent_index": args.solvent,
            "solvent_name": solvent_name,
            "solvent_smiles": solvent_smiles,
        },
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "n_smiles_written": n_written,
        "device_used": str(device),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    meta_json = os.path.join(args.datapath, "metadata.json")
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate molecules using fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)

    parser.add_argument("--strength",   type=int_in_range(0, 4), default=4)
    parser.add_argument("--absorption", type=int_in_range(0, 4), default=0)
    parser.add_argument("--splitting",  type=int_in_range(0, 4), default=0)
    parser.add_argument("--rate",       type=int_in_range(0, 3), default=0)

    parser.add_argument("--solvent", dest="solvent_input_raw", type=str, default="0",
                        help="solvent as index [0–9], SMILES (from map), or name (e.g., toluene)")

    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_return_sequences", type=int, default=180)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    try:
        from argparse import BooleanOptionalAction
        parser.add_argument("--do_sample", action=BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--do_sample", dest="do_sample", action="store_true")
        parser.add_argument("--no-do_sample", dest="do_sample", action="store_false")
        parser.set_defaults(do_sample=True)

    args = parser.parse_args()
    args.solvent = parse_solvent(args.solvent_input_raw)
    main(args)
