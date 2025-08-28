python generate_smiles.py --model_path model_checkpoints/finetuned_coldstart_v1.ckpt --datapath runs/r1 \
  --strength 4 --absorption 0 --splitting 0  --solvent 0 --num_return_sequences 1000 --num_beams 10

python generate_smiles.py --model_path model_checkpoints/finetuned_coldstart_v1.ckpt --datapath runs/r2 \
  --strength 4 --absorption 0 --splitting 0 --rate 0 --solvent dichloromethane --num_return_sequences 10 --num_beams 10

python generate_smiles.py --model_path model_checkpoints/finetuned_coldstart_v1.ckpt --datapath runs/r3 \
  --strength 4 --absorption 0 --splitting 0 --rate 0 --solvent ClCCl --num_return_sequences 10 --num_beams 10
