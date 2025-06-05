mamba create -n oledgen python=3.10
mamba activate oledgen
mamba install -c conda-forge rdkit ipython
pip3 install torch torchvision torchaudio
pip install transformers lightning peft cvxpy gdown