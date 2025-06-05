# list of selected scaffolds (as strings from the table above)
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import sys

# Load SMILES from a text file
def load_smiles(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

class MoleculeBrowser:
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.index = 0
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.display_current()

    def display_current(self):
        self.ax.clear()
        smi = self.smiles_list[self.index]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            self.ax.text(0.5, 0.5, f"Invalid SMILES:\n{smi}", ha='center', va='center')
        else:
            img = Draw.MolToImage(mol, size=(1000, 1000))
            self.ax.imshow(img)
            self.ax.set_title(smi, fontsize=10)
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.smiles_list)
            self.display_current()
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.smiles_list)
            self.display_current()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python browse_smiles.py path_to_smiles.txt")
        sys.exit(1)

    smiles = load_smiles(sys.argv[1])
    if not smiles:
        print("No valid SMILES found.")
        sys.exit(1)

    browser = MoleculeBrowser(smiles)
    plt.show()
