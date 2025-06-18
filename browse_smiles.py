import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from rdkit import Chem
from rdkit.Chem import Draw
import sys
import pyperclip  # ✅ install via `pip install pyperclip`

# Load SMILES from a text file
def load_smiles(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

class MoleculeBrowser:
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.index = 0
        self.current_smiles = ""

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)

        # Button to copy SMILES
        self.copy_button_ax = self.fig.add_axes([0.35, 0.05, 0.3, 0.075])
        self.copy_button = Button(self.copy_button_ax, 'Copy SMILES')
        self.copy_button.on_clicked(self.copy_to_clipboard)

        # Message area
        self.message_ax = self.fig.add_axes([0.1, 0.15, 0.8, 0.05])
        self.message_ax.axis('off')
        self.message_text = self.message_ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=10)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.display_current()

    def display_current(self):
        self.ax.clear()
        smi = self.smiles_list[self.index]
        self.current_smiles = smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            self.ax.text(0.5, 0.5, f"Invalid SMILES:\n{smi}", ha='center', va='center')
        else:
            img = Draw.MolToImage(mol, size=(500, 500))
            self.ax.imshow(img)
            self.ax.set_title(f"{smi}", fontsize=10)
        self.ax.axis('off')
        self.message_text.set_text('')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.smiles_list)
            self.display_current()
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.smiles_list)
            self.display_current()

    def copy_to_clipboard(self, event):
        pyperclip.copy(self.current_smiles)
        self.message_text.set_text("Copied to clipboard!")
        self.fig.canvas.draw()

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
