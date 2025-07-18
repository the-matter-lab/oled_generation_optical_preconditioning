#!/usr/bin/env python3
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator, Draw

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def compute_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)

def compute_morgan_fingerprint(scaffold_smiles, generator):
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is not None:
        return generator.GetFingerprint(mol)
    return None

def generate_fingerprint_array(fingerprints, scaffolds, length=2048):
    return np.array([
        list(fp) if fp is not None and Chem.MolFromSmiles(smiles) is not None
        else np.zeros(length, dtype=int)
        for fp, smiles in tqdm(zip(fingerprints, scaffolds), total=len(fingerprints), desc="Generating fingerprint array")
    ])

def plot_tsne(data, title, output_file, alpha_unselected=0.2):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    
    # Prepare subsets
    selected = data[data['selected'] == True]
    unselected = data[data['selected'] == False]
    
    # Consistent palette across all clusters
    n_clusters = data['cluster'].nunique()
    palette = sns.color_palette("Set3", n_clusters)

    # Plot all unselected molecules in gray
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        data=unselected,
        color="lightgray",
        alpha=alpha_unselected,
        edgecolor=None,
        legend=False,
        marker="o"
    )

    # Plot all selected molecules using cluster-based coloring
    scatter = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster",
        data=selected,
        palette=palette,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.3,
        marker="o",
        legend="full"
    )

    # Set legend to left
    plt.legend(
        title="cluster",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
        frameon=True
    )

    plt.title(title)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
    clustered_data = pd.read_csv('clustered_dataset.csv')
    selected_data = pd.read_csv('final_selection.csv')


    clustered_data['murcko_scaffold'] = clustered_data['smiles'].apply(compute_murcko_scaffold)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    clustered_data['morgan_fingerprint'] = clustered_data['murcko_scaffold'].apply(
        lambda x: compute_morgan_fingerprint(x, morgan_gen)
    )

    clustered_data['selected'] = clustered_data['smiles'].isin(selected_data['smiles'])

    fingerprints_array = generate_fingerprint_array(
        clustered_data['morgan_fingerprint'], clustered_data['murcko_scaffold']
    )


    n_clusters = 16


    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='cosine', learning_rate=200)
    tsne_results = tsne.fit_transform(fingerprints_array)
    clustered_data['tsne-2d-one'] = tsne_results[:, 0]
    clustered_data['tsne-2d-two'] = tsne_results[:, 1]

    # Plot 1: All solid
    clustered_data['selected'] = True
    plot_tsne(
        clustered_data.copy(),
        title="t-SNE of the complete dataset",
        output_file="tsne_all_solid.pdf"
    )

    # Plot 2: Selected vs. unselected
    clustered_data['selected'] = clustered_data['smiles'].isin(selected_data['smiles'])
    plot_tsne(
        clustered_data.copy(),
        title="t-SNE of selected molecules",
        output_file="tsne_selected_vs_unselected.pdf"
    )
    selected_smiles = selected_data['smiles'].tolist()
    selected_mols = [Chem.MolFromSmiles(smiles) for smiles in selected_smiles if Chem.MolFromSmiles(smiles) is not None]

    # Draw grid image
    img = Draw.MolsToGridImage(
        selected_mols,
        molsPerRow=6,
        subImgSize=(300, 300),
        legends=[f"{i+1}" for i in range(len(selected_mols))],
        useSVG=False
    )

    # Save image to file
    img.save("selected_molecules.png")