<html>
<body>

<h1>De novo molecular generation with optical property preconditioning (working project)</h1>

<p>
This repository contains the minimal pipeline for <strong>generating OLED candidate molecules</strong> using a fine-tuned transformer model,
as well as an interactive browser to inspect generated SMILES structures. The final repo will include the pipeline for calculating the optical properties of the molecules.
This is a working project.
</p>

<hr>

<h2>🔧 Installation</h2>

<p>Use the provided <code>installation.sh</code> to create and activate a conda environment:</p>

<pre><code>bash installation.sh</code></pre>

<p>Download the finetuned solvent model from Google Drive using <code>download_model.sh</code>:</p>

<pre><code>bash download_model.sh</code></pre>

<hr>

<h2>🚀 SMILES Generation</h2>

<p>Run the generation script with conditioning flags:</p>

<pre><code>python generate_smiles.py \
  --model_path model_checkpoints/finetuned_coldstart_v1.ckpt \
  --datapath runs/test_run \
  --strength 4 --absorption 0 --splitting 0 --rate 0 --solvent toluene \
  --num_return_sequences 250 --temperature 1.0
</code></pre>

<h3>Key arguments:</h3>
<table border="1" cellspacing="0" cellpadding="6">
<thead>
<tr>
  <th>Flag</th>
  <th>Description</th>
  <th>Default</th>
</tr>
</thead>
<tbody>
<tr><td><code>--strength</code></td><td>Excited-state strength index (0–4)</td><td>4</td></tr>
<tr><td><code>--absorption</code></td><td>Absorption energy index (0–4)</td><td>0</td></tr>
<tr><td><code>--splitting</code></td><td>Energy splitting index (0–4)</td><td>0</td></tr>
<tr><td><code>--rate</code></td><td>Rate index (0–3)</td><td>0</td></tr>
<tr><td><code>--solvent</code></td><td>Solvent (index 0–9, name, or SMILES)</td><td>0</td></tr>
<tr><td><code>--num_return_sequences</code></td><td>Number of molecules to generate</td><td>180</td></tr>
<tr><td><code>--num_beams</code></td><td>Beam search width</td><td>1</td></tr>
<tr><td><code>--temperature</code></td><td>Sampling temperature</td><td>0.8</td></tr>
<tr><td><code>--do_sample</code></td><td>Enable sampling</td><td>True</td></tr>
</tbody>
</table>

<p>The script:</p>
<ul>
  <li>Builds a prompt internally from the conditioning flags.</li>
  <li>Automatically detects and uses a GPU (if available).</li>
  <li>Falls back to CPU if CUDA out of memory is encountered.</li>
  <li>Saves results in <code>smiles.csv</code> and metadata in <code>metadata.json</code> under the chosen <code>--datapath</code>.</li>
</ul>

<hr>

<h2>🧪 Prompt Control Tokens</h2>

<p>Control tokens are constructed automatically, but conceptually they map as follows:</p>

<table border="1" cellspacing="0" cellpadding="6">
<thead>
<tr>
  <th>Property</th>
  <th>Tokens</th>
  <th>Range</th>
</tr>
</thead>
<tbody>
<tr><td>Strength</td><td><code>&lt;strength0&gt;</code> … <code>&lt;strength4&gt;</code></td><td>0–4</td></tr>
<tr><td>Absorption</td><td><code>&lt;absorption0&gt;</code> … <code>&lt;absorption4&gt;</code></td><td>0–4</td></tr>
<tr><td>Splitting</td><td><code>&lt;splitting0&gt;</code> … <code>&lt;splitting4&gt;</code></td><td>0–4</td></tr>
<tr><td>Solvent</td><td><code>&lt;solvent0&gt;</code> … <code>&lt;solvent9&gt;</code></td><td>0–9</td></tr>
</tbody>
</table>

<h3>Solvent Mapping</h3>

<table border="1" cellspacing="0" cellpadding="6">
<thead>
<tr>
  <th>Token</th>
  <th>SMILES</th>
  <th>Solvent</th>
</tr>
</thead>
<tbody>
<tr><td><code>&lt;solvent0&gt;</code></td><td><code>ClCCl</code></td><td>Dichloromethane</td></tr>
<tr><td><code>&lt;solvent1&gt;</code></td><td><code>CC#N</code></td><td>Acetonitrile</td></tr>
<tr><td><code>&lt;solvent2&gt;</code></td><td><code>Cc1ccccc1</code></td><td>Toluene</td></tr>
<tr><td><code>&lt;solvent3&gt;</code></td><td><code>ClC(Cl)Cl</code></td><td>Chloroform</td></tr>
<tr><td><code>&lt;solvent4&gt;</code></td><td><code>C1COC1</code></td><td>Oxetane</td></tr>
<tr><td><code>&lt;solvent5&gt;</code></td><td><code>CO</code></td><td>Methanol</td></tr>
<tr><td><code>&lt;solvent6&gt;</code></td><td><code>CCO</code></td><td>Ethanol</td></tr>
<tr><td><code>&lt;solvent7&gt;</code></td><td><code>CS(C)=O</code></td><td>DMSO</td></tr>
<tr><td><code>&lt;solvent8&gt;</code></td><td><code>C1CCCCC1</code></td><td>Cyclohexane</td></tr>
<tr><td><code>&lt;solvent9&gt;</code></td><td><code>CN(C)C=O</code></td><td>DMF</td></tr>
</tbody>
</table>

<hr>

<h2>📂 Output Structure</h2>

<p>Each run creates a folder under <code>--datapath</code>:</p>

<pre><code>runs/test_run/
    ├── smiles.csv          # Generated SMILES strings (as they come out)
    └── metadata.json       # Generation parameters + metadata
</code></pre>

<hr>

<h2>Contributors</h2>
<ul>
  <li>Hazohe Huang</li>
  <li>Kevin Liu</li>
  <li>Hyun Suk Park</li>
  <li>Manuel Gonzalez-Lastre</li>
  <li>Jorge A. Campos-Gonzalez-Angulo</li>
  <li>Xinjian Liu</li>
  <li>Alán Aspuru-Guzik</li>
</ul>

</body>
</html>
