<html>

<body>

<h1>OLED Molecular SMILES Generator</h1>

<p>
This repository contains the minimal pipeline for <strong>generating OLED candidate molecules</strong> using a fine-tuned transformer model,
as well as an interactive browser to inspect generated SMILES structures.
This is supposed to be an internal repository for The Matter Lab, models developed by Marko and Kevin.
</p>

<hr>

<h2>🔧 Installation</h2>

<p>Use the provided <code>installation.sh</code> to create and activate a conda environment:</p>

<pre><code>bash installation.sh</code></pre>

<hr>

<h2>🚀 SMILES Generation</h2>

<p>To run the generation script, load the model:</p>

<pre><code>python generate_smiles.py --model_path model_checkpoints/finetuned_coldstart_v1.ckpt</code></pre>

<h3>Optional arguments:</h3>
<table border="1" cellspacing="0" cellpadding="6">
<thead>
<tr>
  <th>Flag</th>
  <th>Description</th>
  <th>Default</th>
</tr>
</thead>
<tbody>
<tr><td><code>--prompt</code></td><td>Prompt with property tokens</td><td><code>&lt;bos&gt;&lt;strength4&gt;&lt;absorption0&gt;&lt;splitting0&gt;&lt;solvent0&gt;</code></td></tr>
<tr><td><code>--max_new_tokens</code></td><td>Max tokens to generate per molecule</td><td>150</td></tr>
<tr><td><code>--num_return_sequences</code></td><td>Number of molecules to generate</td><td>180</td></tr>
<tr><td><code>--num_beams</code></td><td>Beam search width</td><td>200</td></tr>
<tr><td><code>--temperature</code></td><td>Sampling temperature</td><td>0.8</td></tr>
<tr><td><code>--do_sample</code></td><td>Use sampling instead of greedy decoding</td><td>True</td></tr>
<tr><td><code>--generated_data</code></td><td>Output directory for results</td><td><code>generated_data</code></td></tr>
</tbody>
</table>

<p>The script:</p>
<ul>
  <li>Automatically detects and uses a GPU (if available).</li>
  <li>Falls back to CPU if CUDA out of memory is encountered.</li>
  <li>Saves results in a timestamped subdirectory:</li>
  <ul>
    <li><code>smiles.txt</code> — all valid generated SMILES</li>
    <li><code>generation_config.yaml</code> — generation metadata and parameters</li>
  </ul>
</ul>

<hr>

<h2>🧬 SMILES Viewer</h2>

<p>Use the interactive viewer to browse generated molecules:</p>

<pre><code>python browse_smiles.py generated_data/20250605-103015/smiles.txt</code></pre>

<p>Controls:</p>
<ul>
  <li><code>→</code> (right arrow): next molecule</li>
  <li><code>←</code> (left arrow): previous molecule</li>
</ul>

<hr>

<h2>🧪 Prompt Control Tokens</h2>

<p>You can condition generation on specific properties using control tokens in the prompt:</p>

<table border="1" cellspacing="0" cellpadding="6">
<thead>
<tr>
  <th>Property</th>
  <th>Tokens</th>
  <th>Range</th>
</tr>
</thead>
<tbody>
<tr><td>Strength</td><td><code>&lt;strength0&gt;</code> to <code>&lt;strength6&gt;</code></td><td>601–607</td></tr>
<tr><td>Absorption</td><td><code>&lt;absorption0&gt;</code> to <code>&lt;absorption6&gt;</code></td><td>583–589</td></tr>
<tr><td>Splitting</td><td><code>&lt;splitting0&gt;</code> to <code>&lt;splitting6&gt;</code></td><td>594–600</td></tr>
<tr><td>Solvent</td><td><code>&lt;solvent0&gt;</code> to <code>&lt;solvent9&gt;</code></td><td>608–617</td></tr>
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

<p>Each generation run creates a folder like:</p>

<pre><code>generated_data/20250605-103015/</code></pre>

<p>Containing:</p>

<pre><code>generated_data/
└── 20250605-103015/
    ├── smiles.txt               # Valid SMILES strings
    └── generation_config.yaml   # Generation parameters + metadata
</code></pre>

</body>
</html>
