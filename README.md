# Lossy-QSCI: Lossy Quantum Selected Configuration Interaction

> Qubit-efficient subspace methods for quantum chemistry via **chemistry-informed Random Linear Encoding (Chem-RLE)** and a **neural-network Fermionic Expectation Decoder (NN-FED)**.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-README-lightgrey.svg)](#-documentation)

---

## ✨ What is this?

**Lossy-QSCI** is a hybrid quantum–classical workflow for ground-state electronic structure that:

* **Compresses** fermionic configurations into **O(N log M)** qubits using a **chemistry-informed Random Linear Encoder (Chem-RLE)**,
* **Decodes** measurements efficiently with a lightweight **neural decoder (NN-FED)** tailored to number-conserving sectors,
* **Selects** compact CI subspaces with **QSCI** (or TE-QSCI/VQE variants) and **classically diagonalizes** the Hamiltonian in that subspace.

> TL;DR: fewer qubits than full JW/Parity encodings, fast classical decoding, and subspace quality competitive with uncompressed QSCI at the same accuracy target.

---

## 🔧 Features

* **Chem-RLE**: chemistry-biased lossy encoder (CAS/MO-ordered priors) to concentrate support near the physically relevant sector.
* **NN-FED**: small feed-forward network (≈O(MN) params) that learns to invert Chem-RLE on the **N-electron** manifold.
* **Lossy-QSCI loop**: encode → prepare → sample → decode → enrich subspace → diagonalize → repeat until energy stalls.
* **Reproducible experiments**: scripts/*.py for C₂ (6-31G), LiH (STO-3G), H₂ (6-31G) incl. noisy bit-flip channel.

---

## 📦 Installation

**Requirements**

* Python **3.10+**
* Optional GPU for faster NN-FED training (PyTorch/CUDA)

**Install (editable):**

```bash
git clone https://github.com/ALS15204/lossy-qsci.git
cd lossy-qsci
python -m venv .venv && source .venv/bin/activate   # or conda
pip install --upgrade pip
pip install -r requirements.txt
```

Typical dependencies (already pinned in `requirements.txt`):

* `numpy`, `scipy`, `pandas`, `matplotlib`
* `pytorch` (CPU or CUDA) for NN-FED
* `pyscf`, `openfermion`, `qiskit` (or your simulator of choice)

---

## 🧠 Concepts (short)

* **Chem-RLE** compresses occupation bitstrings with a **number-conserving** linear map, optionally **biased** by CAS/MO heuristics to keep chemically important determinants and drop likely-irrelevant ones.
* **NN-FED** learns a fast *task-specific* inverse on the N-electron sector, enabling **O(M⁴)**-style measurement post-processing for RDM elements with **negligible lookup memory**.
* **Lossy-QSCI** iteratively enriches the CI subspace: compressed sampling → decoding → subspace diagonalization → accept if energy improves.

---

## ⚠️ Limitations & scope

* **Scalability/universality**: As with **VQE/QSCI** families, demonstrated scales are modest (tens of qubits on well-studied molecules). Chem-RLE injectivity checks and NN-FED generalization may require **system-specific tuning** for large, *unknown* systems. We therefore avoid claiming universality and use “scalable” in the **engineering** sense (improved resource efficiency at fixed accuracy) rather than asymptotic optimality.
* **Lossy encoding**: Aggressive compression can discard small-weight determinants; Lossy-QSCI’s iterative “collect-and-refine” loop is designed to mitigate this, but **monitor energy bias**.
* **Noise**: Number-conserving compression limits unphysical samples; single-bit flips may map between valid codewords—use **post-selection thresholds** and **noise-aware** sampling.

---

## 📊 Benchmarks (at a glance)

* **C₂ (6-31G)**: With **R=50**, Lossy-QSCI approaches uncompressed QSCI (R≈65) as Q increases, beating ED on smaller active spaces at the same qubit budget.
* **LiH (STO-3G)**: **5-qubit** encoding reaches (10,2) chemical accuracy with **\~12** basis states; VQE alone (random compressions) remains above −7.8 Ha.
* **H₂ (6-31G, noisy)**: Lossy-QSCI (4 qubits) hits chemical accuracy with **R≈12**, using fewer samples than QSCI (8 qubits, R≈15).

> Full figures, configurations, and raw CSVs are logged under `figures/` and `experiments/_outputs/`.

---

## 📚 Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{lossyqsci2025,
  title  = {Lossy-QSCI: Qubit-efficient subspace methods with Chem-RLE and NN-FED},
  author = {Chen, Yu-Cheng and Wu, Ronin and Hsieh, Min-Hsiu},
  year   = {2025},
  note   = {GitHub repository: ALS15204/lossy-qsci}
}
```

And the underlying methods you rely on (QSCI/TE-QSCI, FED/RLE, basis sets, etc.).

---

## 🤝 Contributing

PRs and issues are welcome! Please:

1. Open an issue describing the change/bug.
2. Run `pytest` and include/adjust tests.
3. Follow the code style used in `src/lossy_qsci/` (type hints, docstrings).

---

## 📜 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE).

---

## 🙏 Acknowledgements

We thank collaborators and colleagues for discussions and code support. Portions of the workflow build on standard quantum chemistry libraries (PySCF/OpenFermion/Qiskit) and PyTorch for NN-FED.

---

## 💡 FAQ

**Q: Do I need a GPU?**
No—models are small. GPU just speeds up NN-FED training.

**Q: Can I use my own ansatz or state-prep?**
Yes. Swap in your trial state (HF, UCCSD, LUCJ, HEA, TE-QSCI). The driver only needs a sampler returning computational-basis bitstrings.

**Q: How do I set Q?**
Rule-of-thumb: start near **⌈c·N·log₂M⌉** with `c∈[1,2]`. If energy stalls, increase Q or relax the lossy bias.

**Q: What if decoding fails on some samples?**
Filter by decoder confidence or re-train NN-FED with a small hard-example buffer; keep only number-conserving decodes.

---

*Happy compressing & selecting!*
