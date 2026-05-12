# Revisiting German NER with XLM-R Wearing a CRF Hat

**Danny Rechitsky** · Brandeis University, Computational Linguistics

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](paper/german_ner_xlmr_crf.pdf)
![Task](https://img.shields.io/badge/task-Named%20Entity%20Recognition-orange)
![Dataset](https://img.shields.io/badge/dataset-CoNLL--2003-green)
![Model](https://img.shields.io/badge/model-XLM--R%20Base-purple)

---

## Overview

This project asks a focused question: does adding a linear-chain CRF head on top of XLM-R allow it to surpass the Flair + BiLSTM-CRF system of Akbik et al. (2018) on the German CoNLL-2003 NER test set?

We first replicate the XLM-R Base fine-tuning experiments from Conneau et al. (2020) on both English (EN) and German (DE), then run a series of controlled experiments layering a CRF decoder onto XLM-R in different configurations: stacked post-hoc vs. jointly trained from the pretrained checkpoint.

The short answer: **a CRF is worth roughly +4.5 F1 points when XLM-R's fine-tuning capacity is constrained**, but adds almost nothing on top of a fully fine-tuned model, suggesting that end-to-end fine-tuning already internalizes the BIO transition structure that a CRF would otherwise supply.

---

## Key Results

### Replication of XLM-R Base (Table 2)

| Train | Test | Dev F1 | Test F1 | Conneau et al. Test F1 |
|-------|------|--------|---------|------------------------|
| EN    | EN   | 94.64  | **92.55** | 92.25 |
| EN    | DE   | 94.64  | 71.40   | 71.40 |
| DE    | DE   | 90.15  | **87.10** | 85.81 |
| EN+DE | DE   | 90.56  | 86.51   | — |

Our DE→DE replication **exceeds Conneau et al. by +1.29 F1**, attributed to using a corrected UTF-8 encoding of the German CoNLL-2003 data (fixing broken umlaut encoding in the original release).

### Effect of Adding a CRF Head (Table 3)

| Configuration | Test F1 |
|---|---|
| Flair + BiLSTM-CRF — Akbik et al. (2018) *prior SOTA* | 88.32 |
| XLM-R fine-tuned on DE (baseline) | 87.10 |
| CRF stacked on fine-tuned XLM-R (top layer unfrozen) | 87.24 |
| CRF + XLM-R jointly trained (top 1 layer unfrozen) | 80.96 |
| CRF + XLM-R jointly trained (top 2 layers unfrozen) | 83.24 |
| CRF + XLM-R jointly trained (top 3 layers unfrozen) | 83.59 |

### Controlled Head-to-Head (Table 4)

When only the top transformer layer is unfrozen, fixing architecture and hyperparameters and varying only the presence of the CRF:

| Model | Test F1 |
|---|---|
| XLM-R (top layer unfrozen, no CRF) | 75.99 |
| XLM-R (top layer unfrozen) + jointly-trained CRF | **80.53** |

This **~4.5 F1-point gain** is the cleanest evidence of the CRF's contribution: when the encoder has not yet absorbed span-level BIO structure through end-to-end fine-tuning. Making transition constraints explicit pays off.

---

## Broader Takeaway

CRF heads on top of masked language models can recover span-level structure that token-level predictions miss, but only when the encoder has not already absorbed that structure through full fine-tuning. Once XLM-R is fine-tuned end-to-end, ill-formed BIO sequences become rare enough that the CRF has little left to fix.

---

## Methods

- **Encoder**: XLM-R Base (12 layers, 768-dim hidden, 100-language pretraining on 2.5 TB cleaned Common Crawl)
- **Baseline head**: linear projection (768 → 9 BIO labels) via HuggingFace `AutoModelForTokenClassification`
- **CRF head**: linear-chain CRF (`pytorch-crf`); impossible BIO transitions initialized to −10,000; Viterbi decoding at inference
- **Training**: AdamW, lr = 2×10⁻⁵ (encoder) / 10⁻³ (CRF), weight decay 0.01, 3 epochs, seed 1
- **Evaluation**: span-level micro-averaged F1 (`seqeval`), matching the CoNLL-2003 convention

---

## Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `pytorch-crf` · `seqeval`

---

## Paper

[Read the full paper](paper/german_ner_xlmr_crf.pdf)

---

## Code

Training scripts and experiment configs coming soon.
