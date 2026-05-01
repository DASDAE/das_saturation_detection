# Seabed DAS Saturation Detection

This repository provides Python workflows for analyzing seabed Distributed Acoustic Sensing (DAS) data.  
It focuses on:

- First-break picking using robust, noise-adaptive methods  
- Waveform coherence analysis for detecting dynamic range saturation  
- Signal quality evaluation for DAS datasets  

---

## Repository Structure

### 1. First-break picking
- :contentReference[oaicite:0]{index=0}  
Contains multiple picking strategies:

- Bandpass filtering for DAS data  
- Envelope-based picking using Hilbert transform  
- Threshold-based picking methods  
- Robust outlier removal (local + slope constraints)  
- Iterative pick repair using neighbor interpolation  
- Final interpolation (PCHIP) to obtain smooth arrival curves  

---

### 2. Waveform coherence analysis
- :contentReference[oaicite:1]{index=1}  

Implements neighbor-based waveform coherence using arrival-aligned windows:

- Asymmetric window around first-break arrival  
- Multiple coherence metrics:
  - Pearson correlation
  - Absolute correlation
  - Cosine similarity
  - Spearman correlation
  - Cross-correlation with lag tolerance
  - Normalized MSE similarity  
- Aggregation across neighboring channels  

---

## Requirements

```bash
numpy
scipy
dascore
