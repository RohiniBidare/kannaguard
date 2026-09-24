# KannaGuard: Code-Mixed Kannada Content Moderation Framework

KannaGuard is a production-grade content moderation framework tailored for code-mixed Kannada-English (Kanglish) live streaming interactions. The system utilizes a fine-tuned IndicBERTv2 architecture combined with INT8 post-training quantization, post-hoc probability threshold calibration, and a three-tier Human-in-the-Loop (HITL) dispatch pipeline.

---

## System Overview

* **Transformer Backbone:** IndicBERTv2 (12 encoder layers, hidden size $H=768$, 278M parameters) with a 250k WordPiece tokenization vocabulary.
* **Quantization & Efficiency:** Post-training INT8 quantization (`indicbertv2_binary_int8.pt`, 292 MB) delivering an average inference latency of 14.2 ms per comment.
* **Shift Alignment & Calibration:** Corrects prior probability divergence between training data ($P_{\text{src}} = 62.34\%$) and live streaming distributions ($P_{\text{tgt}} \approx 20.59\%$) with an empirical decision boundary calibrated at $\tau = 0.98$.
* **Three-Tier HITL Dispatch:**
  * **Auto-Quarantine ($P \ge 0.98$):** Automatic message suppression.
  * **HITL Review ($0.70 \le P < 0.98$):** Routed to moderation console with SHAP token-level attribution overlays.
  * **Auto-Approved ($P < 0.70$):** Immediate live stream release.
* **Operational Performance:** Saves 71.8% of human moderation time while attaining a System Usability Scale (SUS) benchmark of 82.4/100.

---

## Repository Structure

```text
kannaguard/
├── backend/                  # API server, ingestion engine, and inference endpoints
├── dashboard/                # Moderator console with SHAP visual overlays
├── docs/                     # Documentation and architecture assets
│   └── README.md
├── evaluation/               # Validation benchmarks and test split evaluations
├── experiments/              # Experimental validation suite
│   ├── figures/              # SHAP fairness and attribution plots
│   ├── quantized_models/     # INT8 deployment binaries
│   ├── bias_check.py         # Subgroup demographic bias auditing
│   ├── explainability_shap.py# Token-level SHAP explanation generator
│   ├── load_test.py          # Real-time stream concurrency stress testing
│   ├── model_distillation_quant.py # Quantization routines[cite: 6]
│   ├── speed_test.py         # Inference latency benchmarking[cite: 6]
│   └── sus_evaluation.py     # System Usability Scale metric computation[cite: 6]
├── extension/                # Client-side moderation integration[cite: 6, 8]
├── bias_check_results.csv    # Identity fairness audit logs[cite: 8]
└── docker-compose.yml        # Multi-container local orchestration[cite: 8]
