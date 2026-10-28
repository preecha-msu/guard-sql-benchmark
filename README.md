# GUARD-SQL: Mitigating Adversarial SQL Subquery and Conditional Traps Across 22 Large Language Models via Neuro-Symbolic AST Guardrails and Plan-Guided Self-Correction

**Official Benchmark & Reproducibility Repository for the *IEEE Access* Manuscript (ID: `Access-2026-55112`)**  
**Author:** Preecha Noiumkar (Member, IEEE)  
**Affiliation:** Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (`preecha.n@msu.ac.th`)

---

## 1. Overview
This repository provides the complete enterprise database schemas, the 105-query bilingual adversarial Text-to-SQL benchmark (English V1-V3 and Thai V4-V5 across 7 subquery and conditional trap categories C1-C7), Gold SQL queries, per-query execution logs across all 22 evaluated LLMs ($N = 2,310$ paired evaluations per mode), multi-baseline component ablation logs, 200-query external validation logs, anonymized human practitioner reference baselines, and automated statistical verification scripts reported in the manuscript.

### Key Empirical Findings Verified in This Repository
* **Chain-of-Thought (CoT) Degradation:** Unconstrained zero-shot CoT prompting degrades execution accuracy in **16 out of 22 evaluated LLMs** compared with direct zero-shot prompting (lowering the 22-model macro-average EX from **28.4%** to **25.6%**, $p = 0.027$).
* **Neuro-Symbolic Recovery:** GUARD-SQL elevates the 22-model macro-average execution accuracy from **25.6%** to **72.1%** (**+46.5 percentage points**, 95% seed-cluster bootstrap CI: $[66.6\%, 77.5\%]$, gain CI: $[+40.6\%, +52.0\%]$, 21-seed cluster Wilcoxon $p = 5.94 \times 10^{-5}$) and Valid Efficiency Score (VES) from **19.2** to **59.5**.
* **Frontier Model Parity with Senior DBAs:** Enables frontier flagship models (**Anthropic Claude Sonnet 5.5** and **Google Gemini 4 Argon**) to achieve **100.0% (105/105)** execution accuracy across all seven trap categories (C1-C7).
* **Zero False-Positive Corruption:** Achieves a **0.0% (0/177)** end-to-end false-positive corruption rate on standard analytical queries from BIRD-Dev while repairing 100% (23/23) of targeted dialect/nullable trap queries.

---

## 2. Repository Structure & Manuscript Mapping

| File / Directory | Manuscript Reference | Description |
| :--- | :--- | :--- |
| `verify_paper_statistics.py` | All Tables (Tables 2-7) | Automated verification script that recomputes all paper statistics directly from the CSV logs. |
| `results_105_all22models_combined.csv` | Tables 2, 3, 4, 6, 7 | Combined 2,310-row execution log across all 22 LLMs x 105 queries, including per-query EX, VES, SQLite execution times (`ms`), and LLM call counts. |
| `raw_logs_22_models/results_105_*.csv` | Tables 2, 3, 4 | Individual 105-query execution logs for each of the 22 evaluated LLMs. |
| `summary_all_22_models.csv` | Table 3 | Model-wise EX (%), raw counts, VES, discordant pairs ($b/c$), McNemar's $\chi^2$, and Wilcoxon $Z$. |
| `summary_category_breakdown_22.csv` | Table 4 | Granular category-wise (C1-C7) execution accuracy breakdown across all 22 LLMs. |
| `results_table5_ablation_all22models.csv` | Table 5 & Section VI-C | Full 22-model component ablation study across all 10 configurations (220 model-method pairs). |
| `results_table5_ablation_6models.csv` | Table 5 & Section VI-C | Representative 6-model tier subset ablation summary table. |
| `results_external_validation_200.csv` | Table 5 & Section VI-C | 200-query external validation log (177 standard BIRD-Dev analytical queries + 23 targeted trap queries). |
| `results_human_reference_105.csv` | Table 2 & Section V-A | Anonymized per-query execution logs for Junior/Mid-level SQL Analysts (70.5% EX) and Senior DBAs (99.0% EX). |
| `human_sample_gold_21.csv` | Section III & V-A | Canonical 21 seed queries across the 3 enterprise schemas and 7 trap categories (C1-C7). |
| `benchmark_105.json` | Section III | Complete 105-query bilingual benchmark specification (English V1-V3 and Thai V4-V5) and Gold SQL. |
| `ast_guardrail.py` | Section IV-B | Reference implementation of the Stage 2 zero-token SQLGlot AST normalizer (Rules R1-R3). |
| `figures/` | Figures 1-4 | High-resolution architecture diagram and empirical evaluation figures used in the manuscript. |

---

## 3. Quick Verification
To reproduce and verify all statistical metrics reported in the manuscript, run:

    pip install numpy pandas scipy
    python verify_paper_statistics.py

---

## 4. Funding & Acknowledgment
This research project was financially supported by the **Faculty of Informatics, Mahasarakham University**.
