# GUARD-SQL: Adversarial Text-to-SQL Benchmark & Reproducibility Repository

Official reproducibility repository for the *IEEE Access* regular research article:
**"GUARD-SQL: Mitigating Adversarial SQL Subquery and Conditional Traps Across 22 Large Language Models via Neuro-Symbolic AST Guardrails and Plan-Guided Self-Correction"**
*Author:* Preecha Noiumkar (Department of Information Technology, Faculty of Informatics, Mahasarakham University)

---

## Repository Directory Structure & Manuscript File Mapping

To facilitate independent verification of all empirical findings (Tables II-VII and Figures 1-4), all execution logs, ablation results, and verification scripts cited in the manuscript are organized as follows:

* **`raw_logs_22_models/`**
  * `results_105_*.csv`: Raw per-query execution logs across all 22 evaluated LLMs (N = 105 queries per model; 2,310 paired evaluations per mode across Zero-Shot, Zero-Shot CoT, Stage 2 AST Only, and GUARD-SQL Full Pipeline), corresponding to **Table II, Table III, Table IV, Table VI, and Table VII**.
* **`ablation_and_external_logs/`** (also mirrored in the root directory for convenience)
  * `results_table5_ablation_all22models.csv`: Complete 22-model component ablation and compute-comparable baseline evaluations (N = 2,310 evaluations per mode), corresponding to **Table V (RQ3)**.
  * `results_external_validation_200.csv`: Execution and AST transformation logs on the 200-query external validation suite (177 standard analytical queries + 23 dialect/nullable queries), confirming a `0.0%` (`0/200`) false-positive rewrite rate (**Section V-C & Table V**).
  * `results_human_reference_105.csv`: Anonymized practitioner reference logs across three Junior/Mid-level SQL analysts and two Senior DBAs (**Section IV-A & Table II**).
* **`scripts/`**
  * `verify_paper_statistics.py`: Automated statistical verification suite computing Edwards' continuity-corrected McNemar's chi2, Wilcoxon signed-rank Z, 10,000-iteration seed-cluster bootstrap 95% confidence intervals (N_seeds = 21), and Clopper-Pearson exact binomial intervals.
* **`figures/`**
  * High-resolution publication figures corresponding to **Figures 1-4** in the manuscript.

---

## Quick Statistical Verification

    python scripts/verify_paper_statistics.py

---

## Acknowledgments
This research project was financially supported by the Faculty of Informatics, Mahasarakham University.
