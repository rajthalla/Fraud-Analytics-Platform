# Known Limitations — FinTxn RAG + Text-to-SQL Pipeline

---

## Retrieval and evaluation — known limitations

**Guardrail threshold recalibration.**
The initial out-of-scope threshold (−0.45) was derived from four manual probes during development. When the eval harness was run against the full 25-question set, two finance-adjacent out-of-scope questions — "How do I dispute a wire transfer fee?" and "What are the Basel III capital requirements?" — returned similarity scores of −0.382 and −0.414, both above the threshold, causing the guardrail to silently pass them to the LLM. The eval surfaced this because it included deliberately hard guardrail cases, not just obviously out-of-domain ones. The threshold was tightened to −0.30, which separates the full in-scope population (worst: −0.206) from the full out-of-scope population (best: −0.382) with a 0.176-point gap. Guardrail catch rate after recalibration: 4/4. This is the eval doing its job — a threshold set by eyeballing four probes is not the same as one calibrated against a labeled set.

**Train/test caveat on the threshold.**
The −0.30 threshold was tuned on the same 25-question eval set used to report the 4/4 catch rate. This means the reported guardrail number is optimistic — in production the threshold should be fit on a held-out validation set, not the test set. The current number should be read as "this threshold works on questions I explicitly designed to be hard," not as an unbiased estimate of real-world performance.

**R06 retrieval miss (`fraud_risk_score` column).**
The question "What does `fraud_risk_score` mean and where is the active risk signal?" does not retrieve the `main_marts.fct_transactions` chunk in the top-7 (it sits at rank 8+). The system partially recovers: the `ml.anomaly_scores` chunk (rank 1) contains the redirect to `anomaly_score` and the answer correctly points there. However, the specific `fraud_risk_score` column description — "Placeholder column, currently 0.0. Reserved for a future supervised model." — is missed. Root cause: the query embeds close to `ml.anomaly_scores` content (both concern fraud scores) rather than the `fct_transactions` column list. Finer chunking — one chunk per table column rather than one per table — would surface this column's dedicated description without needing k > 8.

**S06 column ambiguity — resolved correctly.**
The question "How many transactions are flagged?" returned 8,207 (correct: `ml.anomaly_scores.is_anomaly = 1`), not 16 (`is_flagged_fraud`). The schema notes in `text_to_sql.py` that explicitly label `is_flagged_fraud` as "unreliable, almost always 0" and `is_anomaly` as "MODEL-FLAGGED" successfully steered the model to the right column.

**S08 SQL failure — wrong GROUP BY logic.**
The question "How many sender accounts made more than one transaction?" returned 6,353,307 instead of 9,298. The generated SQL produced a wrong count — likely `COUNT(*)` across all rows rather than `COUNT(DISTINCT sender_id)` with a `HAVING COUNT(*) > 1` filter. Fix path: add a few-shot example or schema-level note for velocity/grouping questions that require distinct-account counting rather than row counting.

**R01/R09 RAG accuracy — LLM-as-judge strictness.**
R01 ("What is the anomaly score threshold?") and R09 ("What is the high-value threshold?") were marked INCORRECT by the judge. Spot-checking the actual answers: R01 correctly stated `anomaly_score > 0.0` but the judge penalised it for not fully explaining the Isolation Forest path-length reasoning. R09 gave the $200,000 threshold but omitted "applies independently of anomaly score." Both are partial answers — correct on the core fact, incomplete on surrounding context. R01 is a judge false-negative (answer is factually correct); R09 is a genuine partial answer. This illustrates the known weakness of LLM-as-judge: it tends toward strictness on completeness.

**R10 routing miss — borderline classification.**
"What are the possible values for the txn_type column?" was routed to SQL instead of RAG. Both answers are correct — SQL (`SELECT DISTINCT txn_type`) and RAG (data dictionary entry) both return the same five values. This was labeled RAG-only in the eval set but is genuinely an acceptable-either-way case, making the routing "miss" a labeling artefact rather than a real failure.

---

*recall@7: 0.900 | MRR: 0.833 | guardrail: 4/4 | SQL exact-match: 0.889 (8/9) | RAG accuracy: 0.700 (7/10) | routing: 0.957 (22/23)*
