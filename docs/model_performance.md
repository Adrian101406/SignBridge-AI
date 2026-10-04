# Model and Component Performance

This document records currently available development / controlled benchmark results.

> **Important:** These results come from different datasets and test conditions. They must **not** be combined into a single end-to-end SignBridge accuracy value.

## 1. Recognizers

The current recognizers were evaluated independently because each model performs a different classification task.

| Recognizer | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Medical BIM | 68.75% | 0.712 | 0.688 | 0.680 |
| General BIM | 88.46% | 0.890 | 0.890 | 0.880 |
| Number | 99.83% | 0.998 | 0.998 | 0.998 |

These results should **not be interpreted as direct comparisons between models**, because:

- the models contain different numbers of classes,
- they use different datasets,
- class distributions may differ,
- evaluation sets are different.

---

## Medical Recognizer Evaluation

The Medical BIM Recognizer has been evaluated under more than one testing condition.

### Original Untouched Test Set

```text
Accuracy:        68.75%
Macro Precision: 0.712
Macro Recall:    0.688
Macro F1:        0.680
```

### External Development Benchmark

```text
Top-1 Accuracy: 43.33%
Top-3 Accuracy: 66.67%
```

The lower external benchmark performance demonstrates that real-world generalization remains challenging.

This also highlights the importance of evaluating sign-language models on data collected under different conditions rather than relying only on internal test accuracy.

---

## General Recognizer Evaluation

```text
Accuracy:        88.46%
Macro Precision: 0.890
Macro Recall:    0.890
Macro F1:        0.880
```

---

## Number Recognizer Evaluation

```text
Accuracy:        99.83%
Macro Precision: 0.998
Macro Recall:    0.998
Macro F1:        0.998
```

Detailed evaluation information should be maintained in:

```text
docs/model_performance.md
```

---
## 2. Speech Recognition (Controlled Benchmark)

Controlled ASR benchmark:

| Metric | Result |
|---|---:|
| Average raw WER | **4.86%** |
| Normalized WER | **2.0%** |
| Medical-keyword accuracy | **90%** |

These results were obtained from a small controlled recording benchmark and should not be interpreted as hospital-wide or real-world clinical ASR performance.

---

## 3. MCIE AI V2 (Controlled / Development Benchmark)

| Metric | Result |
|---|---:|
| Intent | 14 / 14 = **100.0%** |
| Medical concept | 12 / 14 = **85.7%** |
| Structured field | 2 / 2 = **100.0%** |
| Overall cases | 12 / 14 = **85.7%** |

These are controlled/development MCIE behaviour results. They are **not BIM recognition accuracy** and are not an end-to-end clinical metric.

The main observed gap was medical-concept extraction/validation in two test cases.

---

## 4. Recognition Intelligence Tests

The context-aware Recognition Intelligence logic behaved as expected on all **8 controlled benchmark cases** used during development.

This means the routing/reranking/safety logic produced the expected software behaviour in those cases. It does **not** mean the sign recognizer achieved 100% accuracy.

---

## 5. End-to-End Evaluation Status

The following should remain separate from controlled component benchmarks until real integrated testing is completed:

- real webcam → recognizer → MCIE testing;
- real microphone → Whisper → MCIE testing;
- MCIE → Text-to-BIM → Unity testing;
- full Wi-Fi-off local runtime test;
- testing with multiple signers and realistic hospital conditions.

When these tests are performed, record:

- scenario;
- hardware;
- input conditions;
- expected output;
- actual output;
- latency;
- whether human confirmation was required;
- success/failure reason.

---

## 6. Reporting Rule

Use component-specific language when presenting results.

Good:

> “Medical V2 achieved 43.33% Top-1 and 66.67% Top-3 on its external development benchmark.”

Good:

> “MCIE achieved 12/14 correct cases on a controlled development benchmark.”

Avoid:

> “SignBridge is 85.7% accurate.”

The latter incorrectly mixes a language-processing benchmark with end-to-end system performance.
