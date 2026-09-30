# Model and Component Performance

This document records currently available development / controlled benchmark results.

> **Important:** These results come from different datasets and test conditions. They must **not** be combined into a single end-to-end SignBridge accuracy value.

## 1. Medical BIM Recognizer V2

External development benchmark reported for the current Medical V2 recognizer:

| Metric | Result |
|---|---:|
| Top-1 | 13 / 30 = **43.33%** |
| Top-3 | 20 / 30 = **66.67%** |

These figures describe the Medical V2 external development benchmark only. They are not end-to-end system accuracy and are not a clinical performance claim.

The recognizer returns Top-K candidates, confidence margin, quality information, and a recognizer-level decision such as `ACCEPT`, `CONFIRM`, or `VERIFY`.

---

## 2. General BIM Recognizer V1

The General V1 recognizer supports **117 classes** and uses a 64-frame pose-and-hand landmark sequence.

A final comparable held-out performance figure has not been documented here. Do not invent a value; add the teammate's verified benchmark when available.

---

## 3. Number BIM Recognizer V1

The Number V1 recognizer is a static isolated-sign classifier for BIM numbers **0–10**.

A final comparable held-out performance figure has not been documented here. Add the verified benchmark when available.

Because it is a closed 11-class classifier, it should only be activated when conversation context expects a number.

---

## 4. Speech Recognition (Controlled Benchmark)

Controlled ASR benchmark:

| Metric | Result |
|---|---:|
| Average raw WER | **4.86%** |
| Normalized WER | **2.0%** |
| Medical-keyword accuracy | **90%** |

These results were obtained from a small controlled recording benchmark and should not be interpreted as hospital-wide or real-world clinical ASR performance.

---

## 5. MCIE AI V2 (Controlled / Development Benchmark)

| Metric | Result |
|---|---:|
| Intent | 14 / 14 = **100.0%** |
| Medical concept | 12 / 14 = **85.7%** |
| Structured field | 2 / 2 = **100.0%** |
| Overall cases | 12 / 14 = **85.7%** |

These are controlled/development MCIE behaviour results. They are **not BIM recognition accuracy** and are not an end-to-end clinical metric.

The main observed gap was medical-concept extraction/validation in two test cases.

---

## 6. Recognition Intelligence Tests

The context-aware Recognition Intelligence logic behaved as expected on all **8 controlled benchmark cases** used during development.

This means the routing/reranking/safety logic produced the expected software behaviour in those cases. It does **not** mean the sign recognizer achieved 100% accuracy.

---

## 7. End-to-End Evaluation Status

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

## 8. Reporting Rule

Use component-specific language when presenting results.

Good:

> “Medical V2 achieved 43.33% Top-1 and 66.67% Top-3 on its external development benchmark.”

Good:

> “MCIE achieved 12/14 correct cases on a controlled development benchmark.”

Avoid:

> “SignBridge is 85.7% accurate.”

The latter incorrectly mixes a language-processing benchmark with end-to-end system performance.
