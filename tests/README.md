# Tests

This directory contains software and integration tests.

Recommended categories:

```text
tests/
├── test_mcie_rules.py
├── test_mcie_ai.py
├── test_recognizer_routing.py
├── test_sentence_buffer.py
├── test_text_to_bim.py
└── test_integration_contracts.py
```

## Test Interpretation

Unit and simulated integration tests verify expected software behaviour and interface contracts.

They do **not** measure real-world BIM recognition accuracy unless the test actually runs the trained recognizer on an independently defined evaluation dataset.

Keep controlled/simulated results clearly separated from real model and end-to-end evaluation.
