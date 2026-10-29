---
id: F34
type: fix
status: done
links: [R13, R16, T05]
date: 2026-10-29
---
# Schema 3 converter and contract test

Evaluation item O-005 (wrapper, auditor and schema shared no contract). tools/to_schema3.py converts a probe result + parsed phase timing into a schema-3 record; tools/results_check.py judges it. Without phase timing the record is marked legacy and check() always returns legacy_no_phase, so it cannot pass as clean. New rule phase_exceeds_wall. Integration test (tests/test_to_schema3.py, 32 checks): PHASE text -> summary -> schema 3 -> checker returns no flags; the real VPS, Colab and Mac 27B results convert (33 counted tokens, read per token = total / 33) and are flagged legacy, not cached. Committed 4d6153c. The probe scripts still write schema 2; switching them to emit phase timing and v3 is the remaining step.
- see [[R13 Itemised work list and test catalog]]
- see [[R16 Screen of the top 22 brainstorm idea families]]
- see [[T05 preflight.sh]]
