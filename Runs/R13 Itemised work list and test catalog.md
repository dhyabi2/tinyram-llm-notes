---
id: R13
type: run
status: done
links: [R10, R12]
date: 2026-10-28
---
# Itemised work list and test catalog

Catalog/WORKLIST.csv + .md: 177 work items, one row each (150 review findings individually, 27 open vault items). Catalog/TESTLIST.csv: 3,267 test cases listed individually with input, expected result and what each needs to run: 3,159 local (no AI, no network), 68 need a host run (VPS, Colab, sudo purge), 40 need the hosted model. Implemented and passing now: 636 (suite T01, cache simulators). Regenerate with python3 Catalog/make_catalog.py. Plan: MASTER_PLAN.md.
- see [[R10 Playbook review of 22 methods]]
- see [[R12 Twelve separate cluster sessions]]
