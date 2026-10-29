---
id: F02
type: fix
status: done
links: [P06,T01]
date: 2026-10-28
---
# probe decimal model size

`--model-gb 12.2` crashed bash integer arithmetic. Size is rounded up with python first.
