---
id: F03
type: fix
status: done
links: [P05,P06,T01]
date: 2026-10-28
---
# download fallback and exact size

No hf CLI on a fresh box: curl resumable fallback. After download the size is compared with the server's content-length; mismatch aborts.
