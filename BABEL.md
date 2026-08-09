# Babel W4H — nvidia-gradient-integrity-quorum

| Dimension | Choice |
|-----------|--------|
| **What** | Finite gradient vote |
| **Where** | allreduce pre-commit |
| **When** | each step |
| **Why** | C finite-check hot path |
| **How** | isfinite + isolate ranks |

**Primary:** `python` · **Companion:** `c`

Independent reference only. No employer affiliation.
