# PantryIQ — project instructions

**Before working, read `docs/PantryIQ_Master_Prompt.md` (the Agent Operating Brief) and
`docs/PantryIQ_Design_Doc.md`.** The brief holds the locked tech decisions (§4), phase gates
(§6), and success metrics to protect (§7). Follow it. Do **not** re-open the §4 locked
decisions without an explicit new decision from the user.

**Working style:** tight step-by-step — propose the next small step, wait for approval, and
show verification after each step. Follow the Karpathy Principles: think before coding,
simplicity first, surgical changes, goal-driven execution.

**At the end of every phase, update `docs/PROJECT_JOURNAL.md`** — the plain-language record of
the whole project. Add a section in the same shape as the others: goal, what was built,
decisions and why, what was measured, what turned out to be wrong. Keep the language simple
enough to read without knowing the codebase, and update the "where the project stands" numbers.

**Toolchain:** uv + Python 3.11. `make setup` / `make test` / `make lint`. Secrets live in
`.env` (gitignored); never commit keys or datasets.
