# Security Policy

## Architectural Security Controls in `veritas-rl`

1. **Zero `eval()` / Zero Arbitrary Code Execution in Step-Level PRM:**
   `src/veritas_rl/ttc/prm_ollama.py` parses natural-language arithmetic equations (`lhs = rhs`) using `ast.parse(..., mode="eval")` and walks a strict whitelist of integer AST nodes (`ast.Constant`, `ast.UnaryOp(USub)`, `ast.BinOp(Add, Sub, Mult, Div, Mod)`). Function calls, attribute lookups, comprehensions, and power operators (`**` DoS vectors) are rejected by construction.
2. **Non-Hackable Symbolic RLVR Rewards:**
   `verify_symbolic_rollout` computes rewards strictly via deterministic modular evaluation over integer token IDs, eliminating neural reward-model prompt injection.
3. **Strict Request Bounds:**
   All FastAPI control-plane inputs (`GRPORequest`, `DoRARequest`, `DistributedRequest`) enforce strict Pydantic `ge`/`le` bounds on group sizes, epochs, ranks, and sequence lengths to prevent resource exhaustion.

## Reporting a Vulnerability
Please report security issues responsibly to `sendemailtonagarjuna@gmail.com`.
