# Benchmark protocol: paired transitions for tool-interface invariance (2026-11-18)

## Source and semantics
- **Tasks:** ToolSchemaSpace synthetic benchmark, commit `d266b7879fd840292770f69e260450866b7b0512`
  (12 domains, 168 native tools, 2,748 tasks with programmatic gold calls).
- **Environment:**
  - TSS `SimEnv` with hard control: off-schema calls are rejected; variant calls are decoded to native
    actions and executed.
  - **State** = multiset of executed native actions, which is TSS's own end-of-episode scoring state.
  - The ordered native-action sequence is tracked separately.
  - Initial state is the empty multiset (`initial_state_hash = h([])`).
  - **Limitation:** the synthetic environment has no deeper device state, so equivalence means "the same
    native operations".
- **Equivalence is verified by execution:**
  - Each response's text is parsed (Hermes `<tool_call>` JSON), executed call by call, and its executed
    state compared with the label.
  - Text or embedding similarity is never used.
  - `scripts/verify_tool_equivalence.py` re-verifies a whole dataset and exits 1 on any violation.

## Sampling
- 15 tasks per domain, seed 0 → **180 tasks**. Five candidates were dropped because fewer than 4 value
  perturbations exist.
- 1–6 gold calls per task.

## Behaviors per task
- **G:** the gold call sequence.
- **S1–S4** (similar syntax, different transition): the same calls with one enum, numeric or boolean
  argument changed.
- **D1–D3** (different syntax, different transition): one call replaced by a valid call of another tool in
  the same domain, taken from gold data.
- All control behaviors reach states distinct from G and from each other (verified).

## Regime R1: one prompt per task, the unit faithful RepExp uses
- **Catalog:** the native tools plus three equivalent alias tools per native tool, deterministically
  shuffled:
  - A: synonym name, e.g. `set_fan` → `configure_fan`;
  - AB: synonym name + renamed arguments (`room` → `location`, …);
  - AC: synonym name + arguments nested in `options`.
- **Preamble:** every response begins with the same short sentence (`PREAMBLES[0]`), except family P.
- **Group of n = 8** (RepExp `rollout.n=8`): {G, S1, S2, S3, D1, D2, D3, f(G)}, one group per surface family f.
- **Surface families** (different syntax, identical transition):
  - **A, AB, AC:** alias tools.
  - **K:** reversed JSON argument-key order. Native tool; tasks with any call of ≥ 2 arguments.
  - **O:** reversed order of the calls. Multi-call tasks only. The state is equal but the ordered
    sequence differs, so this separates "canonical action" from "state" novelty.
  - **P:** a paraphrased preamble sentence; the tool calls are identical.
- **Control family `identical`:** an exact textual duplicate (identical syntax, identical transition).
- **R1_shuf:** the same responses with the catalog shown in a different order (tool-description-order
  control).

## Regime R2: cross-schema memory (persistent novelty memory across prompts)
- **Schemas:**
  - native;
  - A: TSS `rename`, every tool given a synonym name;
  - B: argument renaming of every tool (our alias layer);
  - C: TSS `arg_lower`, arguments of tools with ≥ 2 parameters nested in `options`;
  - M: TSS `merge` with flat encoding, all tools of the domain behind one `execute` dispatcher.
- Each response is conditioned on its own schema's prompt.
- **Memory:** {G, S1–S3, D1–D3} under the native schema.
- **Test:** is G-under-schema-X (a duplicate of memory) scored less novel than S4-native (a new behavior)?

## Pair records (`results/interface_invariance/pairs.jsonl`, 6,596 rows)
- **Fields:** `task_id, initial_state_hash, goal, original_schema, transformed_schema, original_action,
  transformed_action, canonical_transition, final_state_hash, semantic_equivalence_verified, model_id,
  representation_config`, plus `family` and `control`.
- **`control`** is one of:
  - `identical_syntax_identical_transition`;
  - `different_syntax_identical_transition`;
  - `similar_syntax_different_transition`;
  - `different_syntax_different_transition`.
- `model_id` and `representation_config` are null in the benchmark file. The audit outputs record them.

## Tests (`tests/test_interface_invariance.py`, 10 tests, pass)
- execution-verified equivalence of every record;
- surface variants differ in text but not in state (except `identical`);
- controls change state;
- order permutation changes the sequence only;
- a corrupted alias call is detected (negative test);
- unknown tools are rejected;
- alias names never collide with native names;
- deterministic rebuild;
- no text leakage: a control behavior's text never equals any G variant.

## Commands
```
TSS=/path/to/ToolSchemaSpace   # at commit d266b78
python scripts/construct_paired_transitions.py --tss $TSS --per-domain 15 --seed 0
python scripts/verify_tool_equivalence.py --tss $TSS --data results/interface_invariance/benchmark.jsonl
TSS_PATH=$TSS python -m pytest -q tests/test_interface_invariance.py
```

## Result
- **Gate B: PASS.**
- 180 / 180 records verified, 0 violations; dataset hash `3ff2ce3e17e6298b`.
- Response counts:
  - R1 families: 360 each for identical, A, AB, AC, K and P; 296 for O;
  - R2: 180 per schema.
