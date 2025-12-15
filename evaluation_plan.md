### `Evaluation Framework` Plan

**1. Problem Statement**
We need a general-purpose experiment framework for running and comparing ML experiments across tasks and configurations.

**2. Solution Approach**
Implement the design specified in `evaluation_design.md`. Use nbdev workflow (notebook → exported Python).

**3. Key Considerations, Sharp Edges & Nasty Bugs**
- Calling convention magic needs careful signature inspection
- Config name generation must handle edge cases (empty dict, non-primitives)
- Exception serialization for saving/loading
- Intermediate saves for crash recovery

**4. Decision Log**
- All design decisions documented in `evaluation_design.md`

**5. Implementation Plan**

1. [ ] **Data Model** — Core dataclasses
   a. `Evaluation` dataclass
   b. `TaskResult` dataclass  
   c. `ExperimentResults` dataclass with slicing, aggregation, save/load
   d. `config_name()` function

2. [ ] **Protocols** — Interfaces
   a. `Task` protocol
   b. `Evaluator` protocol

3. [ ] **Experiment Class** — Orchestration
   a. `Experiment.__init__` with all parameters
   b. Calling convention detection
   c. `Experiment.run()` with save logic
   d. `run_experiment()` wrapper function

4. [ ] **Utility Functions** — Analysis helpers
   a. Basic plotting functions (success rate bar chart)
   b. Export functions (to_csv)

5. [ ] **Tests** — Validation
   a. Test calling convention detection
   b. Test slicing API
   c. Test save/load round-trip

6. [ ] **Final Review** — Cleanup and polish

**6. References**
- `evaluation_design.md` — Full design specification
- `Runespawn/runespawn/evaluation/` — Original implementation

