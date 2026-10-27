# Instructions for the receiving GPU AI

Read AI_START_HERE.md, docs/SOURCE_AUDIT.md and docs/GPU_RUNBOOK.md before running.
The user has no GPU. Your job is to finish the GPU execution and return usable artifacts.
Work autonomously within the agreed compute resources; never buy compute or invent access.
Do not replace this source with a nearby 0.651 recipe: actual V25 code differs from its prose.
Keep references/upstream and src immutable; verify config/source-lock.json.
Transport/runtime fixes belong in scripts or a documented derivative, with a diff.
Changing coefficients, precision, tokenizer, pool, base model, depth or postprocessing is a new experiment.
Do not run training by default: this task consumes released adapters and fits heads at inference.
No GPU run, full input download, hidden-test score or exact environment reproduction was performed by the sender.
Never equate smoke tests, the visible test or training self-checks with validation or leaderboard success.
Do not claim completion after swallowed component exceptions, missing members or stale output reuse.
Keep secrets and large assets out of Git. Do not upload third-party weights without checking their terms.
Return the checklist in docs/DELIVERY.md and a filled deliverables/REPORT_TEMPLATE.md.
If access to a required weight is blocked, preserve evidence and mark the run blocked/incomplete.
