# Phase One Experimental Plan

## Scope

Use Qwen only for phase one. Treat each checkpoint as a training snapshot, not just a saved artifact. Optimize for a compact, workshop-oriented pass that can scale into activation patching later.

## First Hypotheses

1. Deontological, utilitarian, and game-theoretic tuning preserve a shared strategic backbone and diverge mostly in late-stage readout.
2. Different objectives alter the timing of decision commitment even when they preserve similar earlier representations.
3. Repeated-game state information is part of the moral computation itself, not only a prompt-format nuisance variable.

## Exact Plot Order

1. Behavioral summary plots
2. Layerwise drift plots
3. Probe accuracy by layer and checkpoint
4. Logit-through-depth panels
5. Ranked prompt shortlist for patching

## Most Informative Prompt Subsets

1. High-risk Prisoner's Dilemma prompts where Nash and utilitarian targets diverge
2. Stag Hunt prompts that isolate trust and coordination without many normative ambiguities
3. Chicken prompts where utilitarian, Rawlsian, and Nash-social-welfare labels split
4. Repeated-game betrayal-history prompts
5. Native-tool versus native-notool prompts matched on objective

## Practical Tensor Policy

Save residual stream activations at the decision position for every layer. Save final-token layer states and action logits for every run. Save attention and MLP internals only for shortlisted prompts and selected layers after the drift and probe analyses identify candidate depths.

## Confound Checklist

- Compare objectives at matched behavioral stages, not just matched training steps.
- Keep prompt family balance explicit.
- Normalize action labels before measuring accuracy or logit margins.
- Validate probes with held-out prompts and transfer settings.
- Do not infer causal localization from probe accessibility alone.
