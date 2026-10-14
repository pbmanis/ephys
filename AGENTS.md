# Agents Overview

- This file captures agent-specific instructions. Shared project conventions now live in `CONTRIBUTING.md`.
- Machine-specific adjustments can live in an untracked `AGENTS.local.md`; reference it when present.

## Testing

* ephys uses pytest-style testing; place tests in a `tests/` directory adjacent to the relevant code.

## Project Overview

EPHYS is a platform for neurophysiology analysis, primarily based on data from ACQ$, focusing on patch clamp electrophysiology, optogenetics, and related techniques. It provides tools for data analysis and management features including:

* General elecrophysiology for current clamp (spikes, firing, etc)
* Analysis of optogenetic "Laser scanning photostimulation" data
* Analysis of minature synaptic currents
* Analysis of evoked synaptic currents
* Rudimentary voltage-clamp analysis
* 2-photon imaging

## Collaboration & Workflow

* Favor small, maintainable changes; avoid redesigning large surfaces.
* Preserve existing comments unless demonstrably incorrect.

## Shared Guidance

- Workflow, testing, technology, and architecture expectations are detailed in `CONTRIBUTING.md`; follow them unless explicitly directed otherwise.
- Consult `CONTRIBUTING.md` for environment setup, configuration search paths, and common development patterns before improvising.
- Use `AGENTS.local.md` for any local overrides and note deviations in your journal when they occur.
