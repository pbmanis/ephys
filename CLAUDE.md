# ephys — Claude Code context


## Project overview
Python package (v0.8.x, requires Python 3.13) for analysis of patch-clamp
electrophysiology data recorded with acq4. Not a general-use library — it is
actively developed for a single lab's workflow. Configuration-file-driven: most
free parameters (paths, cell-type lists, selection thresholds, plot parameters)
live in a project-specific TOML/YAML experiment config, not in the code.

## Package layout
```
ephys/
  ephys_analysis/   # raw signal analysis: IV curves, spikes, Rm/tau, VClamp
  tools/            # data assembly, comparison, display, export utilities
  gui/              # datatable GUI (PyQt)
  plotters/         # figure generation (matplotlib)
```

Key files that interact frequently:
| File | Role |
|---|---|
| `gui/data_table_functions.py` | Loads assembled pkl data; builds per-cell DataFrames |
| `tools/assemble_datasets.py` | Combines per-protocol results into per-cell rows |
| `tools/show_assembled_datafile.py` | `apply_select_by`, `populate_columns`, `get_best_and_mean` |
| `plotters/plot_spike_info.py` | `export_r` — writes CSV for downstream R stats |
| `tools/compare_past_analyses.py` | Diffs two exported CSVs to detect regressions |


## Coding conventions
- Never execute git commands.
- Never delete a file or directory.
- Always ask before making any change to a source file, and make changes incrementally (ask for every change, even if they are related).
- When fixing a bug, comment out the original line(s) rather than deleting,
  and add a comment `# Claude fixed YYYY-MM-DD: <reason>` on the replacement.
- `CP("color", message)` is the project's coloured print utility.
- `SAD` is the alias for `show_assembled_datafile` used in plotters/GUI.
- Experiment configuration is accessed via `self.experiment` (a dict);
  thresholds like `maximum_access_resistance` come from there.



## Testing / running
- No automated test suite covers the data pipeline end-to-end.
- A subset of functions/routines are tested in the main "test.py"
- `uv` is used for dependency management (`uv.lock`, `pyproject.toml`).
- Python 3.13 is required (pinned in `pyproject.toml`).
- 
## Data pipeline (current-clamp spike analysis)
1. **Assembly** (`assemble_datasets.py` → `combine_by_cell`): per-protocol
   results are combined into a single row per cell. Many measures are stored
   as **Python lists, one entry per valid protocol** (e.g. `Rs = [8.5, 12.0]`).

2. **Valid protocols** (`valid_prots`): the filtered subset of IV protocols
   that passed exclusion checks, name matching, and data loading. The
   `"protocols"` key in the assembled dict must align index-for-index with
   `Rs`, `srs`, `CNeut`, etc. Misalignment here is the root cause of wrong
   protocol selection downstream.

3. **Selection** (`apply_select_by` in `show_assembled_datafile.py`):
   iterates over `valid_prots`, uses the selector column (usually `"Rs"`) to
   find the protocol with the **lowest** value within `select_limits`, then
   copies the corresponding per-protocol value for every other measure into
   `<measure>_bestRs`. Uses `selector_values` (NaN outside limits) not the raw
   selector list. Equal-minimum ties keep all tied protocols.

4. **Export** (`export_r` in `plot_spike_info.py`): selects `_bestRs` columns,
   renames `Rs_bestRs → Rs` for compatibility with downstream R scripts, applies
   per-column rounding/scaling (CNeut: ×1e12 to pF), and writes CSV. Bracket
   notation for lists is stripped from the CSV with a raw string replace.

5. **Comparison** (`compare_past_analyses.py`): reads two CSVs and diffs
   `_bestRs` columns. Normalises `Rs → Rs_bestRs` at the start of each
   cell comparison to handle old/new CSV format differences.

## Key invariants
- **`valid_prots` alignment**: `row["protocols"]`, `row["Rs"]`, `row["srs"]`,
  `row["CNeut"]` must all be parallel lists of the same length. `"protocols"`
  must be `valid_prots`, not the full `df_cell.IV` key list.
- **`_bestRs` columns are scalars**; the raw measure columns (e.g. `Rs`) are
  lists. Never average the list — use the `_bestRs` scalar.
- **`selector_values`** in `apply_select_by` is a NaN-masked copy of the
  selector list (values outside `select_limits` set to NaN). Use this, not
  `row[select_by][i]`, when tracking the running minimum.
- **CNeut units**: stored internally in Farads; exported to CSV in pF (×1e12).



## Common pitfalls
- **Pandas SettingWithCopyWarning**: always assign to a new variable or
  use `.copy()` when modifying a filtered DataFrame row.
- **`float(series)`**: if a DataFrame has duplicate column names, indexing a
  Series with that label returns a Series, not a scalar. `compare_past_analyses`
  normalises column names up-front to prevent this.
- **CSV bracket stripping**: the `replace(",[", ",").replace("],", ",")` pass
  is a raw-string replace and ignores CSV quoting. List columns must be excluded
  from the export (select scalars only) to avoid corrupting column alignment.
- **`np.allclose` with None/NA**: wrap values with a `_to_numeric` helper
  (converts None/NA → `np.nan`) before passing to `np.allclose`.
