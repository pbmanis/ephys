"""Configuration and path resolution for the cochlear_nucleus_alignment tool.

Precedence (later wins): DEFAULT_CONFIG < packaged cfg < user cfg < explicit
file. The packaged cfg lives in ephys/config/cochlear_nucleus_alignment.cfg;
remembered preferences are written to ~/.config/cn_alignment/cn_alignment.cfg.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Optional, Union

from pyqtgraph import configfile

from .util import cprint

PACKAGED_CFG_NAME = "cochlear_nucleus_alignment.cfg"

DEFAULT_CONFIG: dict = {
    # --- Muniak et al. (2013) CN atlas (ParaView / AVS-UCD) --------------
    "MuniakAtlasRoots": [
        "/Volumes/Pegasus_004/Tamalpais2/Desktop/Muniak_etal_2013 CNmodel ParaView Files",
        "/Volumes/Manis26_001/Lytle_M2/Tamalpais2/Desktop/Muniak_etal_2013 CNmodel ParaView Files",
        "~/Desktop/Muniak_etal_2013 CNmodel ParaView Files",
    ],
    # --- Kim / Chon (2019) Unified Mouse Brain Atlas (NIfTI) -------------
    "KimAtlasRoots": [
        "/Volumes/Pegasus_004/25750983",
        "~/Desktop/25750983",
    ],
    "KimLabelFile": "UnifiedAtlas_Label_v2_20um-isotropic.nii",
    "KimTemplateFile": "UnifiedAtlas_template_coronal_20um-isotropic.nii",
    "KimOntologyFile": "UnifiedAtlas_Label_ontology_v2.csv",
    "KimVoxelSize": 20.0,  # micrometres / voxel (isotropic)
    # --- shared ----------------------------------------------------------
    "CacheDir": "~/.cache/cn_alignment",
    "DefaultAtlas": "Muniak",
    "DefaultFrame": "Original",
    "DefaultColorMode": "Tonotopic",
    "SurfaceOpacity": 0.35,
}


def ephys_config_dir() -> Path:
    """Return the packaged ephys/config directory (independent of cwd)."""
    import ephys

    return Path(inspect.getfile(ephys)).parent / "config"


def user_config_path() -> Path:
    """User-writable config for overrides and remembered preferences."""
    return Path.home() / ".config" / "cn_alignment" / "cn_alignment.cfg"


def get_configuration(configfile_name: Union[str, Path, None] = None) -> dict:
    """Merge built-in defaults with the packaged cfg and the user cfg."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    candidates = [
        ephys_config_dir() / PACKAGED_CFG_NAME,
        user_config_path(),
    ]
    if configfile_name is not None:
        candidates.append(Path(configfile_name))
    for cpath in candidates:
        try:
            if cpath is not None and Path(cpath).is_file():
                loaded = configfile.readConfigFile(str(cpath))
                cfg.update(loaded)
                cprint("c", f"cn_alignment: read config {cpath!s}")
        except Exception as exc:  # noqa: BLE001 - fall through on bad cfg
            cprint("y", f"cn_alignment: could not read {cpath!s}: {exc}")
    # Back-compat: an older cfg stored the Muniak roots under "AtlasRoots".
    if "AtlasRoots" in cfg and "MuniakAtlasRoots" not in cfg:
        cfg["MuniakAtlasRoots"] = cfg["AtlasRoots"]
    return cfg


def save_user_config(prefs: dict) -> None:
    """Persist remembered preferences to the user config file."""
    path = user_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        configfile.writeConfigFile(prefs, str(path))
    except Exception as exc:  # noqa: BLE001
        cprint("y", f"cn_alignment: could not write {path!s}: {exc}")


def resolve_root(cfg: dict, key: str) -> Optional[Path]:
    """Return the first configured path under ``key`` that exists on disk."""
    for cand in cfg.get(key, []):
        p = Path(cand).expanduser()
        if p.is_dir():
            return p
    return None
