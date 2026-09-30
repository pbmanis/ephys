"""
get_reader.py -- data-source dispatch for the ephys package.

Sniffs a data path and returns a *reader instance* whose public API mirrors
``acq4_reader`` (the contract used by ``tools/data_summary.py`` and
``ephys_analysis/analysis_common.py``: ``setProtocol``, ``setDataName``,
``readDirIndex``, ``subDirs``, ``getDataInfo``, ``parseClampHoldingLevel``,
``getData`` ...).  This lets tools that currently hard-code
``acq4_reader.acq4_reader()`` consume other acquisition formats
(ephus / XSG, Axon ABF, matdatac) without changing their call flow.

Design goals
------------
* **Additive and non-breaking.** acq4 paths always route to ``acq4_reader``
  exactly as before.  Importing this module does nothing to existing readers.
* **Honest about gaps.** Formats whose reader does not yet implement the full
  acq4 API (matdatac, abf, datac) are still *detected and dispatched*, but a
  clear, actionable error is raised if the concrete reader class is missing,
  rather than failing obscurely deep in a caller.

Detection precedence (a real dataset is a single format, so this only matters
for stray files): ephus (.xsg) > acq4 (.index/.ma) > abf (.abf) > matdatac
(.mat, excluding ephus's ``lsps_result.mat``).  The raw ``datac`` format has no
reliable filename signature and is intentionally *not* auto-detected yet
(deferred); pass ``fmt="datac"`` explicitly when its reader is ready.

Usage
-----
    from ephys.datareaders.get_reader import get_reader, sniff_format

    fmt = sniff_format(experiment_root)          # "acq4" | "ephus" | ...
    reader = get_reader(experiment_root)         # ready-to-use reader instance
    reader.setProtocol(protocol_path)
    reader.getData()

An explicit override is available for ambiguous or headerless data:
    reader = get_reader(path, fmt="matdatac")
"""

import struct
from collections import deque
from pathlib import Path
from typing import Optional, Union

__all__ = ["sniff_format", "get_reader", "KNOWN_FORMATS"]

KNOWN_FORMATS = ("acq4", "ephus", "abf", "matdatac", "datac")


def _classify_suffix(name: str) -> str:
    """Classify a single filename by suffix/name. Returns a format or 'unknown'."""
    lname = name.lower()
    suf = Path(lname).suffix
    if suf == ".xsg":
        return "ephus"
    if suf == ".abf":
        return "abf"
    if suf == ".ma":
        return "acq4"
    if lname == ".index":
        return "acq4"
    if suf == ".mat":
        # ephus stores its old analysis output as lsps_result.mat; everything
        # else that is a bare .mat here is treated as a matdatac file.
        return "ephus" if lname == "lsps_result.mat" else "matdatac"
    return "unknown"


def _is_datac(path: Union[str, Path]) -> bool:
    """Best-effort *content* sniff for the raw DATAC binary format.

    DATAC files use arbitrary extensions (the recordist's initials, e.g. .PBM,
    .HWF, .JSR), so the suffix is useless.  Validate the 128-byte file header:
    nr_points (uint16 @0), mode (int8 @128), channel count, and that the file
    size is consistent with an integer number of fixed-size records.  Used only
    as a last resort so it never overrides acq4/ephus/abf/matdatac detection.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return False
        size = p.stat().st_size
        FH = 128  # datac file-header length
        if size < FH:
            return False
        with open(p, "rb") as fh:
            head = fh.read(max(FH + 4, 132))
        nr_points = struct.unpack_from("H", head, 0)[0]
        if nr_points <= 0 or nr_points > 1_000_000:
            return False
        if size == FH:
            # A header-only datac file (aborted/empty recording: valid 128-byte
            # header, no data records). Validate that the comment region is
            # text/NUL rather than binary, to avoid false positives. Recognized so
            # it is reported as an (empty) datac file rather than "unknown".
            comment = head[29:FH]
            printable = sum(1 for c in comment if c == 0 or 32 <= c < 127)
            return printable >= int(len(comment) * 0.8)
        if len(head) < FH + 1:
            return False
        mode = struct.unpack_from("b", head, FH)[0]
        if mode < 0 or mode > 20:
            return False
        if mode < 2:
            nch = mode + 2
        else:
            if len(head) < FH + 4:
                return False
            nch = struct.unpack_from("H", head, FH + 2)[0]
        if nch < 1 or nch > 8:
            return False
        datawidth = 4 if mode == 9 else 2
        rec_bytes = nr_points * nch * datawidth + 256  # + recordheaderlen
        nrec = (size - FH) // rec_bytes
        return nrec >= 1
    except Exception:  # noqa: BLE001 - any parse failure -> not datac
        return False


def sniff_format(path: Union[str, Path], maxdepth: int = 4) -> str:
    """
    Inspect ``path`` (a file or a directory tree) and return the data format:
    one of ``KNOWN_FORMATS`` or ``"unknown"``.

    A breadth-first scan is used so that shallow, decisive markers win quickly;
    ``maxdepth`` bounds how far below a directory we look before giving up.
    A ``.xsg`` anywhere is decisive for ephus; ``.index``/``.ma`` at the
    shallowest level containing them is decisive for acq4.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"sniff_format: path does not exist: {p}")

    if p.is_file():
        fmt = _classify_suffix(p.name)
        if fmt == "unknown" and _is_datac(p):
            return "datac"
        return fmt

    seen_abf = False
    seen_mat = False
    seen_lsps = False  # lsps_result.mat: a (weaker-than-.xsg) ephus signal
    q: deque = deque([(p, 0)])
    while q:
        d, depth = q.popleft()
        try:
            entries = list(d.iterdir())
        except (OSError, PermissionError):
            continue
        subdirs = []
        seen_index = seen_ma = False
        for e in entries:
            if e.is_dir():
                subdirs.append(e)
                continue
            fmt = _classify_suffix(e.name)
            if fmt == "ephus":
                if e.suffix.lower() == ".xsg":
                    return "ephus"  # decisive at any depth
                seen_lsps = True  # lsps_result.mat
            if fmt == "acq4":
                if e.name.lower() == ".index":
                    seen_index = True
                else:
                    seen_ma = True
            elif fmt == "abf":
                seen_abf = True
            elif fmt == "matdatac":
                seen_mat = True
        # acq4 markers at this (shallowest reached) level are decisive; an .xsg
        # would already have returned above if present at this depth or shallower.
        if seen_index or seen_ma:
            return "acq4"
        if depth < maxdepth:
            for sd in subdirs:
                q.append((sd, depth + 1))

    if seen_lsps:
        return "ephus"
    if seen_abf:
        return "abf"
    if seen_mat:
        return "matdatac"
    # last resort: raw DATAC files have arbitrary extensions -> content-sniff a
    # bounded sample of the directory's top-level files.
    try:
        for i, f in enumerate(sorted(p.iterdir())):
            if i >= 100:
                break
            if f.is_file() and _is_datac(f):
                return "datac"
    except (OSError, PermissionError):
        pass
    return "unknown"


def get_reader(
    path: Union[str, Path, None] = None,
    dataname: Optional[str] = None,
    defaults: Optional[dict] = None,
    fmt: Optional[str] = None,
    **kwargs,
):
    """
    Return a reader instance for ``path``, chosen by sniffing the data format
    (or forced with ``fmt``).

    Parameters
    ----------
    path : str | Path | None
        A file or directory of the dataset. Required unless ``fmt`` is given
        *and* the reader can be constructed without a path.
    dataname : str | None
        Passed through to readers that accept it (e.g. acq4's
        ``"MultiClamp1.ma"``).
    defaults : dict | None
        Passed through to the ephus reader (config-driven LPF/holding defaults).
    fmt : str | None
        Force a format from ``KNOWN_FORMATS`` instead of sniffing.

    Returns
    -------
    A reader instance duck-typed to the acq4_reader API.

    Raises
    ------
    ValueError
        If the format cannot be determined or is not one of ``KNOWN_FORMATS``.
    NotImplementedError
        If the format is recognized but its reader is not yet available
        (matdatac / abf before their parity adapters land; datac).
    """
    if fmt is None:
        if path is None:
            raise ValueError(
                "get_reader: provide a path to sniff, or an explicit fmt=."
            )
        fmt = sniff_format(path)
    fmt = fmt.lower()

    if fmt == "acq4":
        from . import acq4_reader

        return acq4_reader.acq4_reader(pathtoprotocol=path, dataname=dataname)

    if fmt == "ephus":
        from . import ephus_reader

        return ephus_reader.ephus_reader(
            pathtoprotocol=path, dataname=dataname, defaults=defaults
        )

    if fmt == "abf":
        from . import abf_reader

        cls = getattr(abf_reader, "abf_reader", None)
        if cls is None:
            raise NotImplementedError(
                "abf format detected, but abf_reader.abf_reader is not available "
                "yet (the abf reader class has not been built). "
                f"Path: {path!r}"
            )
        return cls(pathtoprotocol=path, dataname=dataname)

    if fmt == "matdatac":
        from . import matdatac_reader

        cls = getattr(matdatac_reader, "matdatac_reader", None)
        if cls is None:
            raise NotImplementedError(
                "matdatac format detected, but matdatac_reader.matdatac_reader "
                "(the path-based, acq4-API wrapper) is not available yet. "
                f"Path: {path!r}"
            )
        return cls(pathtoprotocol=path, dataname=dataname)

    if fmt == "datac":
        from . import datac_reader

        cls = getattr(datac_reader, "datac_reader", None)
        if cls is None:
            raise NotImplementedError(
                "datac format detected, but datac_reader.datac_reader (the "
                f"acq4-API adapter) is not available. Path: {path!r}"
            )
        return cls(pathtoprotocol=path, dataname=dataname, defaults=defaults)

    raise ValueError(
        f"get_reader: could not determine a supported data format for {path!r} "
        f"(sniffed {fmt!r}). Known formats: {KNOWN_FORMATS}."
    )


if __name__ == "__main__":
    # Read-only helper: print the sniffed format for one or more paths.
    import sys

    for arg in sys.argv[1:]:
        try:
            print(f"{sniff_format(arg):>10s}  {arg}")
        except Exception as exc:  # noqa: BLE001 - CLI convenience
            print(f"{'ERROR':>10s}  {arg}  ({exc})")
