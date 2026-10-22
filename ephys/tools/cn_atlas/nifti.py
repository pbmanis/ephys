"""Minimal, zero-dependency NIfTI-1 reader (enough for the Kim atlas).

Handles the single-file ``.nii`` form (optionally gzipped ``.nii.gz``) with the
standard 348-byte header. Returns the voxel array (memory-mapped when possible)
plus the header fields we care about. We do not apply scl_slope/inter, since the
atlas we read is an integer label volume.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Union

import numpy as np

# NIfTI datatype code -> numpy dtype (base, unqualified by byte order).
_DTYPE = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
    256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def read_nifti(path: Union[str, Path], memmap: bool = True) -> dict:
    """Read a NIfTI-1 volume.

    Returns a dict with ``data`` (ndarray, index order [i,j,k,...]), ``dim``,
    ``pixdim`` (voxel sizes), ``affine`` (4x4; identity if none stored), and
    ``datatype``.
    """
    path = Path(path)
    gz = path.suffix == ".gz"

    # Read the 352-byte header region (first-file layout) to parse fields.
    if gz:
        with gzip.open(path, "rb") as fh:
            hdr = fh.read(352)
    else:
        with open(path, "rb") as fh:
            hdr = fh.read(352)

    # Endianness: sizeof_hdr must be 348.
    bo = "<"
    if struct.unpack_from("<i", hdr, 0)[0] != 348:
        bo = ">"
        if struct.unpack_from(">i", hdr, 0)[0] != 348:
            raise ValueError(f"{path.name}: not a NIfTI-1 header (sizeof_hdr != 348)")

    dim = struct.unpack_from(bo + "8h", hdr, 40)
    datatype = struct.unpack_from(bo + "h", hdr, 70)[0]
    pixdim = struct.unpack_from(bo + "8f", hdr, 76)
    vox_offset = int(struct.unpack_from(bo + "f", hdr, 108)[0])
    sform_code = struct.unpack_from(bo + "h", hdr, 254)[0]

    ndim = dim[0]
    shape = tuple(int(d) for d in dim[1 : 1 + ndim])
    if datatype not in _DTYPE:
        raise ValueError(f"{path.name}: unsupported NIfTI datatype {datatype}")
    dtype = np.dtype(bo + _DTYPE[datatype])

    # NIfTI voxel data is column-major (first axis fastest) -> order="F".
    if gz:
        with gzip.open(path, "rb") as fh:
            fh.seek(vox_offset)
            buf = fh.read()
        data = np.frombuffer(buf, dtype=dtype).reshape(shape, order="F")
    elif memmap:
        data = np.memmap(path, dtype=dtype, mode="r", offset=vox_offset, shape=shape, order="F")
    else:
        data = np.fromfile(path, dtype=dtype, offset=vox_offset).reshape(shape, order="F")

    # Spatial affine (sform preferred). Zero codes -> no transform stored.
    affine = np.eye(4)
    if sform_code > 0:
        affine[0, :] = struct.unpack_from(bo + "4f", hdr, 280)
        affine[1, :] = struct.unpack_from(bo + "4f", hdr, 292)
        affine[2, :] = struct.unpack_from(bo + "4f", hdr, 304)

    return {
        "data": data,
        "dim": shape,
        "pixdim": tuple(float(p) for p in pixdim[1 : 1 + ndim]),
        "affine": affine,
        "datatype": datatype,
        "sform_code": sform_code,
    }
