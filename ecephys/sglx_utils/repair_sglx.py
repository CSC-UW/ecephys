# -*- coding: utf-8 -*-
"""
Repair SGLX metadata_3B2 file following SGLX crash.

Usage:

```python
repair_meta(binpath, dry_run=True)
```
"""

import re
import hashlib
import shutil
import sys
import warnings

from pathlib import Path


__author__ = 'Tom Bugnon <tombugnon@hotmail.com>'
__license__ = 'Public Domain, 2021'


def repair_meta(binpath, dry_run=True, backup_original_meta=True,
                check_SHA1_integrity=False, preview_metadata=True):
    """Add missing fields to SGLX metadata 3B2 files following SGLX crash.

    The following fields may be missing from file following SGLX crash:
        - fileSizeBytes
        - fileTimeSecs
        - firstSample
        - fileSHA1

    NB: Repaired metadata file is saved WITH end-of-line carriage returns and
        may differ from original file depending on the system it was originally
        saved on.

    Args:
        binpath: Path to SGLX bin file

    Kwargs:
        dry_run (bool): Set to False for writing to file (default True)
        backup_original_meta (bool): If true and `dry_run == False`, original
            metadata file is backed up. (default True)
        check_SHA1_integrity (bool): If true and `fileSHA1` exists in metadata,
            we recompute it and compare values (default False)
        preview_metadata (bool): If True, print repaired metadata string to
            stdout.
    """

    # File info
    binpath = Path(binpath)
    metaName = binpath.stem + ".meta"
    metaPath = Path(binpath.parent / metaName)
    meta = readMeta_noparse(metaPath)

    # fileSizeBytes
    print("Check `fileSizeBytes` field.")
    assert not binpath.is_symlink()
    if 'fileSizeBytes' not in meta:
        meta['fileSizeBytes'] = str(binpath.stat().st_size)
        print(f"Derived missing `fileSizeBytes` value: {meta['fileSizeBytes']}")
    fileSizeBytes = int(meta['fileSizeBytes'])
    assert fileSizeBytes == binpath.stat().st_size

    # Check filesize consistency and possibly trim the bin
    nChan = int(meta['nSavedChans'])
    if not fileSizeBytes/2 % nChan == 0:
        msg = 'Inconsistent number of samples across channels! '
        trim_bin = False
        if not trim_bin:
            raise ValueError(
                f"{msg} and trim_bin = False. Set `trim_bin = True` and run"
                " again to trim the raw data (destructive operation)"
            )
        else:
            trim_bin_file(binpath, nChan, bytesPerSamp=2)
            fileSizeBytes = binpath.stat().st_size
    assert fileSizeBytes / 2 % nChan == 0

    # fileTimeSecs
    print("Check `fileTimeSecs` field.")
    sRate = SampRate(meta)
    nFileSamp = int(fileSizeBytes/(2*nChan))  # /2 because int16
    if 'fileTimeSecs' not in meta:
        meta['fileTimeSecs'] = str(nFileSamp / sRate)
        print(f"Derived missing `fileTimeSecs` value: {meta['fileTimeSecs']}")
    assert float(meta['fileTimeSecs']) == nFileSamp / sRate

    # 'firstSample'
    print("Check `firstSample` field.")
    derived_firstSample = derive_first_sample(metaPath)
    if 'firstSample' not in meta:
        if derived_firstSample is not None:
            print(f"Derived missing `firstSample` value: {derived_firstSample}")
            meta['firstSample'] = str(derived_firstSample)
    assert (derived_firstSample is None
            or int(meta['firstSample']) == derived_firstSample)

    # SHA1 hash
    print("Check `fileSHA1` field.", end=" ", flush=True)
    if 'fileSHA1' not in meta:
        print("Missing `fileSHA1` field. Computing... ", end="", flush=True)
        fileSHA1 = get_sha1_hexdigest(binpath).upper()
        print(f"SHA1 hash = {fileSHA1}")
        meta['fileSHA1'] = fileSHA1
    else:
        if check_SHA1_integrity:
            print(f"Checking SHA1 hash integrity...", end=" ", flush=True)
            assert get_sha1_hexdigest(binpath).upper() == meta['fileSHA1']
            print(f"Ok!")
        else:
            print("Not checking SHA1 hash integrity.")

    metadata_string = ""
    for key in sorted(meta.keys()):
        metadata_string += f"{key}={meta[key]}\r\n"

    if preview_metadata:
        print(f'Repaired metadata full string: \n\n"""\n{metadata_string}"""\n')

    if dry_run:
        print("Set `dry_run=False` for overwriting to file`. Copy of original"
              "meta will be saved if `backup_original_meta=True`")
        return

    if backup_original_meta:
        bak_path = metaPath.parent / (metaPath.name + '.bak')
        if bak_path.exists():
            raise FileExistsError(f"Attempting to override existing backup at {bak_path}. Please delete or move this file manually beforehand.")
        print(f"Copying original meta file to {bak_path}")
        shutil.copyfile(metaPath, bak_path)

    print(f"Writing to {metaPath}. If files differ, check for carriage returns.")
    with open(metaPath, 'w') as f:
        f.write(metadata_string)


def get_sha1_hexdigest(binpath):

    BUF_SIZE = 32768 # Read file in 32kb chunks
    sha1 = hashlib.sha1()
    with open(binpath, 'rb') as f:

        while True:
           data = f.read(BUF_SIZE)
           if not data:
              break
           sha1.update(data)
        return sha1.hexdigest()


def derive_first_sample(metapath):
    """Derive the first sample of a trigger metadata from previous trigger."""

    previous_metapath = previous_trigger_metapath(metapath)
    if not previous_metapath.exists():
        warnings.warn(
            "Could not find previous trigger meta file. Ignoring `firstSample`"
            " field"
        )
        return None

    # Previous file endSample
    meta = readMeta_noparse(previous_metapath)
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    startSample = int(meta['firstSample'])

    return startSample + nFileSamp


def previous_trigger_metapath(metapath):

    # Derive previous trigger's meta path
    match = re.match(
        f'\A(.*)_g([0-9]+)_t(.*).imec([0-9]+)(.*).meta\Z',
        str(metapath),
    )
    if match is None:
        raise ValueError("Unrecognize format for metadata path `{metapath}`")
    trig_i = match.groups()[2]
    assert trig_i.isdigit()
    previous_metapath = (
        f"{match.groups()[0]}_g{match.groups()[1]}"
        f"_t{int(trig_i) - 1}"
        f".imec{match.groups()[3]}{match.groups()[4]}.meta"
    )  # Sorry for that :)

    return Path(previous_metapath)


def trim_bin_file(binpath, nChan, bytesPerSamp):
    assert False  # TODO


def readMeta_noparse(metaPath):
    """Read metadata

    No key and value parsing (unlike the original SGLX ReadMeta func)."""
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                currKey = csList[0]
                metaDict[currKey] = csList[1]
    else:
        print("no meta file")
    return(metaDict)


def SampRate(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    else:
        srate = float(meta['niSampRate'])
    return(srate)
