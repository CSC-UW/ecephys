def get_catGT_command(catGT_path, wine_path=None, **kwargs):
    """Build up a CatGT command of the form:
        CatGT -dir=data_dir -run=run_name -g=g -t=ta,tb <which streams> [ options ]
    See the CatGT Readme for details.

    Parameters:
    -----------
    catGT_path: formats str
        Path to the CatGT executable. Ideally the absolute path, but anything your shell can use to find CatGT is fine.
    wine_path: formats as str, optional
        Path to the wine executable, if running on Linux
    dir: formats as str
        Path to the directory containing SpikeGLX run folders. See SpikeGLX directory structure docs for details.
    run: formats as str
        The run name of the data you wish to operate on.
    g: formats as str
        The gate index of the data you wish to operate on.
    t: formats as str
        The trigger index or indices you wish to operate on. E.g. if you wish to concatenate t0 through t6, '0,6'.
    lf: bool, optional
        If true, operate on LFP data. Defaults to `False`
    ap: bool, optional
        If true, operate on AP data. Defaults to `False`
    See CatGT docs for other options.
        Any option of the form -opt=val can be passed as catGT(..., opt=val).
        Any option of the form -flag can be passed as catGT(..., -flag=True).

    Returns:
    --------
    cmd: str
        The shell command to run CatGT with the desired options.

    Examples:
    ---------
    prb = 0
    sync_channel = 384

    cmd = catGT(
        catGT_path=r"/Volumes/scratch/neuropixels/bin/CatGT/CatGT.exe",
        wine_path=r"/usr/bin/wine",
        dir=r"/Volumes/neuropixel_archive/Data/chronic/CNPIX4-Doppio/raw",
        run="3-18-2020",
        g="0",
        t="0,9",
        lf=True,
        prb=prb,
        SY=f"{prb},{sync_channel},6,500",
        prb_fld=True,
        out_prb_fld=True,
        dest="/Volumes/neuropixel/Data/CNPIX4-Doppio/",
    )

    print(cmd)
    >>> /usr/bin/wine /Volumes/scratch/neuropixels/bin/CatGT/CatGT.exe -dir=/Volumes/neuropixel_archive/Data/chronic/CNPIX4-Doppio/raw -run=3-18-2020 -g=0 -t=0,9 -lf -prb=0 -SY=0,384,6,500 -prb_fld -out_prb_fld -dest=/Volumes/neuropixel/Data/CNPIX4-Doppio/

    import subprocess
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    """
    cmd_parts = list()

    if wine_path:
        cmd_parts.append(wine_path)

    cmd_parts.append(catGT_path)

    cmd_parts.append(f"-dir={kwargs.pop('dir')}")
    cmd_parts.append(f"-run={kwargs.pop('run')}")
    cmd_parts.append(f"-g={kwargs.pop('g')}")
    cmd_parts.append(f"-t={kwargs.pop('t')}")

    for opt, val in kwargs.items():
        if type(val) == bool:
            if val:
                cmd_parts.append(f"-{opt}")
        else:
            cmd_parts.append(f"-{opt}={val}")

    cmd = " ".join(cmd_parts)

    return cmd
