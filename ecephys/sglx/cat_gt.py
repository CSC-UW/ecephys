import subprocess
import numpy as np

SYNC_CHANNEL = 384  # Applied to all


CATGT_PARAMS_MANDATORY_KEYS = [
    'gblcar', 'gfix', 'ap', 'lf'
]  # 'gtlist', 'prb' are added programmatically


CATGT_PARAMS_RESERVED_KEYS = [
    'gtlist', 'g', 't', 'prb', 
]


def run_catgt(raw_files, catgt_params, catgt_path, src_dir, tgt_dir, dry_run=True):
    """Run CatGT>=2.4.
    
    Args:
        raw_files (pd.DataFrame): Frame as returned from sglx.session_org_utils.get_files.
            Should be in chronological order and contain a single run & probe.
        catgt_params (dict): CatGT params dict. The following keys are programmatically derived::
            ['gtlist', 'run', 'SY']
        catgt_path (str or pathlib.Path): Path to CatGT's `runit.sh`

    """
    catgt_params = catgt_params.copy()

    # Expected params
    assert all([k in catgt_params.keys() for k in CATGT_PARAMS_MANDATORY_KEYS]), \
        f'Missing key in {catgt_params.keys()}. The following are mandatory: {CATGT_PARAMS_MANDATORY_KEYS}'
    assert all([k not in catgt_params.keys() for k in CATGT_PARAMS_RESERVED_KEYS]), \
        f'Unexpected key in {catgt_params.keys()}. The following are reserved: {CATGT_PARAMS_RESERVED_KEYS}'

    assert len(raw_files)  # Files were found

    assert len(set(raw_files.probe)) == 1  # Single probe
    probe_i = raw_files.probe.values[0].split('imec')[1]

    assert len(set(raw_files.run)) == 1  # Single run
    run_id = raw_files.run.values[0]

    assert len(set(raw_files.acqApLfSy)) == 1 # TODO Properly check SYNC channel
    assert raw_files.acqApLfSy.values[0] == '384,384,1'
    sync_channel = SYNC_CHANNEL
    assert sync_channel == 384

    assert catgt_params['ap']
    assert not catgt_params['lf']

    gtlist = parse_gtlist(raw_files.gate.values, raw_files.trigger.values)
    catgt_params['gtlist'] = gtlist

    cmd = get_catGT_command(
        catGT_path=catgt_path,
        dir=str(src_dir),
        dest=str(tgt_dir),
        run=run_id,
        prb=str(probe_i),
        SY=f"{probe_i},{sync_channel},6,500",
        prb_fld=True,
        out_prb_fld=True,
        **catgt_params,
    )

    print(f"Running {cmd}")
    if dry_run:
        print("Dry run: doing nothing")
        return
    else:
        tgt_dir.mkdir(parents=True, exist_ok=True)
        _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return


def parse_gtlist(g_list, t_list):
    """Return {ga,tstart_a,tend_a}{...}{gn,tstart_end,tend_end}"""

    assert len(g_list) == len(t_list)

    g_i_array = np.array([
        gstring.split('g')[1]
        for gstring in g_list
    ], dtype=int) # eg [0, 1]
    t_i_array = np.array([
        tstring.split('t')[1]
        for tstring in t_list
    ], dtype=int)  # eg [1, 2, 0, 1]

    gtlist = ''
    unique_g_idx = sorted(np.unique(g_i_array))

    for g in unique_g_idx:
        g_t_idx = sorted(t_i_array[np.where(g_i_array == g)[0]])
        assert len(g_t_idx) == len(set(g_t_idx))  # no repeats
        t_start = g_t_idx[0]
        t_end = g_t_idx[-1]
        assert len(g_t_idx) == len(range(t_start, t_end + 1))  # no gaps
        gtlist += '{' + f'g{g},t{t_start},t{t_end}' + '}'
    
    return gtlist


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
    gtlist: formats as str
        For CatGT>=2.5, this option overrides the -g= and -t= options so that you can specify a separate t-range for each g-index. Specify the list like this::
            -gtlist={g0,t0a,t0b}{g1,t1a,t1b}... (include the curly braces).
        if specified, -g and -t should NOT be kwargs.
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

    # CatGT >= 2.4 accepts gtlist option
    assert (
        'gtlist' in kwargs.keys()
    ) or (
        'g' in kwargs.keys() and 't' in kwargs.keys()
    )
    if 'gtlist' in kwargs.keys():
        cmd_parts.append(f"-gtlist={kwargs.pop('gtlist')}")
    else:
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
