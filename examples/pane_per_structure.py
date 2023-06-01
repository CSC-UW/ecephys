import argparse
import subprocess
from wisc_ecephys_tools.sortings import get_completed_subject_probe_structures


# Create a command line argument parser
example_text = """
example:

python pane_per_structure.py experiment alias "conda activate ecephys && python run_off_detection experiment alias "
python pane_per_structure.py --run experiment alias "conda activate ecephys && python run_off_detection experiment alias "

This will create several panes (one for each probe) and type (or execute, if the "--run" flag
is specified) in each:
```conda activate ecephys && python run_off_detection experiment alias subjectName,probe,acronym```
"""
parser = argparse.ArgumentParser(
    description=(
        f"Create one pane per completed subject/probe/acronym within "
        f"an existing tmux session and type/run `<command_prefix>+'<subj>,<prb>,<acronym>'`"
        f"in each pane.\n\n"
    ),
    epilog=example_text,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--run", action="store_true", help="If present, we run the command in each pane.")
parser.add_argument("experiment", type=str, help="Name of experiment we search sortings for")
parser.add_argument("alias", type=str, help="Name of alias we search sortings for")
parser.add_argument("command_prefix", type=str, help="Command to write in the pane. `'subj,prb,acronym'` is appended for each pane.")
args = parser.parse_args()

MAX_PANES_PER_WINDOW = 20

ACRONYMS_TO_IGNORE=[
    'CLA',
    'HY',
    'NAc',
    'NAc-c',
    'NAc-sh',
    'SN',
    'V',
    'ZI',
    'ZI-cBRF',
    'cc-ec-cing-dwn',
    'cfp',
    'eml',
    'ic-cp-lfp-py',
    'ml',
    'wmt',
]

ACRONYMS_TO_INCLUDE = None


# Read the file containing the list of values
subject_probes_structures = get_completed_subject_probe_structures(
    args.experiment,
    args.alias,
    acronyms_to_ignore=ACRONYMS_TO_IGNORE,
    acronyms_to_include=ACRONYMS_TO_INCLUDE,
)

# Add space to prefix string if there isn't any
prefix = args.command_prefix
if not prefix.endswith(" "):
    prefix += " "

for i, val in enumerate(subject_probes_structures):

    if not (i + 1) % MAX_PANES_PER_WINDOW:
        subprocess.run(f"tmux new-window", shell=True)

    # Split the current pane into a new pane and run a command in it
    subprocess.run(f"tmux split-window", shell=True)
    subprocess.run("tmux select-layout even-vertical", shell=True)

    # Format tuple to remove parenthesis/space
    suffix = str(val).replace("(", "").replace(")", "").replace(" ", "")

    if args.run:
        # Append the value to the command line and execute it
        subprocess.run(f"tmux send-keys -t {i+1} '{prefix}{suffix}' Enter", shell=True)
    else:
        # Append the value to the command line without executing it
        subprocess.run(f"tmux send-keys -t {i+1} '{prefix}{suffix}'", shell=True)