# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation
```
git clone https://github.com/CSC-UW/ecephys.git
cd ecephys
```
Then `pip install -e .` or `poetry install` (preferred)

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. kCSD-python, our spikeinterface fork, etc.)
