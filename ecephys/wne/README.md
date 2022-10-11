# Wisconsin Neurophysiology Environment (WNE)

The sysstem used as the Wisconsin Institute for Sleep and Consciousenss to organize neurophysiology projects and data. Similar in many respects to [ONE](https://int-brain-lab.github.io/ONE/one_reference.html). It would be nice to eventually migrate to ONE, but even then some of the funtionality in WNE would need to be ported/replicated.

## Guidelines
- The contents of this subpackage should not depend on the specific subjects, experiments, datapaths, etc. present at WISC. That means there should be no dependencies on on `wisc_ecephys_tools` (to be renamed `wisc_internal` or `wisc_private`).