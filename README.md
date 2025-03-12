# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

Unfortuantely, this package cannot be published to PyPI, because it depends on packages not themselves available on PyPI. 
I suggest cloning and installing with `uv`. 

## Development

For an editable install into an environment managed from a sibling directory, you may want to add the following to that environment's `pyproject.toml`:

```
[tool.uv]
override-dependencies = [
  "ecephys @ file:///${PROJECT_ROOT}/../ecephys",
]

[tool.uv.sources]
ecephys = { path = "../ecephys", editable = true }
```

If you are using VSCode or Cursor, it never hurts to add the following to `.vscode/settings.json`:
```
    "python.analysis.extraPaths": [
        "./ecephys"
    ],
```


