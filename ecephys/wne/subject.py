from pathlib import Path

import yaml


class Subject:
    def __init__(self, subjectYamlFile: Path):
        self.name = subjectYamlFile.stem
        self.doc = Subject.load_yaml_doc(subjectYamlFile)

    @staticmethod
    def load_yaml_doc(yaml_path: Path) -> dict:
        """Load a YAML file that contains only one document."""
        with open(yaml_path) as fp:
            yaml_doc = yaml.safe_load(fp)
        return yaml_doc
