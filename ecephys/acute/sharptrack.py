import pandas as pd
import mat73
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

# TODO: mat73 should be a dependency
# TODO: BrainGlobe should be a depdendency


def _brainglobe_rgb_to_matplotlib_rgba(rgb, alpha):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)


# TODO: Document structures dataframe, especially column names.
class SHARPTrack(object):
    def __init__(self, sharptrack_file):
        self._mat = mat73.loadmat(sharptrack_file)
        self.structures = pd.read_json(self._mat["borders_table_json"])
        self.set_imec_depths()

    def set_imec_depths(self):
        max_electrode_depth_ccf = self._mat["probe_length_histo"]
        shrinkage_factor = self._mat["shrinkage_factor"]

        imec_zero_ccf = max_electrode_depth_ccf * 10
        self.structures["lowerBorder_imec"] = (
            imec_zero_ccf - self.structures["lowerBorder"]
        ) / shrinkage_factor
        self.structures["upperBorder_imec"] = (
            imec_zero_ccf - self.structures["upperBorder"]
        ) / shrinkage_factor

    @classmethod
    def get_atlas(self):
        return BrainGlobeAtlas("allen_mouse_10um", check_latest=False)

    @classmethod
    def get_atlas_colormap(self):
        alpha = 1.0
        return {
            s["acronym"]: _brainglobe_rgb_to_matplotlib_rgba(s["rgb_triplet"], alpha)
            for s in self.get_atlas().structures_list
        }
