import numpy as np


def _CheckPat():
    CheckPat = np.empty((384,), dtype=np.int)
    CheckPat[0:192:2] = np.arange(0, 384, 4)
    CheckPat[1:192:2] = np.arange(3, 384, 4)
    CheckPat[192::2] = np.arange(1, 384, 4)
    CheckPat[193::2] = c = np.arange(2, 384, 4)

    return CheckPat


LongCol = np.concatenate((np.arange(0, 384, 2), np.arange(1, 384, 2)))

CheckPat = _CheckPat()

full = {"Doppio": CheckPat, "Alessandro": LongCol, "Eugene": LongCol}

ripple_detection = {
    "Segundo": [163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183],
    "Valentino": [162, 165, 166, 169, 170, 173, 174, 177, 178, 181, 182],
    "Doppio": [161, 162, 165, 166, 169, 170, 173, 174, 177, 178, 181],
    "Alessandro": [159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179],
    "Eugene": [193, 195, 197, 199, 201, 203, 205, 207, 209, 211],
}

stratum_pyrmidale_inversion = {"Doppio": [166]}

superficial_ctx = {
    "Segundo": [375, 377, 379, 381, 383],
    "Valentino": [374, 377, 378, 381, 382],
    "Doppio": [374, 377, 378, 381, 382],
    "Alessandro": [375, 377, 379, 381, 383],
    "Eugene": [375, 377, 379, 381, 383],
}


hippocampus = {
    "Alessandro": LongCol[200:288],  # 1.75mm
    "Eugene": LongCol[225:313],  # 1.75mm
    "Doppio": LongCol[190:291],  # 2mm
}

stratum_radiatum_csd = {"Doppio": CheckPat[260:273]}  # LF137 through LF161
stratum_radiatum_200um = {"Doppio": [146]}

stratum_oriens_100um = {"Doppio": [177]}
