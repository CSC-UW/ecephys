import numpy as np

CA1 = {
    "Segundo": [163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183],
    "Valentino": [162, 165, 166, 169, 170, 173, 174, 177, 178, 181, 182],
    "Doppio": [161, 162, 165, 166, 169, 170, 173, 174, 177, 178, 181],
    "Alessandro": [159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179],
    "Eugene": [193, 195, 197, 199, 201, 203, 205, 207, 209, 211],
}

superficial_ctx = {
    "Segundo": [375, 377, 379, 381, 383],
    "Valentino": [374, 377, 378, 381, 382],
    "Doppio": [374, 377, 378, 381, 382],
    "Alessandro": [375, 377, 379, 381, 383],
    "Eugene": [375, 377, 379, 381, 383],
}

LongCol = np.concatenate((np.arange(0, 382 + 1, 2), np.arange(1, 383 + 1, 2)))

full = {"Alessandro": LongCol, "Eugene": LongCol}

hippocampus = {"Alessandro": LongCol[200:288], "Eugene": LongCol[225:313]}