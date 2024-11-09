import lightning as L
import torch

def get_label(file_path: str) -> int:
    return int(file_path.split('_')[-1].split('.')[0])

test_path = "02_Homework/mnist_1k/mnist0_2.png"
assert get_label(test_path) == 2, "Fehler bei der Extraktion des Labels (2)"
test_path = "02_Homework/mnist_1k/mnist999_0.png"
assert get_label(test_path) == 0, "Fehler bei der Extraktion des Labels (0)"