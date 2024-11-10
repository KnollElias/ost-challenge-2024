from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch

def get_image_paths(path: str) -> List[Path]:
    image_paths = sorted(Path(path).resolve().rglob("*.png"), key=lambda x: x.name)
    return image_paths

def get_label(file_path: Path) -> int:
    return int(str(file_path).split('_')[-1].split('.')[0])

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Convert to grayscale
        image_tensor = pil_to_tensor(image).float() / 255.0  # Scale to [0, 1] range

        label = get_label(self.image_paths[idx])
        label_tensor = torch.zeros(10, dtype=torch.float32)
        label_tensor[label] = 1.0  # Set the position of the label to 1 for one-hot encoding
        
        return image_tensor, label_tensor


# TESTS ImageDataset

# Arrange (10 Bilder erstellen)
import tempfile
p = Path(tempfile.gettempdir()) / "test2-2"; p.mkdir(exist_ok=True)
images = [Image.new('L', (28, 28), int(25.5 * i)).save(p / f"mnist9{i}9_{i}.png") for i in range(10)]
test_image_paths = get_image_paths(p)

# Act
try:
    ImageDataset(test_image_paths)
except Exception as e:
    assert False, f"Es gab einen Fehler beim Erstellen des Datasets: {e}"
test_dataset = ImageDataset(test_image_paths)
assert len(test_dataset) == 10, "Die Länge des Datasets sollte 10 sein"
assert type(test_dataset[0]) is tuple, "Es sollte ein Tuple zurückgegeben werden"
assert len(test_dataset[0]) == 2, "Das zurückgegebene Tuple sollte 2 Elemente haben"
assert type(test_dataset[0][0]) is torch.Tensor, "Das erste Element des Tuples sollte ein Tensor sein"
assert test_dataset[0][0].shape == (1, 28, 28), "Der Image Tensor sollte die Shape (1, 28, 28) haben"
assert test_dataset[0][0].max() <= 1.0, "Die Werte des Image Tensors sollten zwischen 0 und 1 liegen"
assert test_dataset[0][0].min() >= 0.0, "Die Werte des Image Tensors sollten zwischen 0 und 1 liegen"
assert test_dataset[0][0].dtype == torch.float32, "Der Image Tensor sollte den Datentyp float32 haben"
assert type(test_dataset[0][1]) is torch.Tensor, "Das zweite Element des Tuples sollte ein Tensor sein"
assert test_dataset[0][1].shape == (10,), "Der Label Tensor sollte die Shape (10,) haben"

# Cleanup
import shutil; shutil.rmtree(p)