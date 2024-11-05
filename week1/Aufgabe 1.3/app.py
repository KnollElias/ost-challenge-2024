from PIL import Image
from pathlib import Path

def open_image(path: Path, resize=None) -> Image:
    img = Image.open(path)
    if resize:
        return Image.open(path).resize(resize)
    return img

# Tests

# Arrange
import tempfile
p = Path(tempfile.gettempdir()) / "test1-3"; p.mkdir(exist_ok=True)
img_path = p / "test.jpg"
img = Image.new('RGB', (20, 20), 'white')
img.save(img_path)

# Assert
assert issubclass(type(open_image(img_path)), Image.Image) , "Die Funktion sollte ein Image-Objekt zurückgeben"
assert open_image(img_path).size == (20, 20), "Das Bild sollte nicht verändert werden wenn kein resize angegeben ist"
assert open_image(img_path, (10, 10)).size == (10, 10), "Das Bild sollte verkleinert werden wenn ein resize angegeben ist"

# Cleanup
import shutil; shutil.rmtree(p)

# Dev test
# img_path = Path(r"C:\Users\KNOLLLI\Pictures\Screenshots\test.png")
# image = open_image(img_path)
# print(f"Image size without resizing: {image.size}")
# resized_image = open_image(img_path, resize=(100, 100))
# print(f"Image size with resizing: {resized_image.size}")