from pathlib import Path
from typing import List

def get_image_paths(path: str) -> List[Path]:
    image_paths = sorted(Path(path).resolve().rglob("*.jpg"), key=lambda x: x.name)
    return image_paths

# Tests
# Diese Test könnt ihr benützten, um euren Code zu prüfen. Bitte jedoch nicht ändern

# Arrange
import tempfile
p = Path(tempfile.gettempdir()) / "test1-1"; p.mkdir(exist_ok=True)
test_pats = [p / f"{i}.jpg" for i in range(5)]
[p.touch() for p in reversed(test_pats)]
negative = p / "negative.txt"; negative.touch()

# Assert
assert type(get_image_paths(p)) == list, "Die Funktion sollte eine Liste zurückgeben."
assert len(get_image_paths(p)) == len(test_pats), "Die Funktion sollte im Test genau 5 JPG-Bilder finden"
assert all(p.is_absolute() for p in get_image_paths('.')), "Die Funktion sollte absolute Pfade zurückgeben"
assert all(a==b for a,b in zip(get_image_paths(p), test_pats)), "Die Funktion sollte die korrekten Pfade und sortiert zurückgeben"

# Cleanup
import shutil; shutil.rmtree(p)