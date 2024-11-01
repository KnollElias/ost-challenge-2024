from pathlib import Path
from typing import List

def get_image_paths(path: str) -> List[Path]:
    image_paths = sorted(Path(path).resolve().rglob("*.jpg"), key=lambda x: x.name)
    return image_paths


# Arrange
import tempfile
from pathlib import Path
p = Path(tempfile.gettempdir()) / "test1-1"
p.mkdir(exist_ok=True)
test_pats = sorted([p / f"{i}.jpg" for i in range(5)], key=lambda x: x.name)
[p.touch() for p in reversed(test_pats)]
negative = p / "negative.txt"
negative.touch()

# Assert
returned_paths = get_image_paths(p)
resolved_test_pats = [path.resolve() for path in test_pats]  # Resolve test paths

print("Returned list:", returned_paths)
print("Expected list:", resolved_test_pats)

assert type(returned_paths) == list, "Die Funktion sollte eine Liste zurückgeben."
assert len(returned_paths) == len(resolved_test_pats), "Die Funktion sollte im Test genau 5 JPG-Bilder finden"
assert all(p.is_absolute() for p in returned_paths), "Die Funktion sollte absolute Pfade zurückgeben"
assert all(a == b for a, b in zip(returned_paths, resolved_test_pats)), "Die Funktion sollte die korrekten Pfade und sortiert zurückgeben"
