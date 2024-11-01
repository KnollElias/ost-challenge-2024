def get_image_paths(path: str) -> List[Path]:
    # Collect all .jpg files and sort by name
    image_paths = sorted(Path(path).resolve().rglob("*.jpg"), key=lambda x: x.name)
    return image_paths


from pathlib import Path
from typing import List

def get_image_paths(path: str) -> List[Path]:
    # Collect all .jpg files and sort by name
    image_paths = sorted(Path(path).resolve().rglob("*.jpg"), key=lambda x: x.name)
    return image_paths

# from typing import List
# from pathlib import Path

# def get_image_paths(path: str, n=None) -> [Path]:
#     if n:
#         Paths = []
#     else:
#         Paths = getJPGRecursiveAbsolute(path, "jpg")
#     print("here com epahts")
#     print(Paths)
#     print("here com epahts")
#     strings = str(Paths)
#     alphabeticalStrings = sortStringPaths(strings)
#     paths = generatePaths(alphabeticalStrings)
#     return paths
#     pass

# def getJPGRecursiveAbsolute(path, filetype):
#     allFiles = list(Path(path).resolve().rglob(f'*.{filetype}'))
#     filteredFiles = [file for file in allFiles if file.suffix.lower() == filetype]
#     return filteredFiles

# def sortStringPaths(paths):
#     return sorted(paths)

# def generatePaths(alphabeticalPaths) -> [Path]:
#     return [Path(p) for p in alphabeticalPaths]

# def get_paths_image(path: str) -> List[Path]:
#     image_paths = list(Path(path).resolve().rglob("*.jpg"))
#     image_paths.sort(key=lambda x: x.name)
    
#     return image_paths

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
