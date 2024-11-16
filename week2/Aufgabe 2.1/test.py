from pathlib import Path

image_directory_path = Path("../mnist_1k")

def get_image_paths(path: Path) -> list:
    return list(sorted(path.glob('**/*.png')))

image_paths = get_image_paths(image_directory_path)
print(f"Anzahl der Bilder: {len(image_paths)}")
assert len(image_paths) == 1_000, "Es sollten 1_000 Bilder vorhanden sein"

if not image_directory_path.exists():
    raise Exception(f"Fehler🛑 Der Pfad {image_directory_path} existiert nicht")
else:
    print(f"Alles gut👍 der Pfad {image_directory_path} wurde gefunden")
    paths = get_image_paths(image_directory_path)
    print(len(paths))
