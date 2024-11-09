from pathlib import Path

image_directory_path = Path("../")

if not image_directory_path.exists():
    raise Exception(f"Fehler🛑 Der Pfad {image_directory_path} existiert nicht")
else:
    print(f"Alles gut👍 der Pfad {image_directory_path} wurde gefunden")