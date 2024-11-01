import json
from pathlib import Path

def read_angle_speed(path: Path) -> (float, float):
    with path.open() as file:
        data = json.load(file)
        angle = float(data["angle"])
        speed = float(data["speed"])
    return angle, speed

# Tests

# Arrange
import tempfile
p = Path(tempfile.gettempdir()) / "test1-2"; p.mkdir(exist_ok=True)
json_path = p.resolve() / "data.json"
json_path.write_text('{"angle": 0.5, "speed": 5}')

# assert return is tuple
assert type(read_angle_speed(json_path)) == tuple, "Die Funktion sollte ein Tuple zurückgeben"
assert all(type(x) == float for x in read_angle_speed(json_path)), "Der Tuple sollte zwei floats enthalten"
assert read_angle_speed(json_path) == (0.5, 5), "Die Funktion sollte die korrekten Werte zurückgeben"

# Cleanup
import shutil; shutil.rmtree(p)

# Dev test
# pathString = r"C:\Users\KnollElias\Documents\GitHub\ost-challenge-2024\week1\Aufgabe 1.2\files.json"
# pathPath = Path(pathString)
# getPath = read_angle_speed(pathPath)
# print(getPath)