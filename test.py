from pathlib import Path

def get_image_paths(path: str, n=None) -> [Path]:
    if n:
        Paths = []
    else:
        Paths = getJPGRecursiveAbsolute(path, "png")
    strings = Paths.stringify()
    alphabeticalStrings = sortStringPaths(strings)
    paths = generatePaths(alphabeticalStrings)
    return paths
    pass

def getJPGRecursiveAbsolute(path, filetype):
    allFiles = list(Path(path).resolve().rglob(f'*.{filetype}'))
    filteredFiles = [file for file in allFiles if file.suffix.lower() == filetype]
    return filteredFiles

def sortStringPaths(paths):
    return sorted(paths)

def generatePaths(alphabeticalPaths) -> [Path]:
    return [Path(p) for p in alphabeticalPaths]