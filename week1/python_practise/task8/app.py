# why am I doing this with my life

def ask_name() -> str:
  print(f"Geben Sie Ihren Namen ein:")
  name = input()
  return name

def format_case(name: str, make_lower = False, make_higher = False) -> str:
  if make_lower:
    return name.lower()
  if make_higher:
    return name.upper()
  return name

def get_length(name: str) -> int:
  return len(name)

name = ask_name()
print(f"{format_case(name, make_higher = True)}")
print(f"{format_case(name, make_lower = True)}")
print(f"Anzahl der Buchstaben: {get_length(str(name))}")
