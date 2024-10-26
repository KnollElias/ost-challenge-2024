orders = {
  "Anna": "Pizza",
  "Ben": "Sushi",
  "Clara": "Pasta"
}

def get_display_options():
  return int(input("Option: "))

def display_order(key):
  if key in orders:
    print(f"Lieblingsspeise von {key}: {orders[key]}")
  else:
    print(f"Freund nicht gefunden: {key}")

def display_orders():
  for order in orders:
    print(order)

def add_order(key, order):
  orders[key] = order

print("Wählen Sie eine Option:")
print("1. Lieblingsspeise anzeigen")
print("2. Lieblingsspeise hinzufügen oder ändern")
print("3. Beenden")

while True:
  option = get_display_options()
  match option:
    case 1:
      name = input("Name: ")
      display_order(name)
      continue
    case 2:
      name = input("Wer: ")
      food = input("Lieblingsspeise: ")
      add_order(name, food)
      continue
    case 3:
      print("Programm wird beendet.")
      break