def get_translation(input):
  if word in woerterbuch:
    print("Übersetzung:", woerterbuch[word])
  else:
    print("Übersetzung nicht gefunden")

woerterbuch = {"apple": "Apfel", "banana": "Banane", "cherry": "Kirsche"}
word = input("Geben Sie ein englisches Wort ein: ")
get_translation(input)