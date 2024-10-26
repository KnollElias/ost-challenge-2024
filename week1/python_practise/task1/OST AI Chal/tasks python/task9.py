woerterbuch = {"apple": "apfel", "banana": "banane", "cherry": "kirsche"}

print("gib englisch bwo")
wort = input()
if wort in woerterbuch:
    print(woerterbuch.get(wort))
else:
    print("nixda")
