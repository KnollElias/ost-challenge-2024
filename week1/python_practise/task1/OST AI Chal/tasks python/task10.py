freunde = {"Anna": "Pizza", "Ben": "Sushi", "Clara": "Pasta"}


while True:
    print(
        "1. Lieblingsspeise anzeigen \n2. Lieblingsspeise hinzufügen oder ändern\n3. Beenden"
    )
    eingabe = input()
    if eingabe == ("1"):
        print("wie heisst du?")
        eingabe2 = input()
        if eingabe2 in freunde:
            print(f"du heisst: {eingabe2}\ndu magst: {freunde.get(eingabe2)}")
            break
        else:
            print("kein name bwo")
            break
    elif eingabe == ("2"):
        print("wie heisst du?")
        eingabe3 = input()
        if eingabe3 in freunde:
            print("welche speise bwo?")
            essen = input()
            freunde[eingabe3] = essen
            print(f"dein neues lieblingsessen ist: {freunde .get(eingabe3)}")
            break
        else:
            print("whats your fav food?")
            food = input()
            freunde[eingabe3] = food
            print(freunde)
            break
    elif eingabe == ("3"):
        break
    else:
        break
