import random

geheime_zahl = random.randint(1, 10)
print(geheime_zahl)
print("rate die zahl bwo")


while geheime_zahl == geheime_zahl:
    rate = int(input())
    if geheime_zahl > rate:
        print("zahl zu niedrig")
        continue
    elif geheime_zahl < rate:
        print("zahl zu hoch")
        continue
    elif geheime_zahl == rate:
        print("yippie")
        break
