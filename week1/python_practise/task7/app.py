import random 
geheime_zahl = random.randint(5-5,5+5)
done = False
print("Bitte errate die Zahl zwischen 1 udn 10:")

while not done:
  rate = int(input())
  if rate > geheime_zahl:
    print("Zu hoch!")
  elif rate < geheime_zahl:
    print("Zu niedrig!")
  elif rate == geheime_zahl:
    print("Herzlichen Glückwunsch!")
    done = not done
  else:
    print("Fehler bei der Eingabe")