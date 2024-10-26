# Python Aufgaben Teil 1

## Aufgabe 1: Variablen und Datentypen

Schreiben Sie ein Python-Programm, das folgende Schritte ausführt:

1. Erstellen Sie eine Variable `name` und weisen Sie ihr Ihren eigenen Namen als Zeichenkette zu.
1. Erstellen Sie eine Variable `alter` und setzen Sie sie auf Ihr Alter (ganze Zahl).
1. Erstellen Sie eine Variable `grösse` und setzen Sie sie auf Ihre Körpergrösse in Metern (z. B. 1.75 als Gleitkommazahl).
1. Drucken Sie eine Nachricht, die in etwa so aussieht: „Hallo, ich heisse [Name], bin [Alter] Jahre alt und [Grösse] Meter gross.“

Erwartete Ausgabe:

```
Hallo, ich heisse Max, bin 25 Jahre alt und 1.75 Meter gross.
```

## Aufgabe 2: Bedingungen (if-else)

Schreiben Sie ein Programm, das das Alter eines Benutzers überprüft und entscheidet, ob der Benutzer volljährig ist (18 Jahre oder älter).

1. Erstellen Sie eine Variable `alter` und weisen Sie ihr eine Zahl zu.
1. Verwenden Sie eine if-else-Bedingung, um zu prüfen, ob `alter` grösser oder gleich 18 ist.
1. Geben Sie „Volljährig“ aus, wenn die Bedingung wahr ist, und „Minderjährig“, wenn die Bedingung falsch ist.

Erwartete Ausgabe:

```
Volljährig
```

oder

```
Minderjährig
```

## Aufgabe 3: Schleifen (for und while)

1. Erstellen Sie eine for-Schleife, die die Zahlen von 1 bis 10 nacheinander ausgibt.
1. Erstellen Sie eine while-Schleife, die die Zahlen von 10 bis 1 rückwärts ausgibt.

Erwartete Ausgabe:

```
1
2
3
4
5
6
7
8
9
10
10
9
8
7
6
5
4
3
2
1
```

## Aufgabe 4: Listen und Schleifen

Schreiben Sie ein Programm, das eine Liste von fünf Namen enthält und diese Namen mit einer Schleife einzeln ausgibt.

1. Erstellen Sie eine Liste namens `namen` mit fünf Namen Ihrer Wahl.
1. Verwenden Sie eine for-Schleife, um jeden Namen in der Liste zu drucken.

Erwartete Ausgabe:

```
Anna
Ben
Carla
David
Eva
```

## Aufgabe 5: Funktionen

Schreiben Sie eine Funktion namens `begruessung`, die den Namen und das Alter einer Person als Argumente entgegennimmt und eine Begrüssungsnachricht ausgibt.

1. Definieren Sie die Funktion `begruessung(name, alter)`.
2. Verwenden Sie ein print, um eine Nachricht wie „Hallo [Name], du bist [Alter] Jahre alt.“ auszugeben.
3. Rufen Sie die Funktion mindestens zweimal mit verschiedenen Namen und Altersangaben auf.

Erwartete Ausgabe:

```
Hallo Anna, du bist 20 Jahre alt.
Hallo Ben, du bist 30 Jahre alt.
```

## Aufgabe 6: Einfache Berechnung mit Funktionen

Schreiben Sie eine Funktion namens `addiere`, die zwei Zahlen als Argumente entgegennimmt und deren Summe zurückgibt.

1. Definieren Sie die Funktion `addiere(zahl1, zahl2)`.
2. Die Funktion sollte die Summe von `zahl1` und `zahl2` berechnen und zurückgeben.
3. Rufen Sie die Funktion auf und geben Sie das Ergebnis aus.
   Erwartete Ausgabe:

```
Die Summe ist: 15
(Wobei die 15 das Ergebnis der Summierung ist, die Sie verwenden.)
```

## Aufgabe 7: Zahlen raten (if-else und while)

Schreiben Sie ein einfaches Zahlenspiel:

1. Erstellen Sie eine Variable `geheime_zahl` und setzen Sie sie auf eine zufällige Zahl zwischen 1 und 10.
2. Bitten Sie den Benutzer mit input um eine Zahl und speichern Sie die Eingabe in der Variable `rate`.
3. Verwenden Sie eine while-Schleife, die läuft, solange `rate` nicht gleich `geheime_zahl` ist.
4. Wenn `rate` zu hoch ist, sagen Sie „Zu hoch!“, und wenn `rate` zu niedrig ist, sagen Sie „Zu niedrig!“.
5. Wenn der Benutzer die richtige Zahl errät, sagen Sie „Herzlichen Glückwunsch!“.

Hinweis: Diese Aufgabe kann etwas mehr Zeit in Anspruch nehmen, hilft aber gut beim Verständnis von Schleifen und Bedingungen.

## Aufgabe 8: String-Manipulation

Schreiben Sie ein Programm, das:

1. Den Benutzer nach seinem Namen fragt.
2. Den Namen in Grossbuchstaben und Kleinbuchstaben ausgibt.
3. Die Anzahl der Buchstaben im Namen ausgibt (ohne Leerzeichen).

Erwartete Ausgabe:

```
Geben Sie Ihren Namen ein: Max Mustermann
Ihr Name in Grossbuchstaben: MAX MUSTERMANN
Ihr Name in Kleinbuchstaben: max mustermann
Anzahl der Buchstaben: 12
```

## Aufgabe 9: Wörterbuch durchsuchen

Schreiben Sie ein Programm, das ein Wörterbuch mit englisch-deutschen Übersetzungen durchsucht.

1. Erstellen Sie ein Dictionary namens woerterbuch mit englisch-deutschen Übersetzungen, z. B. {"apple": "Apfel", "banana": "Banane", "cherry": "Kirsche"}.
1. Bitten Sie den Benutzer, ein englisches Wort einzugeben.
1. Wenn das Wort im Wörterbuch vorhanden ist, geben Sie die deutsche Übersetzung aus.
1. Wenn das Wort nicht im Wörterbuch ist, geben Sie „Übersetzung nicht gefunden“ aus.
   Erwartete Ausgabe (Beispiele):

```
Geben Sie ein englisches Wort ein: apple
Übersetzung: Apfel

Geben Sie ein englisches Wort ein: orange
Übersetzung nicht gefunden
```

## Aufgabe 10: break, continue und Dictionaries kombinieren

Schreiben Sie ein Programm, das ein einfaches Menü erstellt, um mit einem Dictionary interaktiv zu arbeiten.

1. Erstellen Sie ein Dictionary namens freunde, das einige Namen und deren Lieblingsspeisen enthält, z. B. {"Anna": "Pizza", "Ben": "Sushi", "Clara": "Pasta"}.
1. Erstellen Sie eine while-Schleife, die das folgende Menü anzeigt:

   - 1. Lieblingsspeise anzeigen
   - 2. Lieblingsspeise hinzufügen oder ändern
   - 3. Beenden

1. Der Benutzer wählt eine Option:
   - Wenn der Benutzer „1“ wählt, bitten Sie ihn um einen Namen und geben Sie die Lieblingsspeise der Person aus.
     - Falls der Name nicht im Dictionary vorhanden ist, zeigen Sie „Freund nicht gefunden“ an.
   - Wenn der Benutzer „2“ wählt, bitten Sie ihn um einen Namen und eine Lieblingsspeise, und fügen Sie diese zum Dictionary hinzu oder ändern sie.
   - Wenn der Benutzer „3“ wählt, verwenden Sie break, um die Schleife zu beenden.

Erwartete Ausgabe (Beispiele):

```
Wählen Sie eine Option:
1. Lieblingsspeise anzeigen
2. Lieblingsspeise hinzufügen oder ändern
3. Beenden
Option: 1
Name: Anna
Lieblingsspeise: Pizza

Option: 2
Name: Max
Lieblingsspeise: Burger

Option: 3
```
