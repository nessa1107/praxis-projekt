# Automatische Analyse von Drohnenaufnahmen nach Naturkatastrophen

Praxisprojekt im Studiengang Informatik
an der Fakultät für Informatik und Ingenieurwissenschaften
der Technischen Hochschule Köln

Dieses Projekt und die dazugehörende Dokumentation befasst sich mit dem FloodNet-Datensatz und wie gut dieser für das Training von Modellen zur semantischen Segmentierung geeignet ist beziehungsweise welches Modell auf dem Datensatz die besten Ergebnisse liefert. Es soll ein Modell auf dem Datensatz trainiert werden, welches die Bilder von Drohnenaufnahmen semantisch segmentiert. Hierbei sollen die einzelnen Pixel einer von 10 Klassen zugeteilt werden, damit schnell erkannt werden kann wo sich überflutete Häuser und Straßen befinden und wo sich unbeschädigte Objekte befinden.

Leider konnte der FloodNet-Datensatz aus technischen Gründen nicht zum Repository hinzugefügt werden. Bitte diesen seperat herunterladen und in das Projekt einfügen. Hierzu den FloodNet-Ordner in FloodNet umbenennen und einfach in den Projektordner ziehen und in den Ordnern "train", "test" und "val" die Ordner "train-label-img", "train-org-img", "test-label-img", "test-org-img" "val-label-img" und "val-org-image" umbenennen, sodass die Ordner jeweils "label-img" und "org-img" heißen.
Es sollte dann folgende Ordner Struktur vorliegen:

![Screenshot 2024-05-31 110947](https://github.com/nessa1107/praxis-projekt/assets/85506778/b40cc1f1-c9bf-4639-8273-ab12cb62f4cd)
