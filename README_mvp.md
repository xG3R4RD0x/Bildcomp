# mvp.py - CLI Tool für Videokompression/Dekompression

# Systemanforderungen

Das Program `mvp.py` erfordert Python3.
Die Abhängigkeit `numba` ist optional, aber wir empfehlen die Installation für eine bessere Performance.
Zum installieren mit pip muss lediglich folgendes Kommando ausgeführt werden:

```sh
python3 -m pip install numba
```

# Ausführen

Das Program kann mit `python3 mvp.py` ausgeführt werden.
Wenn keine weiteren Argumente mitgegeben werden, dann liefert dass Programm eine nützliche Hilfe, indem es alle verfügbaren Sub-Kommandos und Optionen auflistet:

```
  Available Subcommands:
    compress   <input.yuv> <output.vid>   Compress the raw video data from the specified input.yuv file and write the resulting video to the specified output.vid file. Try to extract the metadata from the filename itself. 
    decompress <input.vid> <output.yuv>   Read the compressed video data from the specified input.vid file. Decompress the video and write the resulting raw video data to the specified output.yuv file.

  Available Options:
    --quantization-interval <float>       Only available for the compress subcommand. Specify the desired quantization interval used for compression. We recommend a value between 1.0 and 15.0.
    --help                                Show this help and quit.
```

## Sub-Kommando: compress

Mit `python3 mvp.py compress <input.yuv> <output.vid>` kann man die rohe Video-Datei `<input.yuv>` in die komprimierte `<output.vid>` umwandeln.
Damit das Programm die korrekte Breite und Höhe eines Frames kennt, wird ein bestimmtest Format für den Dateinamen von `<input.yuv>` erwartet.
Das Format für den Dateinamen ist wie folgt:

```
<name>_<width>x<height>.yuv
```

oder 

```
<name>_<width>x<height>_<fps>.yuv
```

Der Teil `<name>` wird vom Programm ignoriert.
Die Teile `<width>` und `<height>` stehen jeweils für die Breite und die Höhe eines Frames.
Der Teil `<fps>` gibt die Bilder pro Sekunde (FPS) an und ist optional.
Falls der Teil `<fps>` weggelassen wird, gehen wir von 25 Bildern pro Sekunde aus.
Die Anzahl der Bilder pro Sekunde ist zwar irrelevant für die Kompression, jedoch nutzen wir diese in der GUI für die korrekte Anzeige des Videos.

Mit Hilfe von der Option `--quantization-interval` kann man die Qualität/Dateigröße beeinflussen.
Höhere Werte für `--quantization-interval` ergben eine niederigere Qualtät und eine niedrigere Dateigröße.
Kleinere Werte ergeben eine bessere Qualtät, aber benötigen hierfür mehr Speicher.
Wir empfehlen einen Wert zwischen 1.0 und 15.0 für einen guten Ausgleich zwischen Qualität und Dateigröße.

## Sub-Kommando: decompress

Mit `python3 mvp.py decompress <input.vid> <output.yuv>` kann man die komprimierte Datei `<input.vid>` in die rohe Video-Datei `<output.yuv>` umwandeln.
Weitere Optionen für dieses Sub-Kommando gibt es nicht.

# Kompressionsalgorithmus

Unser Kompressionsalgorithmus behandelt die Chrominanzkomponenten (U und V) gleichermaßen wie die Luminanzkomponente (Y).
Eine zeitliche Dekorrelation zwischen Frames unterstützen wir nicht.
Jeder Frame wird einzeln komprimiert.
Hierfür wird zunächst jede Komponente in Blöcke unterteilt, welche aus 8x8 Werten bestehen.
Diese Blöcke werden dann in der Reihenfolge von oben-links nach unten-rechs abgearbeitet.
Zuerst wird der Block mit Hilfe einer diskreten Kosinustransformation (DCT) dekorreliert.
Anschließend werden die durch die DCT enstandenen Koeffezienten mit der Intervallbreite `--quantization-interval` quantisiert.
Dann werden die dequantisiert und die es wird eine inverse diskrete Kosinustransformation (IDCT) durchgeführt, um rekonstruierte Frames zu erhalten.
Die rekonstruierten Frames sollten für die Prädiktion verwendet werden, jedoch ist diese nicht Teil der derzeitigen Anwendung.
Nach der Quantisierung werden dann die Auftrittshäufigkeiten für die Symbol bestimmt.
Die Auftrittshäufigkeiten werden genutzt, um optimale Codewörter mit begrenzter Länge für die Symbole des Alphabets zu bestimmen.
Für die Bestimmung der Codewörter verwenden wir den sogenannten Package-Merge-Algorithmus (siehe: https://dl.acm.org/doi/10.1145/79147.79150).
Dann werden die Codewörter für die jeweiligen Symbole Bit für Bit serialisiert und in die Ausgabedatei geschrieben.
Die Serialisierung erfolgt mit Hilfe eines `Bitwriter`s, welcher dafür zuständig ist, Gruppen von Bits zu sammeln bis ein Byte geschrieben werden kann.
Erst wenn dies der Fall ist, dann wird ein Byte in die Ausgabe geschrieben.
Seiteninformationen, welche für die Dekompression benötigt werden, wurden für diese Erklärung zum besseren Verständnis vernachlässigt, sind jedoch im Code enthalten und werden auch in die finale Datei geschrieben.

Für die Dekompression der Daten werden die einzelnen Schritte in umgekehrter Reihenfolge rückgängig gemacht.