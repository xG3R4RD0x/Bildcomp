# CompressorFinal – Videokompression Pipeline

## Überblick

`CompressorFinal` ist die finale Version unseres Videokompressors. Er integriert alle Hauptkomponenten einer modernen Videokompressionspipeline und verarbeitet YUV-Rohvideos zu kompakten Binärdateien. Die Kompression und Dekompression werden vollständig unterstützt.

## Pipeline-Schritte

### 1. Prädiktion

Für jeden Block eines Frames wird zunächst eine Prädiktion durchgeführt. Hierbei wird der beste Prädiktionsmodus für den Block gesucht (`find_best_mode_and_residuals_float`). Das Ziel ist es, die Redundanz im Bild zu reduzieren, indem vorhersehbare Bildinhalte durch Residuen ersetzt werden.

- **Eingabe:** Block aus dem aktuellen Frame
- **Ausgabe:** Residualblock und Modus-Flag

### 2. Transformation (DCT)

Der Residualblock wird mittels diskreter Kosinustransformation (DCT) in den Frequenzbereich transformiert (`dct_block`). Dies ermöglicht eine effizientere Kompression, da viele Koeffizienten nach der Transformation nahe Null liegen.

- **Eingabe:** Residualblock
- **Ausgabe:** DCT-Koeffizienten

### 3. Quantisierung

Die DCT-Koeffizienten werden quantisiert (`quantize`). Dabei werden Werte auf eine reduzierte Anzahl von Stufen abgebildet, was zu Datenverlust, aber auch zu einer starken Reduktion der Datenmenge führt.

- **Eingabe:** DCT-Koeffizienten
- **Ausgabe:** Quantisierte Werte, Minimum, Schrittweite

### 4. Block-Rekonstruktion

Nach der Quantisierung werden die Blöcke dequantisiert und per inverser DCT zurücktransformiert (`idct_block`). Anschließend wird die inverse Prädiktion angewendet (`_decode_block_float`), um den Block zu rekonstruieren. Der rekonstruierte Block ersetzt im gepaddeten Frame den Originalblock, sodass zukünftige Prädiktionen auf bereits rekonstruierten Daten basieren.

- **Eingabe:** Quantisierte Werte, Modus-Flag, Min, Schrittweite
- **Ausgabe:** Rekonstruierter Block

### 5. Huffman-Codierung

Die quantisierten Blöcke werden blockweise Huffman-codiert (`huffman_encode_frame`). Für jeden Frame und Kanal (Y, U, V) werden die Blöcke, die Huffman-Tabelle, Padding-Informationen und weitere Metadaten gespeichert.

- **Eingabe:** Quantisierte Blöcke
- **Ausgabe:** Huffman-codierte Daten, Huffman-Tabelle, Metadaten

## Zusammenfassung des Ablaufs

1. **Frame wird in Y, U, V-Kanäle zerlegt**
2. **Jeder Kanal wird in Blöcke unterteilt und gepaddet**
3. **Für jeden Block:**
   - Prädiktion → Residualbildung
   - DCT → Frequenzbereich
   - Quantisierung → Datenreduktion
   - Dequantisierung & IDCT → Rücktransformation
   - Inverse Prädiktion → Block-Rekonstruktion
   - Ersetze Originalblock im Frame durch rekonstruierten Block
4. **Huffman-Codierung der quantisierten Blöcke**
5. **Speichern aller Metadaten und komprimierten Daten**

## Dekompression

Die Dekompression läuft in umgekehrter Reihenfolge ab: Huffman-Dekodierung, inverse Quantisierung, IDCT, inverse Prädiktion und Zusammenbau der YUV-Frames.

---

**Hinweis:** Die gesamte Pipeline ist so gestaltet, dass sie blockweise arbeitet und die Prädiktion stets auf bereits rekonstruierten Blöcken basiert, was für eine effiziente und konsistente Kompression sorgt.
