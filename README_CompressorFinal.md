# CompressorFinal – Videokompression Pipeline

## Überblick

`CompressorFinal` ist die finale Version unseres Videokompressors. Er integriert alle Hauptkomponenten einer modernen Videokompressionspipeline und verarbeitet YUV-Rohvideos zu kompakten Binärdateien. Die Kompression und Dekompression werden vollständig unterstützt.

## Pipeline-Schritte

### 1. Prädiktion (Vorhersage)

Bei der Prädiktion wird für jeden Block eines Bildes versucht, dessen Werte möglichst genau aus bereits bekannten Nachbarblöcken vorherzusagen. Dazu werden verschiedene Prädiktionsmodi getestet (z.B. Mittelwert, vertikale oder horizontale Fortsetzung), und der Modus mit dem geringsten Fehler wird gewählt (`find_best_mode_and_residuals_float`). Das Ergebnis ist ein Residualblock, der die Differenz zwischen dem Originalblock und der Vorhersage enthält. Diese Residuen sind oft kleiner und weisen weniger Varianz auf als die Originaldaten, was die nachfolgende Kompression deutlich effizienter macht.

- **Theorie:** Prädiktion reduziert Redundanz, indem sie Korrelationen zwischen benachbarten Bildbereichen ausnutzt. Intra-Frame-Prädiktion ist ein zentrales Element moderner Videocodecs (z.B. H.264, HEVC).
- **Eingabe:** Block aus dem aktuellen Frame
- **Ausgabe:** Residualblock (Differenzblock) und Modus-Flag

### 2. Transformation (Diskrete Kosinustransformation, DCT)

Die diskrete Kosinustransformation (DCT) wandelt den Residualblock aus dem Ortsraum in den Frequenzraum um (`dct_block`). Dabei werden die Bilddaten als Summe von Kosinusfunktionen unterschiedlicher Frequenzen dargestellt. In natürlichen Bildern konzentriert sich die Energie meist auf wenige niederfrequente Koeffizienten, während hochfrequente Anteile (z.B. feines Rauschen) oft sehr klein sind oder ganz entfallen.

- **Theorie:** Die DCT ist eine lineare Transformation, die besonders gut für Bild- und Videokompression geeignet ist, da sie die Energie kompakt auf wenige Koeffizienten verteilt (Energiekompaktheit).
- **Eingabe:** Residualblock
- **Ausgabe:** DCT-Koeffizienten (Frequenzanteile)

### 3. Quantisierung

Die Quantisierung reduziert die Präzision der DCT-Koeffizienten, indem sie auf eine endliche Anzahl von Stufen abgebildet werden (`quantize`). Dies geschieht durch Division durch eine Schrittweite und anschließendes Runden. Die Quantisierung ist der Hauptgrund für den Datenverlust in der Pipeline, ermöglicht aber eine drastische Reduktion der Datenmenge, da viele kleine Koeffizienten auf Null fallen.

- **Theorie:** Quantisierung ist ein verlustbehafteter Schritt, der unwichtige Details entfernt und die Kompressionseffizienz erhöht. Die Wahl der Quantisierungsstärke beeinflusst direkt die Bildqualität und die Kompressionsrate.
- **Eingabe:** DCT-Koeffizienten
- **Ausgabe:** Quantisierte Werte, Minimum, Schrittweite

### 4. Block-Rekonstruktion und Ersetzung

Nach der Quantisierung werden die Werte dequantisiert und per inverser DCT zurück in den Ortsraum transformiert (`idct_block`). Anschließend wird die inverse Prädiktion angewendet (`_decode_block_float`), um den ursprünglichen Block möglichst genau zu rekonstruieren. **Wichtig:** Der rekonstruierte Block ersetzt im gepaddeten Frame den Originalblock. Dadurch basieren alle nachfolgenden Prädiktionen auf bereits komprimierten und dekomprimierten (also "verlustbehafteten") Daten, nicht auf den Originaldaten. Dies entspricht dem Vorgehen in modernen Videocodecs und verhindert Fehlerakkumulation zwischen Encoder und Decoder.

- **Theorie:** Die Ersetzung des Originalblocks durch den rekonstruierten Block stellt sicher, dass Encoder und Decoder stets mit denselben Referenzdaten arbeiten. So bleibt die Prädiktion konsistent und es entstehen keine Driftfehler.
- **Eingabe:** Quantisierte Werte, Modus-Flag, Min, Schrittweite
- **Ausgabe:** Rekonstruierter Block

### 5. Huffman-Codierung

Die quantisierten Blöcke werden blockweise mit Huffman-Codierung komprimiert (`huffman_encode_frame`). Huffman-Codierung ist ein verlustfreies, entropiebasiertes Verfahren, das häufig vorkommenden Symbolen kurze Codes und seltenen Symbolen längere Codes zuweist. Für jeden Frame und Kanal (Y, U, V) werden die codierten Blöcke, die Huffman-Tabelle, Padding-Informationen und weitere Metadaten gespeichert.

- **Theorie:** Die Huffman-Codierung nutzt die statistische Verteilung der Werte, um die durchschnittliche Bitrate zu minimieren. Sie ist optimal für bekannte Symbolwahrscheinlichkeiten und wird in vielen Bild- und Videocodecs eingesetzt.
- **Eingabe:** Quantisierte Blöcke
- **Ausgabe:** Huffman-codierte Daten, Huffman-Tabelle, Metadaten

## Ablauf der Kompression im Detail

1. **Frame wird in Y, U, V-Kanäle zerlegt**
2. **Jeder Kanal wird in Blöcke unterteilt und ggf. gepaddet**
3. **Für jeden Block:**
   - Prädiktion: Vorhersage aus Nachbarblöcken, Berechnung des Residuals
   - DCT: Transformation des Residuals in den Frequenzraum
   - Quantisierung: Reduktion der Präzision, viele Werte werden Null
   - Dequantisierung & IDCT: Rücktransformation in den Ortsraum
   - Inverse Prädiktion: Rekonstruktion des Blocks
   - **Ersetze Originalblock im Frame durch den rekonstruierten Block** (wichtig für Konsistenz der Prädiktion)
4. **Huffman-Codierung der quantisierten Blöcke**
5. **Speichern aller Metadaten und komprimierten Daten**

## Dekompression

Die Dekompression läuft in umgekehrter Reihenfolge ab: Zunächst werden die Huffman-codierten Daten dekodiert, dann die Werte dequantisiert, per inverser DCT zurücktransformiert und schließlich mit der inversen Prädiktion die Blöcke rekonstruiert. Die rekonstruierten Blöcke werden zu vollständigen YUV-Frames zusammengesetzt.

---

**Hinweis:** Die gesamte Pipeline arbeitet blockweise und stellt durch die Ersetzung des Originalblocks mit dem rekonstruierten Block sicher, dass sowohl Encoder als auch Decoder immer auf denselben (verlustbehafteten) Referenzdaten aufbauen. Dies ist entscheidend für die Effizienz und Konsistenz der Kompression und entspricht dem Vorgehen moderner Videocodecs.
