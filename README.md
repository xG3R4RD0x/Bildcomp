# VidKomp

# Architektur
> [!NOTE] 
> Manchmal kann die Mermaid Syntax in VSCode nicht angezeigt werden. Dafür einfach die Extension `Markdown Preview Mermaid Support` runterladen.

Einfachshalber würde ich vorschlagen, dass der Player nur YUV und unser VID Format einlesen kann. Falls wir mehr Beipielvideos brauchen können wir ein belibiges Dateiformat in per `ffmpeg` in YUV konvertieren.

```mermaid
---

config:

  layout: dagre

  theme: redux-dark

---

flowchart TD

 subgraph Encoding["Encoding Pipeline"]

        C1["Decorrelation"]

        C2["Quantization"]

        C3["Probability Modelling"]

        C4["Entropy Coding"]

  end

 subgraph Preview["Imagesequence Preview Panel"]

        F1["Für einen direkten visuellen Vergleich können in der GUI zwei Fenster gleichzeitig angezeigt werdenb. Eventuell kann auch ein weiteres Fenster hinzugefügt werden um die PSNR darzustellen"]

        F2["Show Original Video"]

        F3["Show Compressed Video"]

        F4["Calculate PSNR"]

        F5["Display PSNR"]

        F6["Display Relevant Data (Optional)"]

  end

 subgraph Decoding["Decoding Pipeline"]

        I1["Reading Headerfile"]

        I2["Sachen die ein Decoder noch so macht"]

  end

    I1 --> I2

    C1 --> C2

    C2 --> C3

    C3 -- Symbolwahrscheinlichkeiten --> C4

    F3 --> F4

    F2 --> F4

    A["GUI"] --> Preview & B1["Load .YUV File"] & B3["Load .VID File"]

    B3 --> Decoding

    B1 -- choose compression settings --> B2["Save Video"]

    B1 --> D2["Convert YUV to RGB"]

    D2 --> F2

    F4 --> F5

    B2 --> Encoding

    C4 --> H1["Save File to .vid"]

    Decoding --> F3
```