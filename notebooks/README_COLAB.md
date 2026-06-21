# Istruzioni per Google Colab

Questo file spiega come eseguire l'intera pipeline del progetto su **Google Colab**, sfruttando il notebook pronto all'uso fornito nella cartella `notebooks/`.

---

## Metodo Rapido: Notebook Pronto

Nella cartella `notebooks/` trovi il file **`colab_pipeline.ipynb`**, un notebook completo che esegue tutte le 6 fasi della pipeline. Per usarlo:

1. Carica l'intera cartella `biological-data-regression` nel tuo **Google Drive**.
2. Apri Google Colab ([colab.research.google.com](https://colab.research.google.com)).
3. Da Colab, vai su `File → Apri notebook → Google Drive` e seleziona il file `notebooks/colab_pipeline.ipynb`.
4. Esegui le celle in ordine dall'alto verso il basso.

> **⚠️ Importante:** Nella prima cella del notebook c'è la variabile `PROJECT_DIR`. Se hai posizionato la cartella del progetto in una sottocartella del tuo Drive diversa dalla radice, modifica quel path di conseguenza.

---

## Struttura del Notebook

Il notebook è diviso nelle seguenti sezioni:

| Sezione | Cosa fa |
|---|---|
| **0 — Setup** | Monta Google Drive, configura `sys.path`, installa le dipendenze |
| **Fase 1** | Analisi 0-dimensionale (statistiche, box plot, ECDF vs Gaussiana) |
| **Fase 2** | Regressione lineare semplice y = ax + b |
| **Fase 3** | Train/Test split (90/10) e validazione |
| **Fase 4** | Interpolazione ed estrapolazione (con input interattivo) |
| **Fase 5** | Modello multilineare y = X·θ |
| **Fase 6** | Pulizia outlier dinamica (con loop interattivo) |

---

## Note sulle Fasi Interattive (Fase 4 e 6)

Le Fasi 4 e 6 usano `input()` per chiedere dati all'utente. Su Google Colab, `input()` funziona nativamente: apparirà un campo di testo nella cella in cui potrai digitare la tua risposta.

- **Fase 4**: Ti verrà chiesto di inserire un valore numerico di x per testare l'interpolazione/estrapolazione.
- **Fase 6**: Ti verrà chiesto ripetutamente se vuoi rimuovere l'outlier trovato (`s` per sì, `n` per no).

---

## Metodo Manuale: Creare un Notebook da Zero

Se preferisci creare un notebook personalizzato, le celle essenziali di setup sono:

### Cella 1 — Montaggio Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cella 2 — Configurazione Path
```python
import sys
import os

project_dir = "/content/drive/MyDrive/progetto Lucia/biological-data-regression"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.chdir(project_dir)
print("Directory di lavoro impostata:", os.getcwd())
```

### Cella 3 — Import ed Esecuzione
```python
# Esempio: esecuzione Fase 1
from src.data_exploration import run_phase1
df, stats, cov = run_phase1('data/raw/ncbi_muscular_dystrophy_raw.csv', plots_dir=None)
```

La separazione netta tra la logica nel backend `src/` e il notebook frontend (come esploratore visivo) è esattamente ciò che massimizza il voto nell'architettura del software scientifico.
