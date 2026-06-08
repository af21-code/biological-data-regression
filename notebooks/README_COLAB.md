# Istruzioni Colab per l'Uso del Progetto

Questo file spiega come importare coerentemente la pipeline implementata in `src/` all'interno dei notebook di Google Colab, in modo da avere un ambiente ibrido (IDE per il motore algoritmico, Colab per le esplorazioni visive interattive o demo in fase di esame).

## 1. Caricare il progetto in Google Drive

Assicurati che l'intera cartella `biological-data-regression` sia caricata nel tuo Google Drive.

## 2. Montare Google Drive nel Notebook Colab

Crea un nuovo notebook all'interno della cartella `notebooks/` su Google Drive ed esegui questa prima cella:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Impostare il path di sistema (sys.path)

Per fare in modo che Colab riesca ad importare i moduli presenti in `src/` (come se fossimo nel terminale in locale), esegui:

```python
import sys
import os

# Cambia il path a seconda di dove hai posizionato la cartella in Drive
project_dir = "/content/drive/MyDrive/biological-data-regression"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
    
os.chdir(project_dir)
print("Directory di lavoro impostata:", os.getcwd())
```

## 4. Eseguire le analisi richiamando i moduli

A questo punto la coerenza tra il backend in `src/` e il notebook è totale. Puoi testare le singole fasi direttamente in blocchi di codice. Esempio per la Fase 1 e 2:

```python
# Fase 1
from src.data_exploration import run_phase1
df, stats, cov = run_phase1('data/raw/ncbi_muscular_dystrophy_raw.csv', plots_dir=None)

# Fase 2
from src.linear_regression import run_phase2
predittore, a, b, _ = run_phase2(df, 'target_expression', cov, plots_dir=None)

print(f"Modello lineare costruito: y = {a:.4f}x + {b:.4f}")
```

La separazione netta tra la logica nel backend `src/` e il notebook frontend (come esploratore visivo) è esattamente ciò che massimizza il voto nell'architettura del software scientifico.
