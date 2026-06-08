# Comandi per l'Avvio del Progetto

Questo documento riassume i comandi principali per configurare ed eseguire il progetto di Analisi Numerica sul tuo computer.

*Nota: I comandi sono pensati per l'uso da terminale (su macOS, Linux o dal terminale integrato del tuo IDE come IntelliJ o VSCode).*

## 0. Prerequisiti e Controlli Iniziali

Se hai appena scaricato il progetto o installato un nuovo IDE, esegui questi passaggi preliminari:

1. **Apri il Terminale**: All'interno del tuo IDE (es. IntelliJ, VSCode, PyCharm), apri una finestra di Terminale (generalmente si trova nel menu `View -> Terminal` o in basso nella finestra).
2. **Verifica di essere nella cartella giusta**: Assicurati che il terminale punti alla cartella del progetto (es. `.../biological-data-regression`).
3. **Verifica l'installazione di Python**: Controlla se Python è correttamente installato digitando (su macOS/Linux):
   ```bash
   python3 --version
   ```
   Se il comando precedente ti dà errore (molto comune su Windows), prova:
   ```bash
   python --version
   ```
   *(Se il comando restituisce un errore "not found" o simile, devi scaricare e installare Python da [python.org](https://www.python.org/). Su Windows, durante l'installazione ricordati di spuntare l'opzione "Add Python to PATH" o "Add python.exe to PATH").*

---

## 1. Configurazione Iniziale (Una tantum)

Se è la prima volta che avvii il progetto, è fondamentale creare un "ambiente virtuale" per mantenere le librerie isolate dal resto del sistema.

### Crea l'ambiente virtuale:

Su macOS/Linux:
```bash
python3 -m venv .venv
```

Su Windows (o se il comando sopra dà errore):
```bash
python -m venv .venv
```

### Attiva l'ambiente virtuale:

Su macOS/Linux:
```bash
source .venv/bin/activate
```

Su Windows:
```bat
.venv\Scripts\activate
```

### Installa le librerie matematiche necessarie:

Installa Numpy, Pandas, Matplotlib, Scipy e Certifi:
```bash
pip install -r requirements.txt
```

---

## 2. Esecuzione della Pipeline Completa

Questo è il comando principale. Avvia `main.py`, che a sua volta richiama a catena le Fasi dalla 1 alla 6. Il programma stamperà a schermo calcoli e matrici, ti mostrerà i grafici in finestre pop-up, e ti chiederà un input interattivo durante la Fase 4 e la Fase 6.

Esegue l'analisi completa dall'inizio alla fine:
```bash
python main.py
```

---

## 3. Esecuzioni Modulari (Avanzate)

Se vuoi scaricare o rigenerare nuovamente il dataset reale direttamente dal database NCBI GEO (GSE38417), usa lo script di ingestion:

```bash
python src/data_ingestion.py
```

Se vuoi lanciare le singole analisi visive da Google Colab in futuro, fai riferimento alle istruzioni contenute nel file `notebooks/README_COLAB.md`.
