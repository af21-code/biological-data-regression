# Comandi per l'Avvio del Progetto

Questo documento riassume i comandi principali per configurare ed eseguire il progetto di Analisi Numerica sul tuo computer.

*Nota: I comandi sono pensati per l'uso da terminale (su macOS, Linux o dal terminale integrato del tuo IDE come IntelliJ o VSCode).*

## 1. Configurazione Iniziale (Una tantum)

Se è la prima volta che avvii il progetto, è buona norma creare un "ambiente virtuale" per mantenere le librerie pulite e isolate dal resto del sistema.

### Crea e attiva l'ambiente virtuale:
```bash
# Crea l'ambiente virtuale nella cartella ".venv"
python3 -m venv .venv

# Attiva l'ambiente su macOS/Linux:
source .venv/bin/activate

# (SE USI WINDOWS, il comando di attivazione è invece: .venv\Scripts\activate)
```

### Installa le librerie matematiche necessarie:
```bash
# Installa Numpy, Pandas, Matplotlib, Scipy e Certifi
pip install -r requirements.txt
```

---

## 2. Esecuzione della Pipeline Completa

Questo è il comando principale. Avvia `main.py`, che a sua volta richiama a catena le Fasi dalla 1 alla 6. Il programma stamperà a schermo calcoli e matrici, ti mostrerà i grafici in finestre pop-up, e ti chiederà un input interattivo durante la Fase 4 e la Fase 6.

```bash
# Esegue l'analisi completa dall'inizio alla fine
python main.py
```

---

## 3. Esecuzioni Modulari (Avanzate)

Se vuoi scaricare o rigenerare nuovamente il dataset reale direttamente dal database NCBI GEO (GSE38417), usa lo script di ingestion:

```bash
python src/data_ingestion.py
```

Se vuoi lanciare le singole analisi visive da Google Colab in futuro, fai riferimento alle istruzioni contenute nel file `notebooks/README_COLAB.md`.
