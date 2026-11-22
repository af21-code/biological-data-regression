# Analisi Numerica e Modelli di Regressione per la Stima Continua del Decadimento Muscolare

## 📌 Contesto Accademico
Questo repository contiene il codice e la documentazione sviluppati per il **Progetto B: Analisi dati e Regressione - output continuo**.
Il lavoro è stato realizzato come prova d'esame per il **Corso di Analisi Numerica per l'Analisi dei Dati Biologici — LT in Biotecnologie per la salute**, tenuto dai docenti Francesco Calabrò e Luisa D'Amore.

## 🎯 Obiettivo del Progetto
L'obiettivo principale del progetto è lo sviluppo e la validazione matematica di modelli di regressione lineare e multilineare per prevedere l'andamento di parametri clinici continui. Partendo dall'esplorazione statistica dei dati biologici, il sistema implementa algoritmi di predizione valutandone la robustezza tramite metriche di errore e rigide procedure di partizionamento dei dati.

## 🧬 Tema Clinico: La Distrofia Muscolare
Il dominio applicativo scelto per l'analisi è la **Distrofia Muscolare di Duchenne (DMD)**.

**Validazione della scelta:**
La distrofia muscolare si presta perfettamente ai requisiti analitici del progetto per due ragioni fondamentali:
1. **Natura Continua dell'Output:** La progressione della patologia è quantificabile attraverso biomarcatori e parametri clinici strettamente continui (es. livelli di espressione del gene della distrofina, punteggi di forza muscolare). Questo soddisfa pienamente il vincolo progettuale che richiede l'analisi e la stima di un output continuo tramite la costruzione di modelli $y=ax+b$.
2. **Potenziale Predittivo e Applicativo:** La modellazione matematica di questi parametri non è un puro esercizio numerico, ma risponde all'esigenza biotecnologica di monitorare il decadimento muscolare. Stimare la progressione della malattia tramite modelli di regressione permette di valutare l'efficacia di potenziali terapie in fase di trial o di anticipare criticità cliniche.

## 🗂️ Dataset
Il progetto utilizza dati reali di espressione genica scaricati programmaticamente dal database internazionale **NCBI GEO** (Gene Expression Omnibus):

| Proprietà | Valore |
|---|---|
| **Accession** | GSE38417 |
| **Titolo** | *Gene expression profiling of Duchenne muscular dystrophy* |
| **Piattaforma** | GPL570 — Affymetrix Human Genome U133 Plus 2.0 Array |
| **Campioni** | 22 biopsie muscolari (pazienti DMD + controlli sani) |
| **Target (y)** | Espressione del gene DMD (distrofina) — variabile continua |
| **Feature (X)** | 10 geni con correlazione massima al target |

Lo script `src/data_ingestion.py` gestisce il download, il parsing e la selezione automatica delle feature.

## 🛠️ Pipeline dell'Analisi

Il codice è strutturato seguendo una rigorosa pipeline numerica, in risposta alle direttive del corso:

| Fase | Modulo | Descrizione |
|---|---|---|
| **1** | `data_exploration.py` | Analisi 0-dimensionale: statistiche descrittive, intervalli al 90%, covarianza, ECDF vs Gaussiana |
| **2** | `linear_regression.py` | Selezione feature + costruzione algebrica della retta $y = ax + b$ |
| **3** | `linear_regression.py` | Train/Test split (90/10) e validazione oggettiva |
| **4** | `linear_regression.py` | Interpolazione, estrapolazione e prompt interattivo |
| **5** | `multivariate.py` | Modello multilineare $y = X\theta$ con Equazioni Normali |
| **6** | `linear_regression.py` | Pulizia outlier dinamica con loop interattivo |

Tutti i calcoli centrali (regressione, covarianza, metriche) sono implementati **da zero** tramite algebra matriciale con `numpy`, senza l'uso di `scikit-learn` o funzioni preconfezionate.

## 📂 Struttura del Progetto

```
biological-data-regression/
├── main.py                     ← entry point: esegue tutte le fasi in sequenza
├── requirements.txt            ← numpy, pandas, matplotlib, scipy, certifi
│
├── src/                        ← cuore algoritmico del progetto
│   ├── __init__.py
│   ├── data_ingestion.py       ← download e parsing del dataset NCBI GEO
│   ├── data_exploration.py     ← Fase 1: statistiche, covarianza, plot
│   ├── linear_regression.py    ← Fase 2/3/4/6: regressione, split, outlier
│   ├── multivariate.py         ← Fase 5: modello multilineare
│   ├── metrics.py              ← MSE, RMSE, R², residui
│   └── plot_utils.py           ← utilità condivise per i grafici
│
├── notebooks/                  ← notebook per Google Colab
│   ├── colab_pipeline.ipynb    ← pipeline completa pronta per Colab
│   └── README_COLAB.md         ← istruzioni per l'uso su Colab
│
├── data/
│   ├── raw/                    ← dataset originale (non versionato)
│   └── processed/plots/        ← grafici generati dalla pipeline
│
├── docs/                       ← documentazione e spiegazione dei grafici
│   ├── SPIEGAZIONE_GRAFICI.md
│   └── phases/                 ← descrizione dettagliata di ogni fase
│
├── tests/                      ← test unitari
└── RUN_INSTRUCTIONS.md         ← guida passo-passo all'esecuzione
```

## 💻 Requisiti e Utilizzo

Il progetto è compatibile con **macOS, Linux e Windows**. Richiede **Python 3.10+**.

### Installazione

1. Clona il repository e posizionati nella cartella principale.
2. Crea e attiva un ambiente virtuale:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
   ```
3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

### Esecuzione

L'intero progetto (Fasi 1-6) è centralizzato in `main.py`:

```bash
python main.py
```

> **Nota:** Durante l'esecuzione ti verrà chiesto di interagire in due fasi:
> - **Fase 4**: Inserire un valore di test per valutarne l'interpolazione o estrapolazione.
> - **Fase 6**: Un loop iterativo ti permetterà di valutare e rimuovere eventuali outlier individuati dal modello.

Per scaricare nuovamente il dataset reale da NCBI GEO:
```bash
python src/data_ingestion.py
```

### Esecuzione su Google Colab

Il progetto può essere eseguito interamente su Google Colab. Il notebook pronto all'uso si trova in `notebooks/colab_pipeline.ipynb`. Per le istruzioni dettagliate, consulta `notebooks/README_COLAB.md`.
## ⚠️ Limiti e Sviluppi

- **Selezione delle feature prima dello split.** I 10 probe predittori vengono scelti per correlazione con *y* su tutti i 22 campioni, prima della divisione train/test. Le metriche sul test sono quindi ottimistiche: la selezione andrebbe ripetuta solo sui dati di training (ad esempio dentro una leave-one-out).
- **Probe dello stesso gene.** La piattaforma GPL570 ha più probe per il gene DMD; oggi viene escluso solo il probe target, per cui tra le feature possono comparire altri probe della distrofina, quasi duplicati di *y*. Escluderli renderebbe il problema più realistico.
- **Pochi campioni.** Con 22 campioni e uno split 90/10 il test set contiene 2–3 osservazioni: l'R² sul test ha un'incertezza molto alta. Con così pochi dati è più adatta una validazione leave-one-out.
- **Molte variabili, pochi campioni.** Scegliere 10 probe tra decine di migliaia su 22 campioni favorisce correlazioni spurie: i risultati vanno letti come esercizio numerico, non come evidenza biologica.
