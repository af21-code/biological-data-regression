# MANIFESTO DI PROGETTO E LINEE GUIDA PER LO SVILUPPO (CONTEXT_MANIFEST)

## 1. Identità del Progetto e Contesto Accademico

- **Titolo:** Analisi Numerica e Modelli di Regressione per la Stima Continua del Decadimento Muscolare.
- **Corso:** Analisi Numerica per l'Analisi dei Dati Biologici — LT in Biotecnologie per la salute.
- **Docenti:** Francesco Calabrò — Luisa D'Amore.
- **Traccia di Riferimento:** Progetto B — Analisi dati e Regressione — output continuo.

---

## 2. Dominio Clinico e Obiettivo

Il progetto applica l'analisi numerica per modellare la progressione della **Distrofia Muscolare**.
L'obiettivo è sviluppare algoritmi predittivi per stimare parametri clinici continui (es. livelli sierici di specifici biomarcatori o punteggi di mobilità).
La scelta di questo dominio soddisfa il requisito fondamentale della traccia d'esame: lavorare esclusivamente sulla previsione di un **output continuo**.

---

## 3. Regole d'Ingaggio (Il "Sacro Graal" dello Sviluppo)

Per superare la revisione d'esame, il codice generato **DEVE** rispettare queste tre regole auree:

1. **Rigore Algebrico (Bottom-Up):** Vietato l'uso di "black-box" come `scikit-learn.linear_model` per i calcoli centrali. Le regressioni, le matrici di covarianza e le metriche devono essere implementate esplicitamente tramite algebra lineare (es. operazioni matriciali con `numpy`). Bisogna dimostrare di padroneggiare la matematica sottostante.

2. **Validazione Oggettiva:** Ogni modello deve essere validato calcolando esplicitamente gli scarti e le metriche di errore (MSE, R²) sul test set.

3. **Esplorazione Visiva Obbligatoria:** Ogni scelta matematica deve essere supportata da grafici (scatter plot, box plot, istogrammi).

---

## 4. Architettura Ibrida (IntelliJ IDEA + Google Colab)

Lo sviluppo è diviso in due ecosistemi cooperanti:

| Ecosistema | Cartella | Responsabilità |
|---|---|---|
| **IntelliJ IDEA** | `/src` | Motore algoritmico puro. Moduli `.py` con calcolo covarianza, classe di regressione custom, metriche di validazione. |
| **Google Colab** | `/notebooks` | Interfaccia di esplorazione e reportistica. I file `.ipynb` importano le funzioni da `/src` per eseguire l'analisi visiva e generare i plot. |

---

## 5. Struttura delle Cartelle

```
biological-data-regression/
├── CONTEXT_MANIFEST.md         ← questo file
├── README.md                   ← presentazione pubblica del repository
├── requirements.txt            ← numpy, pandas, matplotlib, scipy
│
├── data/
│   ├── raw/                    ← dataset originali (immutabili, non versionati)
│   └── processed/              ← dataset puliti dopo rimozione outlier
│
├── notebooks/                  ← file .ipynb per Google Colab e data exploration
│
├── src/                        ← cuore algoritmico del progetto
│   ├── __init__.py
│   ├── data_exploration.py     ← Fase 1: media, mediana, varianza, covarianza
│   ├── linear_regression.py    ← Fase 2/3/4: y=ax+b, train/test, interp./estrapol.
│   ├── multivariate.py         ← Fase 5: modello multilineare
│   └── metrics.py              ← scarti, MSE, RMSE, R²
│
└── docs/                       ← slide, PDF traccia, documenti di riferimento
```

> **Nota Git:** La cartella `data/` è esclusa dal versionamento via `.gitignore` per non caricare dataset pesanti su GitHub.

---

## 6. Roadmap di Sviluppo (I Checkpoint)

Il lavoro va affrontato e committato seguendo **tassativamente** questa sequenza dettata dalla traccia d'esame.

### ☐ Fase 1 — Analisi 0-dimensionale (`data_exploration.py`)
- Calcolo indicatori statistici per ogni variabile: massimo, minimo, media, mediana, varianza.
- Calcolo degli intervalli centrati nella mediana che contengono il 90% delle istanze, per ogni variabile continua.
- Plot: bar plot, box plot e funzioni di ripartizione empirica (ECDF).
- Confronto visivo della distribuzione empirica con la distribuzione Gaussiana corrispondente.

### ☐ Fase 2 — Regressione Lineare Base (`linear_regression.py`)
- Analisi della matrice di covarianza per selezionare la feature predittiva con correlazione massima all'output.
- Costruzione **algebrica** della retta di regressione $y = ax + b$ tramite minimi quadrati (operazioni matriciali `numpy`).
- Calcolo degli indicatori sintetici del modello.
- Generazione dello scatter plot con la retta sovrapposta.

### ☐ Fase 3 — Train/Test Split (`linear_regression.py`)
- Partizionamento del dataset: **90% Training**, **10% Test** (selezione casuale riproducibile con seed fisso).
- Validazione della similarità delle distribuzioni tra i due set tramite box-plot affiancati.
- Addestramento del modello **esclusivamente** sul training set.
- Valutazione predittiva sul test set tramite il calcolo esplicito degli scarti (residui).

### ☐ Fase 4 — Interpolazione ed Estrapolazione (`linear_regression.py`)
- Predizione nel punto medio delle ascisse e confronto con la media dei valori osservati vicini.
- Predizione dell'ascissa corrispondente all'ordinata media del dataset.
- Predizione dell'output per variabile predittiva = 0 (estrapolazione).
- Implementazione di un prompt interattivo: l'utente inserisce un valore di input e il punto predetto viene plottato sul grafico originale.

### ☐ Fase 5 — Modello Multilineare (`multivariate.py`)
- Sviluppo della regressione multivariata utilizzando **tutte** le feature continue disponibili.
- Implementazione algebrica tramite la formula dei minimi quadrati: $\mathbf{a} = (X^T X)^{-1} X^T \mathbf{y}$.
- Validazione tramite la stessa procedura train/test della Fase 3 e calcolo degli scarti globali.

### ☐ Fase 6 — Pulizia Outlier Dinamica (`linear_regression.py`)
- Script interattivo che individua il dato che si discosta maggiormente dalla retta di regressione (massimo residuo assoluto).
- Output a schermo: `"Il punto [XX] sembra anomalo. Vuoi cancellarlo dal dataset? (s/n)"`.
- **Loop iterativo:** se la risposta è affermativa → rimuovere il punto, ricalcolare la regressione e riproporre la domanda. Se negativa → generare il plot finale con il modello corrente.

### ☐ Fase 7 — Teoria e Integrazione Concettuale
- Integrazione dei commenti teorici nel codice e nella documentazione (approccio globale vs. a tratti).
- Completamento della sezione *Requisiti e Utilizzo* nel `README.md`.
- Revisione finale della coerenza tra i moduli `src/` e i notebook Colab.

---

## 7. Stack Tecnologico Consentito

| Libreria | Uso consentito |
|---|---|
| `numpy` | ✅ Algebra lineare, operazioni matriciali, calcolo numerico |
| `pandas` | ✅ Caricamento e manipolazione del dataset |
| `matplotlib` | ✅ Tutti i plot richiesti |
| `scipy.stats` | ✅ Solo per il confronto con distribuzioni teoriche (Gaussiana) |
| `scikit-learn` | ⛔ **Vietato** per regressione e metriche centrali |

---

## 8. Istruzione per l'AI Assistant (Antigravity)

> Agisci come un **Senior Data Scientist e tutor accademico matematico**.
> Leggi questo manifesto, assimila l'architettura e resta in attesa delle istruzioni dell'utente per iniziare l'implementazione del codice dalla **Fase 1**.
>
> Usa esclusivamente **Python** con `numpy`/`pandas` per la logica algoritmica e `matplotlib` per i plot.
> Non usare mai funzioni di regressione preconfezionate. Mostra sempre la derivazione matriciale o algebrica nei commenti del codice.
> Ogni funzione deve essere documentata con la formula matematica che implementa.
