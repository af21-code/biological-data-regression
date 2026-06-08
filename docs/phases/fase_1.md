# Fase 1 — Analisi 0-Dimensionale

**File di riferimento:** `src/data_exploration.py` · `src/metrics.py` · `src/data_ingestion.py`
**Dataset:** `data/raw/ncbi_muscular_dystrophy_raw.csv`
**Sorgente:** NCBI GEO — dataset reale pubblico (GSE38417, Duchenne Muscular Dystrophy)

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

L'analisi 0-dimensionale è il primo passo obbligatorio di qualsiasi pipeline di
data science: capire la struttura statistica dei dati **prima** di costruire qualsiasi
modello. Ignorare questo step porta a modelli non validi, scelti su basi errate.

---

### Il Dataset — Dati Reali NCBI GEO (GSE38417)

Il dataset non è sintetico: è stato scaricato programmaticamente dal database
**NCBI GEO (Gene Expression Omnibus)**, il più grande archivio pubblico al mondo
di dati di espressione genica, mantenuto dai National Institutes of Health (NIH).

| Attributo | Valore |
|---|---|
| **Accession** | GSE38417 |
| **Titolo** | Gene expression in Duchenne Muscular Dystrophy |
| **Campioni** | 22 biopsie muscolari (pazienti DMD + controlli sani) |
| **Piattaforma** | Microarray Affymetrix (dati di ibridazione normalizzati) |
| **Acquisizione** | `src/data_ingestion.py` via FTP NCBI, SSL/TLS con `certifi` |

#### Struttura del dataset estratto

| Colonna | Tipo | Descrizione |
|---|---|---|
| 10 colonne probe | `float64` | Livelli di espressione genica dei 10 probe più correlati al target (matrice X) |
| `target_expression` | `float64` | Livello di espressione del probe target — **variabile dipendente y ∈ ℝ²²** |

I valori sono in **scala logaritmica** (log₂ delle intensità di fluorescenza
normalizzate): questa è la scala standard per i dati Affymetrix, che garantisce
omogeneità della varianza e simmetria delle distribuzioni.

> **Perché il dataset ha solo 22 campioni?**
> I dataset di espressione genica clinica su pazienti rari come la DMD sono
> costosi e difficili da raccogliere. 22 campioni è una dimensione realistica
> e sufficiente per la validazione matematica richiesta dalla traccia d'esame.

---

### Come è stato acquisito il dataset (`src/data_ingestion.py`)

Il modulo di data ingestion implementa una pipeline bottom-up in 4 step:

#### Step 1 — Download del Series Matrix File
Il file `GSE38417_series_matrix.txt.gz` viene scaricato direttamente dal server
FTP di NCBI tramite `urllib.request` (nessuna libreria di terze parti per il
download). Il file è in formato GEO standardizzato, compresso con gzip.

**Nota tecnica SSL (macOS):** Python installato da python.org su macOS non usa
i certificati SSL di sistema. Il modulo `certifi` fornisce un bundle di certificati
CA aggiornato, utilizzato per costruire il contesto SSL:
```python
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
```

#### Step 2 — Parsing manuale del formato GEO
Il formato GEO Series Matrix ha una struttura testuale definita:
- Righe con `!` = metadati del dataset (titolo, autori, piattaforma…)
- `!series_matrix_table_begin` = inizio della tabella dati
- Prima riga = header con gli ID GSM dei campioni
- Righe successive = `probe_ID \\t valore_campione_1 \\t valore_campione_2 …`

Il parsing è implementato **senza GEOparse** (libreria di alto livello), riga per riga,
per massimizzare il controllo sul processo di estrazione.

#### Step 3 — Selezione del target y e delle feature X
La variabile dipendente $y$ è il livello di espressione del probe più variabile
nel dataset (massima varianza → massima informazione per la regressione).

Le 10 feature predittive $X$ sono i probe con **correlazione di Pearson assoluta
massima** rispetto a $y$, calcolata esplicitamente:

$$r_{ij} = \frac{\mathbf{x}_i^c \cdot \mathbf{y}^c}{n \cdot \sigma_i \cdot \sigma_y}$$

dove $\mathbf{x}_i^c$ e $\mathbf{y}^c$ sono i vettori centrati, senza chiamare
`numpy.corrcoef` o `pandas.corr`.

#### Step 4 — Imputazione NaN e salvataggio
I valori mancanti (segnale sotto soglia di rilevamento nel microarray) vengono
imputati con la media di colonna:

$$\hat{x}_{ij} = \bar{x}_j = \frac{1}{|\{k : x_{kj} \neq \text{NaN}\}|} \sum_{k: x_{kj}\neq\text{NaN}} x_{kj}$$

Il dataset viene salvato in `data/raw/ncbi_muscular_dystrophy_raw.csv`.

---

### Step 2 — Indicatori Statistici Descrittivi

Per ogni variabile numerica vengono calcolati i seguenti indicatori tramite
**formule esplicite con NumPy** (nessuna chiamata a `df.describe()`):

#### Media aritmetica
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

Stima il valore atteso della distribuzione. È sensibile agli outlier.

#### Mediana
Valore che divide la distribuzione ordinata esattamente a metà.
Costruita ordinando il vettore e selezionando l'elemento centrale (o la media
dei due centrali per n pari). **Robusta agli outlier.**

#### Varianza di popolazione
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 = \frac{1}{n} \, \mathbf{d}^\top \mathbf{d}$$

dove $\mathbf{d} = \mathbf{x} - \bar{x}$ è il vettore delle deviazioni.
La divisione per $n$ (e non $n-1$) riflette la **varianza di popolazione**,
coerente con la notazione della traccia d'esame.

#### Deviazione standard
$$\sigma = \sqrt{\sigma^2}$$

Stessa unità di misura dei dati originali: più intuitiva per l'interpretazione.

---

### Step 3 — Intervallo al 90% centrato nella mediana

**Problema:** trovare $\delta$ tale che l'intervallo $[m - \delta,\; m + \delta]$,
centrato sulla mediana $m$, contenga almeno il 90% delle istanze.

**Metodo (non parametrico):**

1. Calcola le distanze assolute dalla mediana: $d_i = |x_i - m|$
2. Ordina le distanze: $d_{(1)} \leq d_{(2)} \leq \ldots \leq d_{(n)}$
3. $\delta = d_{(\lceil 0.90 \cdot n \rceil)}$ — il 90° percentile delle distanze

L'intervallo $[m-\delta, m+\delta]$ contiene automaticamente il 90% dei punti
più vicini alla mediana. **Non assume alcuna distribuzione** — funziona per
qualunque forma della distribuzione.

> **Contesto reale:** i dati di espressione genica in scala log₂ tendono ad
> essere più simmetrici dei dati clinici grezzi (come la creatinchinasi in
> scala lineare). Questo rende l'intervallo centrato nella mediana particolarmente
> affidabile per questo tipo di dataset.

---

### Step 4 — Matrice di Covarianza

La matrice di covarianza $C \in \mathbb{R}^{p \times p}$ misura come ogni
coppia di variabili varia congiuntamente:

$$C_{ij} = \frac{1}{n} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$

**Implementazione matriciale esplicita** (senza `numpy.cov`):

$$C = \frac{1}{n} \, X_c^\top X_c$$

dove $X_c = X - \mathbf{1}\bar{\mathbf{x}}^\top$ è la matrice dei dati centrata.

**Utilizzo nella Fase 2:** la riga/colonna di $C$ corrispondente a `target_expression`
rivela le correlazioni con tutte le altre feature → selezioniamo la feature con
la covarianza assoluta più alta come predittore per la regressione lineare semplice.

---

### Step 5-7 — Visualizzazioni

#### Bar chart degli indicatori
Subplot separato per ogni variabile. Mostra media, mediana e deviazione standard
affiancate. I dati in scala log₂ hanno range simili → il grafico è leggibile senza
problemi di scala.

#### Box plot affiancati
Riassume visivamente la distribuzione di ogni variabile tramite:
- Mediana (linea arancione centrale)
- Scatola: intervallo interquartile [Q1, Q3]
- Whisker: estensione fino a $1.5 \times IQR$
- Punti rossi: outlier (campioni con profilo di espressione anomalo)

#### ECDF vs Gaussiana teorica
La funzione di ripartizione empirica è definita come:
$$\hat{F}(x_{(i)}) = \frac{i}{n}, \quad i = 1, \ldots, n$$

Viene confrontata con la CDF della Gaussiana $\mathcal{N}(\bar{x}, \sigma)$.
Nei dati microarray normalizzati ci si attende una buona aderenza alla gaussiana
— una divergenza segnalerebbe problemi di normalizzazione del dataset.

---

### Modulo `src/metrics.py`

Implementato come modulo isolato e riusabile da tutte le fasi successive:

| Funzione | Formula |
|---|---|
| `residuals(y, ŷ)` | $e_i = y_i - \hat{y}_i$ |
| `mse(y, ŷ)` | $\frac{1}{n}\sum e_i^2 = \frac{1}{n}\,\mathbf{e}^\top\mathbf{e}$ |
| `rmse(y, ŷ)` | $\sqrt{\text{MSE}}$ |
| `r_squared(y, ŷ)` | $1 - \frac{SS_{res}}{SS_{tot}}$ |

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Perché usiamo il Virtual Environment (`.venv`)

Quando si lavora a un progetto Python, si usano librerie esterne — strumenti
già scritti da altri che aggiungiamo al nostro codice. Il problema è che progetti
diversi possono richiedere versioni diverse delle stesse librerie, e se le
installiamo tutte insieme sul computer possono crearsi conflitti.

**Il Virtual Environment è una "stanza isolata"** per il progetto: ogni libreria
che installiamo (numpy, pandas, matplotlib…) viene messa solo in quella stanza,
senza interferire con il resto del sistema. È come avere una cassetta degli
attrezzi dedicata a questo progetto.

Per attivare questa stanza prima di lavorare si esegue:
```bash
source .venv/bin/activate
```

Da quel momento il terminale usa solo gli strumenti di questa stanza, e il
progetto funziona in modo identico su qualsiasi computer del team.

---

### Cosa abbiamo fatto in parole semplici

Invece di inventare dati artificiali, abbiamo scaricato dati **veri e pubblici**
da NCBI GEO — uno dei più grandi archivi scientifici al mondo, gestito dai
National Institutes of Health americani (NIH). Il nostro dataset contiene i
profili genetici di 22 campioni reali di tessuto muscolare, alcuni provenienti
da pazienti con Distrofia Muscolare di Duchenne, altri da persone sane.

Ogni campione è caratterizzato da misurazioni dell'attività di specifici geni
(i probe del microarray). I valori sono già in scala logaritmica: questo significa
che i numeri nel dataset rappresentano "quanto è attivo" ogni gene in ogni
campione, su una scala matematicamente comoda.

**L'obiettivo finale** è costruire una formula che, guardando l'attività di
alcuni geni, preveda l'attività di un altro gene — quello che abbiamo scelto
come variabile target. Questo tipo di analisi può rivelare relazioni biologiche
tra geni, utili per capire i meccanismi della malattia.

---

### Cosa abbiamo calcolato e perché

**Massimo, minimo, media, mediana:**
Sono come i "riassunti" di ogni variabile. La media è la somma di tutti i
valori divisa per quanti sono — come la media dei voti a scuola. La mediana
è invece il valore di mezzo: se metto tutti i campioni in fila ordinati per
livello di espressione di un gene, il campione nel mezzo ha il valore mediano.

Nei dati di espressione genica in scala logaritmica, media e mediana tendono
ad essere simili — un buon segno che i dati sono ben distribuiti.

**Varianza e deviazione standard:**
Misurano "quanto i dati si disperdono" attorno alla media. Un gene con alta
varianza cambia molto da campione a campione — di solito è un gene interessante
biologicamente, perché distingue i malati dai sani.

**Intervallo al 90%:**
Abbiamo trovato un intervallo centrato attorno al valore mediano che contiene
il 90% dei campioni. Questo ci dice qual è il range "tipico" di espressione per
ogni gene, escludendo i casi eccezionali agli estremi.

**Matrice di covarianza:**
Ci dice quali geni "si muovono insieme". Se quando un gene è molto attivo anche
un altro tende ad esserlo, i due hanno covarianza positiva. Questa informazione
è fondamentale: nella Fase 2 useremo proprio questa matrice per scegliere quale
gene è più utile a prevedere il nostro target.

---

### I grafici che abbiamo prodotto

**Grafico a barre:** mostra media, mediana e deviazione standard affiancate per
ogni gene. Con dati in scala logaritmica, le barre hanno altezze paragonabili —
il grafico è immediatamente leggibile.

**Box plot:** è come un riassunto grafico in una scatola. La riga centrale è la
mediana, la scatola contiene il 50% dei campioni "centrali", le linee si estendono
verso i valori estremi, e i puntini rossi sono i campioni anomali. Ci permette
di vedere la distribuzione di tutti i campioni a colpo d'occhio.

**Grafico ECDF vs Gaussiana:** la curva blu mostra come sono realmente distribuiti
i nostri dati, la curva tratteggiata arancione mostra come sarebbero se seguissero
la distribuzione "a campana" (gaussiana). Per i dati microarray normalizzati le
due curve dovrebbero essere molto vicine — è una conferma che il dataset è di
buona qualità.

---

### Perché tutto questo prima di costruire il modello?

Perché un modello costruito senza capire i dati produce previsioni inaffidabili.
Conoscere la distribuzione delle variabili ci aiuta a:
1. Scegliere il gene più correlato al target (Fase 2).
2. Identificare campioni anomali che potrebbero "ingannare" il modello (Fase 6).
3. Giustificare matematicamente ogni scelta davanti alla commissione d'esame.

In bioinformatica — come in medicina — analizzare i dati prima di trarre
conclusioni non è un optional: è il fondamento del metodo scientifico.
