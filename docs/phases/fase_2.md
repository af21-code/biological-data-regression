# Fase 2 — Regressione Lineare Base

**File di riferimento:** `src/linear_regression.py` · `src/metrics.py`
**Dataset:** `data/raw/ncbi_muscular_dystrophy_raw.csv` (22 campioni reali NCBI GEO)

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

Costruire algebricamente la retta di regressione $y = ax + b$ che meglio approssima
la relazione tra il predittore selezionato (il gene più correlato al target) e
la variabile dipendente continua `target_expression`.

---

### Step 1 — Selezione del Miglior Predittore (dalla Matrice di Covarianza)

Dalla matrice di covarianza calcolata in Fase 1, estraiamo la riga corrispondente
a `target_expression` e selezioniamo la feature con **covarianza assoluta massima**:

$$\text{predittore}^* = \arg\max_{j \neq \text{target}} |C_{\text{target}, j}|$$

**Risultato:** il probe `206717_at` ha la covarianza assoluta massima con il target:

| Probe | \|Covarianza con target\| |
|---|---|
| **206717_at** | **16.296** ← selezionato |
| 215389_s_at | 11.595 |
| 205694_at | 10.854 |
| 226213_at | 10.232 |
| ... | ... |

**Motivazione biologica:** il probe `206717_at` con covarianza ~16 indica una forte
co-espressione con il gene target. Nei dati microarray DMD questo tipo di
correlazione riflette spesso pathway biologici comuni (es. geni regolati dalla
stessa via di segnalazione del danno muscolare).

---

### Step 2 — Costruzione Algebrica della Retta (Minimi Quadrati)

**Problema:** trovare $a$ e $b$ che minimizzino la somma dei quadrati dei residui:

$$S(a, b) = \sum_{i=1}^{n} (y_i - ax_i - b)^2$$

**Derivazione (condizioni del primo ordine):**

$$\frac{\partial S}{\partial b} = 0 \implies b = \bar{y} - a\bar{x}$$

$$\frac{\partial S}{\partial a} = 0 \implies a = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Implementazione vettoriale con NumPy** (senza `numpy.polyfit`, `scipy.linregress` o sklearn):

```python
x_c = x - x.mean()           # vettore centrato
y_c = y - y.mean()
a   = dot(x_c, y_c) / dot(x_c, x_c)   # prodotto scalare / norma²
b   = y.mean() - a * x.mean()
```

**Risultato sul dataset reale:**

$$y = 1.1364 \cdot x + (-1.8614)$$

---

### Step 3 — Calcolo delle Metriche (su tutto il dataset)

Dopo aver calcolato $\hat{y}_i = ax_i + b$, le metriche sono delegate a `src/metrics.py`:

| Metrica | Formula | Valore (n=22) |
|---|---|---|
| Residui | $e_i = y_i - \hat{y}_i$ | vettore |
| MSE | $\frac{1}{n}\sum e_i^2$ | **0.0531** |
| RMSE | $\sqrt{\text{MSE}}$ | **0.2305** |
| R² | $1 - \frac{\sum e_i^2}{\sum(y_i - \bar{y})^2}$ | **0.9971** |

**Interpretazione:** R² = 0.997 indica che il modello lineare spiega il **99.7%
della varianza** del target. Questo valore elevato è atteso su dati microarray
normalizzati dove la correlazione tra geni co-espressi è molto forte.

---

### Step 4 — Scatter Plot con Retta di Regressione

Il grafico mostra:
- **Punti blu:** le 22 osservazioni reali (x = probe 206717_at, y = target_expression)
- **Retta arancione:** $y = 1.1364x - 1.8614$ (modello OLS)

Il plot viene salvato in `data/processed/plots/phase2_regression.png`.

---

### Funzioni implementate in `src/linear_regression.py`

| Funzione | Responsabilità |
|---|---|
| `select_best_predictor(df, target, cov_df)` | Sceglie la feature con \|covarianza\| massima |
| `fit_linear_regression(x, y)` | Calcola (a, b) via OLS esplicito |
| `predict(x, a, b)` | Applica il modello: $\hat{y} = ax + b$ |
| `run_phase2(df, target, cov_df, plots_dir)` | Orchestratore completo della Fase 2 |

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Cosa abbiamo fatto

Nella Fase 1 abbiamo studiato i dati uno per uno. Nella Fase 2 cominciamo a
cercare **relazioni tra le variabili**: vogliamo trovare una formula che, dato
il valore di un gene (il predittore), ci permetta di prevedere il valore del
gene target.

Il tipo di formula che cerchiamo è una **retta**:

$$y = a \cdot x + b$$

dove:
- $x$ è il valore del gene predittore (l'input)
- $y$ è il valore del gene target (l'output da prevedere)
- $a$ è la pendenza della retta (quanto cambia y per ogni unità di x)
- $b$ è il punto dove la retta interseca l'asse verticale (valore di y quando x = 0)

---

### Come abbiamo scelto il gene predittore

Avevamo 10 geni a disposizione come predittori. Come abbiamo scelto il migliore?

Abbiamo usato la **matrice di covarianza** calcolata nella Fase 1. La covarianza
misura quanto due variabili "si muovono insieme". Abbiamo scelto il gene che
si muove di più insieme al target — quello con la covarianza assoluta più alta.

Il risultato: il probe `206717_at` è il gene che co-varia di più con il target
(covarianza = 16.3). Questo significa che quando questo gene è molto espresso,
il target tende ad esserlo altrettanto.

---

### Come abbiamo trovato la retta

La retta non è stata scelta "a occhio". Abbiamo usato un metodo matematico
preciso chiamato **Metodo dei Minimi Quadrati (OLS)**:

> Tra tutte le rette possibili, quella che minimizza la somma dei quadrati
> delle distanze verticali tra i punti e la retta stessa.

In parole ancora più semplici: è la retta che "passa più vicino" a tutti i
punti contemporaneamente.

La formula è:
- Calcoliamo quanto ogni valore di x si discosta dalla sua media (deviazione)
- Calcoliamo quanto ogni valore di y si discosta dalla sua media
- La pendenza $a$ è il rapporto tra "deviazioni congiunte" e "varianza di x"
- L'intercetta $b$ garantisce che la retta passi per il punto $(\bar{x}, \bar{y})$

---

### I risultati

La retta trovata è: **y = 1.1364·x − 1.8614**

Questo significa: per ogni unità in più del gene `206717_at`, il gene target
aumenta di circa 1.14 unità (in scala log₂).

**Quanto è buona questa retta?**

Abbiamo usato la metrica **R²** (R-quadro): va da 0 a 1 e misura "quanto bene
la retta spiega i dati". Il nostro R² = 0.997 significa che la retta spiega
il **99.7% della variazione** del target — un risultato eccellente, tipico di
dati biologici co-espressi fortemente correlati.

Il **RMSE** (errore quadratico medio radice) è 0.23: mediamente, le nostre
previsioni si discostano dal valore reale di circa 0.23 unità log₂ — un
errore molto piccolo rispetto al range totale del target (3.1 – 14.9).
