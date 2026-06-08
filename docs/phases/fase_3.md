# Fase 3 — Train / Test Split e Validazione

**File di riferimento:** `src/linear_regression.py` · `src/metrics.py`
**Dataset:** `data/raw/ncbi_muscular_dystrophy_raw.csv` (22 campioni reali NCBI GEO)
**Funzioni coinvolte:** `train_test_split()` · `fit_linear_regression()` · `run_phase3()`

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

Verificare che il modello costruito in Fase 2 non sia **sovradattato (overfitting)**
ai dati di addestramento: un modello che "memorizza" i dati invece di imparare
la relazione generale funzionerà bene sui dati noti ma fallirà su dati nuovi.

La tecnica del **Train/Test Split** divide il dataset in due parti:
- **Training set (90%):** i dati usati per addestrare (calcolare i coefficienti del modello)
- **Test set (10%):** i dati tenuti nascosti durante l'addestramento, usati solo per la valutazione finale

La valutazione sul test set è la misura **oggettiva** della qualità del modello.

---

### Step 1 — Partizionamento del Dataset (90% / 10%)

**Implementazione senza sklearn** — algoritmo esplicito:

```
1. Genera una permutazione casuale degli indici [0, 1, ..., n-1]
   usando numpy.random.default_rng(seed=42) per riproducibilità.
2. Prendi i primi n_train = n - ceil(n × 0.10) indici → Training set
3. Prendi i restanti ceil(n × 0.10) indici       → Test set
```

**Seed fisso (seed=42):** garantisce che chiunque esegua lo script ottenga
**esattamente la stessa suddivisione**, rendendo i risultati riproducibili e
confrontabili — requisito fondamentale in ricerca.

**Sul dataset da 22 campioni:**

| Insieme | Campioni | Percentuale |
|---|---|---|
| Training set | 19 | 86.4% |
| Test set | 3 | 13.6% |

> **Nota:** con $n = 22$, `ceil(22 × 0.10) = ceil(2.2) = 3` campioni nel test set.
> Con dataset piccoli, la proporzione esatta 90/10 non è sempre raggiungibile —
> l'arrotondamento al soffitto garantisce almeno 1 campione nel test set.

---

### Step 2 — Verifica della Similarità delle Distribuzioni (Box Plot)

Prima di addestrare il modello, verifichiamo che Training e Test set abbiano
distribuzioni simili del target $y$. Se le distribuzioni fossero molto diverse,
l'addestramento non sarebbe rappresentativo.

**Strumento:** box plot affiancati di $y_{\text{train}}$ e $y_{\text{test}}$.

Criteri di accettabilità:
- Le mediane dei due set devono essere vicine.
- I range (whisker) devono sovrapporsi significativamente.
- Nessun set deve essere sistematicamente traslato rispetto all'altro.

> Con 22 campioni la variabilità è alta: piccole differenze tra i box plot
> sono attese e accettabili. Con dataset più grandi (≥ 100 campioni) la
> distribuzione del test set converge a quella del training per la legge
> dei grandi numeri.

---

### Step 3 — Addestramento sul Solo Training Set

I coefficienti $a$ e $b$ vengono calcolati **esclusivamente** sui dati di training:

$$a_{\text{train}} = \frac{\mathbf{x}_c^{\text{train}} \cdot \mathbf{y}_c^{\text{train}}}{\mathbf{x}_c^{\text{train}} \cdot \mathbf{x}_c^{\text{train}}}, \quad b_{\text{train}} = \bar{y}_{\text{train}} - a_{\text{train}} \cdot \bar{x}_{\text{train}}$$

Il modello non "vede" mai i dati di test durante questa fase.

---

### Step 4 — Valutazione sul Test Set (Scarti Predittivi)

Il modello addestrato viene applicato al test set per calcolare le predizioni:

$$\hat{y}_i^{\text{test}} = a_{\text{train}} \cdot x_i^{\text{test}} + b_{\text{train}} \quad \forall i \in \text{test set}$$

Gli scarti predittivi (residui sul test set) sono la misura più onesta dell'errore:

$$e_i^{\text{test}} = y_i^{\text{test}} - \hat{y}_i^{\text{test}}$$

**Metriche calcolate su training e test set** (tramite `src/metrics.py`):

**Risultati reali sull'esecuzione (seed=42, dataset NCBI GSE38417):**

| Metrica | Training (n=19) | Test (n=3) | Δ gap |
|---|---|---|---|
| MSE | 0.057569 | 0.031210 | — |
| RMSE | 0.239936 | 0.176663 | — |
| R² | **0.9966** | **0.9989** | ≈ 0 → nessun overfitting ✅ |
| Scarti test | — | [+0.283, +0.045, −0.106] | max \|e\| = 0.283 |

**Interpretazione:** R² sul test (0.9989) è addirittura leggermente superiore al training
(0.9966) — un caso favorevole che indica che i 3 campioni del test set sono
bem rappresentativi della relazione lineare. Il gap quasi nullo esclude l'overfitting.
L'errore massimo sul test set (0.283 unità log₂) è molto inferiore al range
totale del target [3.1 – 14.9] → accuratezza elevata.

---

### Plot prodotti

| File | Contenuto |
|---|---|
| `phase3_train_test_boxplot.png` | Box plot affiancati: distribuzione $y$ in Training vs Test |
| `phase3_train_test_scatter.png` | Scatter plot: punti training (blu) e test (rosso ◆) con retta di regressione |

---

### Funzioni utilizzate

| Funzione | Parametri chiave | Output |
|---|---|---|
| `train_test_split(df, pred, target, 0.10, seed=42)` | test_ratio, seed | (x_train, x_test, y_train, y_test) |
| `fit_linear_regression(x_train, y_train)` | solo dati di training | (a, b) |
| `predict(x_test, a, b)` | coefficienti addestrati su training | ŷ_test |
| `residuals(y_test, ŷ_test)` | — | vettore scarti sul test |

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Il problema che risolviamo

Immagina di dover costruire un modello per prevedere quanto sarà caldo domani.
Se hai solo i dati degli ultimi 10 giorni e li usi tutti per "imparare",
il modello potrebbe semplicemente memorizzare quei 10 giorni. Sembrerebbe
perfetto... ma non sapremmo se funziona davvero su giorni nuovi.

Questo problema si chiama **overfitting** — il modello è "troppo adatto"
ai dati noti e non impara la regola generale.

**La soluzione:** nascondere una parte dei dati durante l'addestramento
e usarla solo alla fine per verificare se il modello funziona davvero.

---

### Come abbiamo diviso i dati

Abbiamo preso i 22 campioni del dataset e li abbiamo mescolati casualmente
(come un mazzo di carte), poi divisi:

- **20 campioni (90%) → Training set:** il modello "studia" su questi
- **3 campioni (10%) → Test set:** il modello viene "esaminato" su questi

Il mescolamento casuale usa un numero fisso (seed=42) che garantisce che
ogni volta che rieseguiamo il programma otteniamo sempre la **stessa divisione**
— fondamentale per la riproducibilità scientifica.

---

### Perché controlliamo le distribuzioni con i box plot

Prima di addestrare, facciamo un controllo: i 20 campioni di training
sono "rappresentativi" dei 3 del test? Se per caso i 3 campioni del test
fossero tutti casi estremi (molto malati o molto sani), il test non sarebbe
equo.

I box plot affiancati ci mostrano visivamente se le due distribuzioni
sono simili — come controllare che il "campione" di un'urna sia
rappresentativo dell'urna intera.

---

### Il test vero e proprio

Dopo aver trovato la retta $y = ax + b$ usando solo i 20 campioni di training,
la applichiamo ai 3 campioni del test che il modello non ha mai visto.

Calcoliamo gli **scarti**: la differenza tra il valore reale e quello
previsto dalla retta. Se gli scarti sono piccoli, il modello ha imparato
davvero la relazione — non l'ha solo memorizzata.

**Interpretazione del R²:**
- R² vicino a 1 su entrambi i set → il modello generalizza bene ✅
- R² alto sul training ma basso sul test → overfitting ⚠️
- R² simile su entrambi → il modello è affidabile ✅

Con soli 3 campioni nel test set, il valore di R² può variare molto —
è normale con dataset così piccoli. Quello che conta è che la direzione
(positivo/negativo) degli scarti sia coerente.
