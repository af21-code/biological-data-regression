# 🗺️ ROADMAP — Analisi Numerica e Modelli di Regressione per la Stima Continua del Decadimento Muscolare

---

## 🎯 Comprensione del Progetto

### Cosa stiamo costruendo e perché

Questo è un progetto d'esame universitario per il corso di **Analisi Numerica per l'Analisi dei Dati Biologici** (LT Biotecnologie per la salute). La traccia assegnata è il **Progetto B: Regressione con output continuo**.

Il dominio applicativo scelto è la **Distrofia Muscolare**: una malattia neuromuscolare progressiva in cui la progressione può essere quantificata attraverso biomarcatori continui misurabili (es. concentrazione sierica di creatinchinasi, punteggi di forza muscolare, espressione proteica). Questa scelta è perfetta perché soddisfa il vincolo della traccia: l'output da prevedere è **strettamente continuo**.

### Cosa deve dimostrare il codice

Il progetto non è una semplice applicazione di librerie. Deve dimostrare la **padronanza matematica** degli algoritmi:

- Implementare la regressione lineare **da zero** usando algebra matriciale (`numpy`), **non** `sklearn.LinearRegression`.
- Calcolare MSE e R² **a mano**, non chiamando `.score()`.
- La matrice di covarianza va costruita esplicitamente, non estratta da funzioni preconfezionate.
- Ogni formula implementata deve essere riconoscibile e derivabile dalla teoria dei minimi quadrati.

### Stack consentito

| Libreria | Uso |
|---|---|
| `numpy` | ✅ Algebra lineare, operazioni matriciali, calcolo numerico |
| `pandas` | ✅ Caricamento e manipolazione del dataset |
| `matplotlib` | ✅ Tutti i plot (scatter, box plot, istogrammi, ECDF) |
| `scipy.stats` | ✅ Solo per il confronto con la distribuzione Gaussiana teorica |
| `scikit-learn` | ⛔ **Vietato** per regressione, metriche e preprocessing centrali |

### Architettura del progetto

```
src/                      ← Logica algoritmica pura (Python modules)
│   data_exploration.py   ← Fase 1
│   linear_regression.py  ← Fase 2, 3, 4, 6
│   multivariate.py       ← Fase 5
│   metrics.py            ← Usato da tutte le fasi

notebooks/                ← Esplorazione visiva su Google Colab
                             I notebook importano le funzioni da src/
```

---

## ✅ ROADMAP — Checkpoint per Fase

---

### FASE 1 — Analisi 0-dimensionale
**File:** `src/data_exploration.py`
**Obiettivo:** Capire i dati prima di modellarli. Calcolare gli indicatori statistici di base e visualizzare le distribuzioni.

- [x] Caricare il dataset con `pandas` e ispezionare tipi, valori nulli e shape.
- [x] Per ogni variabile continua, calcolare:
  - Massimo e minimo
  - Media aritmetica
  - Mediana
  - Varianza (formula esplicita: $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$)
  - Deviazione standard
- [x] Calcolare l'intervallo $[m - \delta, m + \delta]$ centrato nella mediana $m$ che contiene il 90% delle istanze, per ogni variabile continua.
- [x] Calcolare la matrice di covarianza esplicitamente:
  $C_{ij} = \frac{1}{n}\sum_{k=1}^{n}(x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$
- [x] **Plot:** Bar plot degli indicatori statistici per ogni variabile.
- [x] **Plot:** Box plot affiancati per tutte le variabili continue.
- [x] **Plot:** Funzione di ripartizione empirica (ECDF) con sovrapposizione della CDF gaussiana corrispondente (media e varianza calcolate sopra).

---

### FASE 2 — Regressione Lineare Base
**File:** `src/linear_regression.py`
**Obiettivo:** Costruire algebricamente la retta $y = ax + b$ che meglio approssima i dati.

- [x] Usare la matrice di covarianza (Fase 1) per selezionare la **feature con correlazione massima** all'output target.
- [x] Implementare i minimi quadrati per la retta $y = ax + b$:
  $$a = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}, \quad b = \bar{y} - a\bar{x}$$
- [x] Calcolare i valori predetti $\hat{y}_i = ax_i + b$ per ogni istanza.
- [x] Calcolare e stampare: residui, MSE, RMSE, R² (delegati a `metrics.py`).
- [x] **Plot:** Scatter plot dei dati originali con la retta di regressione sovrapposta.

---

### FASE 3 — Train / Test Split e Validazione
**File:** `src/linear_regression.py`
**Obiettivo:** Validare il modello su dati mai visti durante l'addestramento.

- [x] Partizionare il dataset: **90% Training**, **10% Test** (shuffle casuale con seed fisso per riproducibilità).
- [x] **Plot:** Box plot affiancati Training vs Test per verificare similarità delle distribuzioni.
- [x] Addestrare il modello di regressione **esclusivamente** sul training set.
- [x] Applicare il modello al test set e calcolare gli scarti:
  $e_i = y_i - \hat{y}_i \quad \forall i \in \text{test set}$
- [x] Calcolare e stampare MSE e R² sul test set (tramite `metrics.py`).
- [x] **Plot:** Scatter plot con retta di regressione addestrata + punti del test set evidenziati.

---

### FASE 4 — Interpolazione ed Estrapolazione
**File:** `src/linear_regression.py`
**Obiettivo:** Usare il modello per inferire punti non presenti nel dataset originario.

- [x] **Interpolazione nel punto medio:** calcolare $x_{mid} = \frac{x_{max} + x_{min}}{2}$, predire $\hat{y}(x_{mid})$ e confrontare con la media dei valori osservati nell'intorno.
- [x] **Predizione dell'ascissa dell'ordinata media:** trovare $x^* = \frac{\bar{y} - b}{a}$ (inversione della retta).
- [x] **Estrapolazione in zero:** calcolare $\hat{y}(0) = b$.
- [x] **Prompt interattivo:** chiedere all'utente un valore $x_{input}$ da tastiera, calcolare $\hat{y}(x_{input})$ e plottare il punto sul grafico originale con un marcatore distinto.

---

### FASE 5 — Modello Multilineare
**File:** `src/multivariate.py`
**Obiettivo:** Estendere la regressione a tutte le feature continue disponibili.

- [x] Selezionare **tutte** le colonne continue del dataset come feature.
- [x] Costruire la matrice di design $X$ (con colonna di 1 per il termine noto).
- [x] Implementare i minimi quadrati in forma matriciale:
  $$\mathbf{a} = (X^T X)^{-1} X^T \mathbf{y}$$
  usando `numpy.linalg.solve` (più stabile dell'inversione diretta).
- [x] Calcolare $\hat{\mathbf{y}} = X\mathbf{a}$ e gli scarti $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$.
- [x] Applicare la stessa procedura train/test della Fase 3 al modello multivariato.
- [x] Calcolare e stampare MSE e R² su training e test set (tramite `metrics.py`).
- [x] **Confronto:** stampare un riepilogo comparativo tra modello lineare (Fase 2) e multilineare.

---

### FASE 6 — Pulizia Outlier Dinamica (Loop Interattivo)
**File:** `src/linear_regression.py`
**Obiettivo:** Implementare un algoritmo interattivo che individua e rimuove iterativamente i dati anomali.

- [x] Calcolare i residui $|e_i| = |y_i - \hat{y}_i|$ per tutte le istanze.
- [x] Identificare il punto con il residuo assoluto massimo (l'outlier più probabile).
- [x] Stampare a schermo:
  ```
  Il punto [indice XX] sembra anomalo (scarto = YY.YY). Vuoi cancellarlo dal dataset? (s/n):
  ```
- [x] **Se risposta = `s`:** rimuovere il punto, ricalcolare la regressione sul dataset ridotto, riproporre la domanda sul nuovo outlier massimo.
- [x] **Se risposta = `n`:** uscire dal loop e generare il plot finale con il modello corrente e i punti rimossi evidenziati.
- [x] Salvare il dataset pulito in `data/processed/`.

---

### FASE 7 — Teoria e Consolidamento Finale
**File:** `src/` (commenti), `docs/`, `README.md`
**Obiettivo:** Chiudere il progetto con rigore documentale e teorico.

- [x] Aggiungere nei commenti del codice i riferimenti teorici (approccio globale vs. a tratti, stabilità numerica, condizionamento della matrice $X^T X$).
- [x] Completare la sezione *Requisiti e Utilizzo* del `README.md` con istruzioni di installazione e di esecuzione.
- [x] Aggiungere in `docs/` la presentazione e la traccia PDF del progetto.
- [x] Rivedere la coerenza tra i moduli `src/` e i notebook Colab in `notebooks/`.
- [x] Verifica finale: il codice gira senza errori dall'inizio alla fine, con un dataset reale o sintetico di distrofia muscolare.

---

## 📦 Modulo Trasversale — Metriche
**File:** `src/metrics.py`
**Usato da:** tutte le fasi che addestrino un modello.

- [x] `mse(y_true, y_pred)` → $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- [x] `rmse(y_true, y_pred)` → $\text{RMSE} = \sqrt{\text{MSE}}$
- [x] `r_squared(y_true, y_pred)` → $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$
- [x] `residuals(y_true, y_pred)` → vettore degli scarti $e_i = y_i - \hat{y}_i$

---

## 📋 Stato Avanzamento

| Fase | Descrizione | Stato |
|---|---|---|
| Fase 1 | Analisi 0-dimensionale | ✅ Completato |
| Fase 2 | Regressione Lineare Base | ✅ Completato |
| Fase 3 | Train / Test Split | ✅ Completato |
| Fase 4 | Interpolazione ed Estrapolazione | ✅ Completato |
| Fase 5 | Modello Multilineare | ✅ Completato |
| Fase 6 | Pulizia Outlier Dinamica | ✅ Completato |
| Fase 7 | Teoria e Consolidamento | ✅ Completato |
| Trasv. | Modulo Metriche (`metrics.py`) | ✅ Completato |
