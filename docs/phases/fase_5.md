# Fase 5 — Regressione Multilineare

**File di riferimento:** `src/multivariate.py` · `src/metrics.py`
**Funzione principale:** `run_phase5(df, target_col, plots_dir, lambda_reg, simple_metrics)`
**Dataset:** `data/raw/ncbi_muscular_dystrophy_raw.csv` (22 campioni, 10 feature)

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

Estendere la regressione lineare semplice (1 predittore) al caso **multivariato**:
usare **tutte le 10 feature** del dataset per prevedere il target simultaneamente.

Il modello diventa:

$$y = a_1 x_1 + a_2 x_2 + \ldots + a_{10} x_{10} + b$$

In forma vettoriale compatta:

$$\mathbf{y} = X_b \cdot \boldsymbol{\theta}, \quad X_b = [\mathbf{1} \mid X] \in \mathbb{R}^{n \times (m+1)}, \quad \boldsymbol{\theta} = [b,\ a_1,\ldots,a_m]^\top$$

dove $\mathbf{1}$ è il vettore colonna di 1 che modella l'intercetta $b$.

---

### Step 1 — Costruzione della Matrice di Design $X_b$

```
X_b  =  [  1  |  x₁  x₂  x₃  ...  x₁₀  ]   ∈ ℝ^{22 × 11}
          ↑ colonna di 1 (intercetta)
```

| Parametri del modello | Valore |
|---|---|
| Campioni $n$ | 22 |
| Feature $m$ | 10 |
| Parametri totali $m+1$ | 11 |
| Gradi di libertà residui $n-(m+1)$ | 11 |

---

### Step 2 — Verifica del Condizionamento di $X^\top X$

Prima di risolvere il sistema, calcoliamo il **numero di condizionamento** di $X_b^\top X_b$:

$$\kappa(X_b^\top X_b) = \frac{\sigma_{\max}(X_b^\top X_b)}{\sigma_{\min}(X_b^\top X_b)}$$

dove $\sigma$ sono i valori singolari. Un condizionamento alto indica che piccole
perturbazioni nei dati producono grandi variazioni in $\boldsymbol{\theta}$
(instabilità numerica dovuta alla **multicollinearità**).

**Sul dataset NCBI GSE38417:**
$$\kappa(X_b^\top X_b) \approx 4.24 \times 10^6 \quad (\text{accettabile, stabile} ✅)$$

Soglie di riferimento:
- $\kappa < 10^6$: ottimale
- $10^6 \leq \kappa < 10^8$: accettabile (leggera multicollinearità)
- $\kappa \geq 10^8$: ⚠️ instabile → attivare regolarizzazione Ridge

---

### Step 3 — Equazioni Normali

**Derivazione formale:**

Minimizziamo la somma dei quadrati dei residui:

$$S(\boldsymbol{\theta}) = \|\mathbf{y} - X_b \boldsymbol{\theta}\|^2 = (\mathbf{y} - X_b\boldsymbol{\theta})^\top(\mathbf{y} - X_b\boldsymbol{\theta})$$

Espandendo e derivando rispetto a $\boldsymbol{\theta}$:

$$\frac{\partial S}{\partial \boldsymbol{\theta}} = -2X_b^\top(\mathbf{y} - X_b\boldsymbol{\theta}) = \mathbf{0}$$

$$\boxed{X_b^\top X_b \cdot \boldsymbol{\theta} = X_b^\top \mathbf{y}}$$

**Soluzione:**

$$\boldsymbol{\theta} = (X_b^\top X_b)^{-1} X_b^\top \mathbf{y}$$

**Implementazione numerica con** `numpy.linalg.solve`:

```python
A     = X_b.T @ X_b          # (m+1, m+1) — matrice dei coefficienti
b_vec = X_b.T @ y            # (m+1,)     — termine noto
theta = np.linalg.solve(A, b_vec)   # risolve A·θ = b
```

> **Perché `solve` invece di `inv(A) @ b`?**
> `numpy.linalg.solve` utilizza internamente la fattorizzazione LU con pivoting
> parziale — più stabile e rapido dell'inversione esplicita, che amplifica
> gli errori numerici se $A$ è mal condizionata.

---

### Step 4 — Regolarizzazione di Tikhonov / Ridge (opzionale)

Se $\kappa(X_b^\top X_b) \geq 10^8$, aggiungiamo un termine di penalizzazione:

$$\boldsymbol{\theta}_\lambda = (X_b^\top X_b + \lambda I)^{-1} X_b^\top \mathbf{y}$$

**Interpretazione:** invece di minimizzare solo i residui, minimizziamo:

$$S_\lambda(\boldsymbol{\theta}) = \|\mathbf{y} - X_b\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|^2$$

Il termine $\lambda\|\boldsymbol{\theta}\|^2$ penalizza coefficienti grandi,
riducendo la varianza del modello a costo di un piccolo bias (**bias-variance tradeoff**).

> Sul nostro dataset, $\lambda = 0$ (OLS puro) è sufficiente grazie al basso condizionamento.

---

### Step 5 — Confronto Regressione Semplice vs Multilineare

| Metrica | Semplice (F.3) | Multilineare (F.5) | Δ |
|---|---|---|---|
| MSE train | 0.057569 | **0.006252** | −0.051 ↓ migliore |
| RMSE train | 0.239936 | **0.079072** | −0.161 ↓ migliore |
| R² train | 0.996630 | **0.999634** | +0.003 ↑ migliore |
| MSE test | **0.031210** | 1.206570 | +1.175 ↑ peggiore |
| RMSE test | **0.176663** | 1.098440 | +0.922 ↑ peggiore |
| R² test | **0.998865** | 0.956137 | −0.043 ↓ peggiore |

**Interpretazione — Overfitting del modello multilineare:**

Il modello multilineare ottiene R² = 0.9996 sul training set (eccellente),
ma R² = 0.9561 sul test set — un calo significativo rispetto alla regressione semplice
(R² test = 0.9989).

Questo è un **caso classico di overfitting** dovuto alla sproporzione tra
il numero di parametri (11) e i campioni di training (19):

$$\text{ratio parametri/campioni} = \frac{11}{19} \approx 0.58$$

Con un rapporto così alto, il modello "memorizza" i dettagli del training set
invece di imparare la relazione generale — esattamente il problema che il
train/test split era pensato per rilevare.

**Coefficienti segnalatori di instabilità:**
- `230385_at`: $a_4 = -5.25$ — enormemente grande rispetto agli altri → multicollinearità.

**Lezione:** con $n = 22$ campioni e $m = 10$ feature, la regressione multilineare
è sovra-parametrizzata. La regressione semplice (1 feature ben scelta) è **più
robusta** su questo dataset. Con dataset più grandi ($n \geq 200$), il modello
multilineare sarebbe superiore.

---

### Plot prodotti

| File | Descrizione |
|---|---|
| `phase5_actual_vs_predicted.png` | Scatter $y$ reale vs $\hat{y}$. La linea tratteggiata è $y = \hat{y}$ (previsione perfetta). Punti vicini alla linea → modello accurato. |
| `phase5_residuals.png` | Residui $e_i = y_i - \hat{y}_i$ vs $\hat{y}_i$. I residui devono essere centrati attorno a 0 e senza pattern sistematici. |
| `phase5_coefficients.png` | Bar chart dei coefficienti $a_i$. Barre blu = contributo positivo, rosse = negativo. |

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Cosa è cambiato rispetto alla retta semplice

Fino alla Fase 4 usavamo **un solo gene** per prevedere il target.
Nella Fase 5 usiamo **tutti e 10 i geni** contemporaneamente.

Invece di cercare una retta in un piano (2D), cerchiamo un **iperpiano**
in uno spazio a 11 dimensioni (10 geni + intercetta). Non si può visualizzare
direttamente, ma la matematica funziona esattamente allo stesso modo.

**Analogia:** per prevedere il voto a un esame, potresti usare solo le ore di
studio (1 variabile). Ma se aggiungi anche il numero di ore di sonno, l'ansia
misurata, la difficoltà del corso... stai facendo una regressione multilineare.
Più informazioni (se rilevanti) → previsione più accurata.

---

### Come funzionano le Equazioni Normali in parole semplici

La matematica deve trovare i valori di $a_1, a_2, \ldots, a_{10}, b$ che
rendono le previsioni il più accurate possibile. Con una sola variabile, questo
era facile (formula chiusa con media e deviazione standard). Con 10 variabili,
bisogna risolvere un **sistema di 11 equazioni in 11 incognite**.

Le Equazioni Normali sono esattamente quel sistema. Il computer lo risolve
in millisecondi con la fattorizzazione LU — un algoritmo matematico che
"smontava" la matrice in pezzi più semplici da invertire.

---

### Cos'è il numero di condizionamento e perché ci interessa

Immagina di misurare la distanza Roma-Milano con un righello da 30 cm —
dovresti fare migliaia di misure in fila, e ogni piccolo errore si accumula.
Il numero di condizionamento misura qualcosa di simile: quanto si "amplificano"
gli errori di misura nei dati quando calcoliamo i coefficienti.

- **Condizionamento basso (~10⁶):** piccoli errori nei dati → piccoli errori
  nei coefficienti. Il nostro caso ✅.
- **Condizionamento alto (>10⁸):** piccoli errori → grandi errori nei coefficienti.
  In quel caso si usa la **regolarizzazione Ridge** — un trucco matematico che
  introduce un piccolo "peso" artificiale per stabilizzare il calcolo.

---

### Cosa ci dicono i plot

**Actual vs Predicted:** se tutti i punti stanno sulla linea diagonale
tratteggiata (y = ŷ), il modello è perfetto. Più i punti si avvicinano
alla linea, meglio il modello funziona.

**Residui:** i residui (differenza tra valore reale e previsto) devono essere
"rumore casuale" — distribuiti casualmente sopra e sotto lo zero, senza pattern.
Se ci fosse un pattern (es. residui sempre positivi per valori bassi di ŷ),
il modello starebbe sbagliando sistematicamente.

**Coefficienti:** il bar chart mostra quanto ogni gene contribuisce alla
previsione. Un gene con coefficiente alto (in valore assoluto) influenza
molto il target. Un gene con coefficiente vicino a zero contribuisce poco
e potrebbe essere rimosso senza perdere accuratezza.
