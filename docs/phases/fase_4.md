# Fase 4 — Interpolazione ed Estrapolazione

**File di riferimento:** `src/linear_regression.py`
**Funzione principale:** `run_phase4(df, predictor_col, target_col, a, b, plots_dir)`
**Modello usato:** coefficienti $a$, $b$ addestrati sul **training set** (Fase 3)

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

Una volta costruita la retta $y = ax + b$, è importante distinguere due usi fondamentalmente diversi del modello:

| Tipo | Definizione | Affidabilità |
|---|---|---|
| **Interpolazione** | Previsione per $x \in [x_{\min}, x_{\max}]$ (dentro il range osservato) | Alta ✅ |
| **Estrapolazione** | Previsione per $x \notin [x_{\min}, x_{\max}]$ (fuori dal range osservato) | Incerta ⚠️ |

Il modello lineare assume che la relazione $y = ax + b$ sia valida
*ovunque* — ma questa assunzione è empiricamente fondata solo nel range
dei dati con cui è stato addestrato.

---

### Dati del modello (da Fase 3 — training set)

$$y = 1.1437 \cdot x + (-1.9550)$$

**Range osservato del predittore** `206717_at`:

$$x \in [4.1782,\ 14.7482]$$

---

### Step 4.1 — Interpolazione nel Punto Medio delle Ascisse

**Calcolo:**

$$x_{\text{mid}} = \frac{x_{\min} + x_{\max}}{2} = \frac{4.1782 + 14.7482}{2} = 9.4632$$

$$\hat{y}(x_{\text{mid}}) = 1.1437 \cdot 9.4632 + (-1.9550) = 8.8666$$

**Verifica empirica:** cerchiamo i campioni con $|x_i - x_{\text{mid}}| \leq 10\%$ del range
(tolleranza = $0.10 \times (14.7482 - 4.1782) = 1.057$) e ne calcoliamo la media del target.
Se la media osservata è vicina alla previsione, il modello interpola correttamente.

Questo è **interpolazione**: $x_{\text{mid}}$ si trova dentro il range osservato
$[4.1782, 14.7482]$, quindi la previsione è matematicamente fondata.

---

### Step 4.2 — Inversione della Retta: Ascissa dell'Ordinata Media

**Problema inverso:** dato il valore medio del target $\bar{y}$, trovare il valore
di $x$ che il modello prevede corrispondergli.

**Formula (inversione della retta):**

$$y = ax + b \implies x = \frac{y - b}{a}$$

Applicata all'ordinata media $\bar{y}$:

$$x^* = \frac{\bar{y} - b}{a} = \frac{11.1742 - (-1.9550)}{1.1437} = \frac{13.1292}{1.1437} \approx 11.4793$$

**Verifica:** $\hat{y}(x^*) = 1.1437 \cdot 11.4793 + (-1.9550) = 11.1742 = \bar{y}$ ✅

**Interpretazione biologica:** $x^* \approx 11.48$ è il livello di espressione
del probe `206717_at` che il modello associa all'espressione media del target.
È un punto di "equilibrio" della distribuzione.

---

### Step 4.3 — Estrapolazione per x = 0

**Calcolo:**

$$\hat{y}(0) = a \cdot 0 + b = b = -1.9550$$

**Avvertimento:** $x = 0$ è fuori dal range osservato $[4.1782, 14.7482]$, quindi
questa è un'**estrapolazione**. In scala log₂, $x = 0$ corrisponderebbe a
un'intensità di fluorescenza di $2^0 = 1$ — un valore biologicamente irrealistico
per un gene (intensità minima rilevabile è tipicamente intorno a $2^4 = 16$).

> L'estrapolazione è matematicamente valida ma biologicamente non interpretabile:
> segnala il limite del modello lineare fuori dal suo dominio di validità.

---

### Step 4.4 — Input Interattivo dell'Utente

La funzione `run_phase4()` chiede all'utente di inserire un valore $x$ arbitrario,
calcola $\hat{y}(x)$ e indica se si tratta di interpolazione o estrapolazione:

```
Inserisci un valore x (range osservato: [4.1782, 14.7482]): 8.0
→ ŷ(8.0) = 1.1437·8 + (−1.9550) = 7.1946  [interpolazione ✅]

Inserisci un valore x (range osservato: [4.1782, 14.7482]): 20.0
→ ŷ(20.0) = 1.1437·20 + (−1.9550) = 20.9190  [estrapolazione ⚠️]
```

---

### Plot prodotto

**`phase4_interpolation.png`** — scatter plot con i 22 campioni osservati,
la retta di regressione estesa, e marker distinti per ogni punto speciale:

| Marker | Colore | Punto |
|---|---|---|
| ▲ triangolo verde | `#a8ff78` | $x_{\text{mid}}$ → $\hat{y}(x_{\text{mid}})$ |
| ■ quadrato rosa | `#ff78d4` | $x^*(\bar{y})$ → $\bar{y}$ |
| ✕ arancione | `#ff9f43` | $x=0$ → $\hat{y}(0) = b$ |
| ★ rosso | `#ee5a24` | Input utente → $\hat{y}(x_{\text{input}})$ |

---

### Riepilogo numerico (valori reali)

| Operazione | $x$ | $\hat{y}$ | Tipo |
|---|---|---|---|
| Punto medio ascisse | 9.4632 | 8.8666 | Interpolazione |
| Ascissa di $\bar{y}$ | 11.4793 | 11.1742 | Interpolazione |
| Estrapolazione in 0 | 0.0000 | −1.9550 | **Estrapolazione** ⚠️ |
| Input utente (x=8.0) | 8.0000 | 7.1946 | Interpolazione |

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Cosa abbiamo fatto

Nella Fase 4 abbiamo usato la retta trovata nelle fasi precedenti per fare
**previsioni su casi specifici**. La retta è già costruita — dobbiamo solo
"leggere" il suo valore per un dato x.

---

### Interpolazione: come leggere dentro il grafico

Immagina di avere un grafico con 22 punti e una retta che li attraversa.
**Interpolare** significa: scelgo un valore x che si trova *tra* i punti
esistenti, e leggo sull'asse y quanto vale la retta in quel punto.

È come fare una stima tra due misure che già conosci — affidabile,
perché sei nel territorio "coperto" dai tuoi dati.

**Esempio concreto:**
- Il punto medio del range è $x = 9.46$
- La retta in quel punto prevede $\hat{y} = 8.87$
- I campioni reali nell'intorno di 9.46 hanno valori vicini a questo → ✅

---

### Estrapolazione: andare oltre i confini

**Estrappolare** significa: scelgo un valore x che si trova *fuori* dal range
dei miei dati (ad esempio $x = 0$) e applico lo stesso modello.

Questo è pericoloso: la retta è stata costruita sui dati che vanno da 4.18
a 14.75. Non sappiamo se la relazione è ancora lineare per x=0 — potrebbe
cambiare forma completamente. È come prevedere il traffico stradale a Natale
basandosi solo sui dati di un giorno normale di agosto.

> **Analogia:** sai che ogni anno che passa cresci di 5 cm. Se hai 10 anni
> e cresci di 5 cm/anno, *estrapolando* diresti che a 30 anni sarai alto
> 200 cm. Ma il corpo smette di crescere — il modello lineare non vale più.

---

### Il problema inverso: trovare x dato y

Abbiamo anche invertito la formula per rispondere a una domanda diversa:
"*Quale valore di x produce il valore medio del target?*"

Invece di $y = ax + b$ abbiamo risolto $x = (y - b) / a$ sostituendo $y$ con
la media del target. Il risultato ($x^* \approx 11.48$) è il livello del gene
predittore che corrisponde al livello "tipico" del gene target.

---

### Il prompt interattivo

Nella versione interattiva del programma, puoi inserire qualsiasi valore di x
e il programma ti dirà immediatamente:
1. Qual è la previsione $\hat{y}$
2. Se stai interpolando (dentro il range → affidabile) o estrapolando (fuori → da prendere con cautela)

Questo strumento è utile per esplorare "cosa succederebbe se..." senza dover
riaddestrare il modello ogni volta.
