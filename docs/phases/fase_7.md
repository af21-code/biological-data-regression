# Fase 7 — Teoria e Consolidamento Finale

**Obiettivo:** Chiudere il progetto con rigore documentale e teorico. Spiegazione approfondita del background matematico per un'esposizione accademica del lavoro svolto, utile a tutto il team.

---

## Parte 1 — Spiegazione Tecnica (Background Accademico)

Il corso di Analisi Numerica per Dati Biologici richiede la comprensione delle motivazioni e delle criticità che governano i modelli numerici per la regressione, prestando attenzione alle differenze matematiche tra i vari approcci. Di seguito la rassegna dei concetti portanti esplorati e implementati.

### 1.1 Metodo dei Minimi Quadrati Ordinari (OLS) e Derivazione

Il problema della regressione continua è la ricerca del vettore dei parametri $\boldsymbol{\theta}$ che minimizza la funzione di costo quadratica (scarti al quadrato).

Nel caso di una **singola variabile predittiva** (Fase 2, $y = ax + b$), il sistema fornisce formule chiuse per i coefficienti, scalari.
$$a = \frac{Cov(x, y)}{Var(x)}, \quad b = \bar{y} - a\bar{x}$$

Nel **modello multilineare** (Fase 5, $\mathbf{y} = X_b \boldsymbol{\theta}$), il minimo della funzione costo $S(\boldsymbol{\theta}) = \|\mathbf{y} - X_b\boldsymbol{\theta}\|_2^2$ si trova ponendo a zero il gradiente rispetto a $\boldsymbol{\theta}$. Questo conduce alle **Equazioni Normali**:
$$X_b^\top X_b \boldsymbol{\theta} = X_b^\top \mathbf{y}$$

### 1.2 Stabilità Numerica e Condizionamento Matricale

Nel codice abbiamo risolto il sistema matriciale utilizzando `numpy.linalg.solve(A, b_vec)` anziché moltiplicare per l'inversa `numpy.linalg.inv(A) @ b_vec`. 
**Perché?** L'inversione esplicita di una matrice è un'operazione instabile che accumula un enorme errore di arrotondamento macchina quando la matrice è mal condizionata. L'algoritmo di `solve`, basato sulla fattorizzazione LU con pivoting parziale, risulta più rapido e infinitamente più robusto dal punto di vista dell'Analisi Numerica.

Il **Numero di Condizionamento** $\kappa$ di una matrice $A$ definisce quanto l'errore sui dati iniziali venga amplificato dalla risoluzione del sistema.
Nel nostro progetto per la Fase 5, abbiamo calcolato $\kappa(X_b^\top X_b)$. Essendo $\sim 4 \times 10^6$, il sistema risulta accettabilmente stabile e non richiede regolarizzazioni aggiuntive (come il Metodo Tikhonov/Ridge). In caso di $\kappa > 10^8$, avremmo un grave problema di multicollinearità tra le feature.

### 1.3 Apprendimento "Globale" vs. Modelli a "Tratti"

Nel progetto, la regressione implementata è **globale**, in quanto un unico modello analitico (la retta o l'iperpiano) viene applicato su tutto il dominio dei dati.
In scenari in cui i dati manifestano trend fortemente non-lineari o discontinuità, i modelli globali causano un *underfitting* o soffrono l'effetto Runge per polinomi di alto grado. Alternative studiate nell'analisi numerica prevedono le **Spline** o le interpolazioni a tratti (piecewise interpolation), che riducono i problemi di oscillazione usando polinomi di grado basso (es. grado 3, cubic splines) ma definiti su piccoli intervalli del dominio. Data l'elevata correlazione lineare osservata per la DMD ($R^2 > 0.99$), il modello lineare globale si è rivelato la scelta tecnicamente ottima e parsimoniosa.

### 1.4 L'Importanza della Validazione (Train/Test Split e Overfitting)

La differenza documentata tra la regressione semplice e multilineare è il fondamento della statistica moderna (il c.d. *Bias-Variance Tradeoff*). 
Mentre l'algoritmo matematico tende sempre a minimizzare l'errore sui punti forniti (arrivando a un R² $\approx 1$ nel caso multilineare in Fase 5), solo la valutazione su un **Test Set** isolato svela l'incapacità del modello sovra-parametrizzato di generalizzare il suo apprendimento su nuovi pazienti reali. Meno parametri ma ben scelti equivalgono spesso a predizioni cliniche più robuste.

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

### Riassunto Globale del Progetto: Cosa portiamo a casa

Arrivati alla Fase 7, possiamo guardarci alle spalle e capire esattamente la grandezza e lo scopo di ciò che abbiamo costruito, passo dopo passo:

1. **Non ci siamo fidati delle macchine "magiche"**: Esistono programmi gratuiti e librerie (come Scikit-Learn) che in tre righe di codice fanno tutto quello che abbiamo fatto noi. Ma questo progetto d'esame richiedeva di **aprire la scatola nera**. Noi abbiamo costruito l'intelligenza predittiva scrivendo le formule matematiche e le frazioni partendo dalle fondamenta.
2. **Abbiamo tradotto la biologia in algebra**: La malattia studiata, la distrofia muscolare (DMD), si presenta nel nostro progetto non come cellule o tessuti, ma come **vettori numerici e matrici**. Capire la gravità di una patologia basandosi su un gene diventa la ricerca di una "retta che passa in mezzo a dei puntini".
3. **Abbiamo dimostrato i limiti della matematica**: Abbiamo inserito nel programma dei concetti di estrema cautela:
   - **Estrapolazione:** Abbiamo spiegato che la retta non può prevedere cose fuori dal campo dell'osservazione (Fase 4). Come dire: "Sappiamo quanto piove a gennaio e a marzo, ma non possiamo usare questa logica per prevedere il tempo di luglio".
   - **Overfitting:** Abbiamo dimostrato che avere "tanti dati e tanti geni" per prevedere il target può in realtà confondere il modello (Fase 5), se non abbiamo abbastanza pazienti per sostenere quei calcoli complessi. Meglio un solo gene predittore ma buono!
   - **Gli Outlier:** Abbiamo dimostrato quanto anche un singolo "paziente anomalo" possa far sballare tutto, e abbiamo scritto una logica in grado di scovarlo e chiederci se vogliamo espellerlo dalla statistica (Fase 6).

### In Conclusione per il nostro Team
Abbiamo costruito un intero laboratorio numerico che dalla riga di comando ci permette di prendere qualsiasi dataset clinico simile, analizzarlo, pulirlo e calcolarne lo sviluppo. Il risultato è **100% riproducibile** e validato, pronto per la nostra presentazione e per dimostrare una comprensione profonda della materia ai docenti del corso.
