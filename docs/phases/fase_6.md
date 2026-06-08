# Fase 6 — Pulizia Outlier Dinamica

**File di riferimento:** `src/linear_regression.py`
**Funzione principale:** `run_phase6(df, predictor_col, target_col, plots_dir)`
**Dataset iniziale:** `data/raw/ncbi_muscular_dystrophy_raw.csv` (22 campioni)

---

## Parte 1 — Spiegazione Tecnica

### Obiettivo

Nel metodo dei Minimi Quadrati Ordinari (OLS), la funzione di costo da minimizzare
è la somma degli scarti al quadrato: $S = \sum e_i^2$. Poiché gli scarti sono
elevati al quadrato, i punti con errori elevati (outlier) esercitano una
forza di attrazione (leverage) asimmetrica sulla retta, trascinandola verso di sé
e peggiorando le previsioni per la maggioranza dei punti "normali".

L'obiettivo della Fase 6 è un'**eliminazione iterativa e controllata** degli
outlier basata sui residui, per studiare quanto la retta cambia rimuovendo
i campioni più anomali.

---

### Step 1 — Calcolo e Ordinamento dei Residui

A ogni iterazione $k$, il modello viene ri-addestrato sui campioni rimanenti:

$$a^{(k)}, b^{(k)} = \text{fit}(X^{(k)}, y^{(k)})$$

Vengono poi calcolati i residui assoluti per ogni campione:

$$|e_i| = |y_i - (a^{(k)}x_i + b^{(k)})|$$

I campioni vengono ordinati in ordine decrescente di $|e_i|$. Il campione in
cima alla lista è l'outlier principale (quello più distante dal modello attuale).

---

### Step 2 — Rimozione Interattiva

All'utente viene presentata in sequenza l'anomalia peggiore rilevata.
Esempio reale (Dataset NCBI GSE38417, Iterazione 1):

```
  [Iterazione 1]  n = 22 campioni
    MSE corrente = 0.053113  |  R² = 0.997140

    Il campione 'GSM941833' sembra anomalo
      x = 7.3547,  y = 7.1597,  |scarto| = 0.6632
    Vuoi rimuoverlo dal dataset? (s/n):
```

Se l'utente sceglie `s` (sì), il campione viene rimosso (dataset ridotto a 21 campioni)
e il modello viene ricalcolato all'iterazione successiva.

---

### Step 3 — Riaddestramento e Ricalcolo Metriche

Rimossi gli outlier, il modello viene addestrato sul nuovo dataset ridotto
e le metriche ricalcolate.

**Dinamica reale osservata (rimuovendo il campione GSM941833):**
- **Iterazione 1 (22 campioni):** $a = 1.136$, $b = -1.861$, MSE = 0.0531, R² = 0.9971
- **Iterazione 2 (21 campioni):** $a = 1.146$, $b = -2.005$, MSE = **0.0324**, R² = **0.9983**

Rimuovendo un solo campione anomalo, il **MSE è crollato del 39%** (da 0.053 a 0.032)
e l'R² è aumentato. La retta ha modificato leggermente pendenza e intercetta
per adattarsi meglio al resto del dataset "pulito".

Il processo si ripete finché l'utente non decide di fermarsi, permettendo
di pulire progressivamente il segnale dal rumore.

---

### Plot prodotto (Iterativo)

Ad ogni iterazione viene generato e sovrascritto il file:
`phase6_dynamic_outliers.png`

Il grafico mostra:
1. Punti blu (campioni attivi)
2. Retta blu (modello aggiornato)
3. Punti rossi (cerchiati): gli outlier appena rimossi
4. Retta grigia tratteggiata: il modello all'inizio dell'iterazione corrente (per mostrare come la retta "si sposta")

---

### Giustificazione Biologica

In dati microarray trascrittomici (come il nostro dataset NCBI GEO),
gli outlier sono frequentissimi. Cause tipiche:
1. **Artefatti tecnici:** bolle d'aria sul chip microarray, errori di ibridazione.
2. **Eterogeneità biologica:** un paziente con DMD (distrofia muscolare)
   che ha anche un'altra patologia non documentata, o un'infezione acuta in corso
   che altera i profili di espressione in modo aspecifico.

Rimuovere questi outlier (dopo averli identificati in modo puramente matematico
tramite i residui) è una pratica standard in bioinformatica prima di procedure
downstream come il differential expression analysis o il pathway enrichment.

---

---

## Parte 2 — Spiegazione Semplice

*(Per chi non conosce la programmazione o la statistica)*

---

### Cos'è un outlier e perché dà fastidio?

Immagina di voler calcolare lo stipendio medio di 10 persone normali in un bar.
La media potrebbe essere 1.500€. A un certo punto entra Elon Musk: la "nuova
media" delle 11 persone diventa improvvisamente 2 miliardi di euro.

Elon Musk è un **outlier**: un dato così estremo che "tira" verso di sé
il modello matematico, rovinando la previsione per tutti gli altri.

Nella regressione lineare, la retta cerca di stare vicina a tutti i punti.
Ma se un solo punto è sbagliato o anomalo (per esempio per un errore della
macchina da laboratorio), la retta devierà dal suo percorso ideale solo per
"accontentare" quell'unico punto sbagliato.

---

### Come li scoviamo

Il nostro programma ragiona così:
1. Traccia la retta usando tutti i 22 pazienti.
2. Guarda quale paziente è più **lontano in verticale** dalla retta.
3. Lo segnala all'utente: "Ehi, questo paziente si comporta in modo strano
   rispetto agli altri. Vuoi toglierlo?"

I pazienti più lontani dalla retta sono quelli in cui l'errore (la differenza
tra il livello del gene previsto e quello misurato) è massimo.

---

### Il processo iterativo "a matrioska"

Perché non li togliamo tutti in una volta?
Perché togliendo il primo outlier, **la retta si sposta** per tornare al suo
percorso ideale. Spostandosi, un punto che prima sembrava normale potrebbe
improvvisamente rivelarsi un outlier rispetto alla *nuova* retta.

Il processo deve essere fatto a passi:
1. Togli l'outlier più grave.
2. Ricalcoli la retta.
3. Guardi chi è *ora* l'outlier più grave.
4. Ripeti finché la retta non si "stabilizza" e l'errore medio diventa
   accettabile.

È un'operazione simile al lavaggio chirurgico: togli lo sporco evidente, sciacqui, e solo allora riesci a vedere le macchie più piccole sottostanti che prima erano coperte.
