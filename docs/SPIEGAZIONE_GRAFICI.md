# Spiegazione Dettagliata dei Grafici

L'esecuzione del programma genererà (a schermo e all'interno della cartella `data/processed/plots/`) diversi grafici. Poiché in sede d'esame vi potrà essere chiesto di argomentare le vostre scelte visive e i risultati, questo file contiene una spiegazione accurata, dal punto di vista matematico e logico, di ogni plot generato.

---

## FASE 1 — Esplorazione dei Dati

### 1. `bar_stats.png` (Bar plot degli indicatori)
*   **Cosa mostra**: Per ciascuna delle 11 colonne (i geni analizzati), visualizza in un istogramma a barre il valore della Media, Mediana e Deviazione Standard.
*   **Spiegazione da dare all'esame**: "Questo grafico serve per un'analisi di base detta *0-dimensionale*. Ci permette di capire a colpo d'occhio l'ordine di grandezza dei dati (che in log2 vanno circa da 2 a 15) e ci conferma visivamente, paragonando la barra della Media con quella della Mediana, che non vi sono forti asimmetrie nel dataset (infatti media e mediana sono quasi sempre allo stesso livello)."

### 2. `boxplots.png` (Box plot affiancati)
*   **Cosa mostra**: I boxplot di tutte le feature misurate nel dataset, disposte orizzontalmente.
*   **Spiegazione da dare all'esame**: "Il boxplot è lo strumento visivo perfetto per mostrare i *quartili* di una distribuzione. La scatola centrale contiene il 50% dei dati (dal 25° al 75° percentile). La riga in mezzo alla scatola è la mediana. I baffi (whisker) ci fanno capire fino a dove si estendono i dati considerati normali, mentre eventuali pallini solitari fuori dai baffi indicano potenziali *outlier*. È servito per confermare che le varianze (le ampiezze delle scatole) differiscono molto tra un gene e l'altro."

### 3. `ecdf_gaussian.png` (ECDF contro CDF Gaussiana)
*   **Cosa mostra**: Per le singole feature c'è una curva a gradini blu (la funzione di ripartizione empirica ECDF calcolata sui 22 campioni veri) e una curva liscia arancione (la CDF teorica di una Gaussiana con la stessa media e varianza).
*   **Spiegazione da dare all'esame**: "Questo grafico è una verifica fondamentale dell'Analisi Numerica. Volevamo chiederci: *questi dati biologici seguono una distribuzione Normale (Gaussiana)?* Sovrapponendo la ripartizione reale (ECDF) a quella della Gaussiana teorica ideale, notiamo come le due curve siano strettamente aderenti. Questo giustifica matematicamente e statisticamente l'uso successivo dei *Minimi Quadrati* e di modelli lineari, le cui assunzioni lavorano bene su dati normalmente distribuiti."

### 4. `histograms.png` (Istogrammi delle variabili)
*   **Cosa mostra**: Per ogni variabile continua, un istogramma della distribuzione di frequenza con sovrapposta la curva di densità gaussiana calcolata con media e deviazione standard esplicite.
*   **Spiegazione da dare all'esame**: "Gli istogrammi complementano l'analisi ECDF mostrando la *forma* della distribuzione. La curva gaussiana sovrapposta (arancione) permette di valutare visivamente simmetria e modalità. A differenza dell'ECDF che è una CDF cumulata, l'istogramma mostra direttamente dove si concentrano le osservazioni."

### 5. `covariance_heatmap.png` (Heatmap della matrice di covarianza)
*   **Cosa mostra**: Una mappa cromatica della matrice di covarianza $C = (1/n) X_c^T X_c$, con annotazioni numeriche in ogni cella e colormap divergente (rosso per valori positivi, blu per negativi).
*   **Spiegazione da dare all'esame**: "La heatmap della covarianza è cruciale perché mostra le *relazioni lineari* tra ogni coppia di variabili. I colori intensi (rosso o blu) indicano forte covarianza positiva o negativa. La diagonale contiene le varianze. Da questa matrice selezioniamo il predittore con covarianza massima rispetto al target per la Fase 2."

### 6. `correlation_heatmap.png` (Heatmap della matrice di correlazione)
*   **Cosa mostra**: Simile alla heatmap di covarianza, ma normalizzata: $R_{ij} = C_{ij} / (\sigma_i \cdot \sigma_j)$. I valori sono compresi in [-1, +1].
*   **Spiegazione da dare all'esame**: "La correlazione di Pearson è la versione adimensionale della covarianza. Mentre la covarianza dipende dalla scala delle variabili (un gene espresso a livelli alti avrà covarianze più grandi in valore assoluto), la correlazione è sempre in [-1, 1], permettendo un confronto equo. Un valore di |r| > 0.8 indica una forte relazione lineare."

---

## FASE 2 e 3 — Regressione Semplice e Train/Test

### 7. `phase2_regression.png` (Scatter plot base + Retta)
*   **Cosa mostra**: I 22 pazienti distribuiti su assi x/y (x = gene predittore migliore, y = gene target). Attraverso la nube di punti blu, passa la retta OLS arancione calcolata con l'algebra matriciale.
*   **Spiegazione da dare all'esame**: "È la traduzione visiva dell'equazione $y = ax + b$. Dimostra visivamente che il predittore selezionato con la matrice di covarianza intercetta perfettamente il trend lineare del target. I punti non sono disordinati, ma molto aderenti alla retta, coerente con il nostro $R^2 \approx 0.99$."

### 8. `phase3_train_test_boxplot.png` (Boxplot Split)
*   **Cosa mostra**: Due boxplot affiancati per la variabile Target, uno per il gruppo di addestramento (Train) e uno per il collaudo (Test).
*   **Spiegazione da dare all'esame**: "Prima di fare le verifiche sul modello, dovevamo accertarci di aver separato i campioni in modo equo. I due boxplot dimostrano che il Test Set non è finito a campionare solo casi gravi o solo casi lievi: il suo boxplot è grosso modo allineato come altezze con quello del Training Set, dimostrando che la suddivisione stocastica (con `seed=42`) ha generato un set di prova rappresentativo della popolazione."

### 9. `phase3_train_test_scatter.png` (Regressione Train vs Test)
*   **Cosa mostra**: Simile al grafico di Fase 2, ma qui i punti del Test Set sono evidenziati con diamanti rossi.
*   **Spiegazione da dare all'esame**: "Qui mostriamo visivamente il concetto di Validazione. La retta è stata calcolata usando *soltanto* i punti blu (Training). Dopodiché l'abbiamo proiettata sui punti rossi (Test). I punti rossi cadono fisiologicamente a pochissima distanza dalla retta, dimostrando che il modello generalizza correttamente e non c'è *Overfitting*."

---

## FASE 4 — Interpolazione ed Estrapolazione

### 10. `phase4_interpolation.png` (Plot interattivo e previsioni)
*   **Cosa mostra**: È la retta di regressione, estesa ai suoi estremi, costellata di punti con simboli diversi: un quadrato, un triangolo, una X e una stella.
*   **Spiegazione da dare all'esame**: "Qui abbiamo mostrato le operazioni sul modello completato.
    *   Il **triangolo verde** mostra l'interpolazione nel punto medio.
    *   Il **quadrato rosa** mostra il risultato del "problema inverso" (dato un target medio $\bar{y}$, calcola l'incognita $x^*$).
    *   La **X arancione** è forzata a x=0: ci fa vedere come la retta esca drasticamente dai confini biologici osservati per calcolare l'intercetta matematica.
    *   La **Stella rossa** rappresenta l'input dinamico che abbiamo chiesto all'utente da terminale.

---

## FASE 5 — Modello Multivariato

### 11. `phase5_actual_vs_predicted.png` (Valori Reali vs Previsti)
*   **Cosa mostra**: L'asse X riporta i valori veri, l'asse Y le previsioni. C'è una linea tratteggiata diagonale perfetta.
*   **Spiegazione da dare all'esame**: "Nel modello con 10 feature (multilineare) non possiamo tracciare una retta in uno spazio a 11 dimensioni. Dunque l'unico modo per vedere l'accuratezza è plottare $y_{reale}$ contro $y_{stimato}$. I punti blu del Training sono perfettamente *schiacciati* sulla diagonale ideale ($R^2 \approx 0.9996$). I punti rossi del Test (essendo in *Overfitting* dovuto alla sovra-parametrizzazione) appaiono invece un po' più scostati dalla diagonale."

### 12. `phase5_residuals.png` (Grafico dei Residui)
*   **Cosa mostra**: L'asse X mostra i valori stimati, l'asse Y mostra l'errore $e$ (scarto tra reale e stimato). La riga dello 0 centrale è in evidenza.
*   **Spiegazione da dare all'esame**: "L'analisi numerica insegna che per far sì che un modello di regressione OLS sia valido, i residui devono essere rumore bianco. Non devono avere strutture (tipo a imbuto, 'eteroschedasticità', o a onda). I nostri punti appaiono distribuiti casualmente sopra e sotto la linea dello Zero (asse x), senza pattern evidenti, confermando le tesi dell'OLS."

### 13. `phase5_coefficients.png` (Bar chart dei Coefficienti)
*   **Cosa mostra**: Barre blu (positive) e rosse (negative) per ognuno dei 10 geni.
*   **Spiegazione da dare all'esame**: "Una volta calcolato il vettore $\theta$, l'abbiamo plottato. Questo grafico è cruciale perché mostra quali geni "pesano" di più nella stima del target. Permette di individuare eventuali segnali di multicollinearità — coefficienti molto grandi in valore assoluto possono indicare instabilità numerica nel modello non regolarizzato."

### 14. `phase5_residual_histogram.png` (Istogramma dei Residui)
*   **Cosa mostra**: La distribuzione dei residui del modello multilineare, con la curva gaussiana sovrapposta e i residui separati per training (blu) e test (rosso).
*   **Spiegazione da dare all'esame**: "Questa è la verifica dell'assunzione fondamentale dell'OLS: i residui devono essere approssimativamente normali con media zero. L'istogramma mostra se questa assunzione è soddisfatta. Se la distribuzione fosse fortemente asimmetrica o bimodale, il modello OLS non sarebbe appropriato."

### 15. `phase5_model_comparison.png` (Confronto Modelli)
*   **Cosa mostra**: Un grafico a barre raggruppate che confronta MSE, RMSE e R² della regressione semplice vs multilineare, sia su training che su test set.
*   **Spiegazione da dare all'esame**: "Questo è il grafico riepilogativo dell'intero progetto. Mostra che il modello multilineare con 10 feature ottiene un fit quasi perfetto sul training (R² ≈ 1), ma il confronto sul test set rivela che il miglioramento rispetto alla regressione semplice è minore del previsto — segnale classico di overfitting dovuto al basso rapporto n/p (22 campioni / 10 feature)."

---

## FASE 6 — Pulizia Outlier

### 16. `phase6_outlier_clean.png` (Modello finale dopo pulizia)
*   **Cosa mostra**: Lo scatter plot finale con i soli punti rimasti dopo la rimozione degli outlier, con la nuova retta di regressione ricalcolata. Un riquadro in alto a sinistra elenca i campioni rimossi.
*   **Spiegazione da dare all'esame**: "Mostra il risultato finale del loop di pulizia interattivo. I campioni rimossi erano quelli con il residuo assoluto più alto (i più distanti dalla retta). La nuova retta, calcolata esclusivamente sui dati puliti, è più rappresentativa del trend biologico reale, senza essere distorta dalla *Leverage* (forza di leva) dei punti anomali."
