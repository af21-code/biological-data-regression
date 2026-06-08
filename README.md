# Analisi Numerica e Modelli di Regressione per la Stima Continua del Decadimento Muscolare

## 📌 Contesto Accademico
Questo repository contiene il codice e la documentazione sviluppati per il **Progetto B: Analisi dati e Regressione - output continuo**.
Il lavoro è stato realizzato come prova d'esame per il **Corso di Analisi Numerica per l'Analisi dei Dati Biologici - LT in Biotecnologie per la salute**, tenuto dai docenti Francesco Calabrò e Luisa D'Amore.

## 🎯 Obiettivo del Progetto
L'obiettivo principale del progetto è lo sviluppo e la validazione matematica di modelli di regressione lineare e multilineare per prevedere l'andamento di parametri clinici continui. Partendo dall'esplorazione statistica dei dati biologici, il sistema implementa algoritmi di predizione valutandone la robustezza tramite metriche di errore e rigide procedure di partizionamento dei dati.

## 🧬 Tema Clinico: La Distrofia Muscolare
Il dominio applicativo scelto per l'analisi è la **Distrofia Muscolare**.

**Validazione della scelta:**
La distrofia muscolare si presta perfettamente ai requisiti analitici del progetto per due ragioni fondamentali:
1. **Natura Continua dell'Output:** La progressione della patologia è quantificabile attraverso biomarcatori e parametri clinici strettamente continui (es. concentrazione di creatinchinasi nel siero, punteggi di forza muscolare, espressione proteica). Questo soddisfa pienamente il vincolo progettuale che richiede l'analisi e la stima di un output continuo tramite la costruzione di modelli $y=ax+b$.
2. **Potenziale Predittivo e Applicativo:** La modellazione matematica di questi parametri non è un puro esercizio numerico, ma risponde all'esigenza biotecnologica di monitorare il decadimento muscolare. Stimare la progressione della malattia tramite modelli di regressione permette di valutare l'efficacia di potenziali terapie in fase di trial o di anticipare criticità cliniche.

## 🛠️ Architettura e Sviluppo dell'Analisi
Il codice è strutturato seguendo una rigorosa pipeline numerica, in risposta alle direttive del corso:

*   **Esplorazione e Analisi 0-dimensionale:** Analisi statistica descrittiva (massimo, minimo, media, mediana), calcolo della varianza, estrazione di intervalli di confidenza al 90% e confronto visivo con distribuzioni gaussiane.
*   **Selezione delle Feature e Regressione Lineare:** Calcolo della matrice di covarianza per l'individuazione delle variabili maggiormente correlate e costruzione della retta di regressione.
*   **Machine Learning Base (Train/Test Split):** Partizionamento del dataset (90% training, 10% test) per l'addestramento del modello e la validazione oggettiva basata sul calcolo degli scarti predittivi.
*   **Interpolazione ed Estrapolazione:** Utilizzo del modello per inferire punti non presenti nel dataset originario (es. predizione del valore di output per un predittore pari a 0).
*   **Estensione Multivariata:** Costruzione di un modello multilineare sfruttando tutte le colonne di dati continui disponibili per ottimizzare la stima.
*   **Data Cleaning Dinamico:** Implementazione di un algoritmo interattivo per l'individuazione e l'eliminazione dei dati anomali (outlier) basato sullo scostamento dal modello lineare.

## 💻 Requisiti e Utilizzo
*(In questa sezione, da compilare a fine sviluppo, verranno indicate le librerie necessarie, es. `numpy`, `pandas`, `matplotlib`, e i comandi per eseguire lo script).*