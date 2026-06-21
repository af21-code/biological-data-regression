# Dashboard Analisi Numerica — Come avviare

La dashboard è composta da un singolo file HTML statico (`index.html`) che legge in automatico le immagini salvate nella cartella `data/processed/plots/`.

Ci sono due modi per visualizzarla:

## Metodo 1 (Consigliato): Server Locale Python
Eseguire un server locale ti garantisce che il browser carichi correttamente tutte le immagini senza incappare in blocchi di sicurezza locali (CORS restrictions) del browser.

Apri il terminale dalla **radice del progetto** (la cartella principale `biological-data-regression`) ed esegui:
```bash
# Usa python3 se "python" non viene trovato sul tuo Mac
python3 -m http.server 8000
```

*Se ti trovi già all'interno della cartella `dashboard`, torna prima alla radice con questo comando:*
```bash
cd ../../ && python3 -m http.server 8000
```

Dopodiché apri il tuo browser all'indirizzo:
[http://localhost:8000/docs/dashboard/index.html](http://localhost:8000/docs/dashboard/index.html)

> Per spegnere il server, ti basterà premere `CTRL + C` nel terminale.

## Metodo 2: Apertura Diretta
Se stai usando Safari o Firefox, o se Chrome non blocca le immagini locali, puoi semplicemente fare doppio click sul file `index.html` per aprirlo col browser predefinito. 

Nel terminale (su Mac/Linux) puoi eseguire:
```bash
open docs/dashboard/index.html
```

---

*Nota: Assicurati di aver generato i grafici eseguendo prima la pipeline (ad es. tramite `NONINTERACTIVE=1 python main.py`) prima di avviare la dashboard, altrimenti le immagini risulteranno vuote.*
