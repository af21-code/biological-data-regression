"""
plot_utils.py
=============
Funzioni di utilità condivise per la generazione dei plot.

Usato da: linear_regression.py, multivariate.py.
Centralizza la configurazione estetica dei grafici (tema scuro, salvataggio PNG)
per evitare duplicazione di codice tra i moduli.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def base_fig(figsize=(10, 6)):
    """
    Crea una coppia (fig, ax) con il tema scuro del progetto.

    Configurazione
    --------------
    - Sfondo figura  : #0f1117  (quasi nero)
    - Sfondo assi    : #1a1d27  (grigio scurissimo)
    - Bordi (spines) : #333344  (grigio-blu tenue)
    - Tick labels    : bianchi

    Parameters
    ----------
    figsize : tuple, optional
        Dimensioni della figura (larghezza, altezza) in pollici.

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.tick_params(colors="white")
    return fig, ax


def save_or_show(fig, output_path):
    """
    Salva il plot come PNG ad alta risoluzione oppure lo mostra a schermo.

    Parameters
    ----------
    fig         : Figure — la figura matplotlib da salvare/mostrare.
    output_path : str o None — se specificato, salva in PNG; altrimenti plt.show().
    """
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Plot salvato: {output_path}")
    plt.show()
