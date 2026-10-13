"""Vérifie que chaque paire proposée dans bbma_pairs est téléchargeable depuis yfinance.

Usage (dans le dossier du scanner, accès internet requis) :
    python check_pairs.py

Ne lit ni n'écrit le cache disque du scanner. Code de sortie 1 si une paire échoue.
"""
import sys

import bbma_core as core
import bbma_pairs as bp


def main():
    failed = []
    for pair in bp.PAIR_CHOICES:
        df = core._history(pair, "1d", "1mo")          # même appel que le scanner, sans cache
        if df is not None and len(df):
            print(f"OK     {pair:8s} {core.sym(pair):12s} {len(df)} bougies D1, dernière : {df.index[-1]:%Y-%m-%d}")
        else:
            print(f"ÉCHEC  {pair:8s} {core.sym(pair):12s} aucune donnée")
            failed.append(pair)
    print()
    if failed:
        print(f"{len(failed)} paire(s) à retirer de bbma_pairs.py : {', '.join(failed)}")
        return 1
    print(f"Toutes les paires ({len(bp.PAIR_CHOICES)}) sont téléchargeables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
