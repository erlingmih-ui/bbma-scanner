"""Codes et paramètres de la colonne « Qualité ».

Modifier ici uniquement les valeurs. La logique de calcul est dans bbma_quality.py.
Les points maximum par critère doivent totaliser 100 : la vérification en bas du fichier
le contrôle à chaque import.
"""

# --- Périodes des indicateurs ---
MID_PERIOD = 20              # période de la MID BB (moyenne des Bandes de Bollinger)
ATR_PERIOD = 14              # période de l'ATR (normalisation de la distance)

# --- Points maximum par critère (somme = 100) ---
MAX_CONFLUENCE = 30          # part des TF scannés dans le même sens que l'entrée
MAX_SIGNAL = 25              # type de signal d'entrée
MAX_RSI = 15                 # RSI dans le sens du trade
MAX_CHASE = 20               # anti-poursuite (distance à la MID BB en ATR)
MAX_D1 = 10                  # accord avec le filtre D1 (EMA50)

# --- Signal d'entrée : points par code (chaque valeur <= MAX_SIGNAL) ---
CODE_WEIGHT = {"RE": 25, "MHV": 22, "CSAK": 18, "EXT": 12, "EXM": 10, "MOM": 5}

# --- RSI (du point de vue du trade : élevé = favorable au sens) ---
RSI_ZONE_OVERBOUGHT = 70     # au-dessus : suracheté dans le sens du trade
RSI_ZONE_PULLBACK = 50       # en dessous : marge de mouvement, pullback
RSI_POINTS = {"overbought": 0, "neutral": 8, "pullback": MAX_RSI}
RSI_MISSING = 7              # RSI indisponible : demi-points

# --- Anti-poursuite : distance signée close -> MID BB, en ATR ---
# Paliers (seuil en ATR, points) : premier seuil atteint = points donnés.
# Au-delà du dernier seuil : 0 point. Prix du mauvais côté de la MID BB : CHASE_BELOW.
CHASE_STEPS = ((0.5, 20), (1.5, 12), (2.5, 5))
CHASE_BELOW = 15
CHASE_MISSING = 5            # distance indisponible : demi-points

# --- Filtre D1 ---
D1_AGREE = MAX_D1            # D1 dans le même sens que l'entrée
D1_UNKNOWN = 4               # D1 indisponible
D1_DISAGREE = 0              # D1 dans le sens contraire

# --- Seuils des grades : score >= valeur ; sinon grade C ---
GRADE_MIN = {"A": 75, "B": 55}

# --- Libellés affichés dans l'infobulle ---
QUAL_LABELS = {"confluence": "Confluence", "signal": "Signal", "rsi": "RSI",
               "chase": "Anti-poursuite", "d1": "D1"}


# --- Vérification de cohérence (exécutée à l'import) ---
assert MAX_CONFLUENCE + MAX_SIGNAL + MAX_RSI + MAX_CHASE + MAX_D1 == 100, "les critères doivent totaliser 100"
assert max(CODE_WEIGHT.values()) <= MAX_SIGNAL, "un poids de signal dépasse MAX_SIGNAL"
assert max(RSI_POINTS.values()) <= MAX_RSI, "un point RSI dépasse MAX_RSI"
assert max(p for _, p in CHASE_STEPS) <= MAX_CHASE and CHASE_BELOW <= MAX_CHASE, "anti-poursuite hors plafond"
assert D1_AGREE <= MAX_D1, "D1_AGREE dépasse MAX_D1"
assert GRADE_MIN["A"] > GRADE_MIN["B"], "seuil A doit être supérieur au seuil B"
