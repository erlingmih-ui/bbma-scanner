"""Sélection des paires depuis une liste prédéfinie, propre à chaque utilisateur.

Isolation :
  - Les choix sont dans st.session_state : ils ne concernent que la session de navigateur courante.
  - Si l'hébergeur fournit un utilisateur connecté (st.user.email), le choix est aussi enregistré
    dans user_pairs/<empreinte de l'e-mail>.json, et retrouvé à la prochaine visite.
  - Aucune variable de module n'est modifiée : PAIR_CHOICES et DEFAULT_LIST restent fixes pour tous.
Les paires par défaut ne changent jamais ; « Réinitialiser » restaure ces 12 paires pour l'utilisateur.
"""
import hashlib
import json
import logging
import uuid
from pathlib import Path

import streamlit as st

log = logging.getLogger("bbma.pairs")

# Les 12 paires par défaut (identiques à DEFAULT_PAIRS de app.py). Lecture seule.
DEFAULT_LIST = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "XAUUSD", "BTCUSD",
)

# Paires supplémentaires. Format yfinance identique aux paires par défaut (PAIRE=X) ;
# vérifiez-les avec check_pairs.py avant de les proposer.
EXTRA_LIST = (
    "AUDJPY", "CHFJPY", "CADJPY", "NZDJPY",
    "EURAUD", "EURCAD", "EURCHF", "GBPAUD", "GBPCHF",
    "XAGUSD",
)

PAIR_CHOICES = DEFAULT_LIST + tuple(p for p in EXTRA_LIST if p not in DEFAULT_LIST)
MAX_PAIRS = 12

USER_DIR = Path(__file__).resolve().parent / "user_pairs"

CHECK_KEY = "pairs_use_list"     # case « Choisir dans la liste »
WIDGET_KEY = "pairs_widget"      # multiselect (effacé par Streamlit quand la case est décochée)
SAVED_KEY = "pairs_saved"        # choix de l'utilisateur pour la session (survit au décochage)
ID_KEY = "pairs_user_id"         # empreinte de l'utilisateur, ou None si non connecté
RESET_KEY = "pairs_reset"        # bouton de réinitialisation


# --- Identité et stockage ----------------------------------------------------------------

def _user_id():
    """Empreinte stable de l'utilisateur connecté, ou None si l'hébergeur ne fournit pas d'e-mail."""
    if ID_KEY not in st.session_state:
        uid = None
        try:
            email = getattr(getattr(st, "user", None), "email", None)
            if email and "@" in str(email):
                uid = hashlib.sha256(str(email).strip().lower().encode("utf-8")).hexdigest()
        except Exception:
            uid = None
        st.session_state[ID_KEY] = uid
    return st.session_state[ID_KEY]


def _path(uid):
    return USER_DIR / f"{uid}.json"


def _load_saved(uid):
    """Paires enregistrées pour cet utilisateur (liste, éventuellement vide), ou None si rien."""
    if not uid:
        return None
    try:
        data = json.loads(_path(uid).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning("choix de paires illisible, défauts utilisés : %s", e)
        return None
    if not isinstance(data, list):
        return None
    return [p for p in data if isinstance(p, str) and p in PAIR_CHOICES][:MAX_PAIRS]


def _write_saved(uid, pairs):
    """Écriture atomique : un fichier temporaire unique, puis remplacement."""
    if not uid:
        return
    try:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        path = _path(uid)
        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(list(pairs)), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        log.warning("enregistrement des paires impossible : %s", e)


def _delete_saved(uid):
    if not uid:
        return
    try:
        _path(uid).unlink(missing_ok=True)
    except Exception as e:
        log.warning("suppression du choix de paires impossible : %s", e)


# --- Callbacks ---------------------------------------------------------------------------

def _save():
    """Appelé quand l'utilisateur modifie sa sélection."""
    pairs = list(st.session_state[WIDGET_KEY])
    st.session_state[SAVED_KEY] = pairs
    _write_saved(_user_id(), pairs)


def _reset():
    """Restaure les paires par défaut pour cet utilisateur uniquement."""
    st.session_state[SAVED_KEY] = list(DEFAULT_LIST)
    st.session_state[WIDGET_KEY] = list(DEFAULT_LIST)
    _delete_saved(_user_id())


# --- Interface ---------------------------------------------------------------------------

def selector(text_pairs):
    """Retourne la liste de paires à scanner pour cet utilisateur.

    text_pairs : paires lues dans la zone de texte existante.
    Case décochée : text_pairs est renvoyée telle quelle (comportement d'origine).
    Case cochée : choix multiple limité à MAX_PAIRS. Ordre de préremplissage :
    choix enregistré de l'utilisateur, puis paires de la zone de texte, puis défauts.
    """
    use_list = st.sidebar.checkbox(
        "Choisir dans la liste", key=CHECK_KEY,
        help="Remplace la saisie libre par une sélection parmi les paires proposées. Votre choix ne concerne que vous.")
    if not use_list:
        return text_pairs

    options = list(PAIR_CHOICES) + [p for p in text_pairs if p not in PAIR_CHOICES]

    # Première ouverture de la session : charger le choix enregistré, sinon zone de texte, sinon défauts.
    if SAVED_KEY not in st.session_state:
        saved = _load_saved(_user_id())
        if saved is not None:
            initial = saved
        else:
            initial = [p for p in text_pairs if p in options][:MAX_PAIRS] or list(DEFAULT_LIST)
        st.session_state[SAVED_KEY] = [p for p in initial if p in options]

    # Filtrage à chaque run : la zone de texte peut changer les options entre deux exécutions.
    st.session_state[SAVED_KEY] = [p for p in st.session_state[SAVED_KEY] if p in options]
    if WIDGET_KEY not in st.session_state:
        st.session_state[WIDGET_KEY] = list(st.session_state[SAVED_KEY])
    else:
        st.session_state[WIDGET_KEY] = [p for p in st.session_state[WIDGET_KEY] if p in options]

    st.sidebar.button("Réinitialiser (paires par défaut)", key=RESET_KEY, on_click=_reset)
    return st.sidebar.multiselect(
        f"Paires disponibles (max {MAX_PAIRS})", options,
        max_selections=MAX_PAIRS, key=WIDGET_KEY, on_change=_save)
