"""Colonne « Pattern MTF » : les signaux des 3 timeframes supérieurs actifs, traduits en lettres Oma Ally.

Lettres :  R = Reentry (RE) · E = Extreme (EXM ou EXT) · M = MHV · C = CSAK · O = Momentum (MOM)
Chaque timeframe affiche sa lettre, colorée selon le sens (vert = achat, rouge = vente), son nom en dessous.
Sous les pastilles, une ligne de lecture : sens majoritaire et nombre de TF actifs (ex. « ▲ 3/3 »).
« ⇅ » signale que les TF ne vont pas tous dans le même sens.

Filtre : la colonne n'affiche QUE les patterns de PATTERN_CODES (par défaut REM, RRE, REE), lus dans l'ordre
des timeframes (D1 → H4 → H1 : la 1re lettre est le TF le plus haut). Aucun autre pattern n'est affiché : la
cellule reste vide (« – ») et l'infobulle donne la lecture actuelle. PATTERN_CODES = None : tous les patterns.
Pattern INCOMPLET : il s'affiche dès qu'au moins un signal est à la bonne place, c'est-à-dire que la lettre
d'un timeframe est celle du pattern cible à cette position (ex. D1 = R pour REM / RRE / REE ; H4 = E pour REM / REE ;
H1 = M pour REM, H1 = E pour REE). Le pattern cible retenu est celui qui compte le plus de signaux bien placés ;
la ligne de lecture le montre avec des points pour les places vides ou mal tenues (ex. « ▲ R·· », « ▼ ·E· »).
Réglages : ALLOW_PARTIAL (activer / couper), MIN_HITS (signaux bien placés minimum), PARTIAL_STRICT (écarter
le pattern si une lettre est mal placée). Sur les chips : lettre encadrée = bien placée, atténuée = mal placée.
Un pattern complet garde son affichage d'origine (« ▲ REM ») et son cadre coloré.
REQUIRE_SAME_SIDE = True : le pattern doit en plus aller dans le même sens (achat ou vente) sur les 3 TF.

Choix des 3 timeframes (PATTERN_TFS) :
  - par défaut D1, H4, H1, à condition qu'ils soient cochés dans la barre latérale ;
  - s'il en manque, on complète avec les autres timeframes cochés (du plus haut au plus bas) ;
  - PATTERN_TFS = None : on prend simplement les 3 timeframes cochés les plus hauts (ex. MN, W1, D1).

Signaux récents : si `recent` est fourni (case « Afficher les signaux récents »), un TF sans signal sur la
dernière bougie reprend son dernier signal récent (core.recent_signal, jusqu'à RECENT_BARS bougies). Il est
affiché en grisé avec son âge (« −2 ») et sa lettre en minuscule dans `letters` (ex. « CeR »). Il compte dans le
sens majoritaire, mais le cadre « tous alignés » n'est donné que si tous les signaux sont sur la dernière bougie.

Module ajouté, sans Streamlit et sans téléchargement : il relit les signaux déjà calculés par le scan
(`res` de bbma_core / bbma_nuclear). bbma_core, bbma_quality, bbma_nuclear et les autres modules ne sont
pas modifiés. Ce module n'agit ni sur la synthèse, ni sur la confiance, ni sur la qualité, ni sur le tri.

Intégration dans app.py : voir build_table (colonne), main (CSS) et legend_html (légende).
"""
import html

import bbma_core as core

# --- Paramètres -------------------------------------------------------------------------------
PATTERN_TFS = ("D1", "H4", "H1")       # None : les N timeframes cochés les plus hauts
N_TFS = 3
PATTERN_CODES = ("REM", "RRE", "REE")   # R = Reentry, E = Extreme, M = MHV ; None : aucun filtre
REQUIRE_SAME_SIDE = True                # les 3 lettres doivent aller dans le même sens (achat ou vente)
ALLOW_PARTIAL = True                    # False : seuls les patterns complets sont affichés (comportement d'origine)
MIN_HITS = 1                            # signaux bien placés minimum pour afficher un pattern incomplet (1 à 3)
PARTIAL_STRICT = False                  # True : un signal mal placé (ex. « O » en 2e position) écarte le pattern incomplet

LETTER = {"RE": "R", "EXM": "E", "EXT": "E", "MHV": "M", "CSAK": "C", "MOM": "O"}
CODE_NAMES = {
    "R": "Reentry (RE)",
    "E": "Extreme (EXM / EXT)",
    "M": "Market Hilang Volume (MHV)",
    "C": "Candle Arah Kukuh (CSAK)",
    "O": "Momentum (MOM)",
}
ARROW = {"B": "▲", "S": "▼"}
SIDE_TEXT = {"B": "achat", "S": "vente"}


# --- Détection (fonctions pures) --------------------------------------------------------------

def select_tfs(tfs, preferred=PATTERN_TFS, n=N_TFS):
    """Les timeframes analysés, du plus haut au plus bas, parmi les timeframes affichés (tfs)."""
    shown = [t for t in core.TF_ORDER if t in set(tfs)]
    if preferred:
        chosen = [t for t in core.TF_ORDER if t in preferred and t in shown]
        # Complément : d'autres TF cochés, du plus haut au plus bas, jusqu'à n.
        chosen += [t for t in shown if t not in chosen][: max(0, n - len(chosen))]
    else:
        chosen = shown[:n]
    return [t for t in core.TF_ORDER if t in chosen][:n]


def _best_fit(items, codes):
    """(nombre de signaux bien placés, pattern cible, liste de booléens par TF) pour le meilleur pattern de `codes`.
    Un signal est bien placé si sa lettre est celle du pattern cible à la même position."""
    best, best_key = (0, None, None), None
    for code in codes or ():
        hits = [bool(i["letter"]) and k < len(code) and i["letter"] == code[k] for k, i in enumerate(items)]
        n = sum(hits)
        bad = sum(1 for i, h in zip(items, hits) if i["letter"] and not h)    # signaux mal placés
        key = (n, -bad)                    # à égalité de signaux bien placés, le moins de signaux mal placés
        if n > 0 and (best_key is None or key > best_key):
            best, best_key = (n, code, hits), key
    return best


def detect_pattern(cells, tfs, avail=None, preferred=PATTERN_TFS, n=N_TFS, recent=None,
                   codes=PATTERN_CODES, same_side=REQUIRE_SAME_SIDE,
                   partial=ALLOW_PARTIAL, min_hits=MIN_HITS, strict=PARTIAL_STRICT):
    """Pattern MTF d'une paire.

    cells : {tf: (code, sens)} (le résultat du scan) ; tfs : timeframes affichés ;
    avail : timeframes ayant des données (None = tous) ; recent : {tf: (code, sens, age) ou None}, derniers signaux
    des bougies précédentes (None = signaux récents ignorés). Retourne None si aucun timeframe n'est utilisable,
    sinon un dict :
      tfs      timeframes analysés (du plus haut au plus bas)
      items    liste de {tf, state ('signal' | 'recent' | 'none' | 'absent'), code, letter, sens, age}
      letters  lettres concaténées, minuscule = signal récent, '-' = aucun, '?' = pas de données (ex. 'Cr-')
      side     'B' / 'S' (sens majoritaire), 'X' (égalité) ou None (aucun signal)
      buy, sell, active, total   décomptes
      live     nombre de signaux sur la dernière bougie (active - live = signaux récents grisés)
      aligned  True si tous les signaux actifs vont dans le même sens
      match    True si le pattern fait partie de `codes`, complet ou incomplet (toujours True si codes est None)
      code     le pattern COMPLET en majuscules (ex. 'REM') s'il correspond, sinon None
      partial  True si le pattern est incomplet : au moins un signal bien placé, mais pas le pattern entier
      hits     nombre de signaux bien placés (pattern incomplet)
      target   pattern cible d'un pattern incomplet (ex. 'REM'), sinon None
      shape    lecture d'un pattern incomplet, '·' pour une place vide ou mal tenue (ex. 'R··')
    """
    sel = select_tfs(tfs, preferred, n)
    if not sel:
        return None
    have = None if avail is None else set(avail)
    items = []
    for t in sel:
        if have is not None and t not in have:
            items.append({"tf": t, "state": "absent", "code": None, "letter": None, "sens": None, "age": 0})
            continue
        code, sens = (cells or {}).get(t, (None, None))
        letter = LETTER.get(code) if sens in ("B", "S") else None
        rec = (recent or {}).get(t)
        if letter:
            items.append({"tf": t, "state": "signal", "code": code, "letter": letter, "sens": sens, "age": 0})
        elif rec and LETTER.get(rec[0]) and rec[1] in ("B", "S"):
            items.append({"tf": t, "state": "recent", "code": rec[0], "letter": LETTER[rec[0]], "sens": rec[1],
                          "age": int(rec[2])})
        else:
            items.append({"tf": t, "state": "none", "code": None, "letter": None, "sens": None, "age": 0})
    nb = sum(i["sens"] == "B" for i in items)
    ns = sum(i["sens"] == "S" for i in items)
    side = None if nb + ns == 0 else ("B" if nb > ns else "S" if ns > nb else "X")
    letters = "".join(
        (i["letter"].lower() if i["state"] == "recent" else i["letter"]) if i["letter"]
        else ("?" if i["state"] == "absent" else "-") for i in items)
    aligned = nb + ns > 0 and (nb == 0 or ns == 0)
    pattern = letters.upper()
    full = codes is None or pattern in codes
    hits_n, target, hits = (0, None, None) if codes is None else _best_fit(items, codes)
    if hits:
        for i, h in zip(items, hits):
            i["fit"] = h
    misplaced = hits is not None and any(i["letter"] and not h for i, h in zip(items, hits))
    is_partial = (partial and not full and hits_n >= max(1, min_hits)
                  and not (strict and misplaced))
    match = (full or is_partial) and (aligned or not same_side)
    shape = "".join(target[k] if hits[k] else "·" for k in range(len(items))) if is_partial else None
    return {"tfs": sel, "items": items, "letters": letters, "side": side, "buy": nb, "sell": ns,
            "active": nb + ns, "live": sum(i["state"] == "signal" for i in items), "total": len(items),
            "aligned": aligned, "match": match, "code": pattern if match and full and codes else None,
            "partial": bool(match and is_partial), "hits": hits_n if is_partial else 0,
            "target": target if is_partial else None, "shape": shape}


# --- Rendu HTML -------------------------------------------------------------------------------

def tooltip(p):
    """Infobulle : une ligne par timeframe, puis la lecture d'ensemble."""
    if p is None:
        return "Pattern MTF : aucun timeframe disponible"
    order = ' → '.join(p['tfs'])
    if p["match"] and p["code"]:
        lines = [f"Pattern {p['code']} ({order})"]
    elif p["match"] and p.get("partial"):
        lines = [f"Pattern incomplet {p['target']} ({order}) : {p['letters']}",
                 f"Signaux bien placés : {p['hits']}/{p['total']}"]
    elif p["match"]:
        lines = [f"Pattern MTF ({order}) : {p['letters']}"]
    else:
        lines = [f"Aucun pattern {' / '.join(PATTERN_CODES)}", f"Lecture actuelle ({order}) : {p['letters']}"]
        if p["letters"].upper() in PATTERN_CODES and not p["aligned"]:
            lines.append("Pattern écarté : sens différents d'un TF à l'autre")
    for i in p["items"]:
        ok = ""
        if p.get("partial") and i.get("letter"):
            ok = " ✓ bien placé" if i.get("fit") else " ✗ mal placé"
        if i["state"] == "signal":
            lines.append(f"{i['tf']} : {CODE_NAMES[i['letter']]} · {SIDE_TEXT[i['sens']]} {ARROW[i['sens']]}{ok}")
        elif i["state"] == "recent":
            lines.append(f"{i['tf']} : {CODE_NAMES[i['letter']]} · {SIDE_TEXT[i['sens']]} {ARROW[i['sens']]}"
                         f" · récent, il y a {i['age']} bougie(s) (grisé){ok}")
        elif i["state"] == "absent":
            lines.append(f"{i['tf']} : données absentes")
        else:
            lines.append(f"{i['tf']} : aucun signal")
    if p["side"] in ("B", "S"):
        verdict = f"Sens majoritaire : {SIDE_TEXT[p['side']]} ({max(p['buy'], p['sell'])}/{p['total']})"
        lines += ["", verdict + ("" if p["aligned"] else " · timeframes contradictoires")]
    elif p["side"] == "X":
        lines += ["", "Signaux opposés à égalité"]
    return "&#10;".join(html.escape(x) for x in lines)


def _chip(i, partial=False) -> str:
    """Pastille d'un TF : lettre et nom ; signal récent grisé avec son âge (« D1 −2 »)."""
    tf = html.escape(i["tf"])
    mark = ""
    if partial and i["letter"]:                     # pattern incomplet : bien placé (cadre) / mal placé (atténué)
        mark = " ok" if i.get("fit") else " bad"
    if i["state"] == "recent":
        return f'<span class="pc {i["sens"]} g{mark}"><b>{i["letter"]}</b><i>{tf} −{i["age"]}</i></span>'
    letter = i["letter"] or ("?" if i["state"] == "absent" else "–")
    return f'<span class="pc {i["sens"] or ""}{mark}"><b>{letter}</b><i>{tf}</i></span>'


def pattern_cell(p) -> str:
    """Cellule « Pattern MTF » : une pastille lettre + timeframe par TF, puis la ligne de lecture."""
    tip = tooltip(p)
    if p is None or not p["match"]:                # hors patterns autorisés : cellule vide
        return f'<td class="pat none" title="{tip}">–</td>'
    if p["active"] == 0:                           # aucun signal : pastilles grises, sans ligne de lecture
        chips = "".join(
            f'<span class="pc"><b>{"?" if i["state"] == "absent" else "–"}</b><i>{html.escape(i["tf"])}</i></span>'
            for i in p["items"])
        return f'<td class="pat none" title="{tip}"><span class="pt">{chips}</span></td>'
    chips = "".join(_chip(i, p.get("partial")) for i in p["items"])
    if p["code"] and p["side"] in ("B", "S"):
        read = f'{ARROW[p["side"]]} {p["code"]}' + ("" if p["aligned"] else " ⇅")
    elif p.get("partial") and p["side"] in ("B", "S"):
        read = f'{ARROW[p["side"]]} {html.escape(p["shape"])}' + ("" if p["aligned"] else " ⇅")
    elif p["side"] in ("B", "S"):
        read = f'{ARROW[p["side"]]} {max(p["buy"], p["sell"])}/{p["total"]}' + ("" if p["aligned"] else " ⇅")
    else:
        read = "⇅ égalité"
    side_cls = p["side"] if p["side"] in ("B", "S") else "X"
    full = " full" if p["aligned"] and p["live"] == p["total"] and not p.get("partial") else ""
    mix = "" if p["aligned"] else " mix"
    return (f'<td class="pat {side_cls}{full}{mix}" title="{tip}"><span class="pt">{chips}</span>'
            f'<div class="pa {side_cls}">{read}</div></td>')


# --- CSS --------------------------------------------------------------------------------------

_PATTERN_CSS_RULES = """table.bb td.pat{min-width:92px;padding:6px 6px;background:#2a2e39}
table.bb td.pat .pt{display:inline-flex;gap:3px;vertical-align:middle}
table.bb td.pat .pc{display:flex;flex-direction:column;align-items:center;min-width:24px;padding:3px 3px 2px;border-radius:5px;background:#1e222d;color:#5c6070}
table.bb td.pat .pc b{font-size:14px;line-height:1.1}
table.bb td.pat .pc i{font-style:normal;font-size:8px;font-weight:600;margin-top:1px;opacity:.85}
table.bb td.pat .pc.B{background:rgba(38,166,154,.22);color:#26a69a}
table.bb td.pat .pc.S{background:rgba(239,83,80,.22);color:#ef5350}
table.bb td.pat .pc.g{opacity:.55;box-shadow:inset 0 0 0 1px currentColor;background:transparent}
table.bb td.pat .pc.ok{box-shadow:0 0 0 1px currentColor}
table.bb td.pat .pc.bad{opacity:.4}
table.bb td.pat .pa{margin-top:4px;font-size:10px;font-weight:700;color:#787b86}
table.bb td.pat .pa.B{color:#26a69a}
table.bb td.pat .pa.S{color:#ef5350}
table.bb td.pat.B.full{box-shadow:inset 0 0 0 1px #26a69a}
table.bb td.pat.S.full{box-shadow:inset 0 0 0 1px #ef5350}
table.bb td.pat.mix .pa{color:#BA7517}
table.bb td.pat.none{color:#5c6070;font-weight:400}
table.bb td.pat .pa{letter-spacing:.04em}
"""
PATTERN_CSS = "<style>" + _PATTERN_CSS_RULES + "</style>"   # usage autonome

# Ajuste le tableau à la largeur de l'écran : plus de largeur minimale fixe, marges réduites, police et
# espacements proportionnels à la fenêtre (clamp). Sur écran très étroit, le défilement horizontal reste
# actif (règle d'origine de app.py) avec la colonne Pair figée.
_FIT_CSS_RULES = """.block-container{padding-left:1rem!important;padding-right:1rem!important;max-width:100%!important}
.bb-wrap{padding:3px}
table.bb{min-width:0;border-spacing:2px;font-size:clamp(10px,.8vw,13px)}
table.bb th{padding:6px 2px;font-size:clamp(8px,.62vw,11px);letter-spacing:.02em}
table.bb th.pair{width:auto;min-width:74px;font-size:clamp(10px,.8vw,13px)}
table.bb td{padding:5px 3px;border-radius:6px;font-size:clamp(10px,.8vw,13px)}
table.bb td.strong{box-shadow:inset 2px 0 0 currentColor}
table.bb .tpw{margin-left:3px;padding:0 3px;font-size:8px}
table.bb .rsi{font-size:9px;margin-top:2px}
table.bb td.synth,table.bb td.conf,table.bb td.qual{min-width:0}
table.bb td.synth{font-size:clamp(10px,.75vw,12px)}
table.bb td.gauge{min-width:0;max-width:none;padding:5px 4px}
table.bb .gsplit{font-size:clamp(9px,.7vw,12px)}
table.bb td.nuc{min-width:0;max-width:none;padding:5px 3px;font-size:clamp(9px,.72vw,12px)}
table.bb td.nuc .ntf{font-size:clamp(9px,.68vw,11px)}
table.bb td.pat{min-width:0;padding:5px 3px}
table.bb td.pat .pc{min-width:clamp(18px,1.5vw,24px)}
table.bb td.pat .pc b{font-size:clamp(11px,.9vw,14px)}
.fdot{width:6px;height:6px;margin-left:4px}
"""
FIT_CSS = "<style>" + _FIT_CSS_RULES + "</style>"   # usage autonome


def with_css(base_css, fit=False) -> str:
    """Insère les règles de la colonne (et, si fit, l'ajustement à l'écran) DANS le bloc <style> de base_css.

    Un seul bloc <style> est indispensable : st.markdown suit CommonMark, et deux blocs <style> collés
    (« </style><style> ») font lire le CSS suivant comme du texte Markdown, ce qui casse l'affichage du tableau."""
    extra = _PATTERN_CSS_RULES + (_FIT_CSS_RULES if fit else "")
    i = base_css.rindex("</style>")
    return base_css[:i] + extra + base_css[i:]


def legend_section() -> str:
    """Section de légende « Pattern MTF » (à insérer dans legend_html de app.py)."""
    letters = "".join(f"<div><b>{k}</b> · {v}</div>" for k, v in CODE_NAMES.items()
                      if not PATTERN_CODES or k in "".join(PATTERN_CODES))
    tfs = "les 3 TF cochés les plus hauts" if not PATTERN_TFS else " → ".join(PATTERN_TFS)
    pats = " · ".join(PATTERN_CODES) if PATTERN_CODES else "tous"
    return (
        '<h4>Pattern MTF</h4>'
        '<div class="chips"><span class="chip buy">▲ REM</span><span class="chip sell">▼ REE</span>'
        '<span class="chip buy">▲ R··</span>'
        '<span class="chip low">– aucun pattern ou sens mixtes</span></div>'
        '<div class="sigs">' + letters + '</div>'
        f'<div class="note">Seuls les patterns <b>{html.escape(pats)}</b> sont affichés, lus dans l’ordre des timeframes '
        f'({html.escape(tfs)}) : la 1re lettre est le TF le plus haut. Chaque lettre est colorée selon son sens ; '
        'les 3 lettres doivent aller dans le même sens (achat ou vente), sinon rien n’est affiché ; cadre coloré : les 3 '
        'sont sur la dernière bougie. Un pattern incomplet (ex. « R·· ») s’affiche dès qu’un signal est à la bonne place ; '
        'les points marquent les places vides ou mal tenues. Lettre encadrée : bien placée ; lettre atténuée : mal placée. Si « Afficher les signaux récents » est coché, un TF sans signal reprend son dernier '
        'signal, grisé avec son âge. Information seulement : n’entre ni dans l’entrée BBMA, ni dans la confiance, '
        'ni dans la qualité, ni dans le tri.</div>'
    )
