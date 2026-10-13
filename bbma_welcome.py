"""Page d'accueil du scanner BBMA OA MTF.

Affichée une fois par session avant le scanner : présentation, mode d'emploi rapide et
acceptation de l'avertissement. Module ajouté : aucune fonction existante n'est modifiée.

Branchement dans app.py (dans main(), juste après st.set_page_config) :
    if not welcome.passed():
        welcome.render()
        st.stop()
"""
import html

import streamlit as st

SESSION_KEY = "welcome_ok"

CSS = """<style>
.stApp{background:#131722}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
.wl{max-width:980px;margin:0 auto;font:15px/1.6 -apple-system,"Trebuchet MS",Roboto,sans-serif;color:#b2b5be}
.wl-hero{text-align:center;padding:34px 16px 8px}
.wl-hero h1{margin:0;font-size:38px;font-weight:800;color:#d1d4dc;letter-spacing:-.01em}
.wl-hero h1 span{color:#2962ff}
.wl-hero p{margin:10px auto 0;max-width:640px;color:#787b86;font-size:16px}
.wl-hi{display:inline-block;margin-top:16px;padding:5px 14px;border-radius:999px;background:#1e222d;border:1px solid #2a2e39;font-size:13px;color:#b2b5be}
.wl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:26px 0 8px}
.wl-card{background:#1e222d;border:1px solid #2a2e39;border-radius:14px;padding:16px 18px}
.wl-card h3{margin:0 0 6px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#787b86}
.wl-card p{margin:0;font-size:14px}
.wl-card b{color:#d1d4dc}
.wl-steps{background:#1e222d;border:1px solid #2a2e39;border-radius:14px;padding:16px 20px;margin:12px 0}
.wl-steps h3{margin:0 0 8px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#787b86}
.wl-steps ol{margin:0;padding-left:20px}
.wl-steps li{margin:4px 0}
.wl-steps b{color:#d1d4dc}
.wl .buy{color:#26a69a;font-weight:700}
.wl .sell{color:#ef5350;font-weight:700}
.wl-warn{margin:12px 0 4px;padding:14px 18px;background:#1e222d;border:1px solid #2a2e39;border-left:3px solid #BA7517;border-radius:10px;font-size:13px;color:#787b86}
.wl-warn b{color:#b2b5be}
</style>"""


def passed() -> bool:
    """True si l'utilisateur a déjà accepté l'avertissement pendant cette session."""
    return bool(st.session_state.get(SESSION_KEY, False))


def _email():
    """E-mail du compte connecté si l'hébergeur le fournit (app privée), sinon None."""
    try:
        user = getattr(st, "user", None) or getattr(st, "experimental_user", None)
        email = getattr(user, "email", None)
        return str(email) if email and "@" in str(email) else None
    except Exception:
        return None


def _page_html() -> str:
    email = _email()
    hello = f'<div class="wl-hi">Connecté : {html.escape(email)}</div>' if email else ""
    return (
        '<div class="wl">'
        '<div class="wl-hero"><h1>BBMA <span>OA</span> MTF Scanner</h1>'
        '<p>Un tableau de bord qui scanne plusieurs paires sur plusieurs timeframes et résume '
        'les signaux BBMA en un coup d’œil.</p>' + hello + '</div>'

        '<div class="wl-grid">'
        '<div class="wl-card"><h3>Multi-timeframes</h3><p>De <b>MN</b> à <b>M5</b> sur forex, or et Bitcoin : '
        'signaux MOM, EXM, EXT, MHV, CSAK et RE sur chaque timeframe.</p></div>'
        '<div class="wl-card"><h3>Synthèse d’entrée</h3><p>La colonne <b>Entrée BBMA</b> combine le timeframe '
        'directeur et un timeframe d’entrée, avec le filtre de tendance D1.</p></div>'
        '<div class="wl-card"><h3>Notes de lecture</h3><p><b>Confiance</b>, <b>Jauge</b> Buy / Sell et '
        '<b>Qualité</b> (A, B, C) aident à comparer les situations. Le détail s’affiche au survol.</p></div>'
        '</div>'

        '<div class="wl-steps"><h3>Prise en main</h3><ol>'
        '<li>Choisissez vos <b>paires</b> et vos <b>timeframes</b> dans la barre latérale.</li>'
        '<li>Lisez le tableau : <span class="buy">▲ vert = achat</span>, <span class="sell">▼ rouge = vente</span>. '
        'Seules les bougies <b>clôturées</b> sont utilisées.</li>'
        '<li>La pastille à côté de chaque paire indique la <b>fraîcheur des données</b>.</li>'
        '<li>Le bouton <b>Afficher la légende</b>, sous le tableau, explique chaque code et chaque colonne.</li>'
        '<li>Le scan se relance automatiquement ; <b>Rafraîchir</b> force une mise à jour.</li>'
        '</ol></div>'

        '<div class="wl-warn"><b>Avertissement.</b> Cet outil d’analyse technique est fourni à titre purement '
        'informatif et éducatif. Il ne constitue ni un conseil en investissement, ni une recommandation d’achat '
        'ou de vente. Le trading comporte un risque élevé de perte, pouvant dépasser le capital engagé. '
        'Les signaux peuvent être erronés ou retardés et les performances passées ne préjugent pas des résultats '
        'futurs. <b>Vous restez seul responsable de vos décisions et de vos pertes éventuelles.</b></div>'
        '</div>'
    )


def render():
    """Affiche la page d'accueil. Le bouton d'accès n'est actif qu'après acceptation de l'avertissement."""
    st.markdown(CSS + _page_html(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        accepted = st.checkbox("J’ai lu l’avertissement et je comprends que je suis seul responsable "
                               "de mes décisions.", key="welcome_accept")
        if st.button("Accéder au scanner", type="primary", disabled=not accepted, use_container_width=True):
            st.session_state[SESSION_KEY] = True
            st.rerun()
