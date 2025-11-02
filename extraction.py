import re
from collections import Counter
import pandas as pd

# ==============================
# 🔍 Regex patterns
# ==============================

# Extraction du nom de l’écran (entre parenthèses)
pattern_ecran = re.compile(r"\((.*?)\)")

# Extraction de la configuration de l’écran (entre chevrons <...>)
pattern_conf_ecran = re.compile(r"<(.*?)>")

# Extraction de la chaîne (entre symboles $...$)
pattern_chaine = re.compile(r"\$(.*?)\$")

# Extraction d’un temps (t suivi d’un nombre)
pattern_temps = re.compile(r"t(\d+)")

# Extraction du type d’action principale (avant les symboles spéciaux)
pattern_action = re.compile(r"^[^\(<\$1]+")


# ==============================
# 🧩 Fonctions utilitaires
# ==============================

def extract_ecran(value: str) -> str:
    """
    Extrait le nom de l’écran à partir d’une chaîne d’action utilisateur.

    Exemple :
        >>> extract_ecran("Création d'un écran(infologic.core.accueil.Accueil)")
        'infologic.core.accueil.Accueil'

    Args:
        value (str): La chaîne représentant une action (ex: 'Création d'un écran(...)').

    Returns:
        str: Le nom de l’écran extrait, ou None si non trouvé.
    """
    match = pattern_ecran.search(str(value))
    return match.group(1) if match else None


def extract_conf_ecran(value: str) -> str:
    """
    Extrait la configuration d’écran à partir d’une action contenant des chevrons <...>.

    Exemple :
        >>> extract_conf_ecran("Saisie dans un champ<DEF_03/24>")
        'DEF_03/24'

    Args:
        value (str): Chaîne d’action utilisateur.

    Returns:
        str: La configuration d’écran si présente, sinon None.
    """
    match = pattern_conf_ecran.search(str(value))
    return match.group(1) if match else None


def extract_chaine(value: str) -> str:
    """
    Extrait la chaîne (catégorie de fiche) à partir d’une action contenant des symboles $...$.

    Exemple :
        >>> extract_chaine("Exécution d'un bouton$CLIENT$")
        'CLIENT'

    Args:
        value (str): Chaîne d’action utilisateur.

    Returns:
        str: La chaîne extraite, ou None si absente.
    """
    match = pattern_chaine.search(str(value))
    return match.group(1) if match else None


def extract_temps(value: str) -> int:
    """
    Extrait le temps indiqué dans une action (e.g., 't10' → 10).

    Exemple :
        >>> extract_temps("t25")
        25

    Args:
        value (str): Chaîne d’action utilisateur.

    Returns:
        int: Temps extrait en secondes, ou None si non trouvé.
    """
    match = pattern_temps.match(str(value))
    return int(match.group(1)) if match else None


def filter_action(value: str) -> str:
    """
    Nettoie une action en supprimant les parties contextuelles (parenthèses, balises, chaînes, indices numériques).

    Exemple :
        >>> filter_action("Exécution d'un bouton(MAINT)<DEF_03/24>$FICHE$")
        "Exécution d'un bouton"

    Args:
        value (str): Chaîne d’action utilisateur.

    Returns:
        str: Action nettoyée.
    """
    if not isinstance(value, str):
        return value
    for delim in ["(", "<", "$", "1"]:
        if delim in value:
            value = value.split(delim)[0]
    return value.strip()


def count_actions(series: pd.Series) -> Counter:
    """
    Compte la fréquence des actions principales dans une série de chaînes.

    Exemple :
        >>> count_actions(pd.Series(["Exécution d'un bouton", "Affichage d'une dialogue", "Exécution d'un bouton"]))
        Counter({"Exécution d'un bouton": 2, "Affichage d'une dialogue": 1})

    Args:
        series (pd.Series): Série Pandas contenant des actions.

    Returns:
        Counter: Dictionnaire des fréquences par action.
    """
    actions = [filter_action(v) for v in series.dropna().tolist()]
    return Counter(actions)


def most_common_ecran(actions: pd.Series) -> str:
    """
    Renvoie l’écran le plus fréquemment utilisé dans une série d’actions.

    Args:
        actions (pd.Series): Série contenant les actions d’un utilisateur.

    Returns:
        str: Nom de l’écran le plus utilisé, ou None si aucun trouvé.
    """
    ecrans = [extract_ecran(a) for a in actions.dropna()]
    ecrans = [e for e in ecrans if e]
    return Counter(ecrans).most_common(1)[0][0] if ecrans else None


def most_common_conf(actions: pd.Series) -> str:
    """
    Renvoie la configuration d’écran la plus fréquente dans une série d’actions.

    Args:
        actions (pd.Series): Série contenant les actions d’un utilisateur.

    Returns:
        str: Configuration la plus fréquente, ou None si aucune.
    """
    confs = [extract_conf_ecran(a) for a in actions.dropna()]
    confs = [c for c in confs if c]
    return Counter(confs).most_common(1)[0][0] if confs else None


def most_common_chaine(actions: pd.Series) -> str:
    """
    Renvoie la chaîne (catégorie de fiche) la plus fréquente dans une série d’actions.

    Args:
        actions (pd.Series): Série contenant les actions d’un utilisateur.

    Returns:
        str: Chaîne la plus fréquente, ou None si aucune trouvée.
    """
    chaines = [extract_chaine(a) for a in actions.dropna()]
    chaines = [c for c in chaines if c]
    return Counter(chaines).most_common(1)[0][0] if chaines else None
