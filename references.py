import re
from collections import Counter

def extraire_noms_numeros(text):
    """
    Extrait les noms des auteurs et les numéros de référence du texte.

    Parameters:
    text (str): Le texte à analyser.

    Returns:
    list: Une liste des chaînes de caractères contenant les noms des auteurs et les numéros de référence.
    """
    # Utilisation de l'expression régulière pour extraire les noms des auteurs avec les numéros de référence
    references = re.findall(r'\b\w+\set\sal\.\s\[\d+\]', text)
    
    # Initialisation de la liste des noms et numéros
    noms_numeros = []

    # Parcours des références extraites
    for reference in references:
        noms_numeros.append(reference)

    # Retourner la liste des noms et numéros
    return noms_numeros

def extraire_references_par_nom_et_numero(liste_references, noms_numeros):
    """
    Extrait les références complètes à partir des noms d'auteurs et des numéros de référence.

    Parameters:
    liste_references (list): La liste complète des références.
    noms_numeros (list): Une liste des chaînes de caractères contenant les noms des auteurs et les numéros de référence.

    Returns:
    list: Une liste des références complètes correspondantes.
    """
    references_extraites = []

    for nom_numero in noms_numeros:
        # Recherche des correspondances avec le motif régulier
        matches = re.findall(r"([\w\s]+) et al. \[(\d+)\]", nom_numero)
        # Vérification si des correspondances ont été trouvées
        if matches:
            nom_auteur, numero_reference = matches[0]
            for reference in liste_references:
                if nom_auteur in reference and f"[{numero_reference}" in reference:
                    references_extraites.append(reference)

    return references_extraites

def extract_solo_references(text):
    """
    Extrait les références seules du texte.

    Parameters:
    text (str): Le texte à analyser.

    Returns:
    list: Une liste des chaînes de caractères contenant les références seules.
    """
    # Utilisation d'une expression régulière pour rechercher les références seules
    solo_references = re.findall(r'\[\d+\]', text)
    
    # Affichage des références seules
    print("\nRéférences seules:")
    for reference in solo_references:
        print(reference)
        
    # Écriture des références seules dans un fichier
    with open("solo_references.txt", "w") as file:
        for reference in solo_references:
            file.write(reference + "\n")
    
    return solo_references

def extract_unique_references(text, liste_references):
    """
    Extrait les références complètes uniques du texte.

    Parameters:
    text (str): Le texte à analyser.
    liste_references (list): La liste complète des références.

    Returns:
    set: Un ensemble des références complètes uniques.
    """
    # Utilisation de la fonction extract_solo_references pour obtenir les numéros de référence uniques
    solo_references = extract_solo_references(text)
    
    # Création d'un dictionnaire pour compter le nombre d'occurrences de chaque référence dans le texte
    text_reference_counts = Counter(solo_references)
    
    # Création d'un dictionnaire pour compter le nombre d'occurrences de chaque référence dans la liste de références
    full_reference_counts = Counter()
    for full_reference in liste_references:
        reference_number = re.search(r'\[\d+\]', full_reference).group()
        full_reference_counts[reference_number] += 1
    
    # Création d'un ensemble pour stocker les références complètes uniques
    unique_full_references = set()
    
    # Parcours de chaque référence complète
    for full_reference in liste_references:
        reference_number = re.search(r'\[\d+\]', full_reference).group()
        # Vérification si le numéro de référence est unique dans le texte et dans la liste de références
        if text_reference_counts[reference_number] == 1 and full_reference_counts[reference_number] == 1:
            unique_full_references.add(full_reference)
    
    return unique_full_references
