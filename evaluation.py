from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le modèle de transformation de texte pour l'encodage de phrases
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Charger le modèle d'évaluation d'hallucination
modelf = CrossEncoder("vectara/hallucination_evaluation_model")

def score_hallucination(generated_state_of_the_art, text):
    """
    Calcule le score d'hallucination entre un texte généré et un texte de référence.
    
    Parameters:
    generated_state_of_the_art (str): Le texte généré représentant l'état de l'art.
    text (str): Le texte de référence avec lequel comparer le texte généré.
    
    Returns:
    float: Le score d'hallucination entre les deux textes.
    """
    # Prédiction du score d'hallucination à l'aide du modèle d'évaluation
    scores = modelf.predict([
        [generated_state_of_the_art, text]
    ])

    return scores

def calculate_cosine_similarity(generated_state_of_the_art, text):
    """
    Calcule la similarité cosinus entre un texte généré et un texte de référence.
    
    Parameters:
    generated_state_of_the_art (str): Le texte généré représentant l'état de l'art.
    text (str): Le texte de référence avec lequel comparer le texte généré.
    
    Returns:
    float: La similarité cosinus entre les deux textes.
    """
    # Convertir les textes en vecteurs d'embedding
    embedding1 = model.encode(generated_state_of_the_art)
    embedding2 = model.encode(text)
    
    # Redimensionner les vecteurs pour s'assurer qu'ils ont la même forme
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    
    # Calculer la similarité cosinus entre les deux vecteurs d'embedding
    similarity = cosine_similarity(embedding1, embedding2)
    
    return similarity[0][0]
def list_to_text(lst):
    """
    Convertit une liste de chaînes de caractères en une seule chaîne concaténée.
    
    Parameters:
    lst (list): Liste de chaînes de caractères.
    
    Returns:
    str: Chaîne unique concaténée.
    """
    return ' '.join(lst)
