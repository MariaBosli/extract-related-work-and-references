from langchain import HuggingFaceHub

# Définition de la fonction `llm` qui crée et retourne un modèle de langage basé sur Hugging Face Hub
def llm():
    """
    Crée et retourne un modèle de langage basé sur le dépôt Hugging Face Hub spécifié.

    Utilise une clé API pour authentifier l'accès au dépôt du modèle.

    Returns:
        HuggingFaceHub: Une instance configurée de la classe HuggingFaceHub représentant le modèle de langage.
    """
    # Clé API Hugging Face
    api_token = "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"

    # Création d'une instance de la classe HuggingFaceHub avec la clé API spécifiée
    llm_model = HuggingFaceHub(
        repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',  # Identifiant du dépôt du modèle
        model_kwargs={'max_length': 700, 'temperature': 0.6},  # Arguments du modèle (longueur maximale et température)
        huggingfacehub_api_token=api_token  # Clé API pour l'authentification
    )

    # Retour de l'instance du modèle de langage mixte créée
    return llm_model
