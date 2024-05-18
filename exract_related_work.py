from llm import llm
from references import extract_unique_references, extraire_noms_numeros, extraire_references_par_nom_et_numero
from evaluation import score_hallucination,list_to_text, calculate_cosine_similarity
mistral_llm = llm()  # Créez une instance de votre modèle de langage 

def extract_state_of_the_art(related_works):
    # Construction du prompt pour la génération de la demande d'état de l'art
    prompt = "[INST] Tu es un chercheur académique. Ton rôle est de synthétiser l'état de l'art dans un domaine de recherche spécifique. En te basant sur les travaux connexes suivants :"

    # Ajout des travaux connexes au prompt
    for work in related_works:
        prompt += f"\n* {work}"

    # Découpage du prompt en chunks pour respecter la limite de longueur
    max_length = 512
    prompt_chunks = [prompt[i:i+max_length] for i in range(0, len(prompt), max_length)]

    # Génération du texte de demande d'état de l'art à partir du prompt en utilisant le modèle de langage mixte
    generated_text_chunks = []
    for prompt_chunk in prompt_chunks:
        output = mistral_llm(prompt_chunk)
        generated_text_chunks.append(output)

    generated_text = ''.join(generated_text_chunks)
    generated_text = '\n'.join([p for p in generated_text.split('\n')[1:] if len(p) > 0])

    # Retour du texte généré
    return generated_text

def main():
    # Liste de travaux connexes(il faut automatiser cette tache)
    related_works = [ """Many studies have been suggested in the literature
    handling DL techniques for the purpose of diagnosing and
    diagnosing patients with COVID-19 virus.
    Soares et al. [6] conducted a study to determine Covid-19
    using deep learning methods, xDNN, ResNet, GoogleNet,
    VGG16, AlexNet, Decision Tree and AdoBoost. The dataset
    used is 80:20 reserved for training and testing. Consequently
    of the studies, the xDNN deep learning model reached the
    highest accuracy value of 97.38%. Cifci [7] also tried to detect
    COVID-19 with AlexNet and Inception-V4 models. It used
    4640 (80%) CT images to train the model and 1160 (20%) CT
    images to test it. Looking at the experimental results, AlexNet
    performed better than Inceptionv4. AlexNet achieved an
    accuracy of 94.74%.
    Wang et al. [8] experimented on pathogen-confirmed
    COVID-19 cases in 1,119 CT scans. Using the Inception
    model, they put into practice the transfer learning techniques
    of deep learning and achieved 89.5% accuracy. Lawton et al.
    [9] evaluated the performance of standard histogram
    equalization and contrast limited adaptive histogram
    equalization as well as transfer learning models to determine
    Covid-19 with deep learning methods. In the Sars Cov 2 CT
    Scan dataset, the highest performing model was VGG19 with
    95.75%, which was applied with contrast bound adaptive
    histogram equalization.
    In the study directed by Rahimzadeh et al. [10] on the
    detection of COVID-19 in Iran, a new dataset containing
    48260 CT (tomography) scan images was used. Image
    classification with ResNet50V2, a 50-layer network trained on
    the ImageNet dataset, achieved 98.49% accuracy in CT
    scanning. In the study conducted by Pathak et al. [11],
    COVID-19 detection was made from computed tomography
    images based on deep transfer learning. The developed model
    achieved training and testing accuracy of 96.22% and 93.01%,
    respectively.
    Bansal et al. [12] tried to detect Covid-19 using deep
    learning methods ResNet50, VGG16, SVM models. The
    dataset used is 80:20 reserved for training and testing. Thus,
    496 data were used for the 1985 test for training. During the
    training phase, the 5-fold cross-validation method was used.
    The model with the highest accuracy was ResNet50 with
    95.16%. In the study conducted by Bukharia et al. [13],
    COVID-19 detection was made using the ResNet50 technique
    from X-Ray images. During the diagnosis process, They
    obtained 98.18%, 98.14%, 98.24% and 98.19% accuracy,
    precision, recall and F1-Score from the model, respectively.
    Jaiswal et al. [14] used DenseNet201, ResNet152V2,
    VGG16 and InceptionResNet models from DL methods to
    detect Covid-19. As a result of the studies, the test validation
    rate for DenseNet201 was 96.25%. Silva et al. [15], unlike
    others, performed cross dataset analysis with Sars Cov 2 CT
    Scan dataset and CovidCT dataset. EfficientNet, one of the
    deep learning models, was used in the study, and the highest
    accuracy value was 87.68.""",

        """COVID-19 detection methods can be categorized into two
    groups: 1) unimodal detection methods, 2) multi-modal
    methods. We discuss the symptoms and techniques used within
    these categories below.
    A. Unimodal Automated Methods for COVID-19 Symptom
    Detection
    At the early stages of the pandemic, many studies focused on
    unimodal automated methods for COVID-19 detection, in
    particular on X-ray [14], CT [15], or ultrasound [21] images of
    patients’ chests to identify anomalies [25].
    Despite the high accuracy of AI enhanced imaging
    identification, these diagnosis methods are not portable and must
    be performed in hospitals or medical centers where the
    equipment is available. Furthermore, specially trained
    technicians are needed in order to operate these systems, which
    further restricts the testing capability for identifying COVID-19
    in the field. Therefore, researchers have been investigating the
    use of more widely available signals for development of early
    detection of COVID symptoms [26]. These include, for
    example, vocal signals such as coughing [22], [27]–[29] or
    breathing [23], [30], [31].
    Coughing is a very common and natural physiological
    behavior which is associated with several viral respiratory
    infections including COVID-19 [32], [33]. Depending on the
    infected and the irritant locations within the respiratory system,
    the coughing sound produced by different respiratory infections
    have distinct features [34]–[39]. Recent studies have found that
    COVID-19 infects the respiratory system in a distinct way [27],
    [40]–[42]. By comparing the CT analysis of COVID-19 infected
    patients with non-COVID pneumonia, the COVID-19 patients
    are more likely to develop a peripheral distribution, groundglass
    opacity, vascular thickening, and reverse halo sign [16],
    [40].
    Based on the difference of changes in the respiratory system,
    a COVID-19 patient would likely produce a distinct coughing
    sound that can be identified by learning algorithms [16]. For
    example, in [28], Convolutional Neural Networks (CNN) were
    used to identify people with COVID through forced coughing as
    input signals. When processing the raw coughing signal, four
    distinct biomarkers including muscular degradation, changes in
    vocal cords, changes in sentiment/mood, and changes in the
    lungs/respiratory tract were utilized as part of the detection
    criteria.Another common symptom of COVID-19 infection is
    changes in the breathing rhythm or shortness of breath. In [23],
    frequency-modulated continuous wave signals were classified
    using a XGBoost classifier and a Mel-frequency cepstral
    coefficient (MFCC) feature extractor. Five different breathing
    patterns were analyzed: normal breathing, deep/quick breathing,
    deep breathing, quick breathing, and holding the breath.
    Despite the fact that fever is the most common symptoms of
    COVID-19 [20], a unimodal temperature measurement is not
    sufficient to determine whether a person is infected by the virus,
    as fever alone is a very common symptom of many illnesses
    [43]. Furthermore, fever can be less common in younger age
    groups that have been diagnosed with mild or moderate COVID
    [44]. As a result, temperature can only be utilized in combination
    with other modes (symptoms) in multimodal analysis.
    B. Multi-modal Automated Methods for COVID-19 Detection
    Since COVID-19 patients are usually presented with multiple
    symptoms [24], multi-modal deep learning classification can be
    used to improve the accuracy of COVID-19 diagnoses [17]. For
    example, in [30], a COVID-19 Identification ResNet (CIdeR)
    classifier used cough and breathing together to determine the
    probability of COVID-positivity. In [17], more modes were
    utilized. Namely, Logistic Regression (LR) and Support Vector
    Machines (SVM) were used to classify breathing, coughing, and
    speech signals separately and a Decision Tree classifier was used
    to classify a group of other symptoms such as fatigue, muscle
    pain, and loss of smell. The outputs of each classifier were
    prediction scores and were all given equal weights to determine
    the overall probability of COVID.
    However, when investigating clinically obtained statistical
    datasets for COVID-19, it has been found that varying symptoms
    have different prevalence, as seen in the MIT dataset [22]. In this
    paper we introduce a unique weighting system that can utilize
    multiple modes (non-fixed parameters) to determine the
    influence of each individual symptom for COVID positivity.:""",
        """ Extensive research work is going on for classifying COVID-
    19 patient image data. Few researchers have proposed different
    DL models for classifying chest x-ray images whereas some
    others have taken CT images into consideration. Narin et. al
    proposed three pretrained CNN models based on ResNet50,
    InceptionV3 and Inception-ResNetV2 for detecting COVID-
    19 patient from chest X-ray radiographs [24]. It is found that
    ResNet 50 gives the classifying accuracy of 98% whereas InceptionV3
    and Inception-ResNetV2 perform with the accuracy
    of 97% and 87% respectively. But these models have taken
    only 100 images (50 COVID-19 and 50 normal Chest X-rays)
    into consideration for training which might result in declined
    accuracy for a higher number of training images. Zhang et
    al. propose a DL model for Coronavirus patient screening
    using their chest X-ray images [25]. This research group has used 100 chest X-ray images of 70 COVID-19 patients and
    1431 X-ray images of other pneumonia patients where they
    are classified as COVID-19 and non-COVID-19 respectively.
    This model is formed of three main parts: backbone networks,
    classification head, and anomaly detection head. The backbone
    network is a 18 residual CNN layer pre-trained on ImageNet
    dataset and it is mentionable that ImageNet provides a huge
    number of a generalized dataset for image classifications. This
    model can diagnosis COVID-19 and non-COVID-19 patients
    with an accuracy of 96% and 70.65% respectively. Hall et al.
    also worked on finding COVID-19 patients from a small set of
    chest X-ray images with DL [26]. They have used pre-trained
    ResNet50 and VGG 16 along with their own CNN and this
    model generates the overall accuracy of 91.24%. Sethy and
    Behea have also utilized deep features for Coronavirus disease
    detection [27]. Their model is based on ResNet50 plus SVM
    which achieved the accuracy and F1-score of 95.38% and
    91.41% respectively. Apostolopoulos and Mpesiana utilized
    CNN transfer learning for detecting COVID-19 with X-ray
    images [28]. This work has considered 224 chest X-ray images
    of COVID-19 infected people, 714 images with Pneumonia
    and 504 images of normal people for training their model.
    This model achieved the accuracy of 96.78% and sensitivity
    and specificity of 98.66% and 96.46% respectively. Li et al.
    used the patients’ chest CT images for detecting COVID-19
    with the developed CNN architecture called COVNet [29].
    This research group has obtained sensitivity, specificity and
    126 Area Under the Receiver Operating Curve (AUC) of 90%,
    96% and 0.96 respectively. Other researchers have also put an
    effort to detect COVID-19 patient from chest X-ray images in
    [30] [31].""",
        # Ajoutez d'autres titres d'articles ici
    ]
  #il faut automatiser l'extraction des references aussi
    liste_references = [
    "[1] R. M. Pereira, D. Bertolini, L. O. Teixeira, C. N. Silla, and Y. M. G. Costa, “COVID-19 identification in chest X-ray images on flat and hierarchical classification scenarios,” Comput. Methods Programs Biomed., vol. 194, p. 105532, Oct. 2020, doi: 10.1016/j.cmpb.2020.105532.",
    "[2] L. Brunese, F. Mercaldo, A. Reginelli, and A. Santone, “Explainable Deep Learning for Pulmonary Disease and Coronavirus COVID-19 Detection from X-rays,” Comput. Methods Programs Biomed., vol. 196, p. 105608, Nov. 2020, doi: 10.1016/j.cmpb.2020.105608.",
    "[3] H. Panwar, P. K. Gupta, M. K. Siddiqui, R. Morales-Menendez, and V. Singh, “Application of deep learning for fast detection of COVID-19 in X-Rays using nCOVnet,” Chaos, Solitons and Fractals, vol. 138, p. 109944, Sep. 2020, doi: 10.1016/j.chaos.2020.109944.",
    "[4] Y. Tung-Chen et al., “Correlation between Chest Computed Tomography and Lung Ultrasonography in Patients with Coronavirus Disease 2019 (COVID-19),” Ultrasound Med. Biol., vol. 46, no. 11, pp. 2918–2926, Nov. 2020, doi: 10.1016/j.ultrasmedbio.2020.07.003.",
    "[5] A. I. Khan, J. L. Shah, and M. M. Bhat, “CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest xray images,” Comput. Methods Programs Biomed., vol. 196, p. 105581, Nov. 2020, doi: 10.1016/j.cmpb.2020.105581.",
    "[6] E. Soares, P. Angelov, S. Biaso, M. H. Froes, and D. K. Abe, “SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification,” medRxiv, pp. 1–8, 2020.",
    "[7] M. Akif Cifci, “Deep Learning Model for Diagnosis of Corona Virus Disease from CT Images,” Int. J. Sci. Eng. Res., vol. 11, no. 4, pp. 273–278, 2020, [Online]. Available: http://www.ijser.org.",
    "[8] B. X. Shuai Wang, Bo Kang, Jinlu Ma, Xianjun Zeng5, Mingming Xiao1, Jia Guo, Mengjiao Cai, Jingyi Yang, Yaodong Li, Xiangfei Meng, “A deep learning algorithm using CT images to screen for Corona Virus Disease (COVID-19),” pp. 1–27, 2020.",
    "[9] S. Lawton and S. Viriri, “Detection of COVID-19 from CT Lung Scans Using Transfer Learning,” Comput. Intell. Neurosci., vol. 2021, 2021, doi: 10.1155/2021/5527923.",
    "[10] M. Rahimzadeh and A. Attar, “A NEW MODIFIED DEEP CONVOLUTIONAL NEURAL NETWORK FOR DETECTING COVID-19 FROM X-RAY IMAGES,” arXiv, vol. 19. arXiv, p. 100360, Apr. 16, 2020, doi: 10.1016/j.imu.2020.100360.",
    "[11] Y. Pathak, P. K. Shukla, A. Tiwari, S. Stalin, and S. Singh, “Deep Transfer Learning Based Classification Model for COVID-19 Disease,” IRBM, May 2020, doi: 10.1016/j.irbm.2020.05.003.",
    "[12] A. Bansal, G. Thakur, and D. Verma, “Detection of covid-19 using the ct scan image of lungs,” CEUR Workshop Proc., vol. 2786, pp. 219–227, 2021.",
    "[13] S. U. khalid Bukhari, S. S. K. Bukhari, A. Syed, and S. S. H. Shah, “The diagnostic evaluation of convolutional neural network (CNN) for the assessment of chest X-ray of patients infected with COVID-19,” medRxiv, 2020, doi: 10.1101/2020.03.26.20044610.",
    "[14] A. Jaiswal, N. Gianchandani, D. Singh, V. Kumar, and M. Kaur, “Classification of the COVID-19 infected patients using DenseNet201 based deep transfer learning,” Journal of Biomolecular Structure and Dynamics. 2020, doi: 10.1080/07391102.2020.1788642.",
    "[15] P. Silva et al., “COVID-19 detection in CT images with deep learning: A voting-based scheme and cross-datasets analysis,” Informatics in Medicine Unlocked, vol. 20. 2020, doi: 10.1016/j.imu.2020.100427.",
    "[16] “Antic, J.: Deoldify (2018). https://github.com/jantic/DeOldify.",
    "[17] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” Aug. 2016, Accessed: May 20, 2021. [Online]. Available: https://arxiv.org/abs/1608.06993.",
    "[1] “Coronavirus disease (COVID-19).” Jun. 2021, [online] Available: https://www.who.int/emergencies/diseases/novelcoronavirus-2019.",
    "[2] N. Shahhosseini, G. Babuadze, G. Wong, and G. P. Kobinger, “Mutation signatures and in silico docking of novel sars-cov-2 variants of concern,” Microorganisms, vol. 9, no. 5, p. 926, May 2021.",
    "[3] B. Hu, H. Guo, P. Zhou, and Z. L. Shi, “Characteristics of SARS-CoV-2 and COVID-19,” Nature Reviews Microbiology, vol. 19, no. 3. Nature Research, pp. 141–154, Mar. 01, 2021.",
    "[4] “Scientific Brief: SARS-CoV-2 Transmission | CDC.” May. 2021. [online] Available: https://www.cdc.gov/coronavirus/2019-ncov/science/science-briefs/sars-cov-2-transmission.html",
    "[5] V. M. Corman et al., “Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR,” Eurosurveillance, vol. 25, no. 3, p. 2000045, Jan. 2020.",
    "[6] A. Afzal, “Molecular diagnostic technologies for COVID-19: Limitations and challenges,” Journal of Advanced Research, vol. 26, pp. 149–159, Nov. 2020.",
    "[7] K. Syal, “Guidelines on newly identified limitations of diagnostic tools for COVID-19 and consequences,” Journal of Medical Virology, vol. 93, no. 4, pp. 1837–1842, Apr. 2021.",
    "[8] K. Ramdas, A. Darzi, and S. Jain, “‘Test, re-test, re-test’: using inaccurate tests to greatly increase the accuracy of COVID-19 testing,” Nature Medicine 2020 26:6, vol. 26, no. 6, pp. 810–811, May 2020.",
    "[9] W. Han et al., “The course of clinical diagnosis and treatment of a case infected with coronavirus disease 2019,” Journal of Medical Virology, vol. 92, no. 5, p. 461, May 2020.",
    "[10] A. Alimadadi, S. Aryal, I. Manandhar, P. B. Munroe, B. Joe, and X. Cheng, “Artificial intelligence and machine learning to fight covid-19,” Physiological Genomics, vol. 52, no. 4. American Physiological Society, pp. 200–202, Apr. 01, 2020.",
    "[11] A. W. Salehi, P. Baglat, and G. Gupta, “Review on machine and deep learning models for the detection and prediction of Coronavirus,” in Materials Today: Proceedings, Jan. 2020, vol. 33, pp. 3896–3901.",
    "[12] J. Rasheed, A. Jamil, A. A. Hameed, F. Al-Turjman, and A. Rasheed, “COVID-19 in the Age of Artificial Intelligence: A Comprehensive Review,” Interdisciplinary Sciences: Computational Life Sciences, vol. 13, no. 2. Springer Science and Business Media Deutschland GmbH, pp. 153–175, Jun. 01, 2021.",
    "[13] M. H. Tayarani N., “Applications of artificial intelligence in battling against covid-19: A literature review,” Chaos, Solitons and Fractals, vol. 142. Elsevier Ltd, p. 110338, Jan. 01, 2021.",
    "[14] “COVIDX-Net: A Framework of Deep Learning Classifiers to Diagnose COVID-19 in X-Ray Images,” Mar. 2020, [Online]. Available: http://arxiv.org/abs/2003.11055",
    "[15] “Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans,” Apr. 2020. [Online]. Available: https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1",
    "[16] A. Imran et al., “AI4COVID-19: AI enabled preliminary diagnosis for COVID-19 from cough samples via an app,” Informatics in Medicine Unlocked, vol. 20, p. 100378, Jan. 2020.",
    "[17] S. R. Chetupalli et al., “Multi-modal Point-of-Care Diagnostics for COVID-19 Based On Acoustics and Symptoms,” Jun. 2021. [Online]. Available: http://arxiv.org/abs/2106.00639",
    "[18] M. W. Hasan, “Covid-19 fever symptom detection based on IoT cloud,” International Journal of Electrical and Computer Engineering, vol. 11, no. 2, pp. 1823–1829, Apr. 2021.",
    "[19] H. Zare-Zardini, H. Soltaninejad, F. Ferdosian, A. A. Hamidieh, and M. Memarpoor-Yazdi, “Coronavirus Disease 2019 (COVID-19) in Children: Prevalence, Diagnosis, Clinical Symptoms, and Treatment,” International Journal of General Medicine, vol. 13, p. 477, 2020.",
    "[20] J. Yang et al., “Prevalence of comorbidities and its effects in patients infected with SARS-CoV-2: a systematic review and meta-analysis,” International Journal of Infectious Diseases, vol. 94, pp. 91–95, May 2020.",
    "[21] G. Soldati et al., “Is There a Role for Lung Ultrasound During the COVID‐19 Pandemic?,” Journal of Ultrasound in Medicine, vol. 39, no. 7, pp. 1459–1462, Jul. 2020.",
    "[22] Bertsimas, Dimitris et al., “An Aggregated Dataset of Clinical Outcomes for COVID-19 Patients” 2020, [Online] Available: http://www.covidanalytics.io/dataset_documentation.",
    "[23] A. T. Purnomo, D. B. Lin, T. Adiprabowo, and W. F. Hendria, “Non-contact monitoring and classification of breathing pattern for the supervision of people infected by covid-19,” Sensors, vol. 21, no. 9, p. 3172, May 2021.",
    "[24] J. Li et al., “Epidemiology of COVID-19: A systematic review and meta-analysis of clinical characteristics, risk factors, and outcomes,” Journal of Medical Virology, vol. 93, no. 3, pp. 1449–1458, Mar. 2021.",
    "[25] “Collaborative Federated Learning For Healthcare: Multi-Modal COVID-19 Diagnosis at the Edge,” Jan. 2021. [Online]. Available: https://arxiv.org/abs/2101.07511v1",
    "[26] A. N. Belkacem, S. Ouhbi, A. Lakas, E. Benkhelifa, and C. Chen, “End-to-End AI-Based Point-of-Care Diagnosis System for Classifying Respiratory Illnesses and Early Detection of COVID-19: A Theoretical Framework,” Frontiers in Medicine, vol. 0, p. 372, Mar. 2021.",
    "[27] “The COUGHVID crowdsourcing dataset: A corpus for the study of large-scale cough analysis algorithms,” Sep. 2020. [Online]. Available: http://arxiv.org/abs/2009.11644",
    "[28] A. Pal and M. Sankarasubbu, “Pay attention to the cough: Early diagnosis of COVID-19 using interpretable symptoms embeddings with cough sound signal processing,” in Proceedings of the ACM Symposium on Applied Computing, Mar. 2021, pp. 620–628.",
    "[29] P. Mouawad, T. Dubnov, and S. Dubnov, “Robust Detection of COVID-19 in Cough Sounds,” SN Computer Science 2021 2:1, vol. 2, no. 1, pp. 1–13, Jan. 2021.",
    "[30] H. Coppock, A. Gaskell, P. Tzirakis, A. Baird, L. Jones, and B. Schuller, “End-to-end convolutional neural network enables COVID-19 detection from breath and cough audio: A pilot study,” BMJ Innovations, vol. 7, no. 2, pp. 356–362, Apr. 2021.",
    "[31] M. Faezipour and A. Abuzneid, “Smartphone-based self-testing of COVID-19 using breathing sounds,” Telemedicine and e-Health, vol. 26, no. 10, pp. 1202–1205, Oct. 2020.",
    "[32] R. S. Irwin and J. M. Madison, “The Diagnosis and Treatment of Cough,” New England Journal of Medicine, vol. 343, no. 23, pp. 1715–1721, Dec. 2000.",
    "[33] A. B. Chang et al., “Cough in children: definitions and clinical evaluation,” Medical Journal of Australia, vol. 184, no. 8, pp. 398–403, 2006.",
    "[34] M. Soliński, M. Łepek, and Ł. Kołtowski, “Automatic cough detection based on airflow signals for portable spirometry system,” Informatics in Medicine Unlocked, vol. 18, p. 100313, Jan. 2020.",
    "[35] R. X. Adhi Pramono, S. Anas Imtiaz, and E. Rodriguez-Villegas, “Automatic Cough Detection in Acoustic Signal using Spectral Features,” in Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, Jul. 2019, pp. 7153–7156.",
    "[36] M. You et al., “Novel feature extraction method for cough detection using NMF,” IET Signal Processing, vol. 11, no. 5, pp. 515–520, Jul. 2017.",
    "[37] C. Infante, D. Chamberlain, R. Fletcher, Y. Thorat, and R. Kodgule, “Use of cough sounds for diagnosis and screening of pulmonary disease,” in GHTC 2017 - IEEE Global Humanitarian Technology Conference, Proceedings, Dec. 2017, vol. 2017-January, pp. 1–10.",
    "[38] H. Chatrzarrin, A. Arcelus, R. Goubran, and F. Knoefel, “Feature extraction for the differentiation of dry and wet cough sounds,” in MeMeA 2011 - 2011 IEEE International Symposium on Medical Measurements and Applications, Proceedings, 2011, pp. 162–166.",
    "[39] W. Thorpe, M. Kurver, G. King, and C. Salome, “Acoustic analysis of cough,” in ANZIIS 2001 - Proceedings of the 7th Australian and New Zealand Intelligent Information Systems Conference, 2001, pp. 391–394.",
    "[41] W. C. Dai et al., “CT Imaging and Differential Diagnosis of COVID-19,” Canadian Association of Radiologists Journal, vol. 71, no. 2, pp. 195–200, May 2020.",
    "[42] “COVID-19 Cough Classification using Machine Learning and Global Smartphone Recordings,” Computers in Biology and Medicine, vol. 135, p. 104572, Dec. 2020. [Online]. Available: https://arxiv.org/abs/2012.01926v2",
    "[43] A. Grünebaum, F. A. Chervenak, L. B. McCullough, J. W. Dudenhausen, E. Bornstein, and P. A. Mackowiak, “How fever is defined in COVID-19 publications: a disturbing lack of precision,” Journal of Perinatal Medicine, vol. 49, no. 3, pp. 255–261, Mar. 2021."

    ]
   

    # Appel de la fonction extract_state_of_the_art avec la liste de travaux connexes
    etat_art = extract_state_of_the_art(related_works)

    # Affichage du résultat
    print(etat_art)

    #calculer le score d'hallusination
    related_works_text = list_to_text(related_works)
    hallucination_score = score_hallucination(etat_art, related_works_text)

    # Affichez le score d'hallicination obtenu
    print("Score d'hallicination :", hallucination_score)

  #calculer la cosine_similarity
    cosine_similarity = calculate_cosine_similarity(etat_art, related_works_text)
 # Affichez le score cosine_similarity
    print("cosine similarity :", cosine_similarity)

 # Appel de la fonction pour extraire les références complètes uniques 
    unique_references = extract_unique_references(etat_art, liste_references)

    # Affichage des références complètes correspondantes pour les références uniques
    print("\nRéférences complètes correspondantes:")
    for reference in unique_references:
        print(reference)

    
    
    print(" Appel de la fonction pour extraire les références")

    noms_numeros = extraire_noms_numeros(etat_art)
    print(noms_numeros)

    references_extraites = extraire_references_par_nom_et_numero(liste_references, noms_numeros)

    # Affichage des références extraites avec une ligne vide entre chaque référence
    print("\nRéférences extraites:")
    for reference in references_extraites:
        print(reference)
        print()

if __name__ == "__main__":
    main()