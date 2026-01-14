# Raport: Analiza Sentimentului pe Text în Limba Română

## 1. Introducere și Descriere Generală

### Scopul Proiectului
Acest proiect are ca scop dezvoltarea și evaluarea unui subsistem de analiză a sentimentului (Sentiment Analysis) pentru limba română, utilizând rețele neurale recurente (RNN, LSTM). Obiectivul principal este clasificarea automată a recenziilor în două categorii: **pozitiv** și **negativ**.

### Setul de Date
Pentru antrenarea și evaluarea modelelor, s-a utilizat setul de date **`ro_sent`**, disponibil public. Acesta este constituit din recenzii de produse și filme, fiind un benchmark relevant pentru prelucrarea limbajului natural (NLP) în limba română.

- **Dimensiune**:
    - Set de antrenare: 17.941 exemple.
    - Set de testare: 11.005 exemple.

Problema este abordată ca o sarcină de clasificare binară supervizată.

---

## 2. Explorarea Datelor (EDA)

În această etapă, am analizat distribuția datelor și caracteristicile textuale pentru a înțelege mai bine natura task-ului și eventualele provocări.

### 2.1 Analiza Echilibrului de Clase

![Class Balance](./plots/eda/class_balance.png)

Dat fiind dezechilibru semnificativ între clasile pozitive și negative (aproximativ 2:1), modelele ar putea tinde să favorizeze clasa majoritară. Pentru a mitiga acest risc, în antrenarea fara augmentare am folosit`WeightedRandomSampler`, iar ulterior, dupa ce am aplicat augmentare offline pentru clasa minoritară si am obținut o distribuție echilibrată, am renuntat la aceasta tehnica`.

### 2.2 Statistici despre Text

![Text Length Distribution](./plots/eda/word_count_distribution.png)


Majoritatea recenziilor au o lungime moderată, ceea ce justifică utilizarea unei lungimi maxime a secvenței (`max_seq_len`) de 160 de tokeni pentru a acoperi 95% din recenziile din setul de date.

### 2.3 Analiza Lexicală

Pentru a intui ce cuvinte sunt discriminative pentru fiecare clasă, am vizualizat cele mai frecvente cuvinte (excluzând stop-words banale, dar păstrând cuvintele relevante pentru sentimente).

![Top Words](./plots/eda/top_words_filtered.png)

De asemenea, norii de cuvinte (Word Clouds) oferă o perspectivă vizuală asupra vocabularului specific fiecărui sentiment:

![Word Clouds](./plots/eda/wordclouds.png)

De asemenea, graficul de mai jos evidențiază cuvintele cele mai distinctive (cu cea mai mare diferență de frecvență relativă) pentru fiecare clasă:

![Distinctive Words](./plots/eda/distinctive_words.png)

Această analiză confirmă că anumite cuvinte (adjective precum "rau" vs "bun"/"bine"/"bună", "recomand", "multumit" vs. "slab", "evitati", "pierdere", "ingrozitor") sunt indicatori puternici ai sentimentului, justificând abordarea bazată pe embeddings și rețele recurente care pot capta contextul acestor cuvinte.

## 3. Preprocesare și Embeddings

Procesarea textului este esențială pentru a transforma recenziile brute într-un format compatibil cu rețelele neurale. Pipeline-ul de preprocesare (`src/preprocessing/text.py`) a inclus următorii pași:

1.  **Curățarea Textului**:
    - Normalizare Unicode.
    - Standardizarea diacriticelor.
    - Păstrarea punctuației relevante și eliminarea caracterelor speciale(ex. taguri html) care introduc zgomot.

2.  **Tokenizare**:
    - Utilizarea **spaCy** (`ro_core_news_sm`) pentru o segmentare corectă a cuvintelor.

3.  **Gestionarea Vocabularului și Embeddings**:
    - **Vocabular**: Construit pe baza setului de antrenare, păstrând cuvintele cu o frecvență minimă de 2 apariții pentru a reduce zgomotul.
    - **Embeddings**: Am utilizat vectori pre-antrenați **FastText** (`cc.ro.300.bin`), cu o dimensiune de 300. Aceștia captează relații semantice între cuvinte și permit modelului să generalizeze mai bine, chiar și pentru cuvinte care apar rar în setul de antrenare.
    - Stratul de embedding a fost complet înghețat. Cuvintele din afara vocabularului (OOV) au primit reprezentări generate din sub-cuvinte (folosind FastText), iar acestea nu au fost ajustate în timpul antrenării.

---

## 4. Arhitecturi de Modele

Am experimentat cu următoarele arhitecturi de bază, toate în configurație **Bidirecțională**:

### 4.1 RNN (Recurrent Neural Network)
Am început cu un model de bază (Simple RNN) pentru a stabili o referință.
- **Arhitectură**: Embedding -> RNN -> FC Layer.
- **Configurația de bază**: Bidirecțional, 2 straturi, Hidden Dim 128, Dropout 0.5.
- **Limitări**: RNN-urile simple suferă de problema "vanishing gradient", având dificultăți în captarea dependențelor pe termen lung în texte lungi.

### 4.2 LSTM (Long Short-Term Memory)
LSTM-urile au fost arhitectura principală, fiind concepute special pentru a rezolva problema gradientului evanescent.
- **Configurația de bază**: Bidirecțional, 2 straturi, Hidden Dim 128, Dropout 0.5.
- **Mecanisme adiționale**:
    - **Attention Mechanism**: Am implementat un model `BiLSTMWithAttention` care calculează o sumă ponderată a stărilor ascunse ale LSTM-ului. Aceasta permite modelului să se "concentreze" pe cuvintele cele mai relevante pentru sentiment (de exemplu, "excelent", "oribil") atunci când generează predicția finală.

---

## 5. Augmentarea Datelor

Pentru a îmbunătăți generalizarea și a combate overfitting-ul, am aplicat tehnici de augmentare a datelor, atât offline (înainte de antrenare), cât și online (dinamic, în timpul antrenării):

### 5.1 Tehnici Utilizate
1.  **Fără Augmentare (No Aug)**: Setul de date original.
2.  **Augmentare Offline (Balanced)**: 
    - Am aplicat **Back-Translation** (Română -> Engleză -> Română) **doar pentru clasa minoritară (Negativ)**.
    - Scopul a fost atingerea echilibrului perfect între clase înainte de antrenare.
3.  **Full Augmentation (EDA+)**: 
    - Pe lângă setul echilibrat offline, am aplicat augmentări **online** dinamice (Random Swap, Delete, Synonym Replacement, contextual insert, contextual replace) în timpul antrenării, cu o probabilitate de 10%.

---

## 6. Evaluare și Rezultate

### 6.1 Evoluția Modelelor (Best Configurations)

Pentru fiecare tip de arhitectură, prezentăm curbele de învățare pentru cea mai bună configurație (Full Augmentation).

#### Simple RNN (Bidirecțional)
![Simple RNN Learning Curves](./plots/simple_rnn_bi_hd128_nl2_eda_plus_p0.1/loss_curves.png)
*Figura 4: Loss (Train vs Val) pentru Simple RNN.*

#### LSTM (Bidirecțional)
![LSTM Learning Curves](./plots/lstm_bi_hd128_nl2_eda_plus_p0.1/loss_curves.png)
*Figura 5: Loss (Train vs Val) pentru LSTM.*

#### BiLSTM cu Atenție
![Attention Model Learning Curves](./plots/bilstm_attention_hd128_nl2_eda_plus_p0.1/loss_curves.png)
*Figura 6: Loss (Train vs Val) pentru BiLSTM Attention. Se observă o convergență stabilă și un loss mai mic pe validare față de celelalte modele.*

### 6.2 Analiza Augmentării

Comparăm impactul celor trei strategii de augmentare (No Aug vs. Offline Balanced vs. Full EDA) asupra performanței (F1-Score pe Validare).

![Augmentation Comparison RNN](./plots/comparison/augmentation/augmentation_comparison_val_f1.png)

*Figura 7: Impactul augmentării asupra F1-Score.*

Se observă că **echilibrarea offline (Offline Augmentation)** aduce un câștig major de performanță față de varianta fără augmentare (linia neagră/albastră), în special pentru modelele RNN și BiLSTM. Adăugarea augmentării online (Full Augmentation) aduce un plus de regularizare, prevenind overfitting-ul pe termen lung.

### 6.3 Comparație Finală

Ierarhia finală a modelelor, ordonată după F1-Score pe setul de test.

![Final Ranking](./plots/comparison/comparison_f1_sorted.png)

*Figura 8: Comparația metricilor pe setul de Test.*

### 6.4 Tabel Rezultate

| Experiment | Precizie | Recall | F1-Score | Acuratețe |
| :--- | :--- | :--- | :--- | :--- |
| **BiLSTM Attention (Balanced)** | 0.8550 | 0.9289 | **0.8904** | 0.8717 |
| **BiLSTM Attention (Full Aug)** | 0.8952 | 0.8689 | 0.8819 | 0.8693 |
| **LSTM (Balanced)** | 0.9128 | 0.8460 | 0.8782 | 0.8682 |
| **LSTM (Full Aug)** | 0.9146 | 0.8443 | 0.8780 | 0.8683 |
| **LSTM (No Aug)** | 0.8880 | 0.8650 | 0.8763 | 0.8630 |
| **Simple RNN (Full Aug)** | 0.8588 | 0.8705 | 0.8646 | 0.8470 |
| **Simple RNN (Balanced)** | 0.8465 | 0.8773 | 0.8616 | 0.8418 |
| **BiLSTM Attention (No Aug)** | 0.9242 | 0.8010 | 0.8582 | 0.8514 |
| **Simple RNN (No Aug)** | 0.8753 | 0.8161 | 0.8447 | 0.8315 |

*Tabel 1: Rezultate detaliate pe setul de testare. Ordinea este descrescătoare după F1-Score.*

---

## 7. Concluzii

1.  **Dinamica Augmentării**: Echilibrarea setului de date prin Back-Translation (Offline) a fost **factorul decisiv** pentru performanță, crescând semnificativ Recall-ul (capacitatea de a detecta recenziile negative). Modelele fără augmentare tind să aibă o precizie mare dar un recall mic (bias spre clasa pozitivă).
2.  **Superioritatea Atenției**: Modelul **BiLSTM cu Atenție** a obținut cele mai bune rezultate (**F1 ~88%**), demonstrând capacitatea de a selecta informația relevantă.
3.  **Eficiența RNN**: Chiar și un RNN simplu, dacă este antrenat pe date echilibrate și augmentate, poate atinge performanțe respectabile (F1 86.5%), depășind modele complexe antrenate pe date nebalansate.
