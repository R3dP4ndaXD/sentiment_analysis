# Învăţare Automată

## 1. Introducere/Descriere Generală

Scopul este acela de a implementa un flux complet de analiză și procesare pentru rezolvarea sarcini de analiza de sentiment pe text în limba română utilizând rețele neurale pentru consolidarea înțelegerii antrenării, testării, analizei şi îmbunătățirii modelelor. 

---

### Analiza de Sentiment pe Text in Română

Veți rezolva o sarcină de analiză de sentiment pe text în limba română, folosind rețele neurale recurente pentru a clasifica recenzii în 2 categorii de sentiment: pozitiv sau negativ. 

Pentru problema de clasificare veți folosi setul de date `ro_sent`, disponibil pe HuggingFace. Acest set de date conține recenzii de produse și filme scrise în limba română, fiecare etichetată cu sentiment (pozitiv sau negativ). Setul de date include 17.941 de exemple pentru antrenare şi 11.005 exemple pentru testare. 

**Link către setul de date:** [https://huggingface.co/datasets/dumitrescustefan/ro_sent](https://huggingface.co/datasets/dumitrescustefan/ro_sent) 
**Link descărcare:**

* 
[train.csv](https://www.google.com/search?q=https://raw.githubusercontent.com/dumitrescustefan/Romanian-Transformers/examples/examples/sentiment_analysis/ro/train.csv) 


* 
[test.csv](https://www.google.com/search?q=https://raw.githubusercontent.com/dumitrescustefan/Romanian-Transformers/examples/examples/sentiment_analysis/ro/test.csv) 



#### Explorarea Datelor

1. 
**Analiza echilibrului de clase:** Realizați un grafic al frecvenței de apariție a fiecărei etichete (pozitive / negative) în setul de date de antrenare / test, folosind bar plot/count plot. 


2. **Statistici despre text:**
* Afişaţi distribuția lungimii textelor (număr de cuvinte sau caractere) pentru fiecare clasă de sentiment. 


* Identificați și vizualizați cele mai frecvente cuvinte pentru fiecare clasă. 


* 
*Notă: Diagramele din această secțiune sunt cele minimal cerute: Nu sunt singurele pe care le puteți face.* 





#### Tokenizare și Embedding Layer

Implementați procesul de preprocesare a textului şi embedding: 

* 
**Curăţarea datelor:** Primul pas constă în preprocesarea textului brut - eliminarea caracterelor speciale, normalizarea textului, eliminarea stopwords (opțional), etc. 


* **Tokenizare:** Transformaţi textele în secvenţe de token-uri (indici numerici). Specificați vocabularul utilizat şi cum sunt tratate cuvintele necunoscute. *Hint: spacy (Tokenization Using Spacy - GeeksforGeeks)* 


* 
**Embedding Layer:** Importati un model de embedding care transformă indicii în vectori denși. *Hint: fasttext (How to Use Pretrained FastText Word Vectors for English fxis.ai)* 


* 
**Padding:** Normalizați lungimea secvențelor la o lungime fixă. 



#### Utilizarea modelelor de Rețele Neurale Recurente

Sunt propuse spre evaluare următoarele arhitecturi de rețele neurale recurente: 

* 
**Arhitectură de tip RNN simplu**: Experimentați cu numărul de straturi, dimensiunea acestora şi dimensiunea hidden state-ului. 


* 
**Arhitectură de tip LSTM**: Puteți genera propria voastră arhitectură explorând: 


* Folosind straturi LSTM unidirecționale sau bidirecționale. 


* Numărul de straturi LSTM şi dimensiunea hidden state-ului. 


* Combinarea cu straturi liniare pentru partea finală a reţelei. 





Explorați modalități de îmbunătățire a performanței, având în vedere că aceasta este influențată nu doar de arhitectură, ci și de alegerea hiperparametrilor (e.g., optimizator, learning rate, batch size, regularizare), folosirea de straturi de normalizare (ex. Batch Norm), de straturi de tip pooling, tehnicile de augmentare folosite etc. 

Pentru îmbunătățirea performanței și a capacității de generalizare a modelului, explorați tehnici de augmentare specifice datelor text. Câteva tehnici recomandate: 

* 
**Random Swap/Delete/Insert:** Operații aleatorii asupra cuvintelor din propoziție. 


* 
**Back-Translation:** Traducerea textului în engleză și înapoi în română pentru a genera variaţii. 


* 
**Contextual Word Embeddings:** Folosirea unui model BERT românesc pentru înlocuirea contextuală a cuvintelor. 



Pentru augmentări, puteţi urmări indicaţii și exemple din: [https://neptune.ai/blog/data-augmentation-nlp](https://neptune.ai/blog/data-augmentation-nlp) 

> **ATENȚIE:** În raportul final trebuie să dați o justificare fiecărei alegeri arhitecturale, de augmentare sau de optimizare făcute. Altfel spus, trebuie să spuneți ce problemă avea antrenarea voastră, care v-a făcut să schimbați arhitectura, augmentările folosite, sau configurația optimizatorului. 
> 
> 

Pentru augmentări, arătați pe același grafic rezultatele curbelor de loss sau performanță, rezultate în urma folosirii vs. nefolosirii augmentărilor în cauză. 

---

### 2.3 Evaluarea modelelor

Raportul final trebuie să includă următoarele: 

* Pentru fiecare model antrenat trebuie raportat setup-ul de antrenare: descrierea arhitecturii utilizate; detalierea configuraţiei, incluzând cel puțin: optimizatorul folosit, learning rate (şi eventual scheduler), dimensiunea batch-urilor, numărul de epoci de antrenare, metodele de regularizare, precum şi orice alți hiperparametri relevanți utilizați. 


* Prezentați curbele de loss pentru antrenare şi validare pe același grafic, iar separat, pe un alt grafic, curbele pentru cel puţin o metrică relevantă pentru task, tot pentru antrenare şi validare. 


* Realizați un tabel în care liniile reprezintă configuraţiile arhitecturale și de optimizare, iar coloanele includ metricele de performanță (e.g., acuratețe, F1). 


* Creați matricea de confuzie corespunzătoare clasificării realizate, cel puțin pentru cea mai bună configurație asociată unui tip de arhitectură. 



---

## 3. Predarea Temei

Tema va fi încărcată pe Moodle însoțită de un raport sub formă de fişier PDF, care include: 

* Vizualizări şi diagrame relevante. 


* Raportarea completă a evaluării rețelelor antrenate. 


* Toate rezultatele, cantitative şi calitative, trebuie însoțite de interpretări și analiză în text (e.g., care este influența arhitecturii, cât de puternic este impactul hiperparametrilor asupra performanţei, care sunt clasele cu cele mai bune predicții). 


* Textul trebuie să includă toate elementele necesare reproducerii setup-ului experimental (e.g., tipuri de augmentări, proceduri de antrenare, valori ale hiperparametrilor). 
