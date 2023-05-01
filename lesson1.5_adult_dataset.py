#!/usr/bin/env python
# coding: utf-8

# # [Adult Dataset](https://bit.ly/3rvRB8j)
# 
# ### Celem jest zrobić predykcję, czy dana osoba osiągnie przychód ponad 50 tysięcy (dolarów) w rok.
# 
# Innymi słowy, z punktu widzenia uczenie maszynowego należy wykonać klasyfikację binarną. W praktyce to oznacza, że są dwie możliwe odpowiedzi: np. "tak" czy "nie" lub "kot" czy "pies" itd.
# 

# ### Krok po kroku 
# 
# Jeśli wolisz najpierw słuchać i oglądać, to obejrzyj nagranie poniżej, które omawia tę lekcję. 

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe style="height:500px;width:100%" src="https://bit.ly/31v8THY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


# In[1]:


import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

import qgrid

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Wczytujemy dane
# Dane są przechowywane w formacie [HDF5](https://bit.ly/3w6jbwk). To jest binarny format, który jest dość wygodny (zwykle trzymamy dane w tym formacie zamiast .csv). Między innymi umożliwia to zapisanie więcej niż jeden zbiór danych do jednego pliku.

# In[2]:


train = pd.read_hdf('../input/train.adult.h5')


# ## Pytania 1.5.1
# O zbiorze danych:
# 1. Ile jest obiektów (wierszy)?
# 2. Ile jest cech/features (column)?
# 3. Ile jest potrzebnej pamięci?
# 4. Jakie są dane?
#     4.1 Czy są zmienne **ilościowe** lub **ciągłe** (takie jak wiek, wzrost, waga...) i jak wygląda ich rozkład?
#     4.2 Czy są zmienne **nominalne** (i jak dużo unikalnych wartości ma każda z nich)?
#     4.3 Czy są zmienne **porządkowe** (np. bardzo dobry, dobry, zły)?
#     4.4 Czy są daty?
# 5. Co jest **zmienną docelową** (ang. *target variable*)?
#     5.1 Jakiego typu jest ta zmienna?
#     5.2 Czy są jakieś zmienne, które są mocno skorelowane ze zmienną docelową?
# 6. Jak dużo zmiennych ma brak danych?
# 7. Czy jest zauważalna [(liniowa) korelacja](https://bit.ly/3lZa1gq) pomiędzy cechami (w szczególności powiązane ze zmienną docelową)?

# ## Odpowiedzi/wnioski 1.5.1
# 
# 1.  32561
# 2.  15
# 3.  2.9+MB
# 4.  Dane opisują cechy ludzi: płeć,wykształcenie,zawód, stopień edukacji, ilość godzin pracy, kraj pochodzenia itp.    
#     4.1  Numerical fetures: (Age, fnlwgt, Capital Gain, Capital Loss, Hours per week) 
#     4.2  Categorical fetures: (Workclass, Martial Status, Occupation, Relationship, Race, Sex, Country, Target)
#     4.3  zmienne są skategoryzyowane, ale posiadają wewnętrzną rangę    
#     4.4  Nie    
# 5.  Informacje, czy osoba osiągnie próg dochodu    
#     5.1  Zmianna logiczna o charakterze binarnym    
#     5.2  Tak, największa korelacja jest dla Martial Status: Married-civ-spous i Edukacion, najmnijesza: Never-married     
# 6.  Workclass, Occupation, Country   
# 7.  Education-Num, Sex

# **Podpowiedź**: spróbuj użyć `train.info()`, `train.describe()` i `train.corr()`, żeby znaleźć odpowiedzi. Również funkcje jak `.unique()` lub `.nunique()` mogą być przydatne (pierwsza zwraca wszystkie unikalne wartości dla danej cechy, druga ilość unikalnych cech), np. `train.Age.nunique()` zwróci 73, to oznacza, że mamy 73 różne wartości dla cechy wiek.
# 
# Dość często patrząc na dane, funkcja `.value_counts()` jest także przydatna, bo zlicza, ile razy pojawiła się konkretna wartość (np. ile mamy ludzi, którzy są w wieku 36 lat itd), można spróbować w ten sposób `train.Age.value_counts()`.
# 
# *Nawiasem mówiąc*: `pd.describe()` pomaga zrozumieć tak zwane [5 liczb statystycznych](https://bit.ly/2O2caeV).

# In[3]:


#info


# In[4]:


#describe


# In[5]:


#corr


# In[ ]:


*Nawiasem mówiąc*: `pd.corr()` wygląda ciekawiej, jeśli to zwizualizujemy. Na szczęście da się to zrobić bardzo prosto.


# In[7]:


plt.rcParams['figure.figsize']=(20,10)
sns.heatmap(train.corr(), vmax=1., vmin=-1., annot=True, linewidths=.8, cmap="YlGnBu");


# Mówiąc o korelacji, przede wszystkim chodzi o korelację liniową (domyślnie `pd.corr()` używa [Pearsona](https://bit.ly/3lZa1gq)). 
# 
# Po drugie, chodzi o korelację pomiędzy zmiennymi ciągłymi (numerycznymi).
# 
# Co zrobić ze zmiennymi kategorialnymi (albo jeszcze ciekawszą kombinacją: zmienna kategorialna i zmienna ciągła)? To jest większa "rozkmina", która wykracza poza ten kurs. Natomiast na tej [stronie](https://bit.ly/2PxcORX) można zobaczyć różne kombinacje i inspiracje, jak sobie z tym radzić.

# In[8]:


#unique/nunique/value_counts


# ## Popatrz na dane
# Pierwsze 5 czy 10 wierszy
# 
# *Inaczej mówiąc*: funkcja `.head()` domyślnie pokazuje tylko pierwsze **5 wierszy**, ale możesz to zmienić przekazując parametr, np. pokazać pierwsze **20 wierszy** `.head(20)`. 
# 
# Możesz również spróbować użyć funkcji `.sample(10)`. Ta funkcja będzie losować `n` wierszy, gdzie `n` podajesz jako argument. Jeśli zbiór danych jest relatywnie mały (mniej niż milion), to może być to lepszą opcją, bo po uruchomieniu za każdym razem możesz zobaczyć coś więcej niż tylko pierwsze 10 wierszy. 
# 
# W ten sposób możesz "wyłapać" ciekawsze przypadki mniejszym kosztem (spróbuj kilka razy uruchomić tę linijkę).

# In[9]:


train.sample(10)


# ### Oficjalny opis danych
# 
# - **Age** – Wiek osoby.
# - **Workclass** – Rodzaj pracodawcy, jaki ma dana osoba. Niezależnie od tego, czy są to instytucje rządowe, wojskowe, prywatne i inne.
# - **fnlwgt** – Pewna liczba (mało istotne jaka). Będziemy ignorować tę zmienną.
# - **Education** – Najwyższy poziom wykształcenia osiągnięty dla tej osoby.
# - **Education-Num** – Najwyższy poziom wykształcenia w formie liczbowej.
# - **Marital** – Stan cywilny osoby.
# - **Occupation** – Wykonywany zawód.
# - **Relationship** – Trochę trudniejsze do wytłumaczenia. Zawiera wartości związane z rodziną, takie jak mąż, ojciec itd.
# - **Race** – opisy poszczególnych ras człowieka. 
# - **Sex** – Płeć.
# - **Capital Gain** – Zyski kapitałowe rejestrowane.
# - **Capital Loss** – Straty kapitałowe rejestrowane.
# - **Hours per week** – Liczba godzin przepracowanych w tygodniu.
# - **Country** – Kraj pochodzenia danej osoby.
# - **Target** – Zmienna logiczna (mniejsza, równa się lub większa). Niezależnie od tego, czy dana osoba zarabia więcej niż 50 000$ rocznie.
# 
# 
# Więcej o danych można przeczytać [tutaj](https://bit.ly/39lBFPy).
# 
# Spędź proszę chwilę czasu zastanawiając się nad danymi. Pomyśl (na razie teoretycznie), która cecha może mieć większy lub mniejszy wpływ na jakość modelu. Będzie dobrze, jeśli zanotujesz swoje pomysły, a później to będziemy weryfikować. 
# 
# 
# ## QGrid
# Ciekawostką jest to, że jest narzędzie [qgrid](https://bit.ly/31rXj0j) i funkcja `.show_grid()`, które umożliwia analizowanie danych bez znajomości `pandas`. To może być krok przejściowy :). Zobacz, jak to wygląda.

# In[10]:


qgrid.show_grid(train, show_toolbar=True)


# Teraz przełączamy na się kodowanie, bo jednak warto to poznać, bo daje większe możliwości.
# 
# 
# ## Braki w danych
# Jest kilka cech w których są braki. Kod poniżej wypisze, które to cechy i jak dużo jest braków?

# In[11]:


def check_missing():
    for column in train.columns:
        missing = column, train[column].isnull().sum()
        if missing[1] == 0: continue

        print(missing)
        
check_missing()


# ### Powstaje pytanie, co z tym robić? 
# Jest wiele różnych technik, ale na początek możemy zastosować najprostszą: "olać to" :). Tylko życie jest trochę bardziej skomplikowane i później "wysypie się" kod w innym miejscu, bo oczekuje tego, że będą dane. Dlatego zróbmy coś bardzo prostego, a później (jak zbudujemy cały workflow do testowania w praktyce), będziemy patrzeć, co jest lepsze.
# 
# Na początek zamiast wszystkich brakujących wartości wstawimy: -1. 
# 
# ### Dlaczego -1? 
# Coś trzeba wstawić :), ale liczba ujemna dlatego, żeby nie powtarzać się. Na przykład dość częstym błędem jest wstawianie 0. Problem pojawia się wtedy, gdy jest różnica pomiędzy wartością 0, która była od samego początku i wartością 0, która pojawiła się na skutek zmiany brakującej wartości. Innymi słowy, niechcący będziemy wtedy "oszukiwać" nasz model.
# 
# Wywołajmy funkcję z pandas `.fillna()`, która wypełnia wszystkie brakujące wartości liczbą podaną przez nas.

# In[12]:


train = train.fillna(-1)


# Sprawdźmy na wszelki wypadek, czy to zadziała dobrze...
# 
# *Nawiasem mówiąc*, wypracuj nawyk weryfikowania wszystkiego co najmniej raz (tym bardziej tych oczywistych rzeczy), co jakiś czas wynik będzie zaskakiwać.

# In[13]:


check_missing()


# Wygląda, że jest OK - mam wszystkie wartości :). Na wszelki wypadek można jeszcze zobaczyć `train.info()` (ale już bez przesady).

# ## Metryka sukcesu
# 
# 
# Trzeba wybrać, w jaki sposób mierzyć jakość modelu. Mówimy o klasyfikacji (binarnej), więc zwykle pierwszą rzeczą, która przychodzi do głowy to dokładność (ang. *accuracy*). Nie zawsze ta metryka jest dobra (zwłaszcza, jeśli mówimy o skrzywionych [niezbilansowanych] zbiorach danych). 
# 
# Jest tak zwana [tablica pomyłek](https://bit.ly/3w751er) ([confusion matrix](https://bit.ly/3sJrHz6)). Na razie tylko wspominam o istnieniu tej tablicy, ale jeszcze do niej wrócimy :).
# 
# Dokładność (ang. *accuracy*) liczy się bardzo prosto:
# $$ \frac{wszystkie\ poprawne\ odpowiedzi}{wszystkie\ odpowiedzi}$$
# 
# Pobawmy się trochę, żeby wyczuć to lepiej. W każdym wierszu są dwie listy, z lewej strony prawidłowa odpowiedź, a z prawej strony jest predykcja (czyli "odpowiedź" modelu).

# In[14]:


values = [
      #odpowiedź          #predykcja
    ([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]), #dokładność 100% 
    ([1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]), #dokładność 50% 
    ([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]), #dokładność 50% 
    ([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1]), #dokładność ~67% 
]

for true_values, pred_values in values:
    score = accuracy_score(true_values, pred_values)
    print("score: [{0}], true: {1}, pred: {2}".format(score, true_values, pred_values))


# ## Pytania 1.5.2
# 
# 1. Jaka jest minimalna i maksymalna wartość dokładności?
# 2. Jaka jest minimalna i maksymalna wartość dokładności w przykładzie, który podałem (zakładając, że odpowiedzi są stałe)?
# 3. Kiedy dokładność jest równa zero?
# 4. Jak zachowuje się dokładność na niezbilansowanych zbiorach danych (np. prawie wszystko 0 i tylko kilka jedynek, standardowy przykład dla zadań typu `fraud detection`)?

# ## Odpowiedzi/wnioski 1.5.2
# 1. od 0 do 1    
# 2. od 0,5 do 1  
# 3. Kiedy wszystkie odpowiedzi są błędne    
# 4. Fałszywe wysokie wyniki, ze względu na przypisanie wszystkim rekordom tego samego wyniku    

# ## Basic Model
# 
# Zbuduj najprostszy model...
# 
# **Podpowiedź:** użyj [DummyClassifier](https://bit.ly/3w7O17T)
# 
# Przed trenowaniem modelu trzeba przygotować dane. Zróbmy to najszybciej, jak tylko się da. Dla `DummyClassifier` wszystko jedno jaka jest cecha (ważne, żeby tylko rozmiar macierzy się zgadzał). Jest to istotne, ponieważ model patrzy tylko na zmienną docelową i w zależności od strategii zwraca zawsze wartość średnią lub medianę (dla przykładu załóżmy, że strategią będzie wartość średnia, więc bierzemy wszystkie odpowiedzi, znajdujemy wartość średnią i model zawsze będzie "odpowiadał", że wynik to wartość średnia).
# 
# **Zapamiętaj**: wszystkie modele wewnątrz oczekują na liczby (a jeśli jest inaczej, to oznacza, że model jest na tyle sprytny, że jest w stanie sam zrobić to przekształcenie). W naszym przypadku `Target` jest w tej chwili tekstowy, więc trzeba to przerzucić do klasy binarnej. 
# 
# Naszym zadaniem było przewidzieć, czy człowiek będzie zarabiał więcej niż 50K, odpowiedź "tak" jest dla tego warunku: `train['Target'] != '<=50K'`.
# 
# *Nawiasem mówiąc*, `X` to jest macierz, ale `y` to wektor i to dlatego `X` jest pisany z dużej litery (`y` z małej), taka jest konwencja.

# In[15]:


train['target_cat'] = (train['Target'] != '<=50K').astype('int8')

X = train[ ['Age'] ].values ##to macierz, dlatego są podwójne nawiasy, natomiast teraz to jest macierz z jedną kolumną
y = train[ 'target_cat' ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)


# Zamiast `YOUR CODE HERE` trzeba podać model,  którego będziesz używać (np. dummy czy liniowy).

# In[17]:


#model = ...YOUR CODE HERE

model.fit(X_train, y_train)    #podajemy X_train, y_train
y_pred = model.predict(X_test) #sprawdzamy na X_test

print(accuracy_score(y_test, y_pred))


# Dobrą praktyką będzie poprzednią komórkę "wrzucić" do jednej funkcji i jako argument będzie podawany model.

# In[18]:


def train_and_predict(model, X, y, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
    
    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test) 

    return accuracy_score(y_test, y_pred)


# Sprawdź, czy wynik jest podobny.

# In[19]:


train_and_predict(DummyClassifier(strategy = 'stratified'), X, y, test_size=0.33)


# ## Model Basic+
# 
# Zróbmy trochę bardziej złożony model. Zwykle bierze się liniowy. Teraz już model "patrzy" na cechy, więc na początek można używać nadal jednej ("`Age`"), ale w kolejnych iteracjach dodawać kolejne cechy (ale pamiętaj, o tym, że jeśli cecha nie jest liczbowa, to trzeba zrobić transformację).
# 
# **Podpowiedź**: użyj [LogisticRegression](https://bit.ly/3rCEu5q)
# 
# *Nawiasem mówiąc*, problem który rozwiązujemy jest klasyfikacją (binarną). Czemu zatem model zawiera nazwę "regression"? Jest to mylące, ale nazwa już się przyczepiła ... W pewnym sensie da się wytłumaczyć, dlaczego akurat nazwa jest taka, ale fakt jest faktem, jest ona myląca :).

# In[20]:


#train_and_predict(..., X, y) #zamiast kropek, trzeba wstawić model np. 


# ## Feature Engineering
# Zacznijmy generować nowe cechy.
# 
# Zmienne kategorialne będą zamienione na liczby. Możemy to zrobić co najmniej na dwa sposoby.
# 
# ### 1. Label Encoding
# Łatwiejszy sposób polega na tym, aby do każdej unikalnej wartości przypisać ID
# np. `jabłko => 1, gruszka => 2, pomarańcza => 3` itd.
# 
# ##### Zalety
# Jest szybkie i proste. Tanie z punktu widzenia zasobów (nadal mamy jedną kolumnę, zwykle potrzebujemy mniej miejsca niż przed tym).
# 
# ##### Wady
# `jabłko` to nie jest to samo co `1`, chociażby dlatego że `1` jest mniejsze niż `2`, czy jabłko jest mniejsze niż gruszka? Innymi słowy, niechcący próbujemy "oszukać" nasz model skrzywiając rzeczywistość.
# 
# ### 2. [One Hot](https://bit.ly/39j8GMv) Encoding
# Dlatego pojawia się od razu co najmniej jeszcze jedna alternatywa, która próbuje rozwiązać problem poprzedniego sposobu (ale pamiętasz, że jak to zwykle bywa "coś za coś").
# 
# Działa to tak, że dla każdej unikalnej wartości pojawia się pytanie: czy to jest X? Na przykład, "czy to jest jabłko?" lub "czy to jest gruszka?".
# 
# 
# ##### Zalety
# Rozwiązaliśmy poprzedni problem (kiedy "jabłko" mogło być "mniejsze", niż gruszka).
# 
# ##### Wady
# Potrzebuję znacznie więcej zasobów. Dla każdej unikalnej wartości pojawia się nowa kolumna (szczególnie to zaczyna boleć, gdy unikalnych wartości są dziesiątki, a tym bardziej setki).
# Również przy większej ilości cech model jest bardziej podatny na przeuczenie się.
# 
# **Zapamiętaj** Dla obu przypadków trzeba z góry znać wszystkie unikalne wartości, bo inaczej pojawia się problem z interpretacją (i tak też się dzieje, są na to dodatkowe techniki, do tego jeszcze wrócimy).
# 
# *Nawiasem mówiąc*, nie ma jednej dobrej porady, co jest lepsze (bo inaczej istniałaby tylko jedna [dobra] technika). Jak to zwykle bywa w życiu, trzeba próbować zaczynając od najprostszych rzeczy, jeśli jest to wystarczająco dobre, to cieszymy się, jeśli nie - próbujemy innych metod bardziej złożonych.
# 
# 
# Zróbmy eksperyment, żeby lepiej zrozumieć w praktyce, jak to działa... Dla przykładu bierzemy kolumnę (cechę) `Race`, która ma 5 unikalnych wartości. Po zastosowaniu `one-hot encoding` pojawi się pięć nowych kolumn, gdzie wartości w kolumnach będą 0 lub 1. Na przykład, `isWhite`: 0 lub 1 (tak lub nie) itd.

# In[21]:


train['Race'].unique()


# W pandas jest funkcja `.get_dummies()`, która umożliwia w prosty sposób uruchomić `one-hot-encoding` transformację.

# In[22]:


pd.get_dummies( train['Race'] ).head()


# ## Label Encoding
# Jednym z najprostszych sposobów jest użycie funkcji pandas `pd.factorize()` i pobranie pierwszego elementu (drugim jest label, a nam jest potrzebne tylko ID).
# 
# Również można używać funkcji z `sklearn`: [LabelEncoder](https://bit.ly/3ruF591). Tylko wtedy będzie więcej linii kodu :).

# In[23]:


pd.factorize( ['a', 'b', 'c', 'a', 'a', 'c'] )


# Funkcja `pd.factorize()` przyjmuje listę (zwykle chodzi o ciąg znaków, oczywiście również może być to ciąg liczb, tylko po co konwertować liczby w liczby?). Przypisuje unikalny ID dla danej wartości i zwracane są dwie listy. Pierwsza lista z ID'kami, a druga to lista unikalnych wartości.
# 
# W przypadku powyżej, unikalne ID wyglądają tak:
# - a => 0
# - b => 1
# - c => 2
# 
# #### input
# `['a', 'b', 'c', 'a', 'a', 'c']`
# 
# #### output
# `[0, 1, 2, 0, 0, 2]`
# 
# W jaki sposób `pd.factorize()` to robi? 
# 1. Na początku funkcja tworzy pustą listę `[]`.
# 2. Następnie bierze naszą kolumnę i po kolei sprawdza każdy element z tej kolumny.
# 3. Dla każdego elementu sprawdza, czy ten element widzi po raz pierwszy, czy po raz kolejny.
# 4. Jeśli po raz pierwszy, to dodaje ten element do tej listy, którą utworzył na początku i indeks tej wartości w tej liście będzie liczbą, którą będzie od tej pory przypisywał do tej wartości. 
# 5. Jeśli natomiast trafia na element, który już był wcześniej, to po prostu przypisuje mu jego indeks z tej listy.
# 
# Wynik na końcu będzie składał się z 2-óch elementów.
# Pod indeksem 0 będzie nasza kolumna zamieniona na liczby. Pod indeksem 1 będzie nasza lista unikalnych wartości z kolumny.
# Jeśli więc zrobisz np. `pd.factorize(..)[1][0]` to zobaczysz, jaka wartość została zamieniona na 0 :).
# 
# Wracamy do naszego przykładu.

# In[24]:


pd.factorize( train['Workclass'] )[0]


# Spróbujmy `LabelEncoder`, ale od razu powiem, że zaraz wyskoczy błąd. Nie przejmuj się tym. Za chwilę zrozumiemy, co się stało.
# 
# *Nawiasem mówiąc*, jak pojawia się jakiś błąd, to staraj się przewijać na sam koniec, tam zwykle są najciekawsze informacje (np. w ostatniej linijce).

# In[26]:


le = LabelEncoder()
le.fit(train['Workclass'])
le.transform(train['Workclass'])


# No i właśnie `TypeError: Encoders require their input to be uniformly strings or numbers` narzeka na to, że mamy `str` i `int`.  To popatrzmy, jakie mamy unikalne wartości dla `Workclass`.

# In[27]:


train.Workclass.unique()


# Pamiętaj, że `Workclass` miał brakujące wartości i na szybko wrzuciliśmy tam -1? No właśnie w tym przypadku oczekuje od nas tylko `string`. Szybko to naprawiamy i...

# In[28]:


train.Workclass = train.Workclass.map(lambda x: str(x))

le = LabelEncoder()
le.fit(train['Workclass'])
le.transform(train['Workclass'])


# Jak widać, działa :). Super! Tylko mało tego, że trzeba napisać więcej kodu, to jeszcze musieliśmy trzymać wszystkie dane jednego typu (`string`), natomiast `pd.factorize()` poradził sobie z tym bardzo dobrze. Również `.factorize()` jest szybszy, to jest bardzo zauważalne na większych zbiorach danych (dlatego lubię `.factorize()`).
# 
# Ale...
# 
# `LabelEncoder()` ma też swoje zalety, jedną z największych jest łączenie różnych elementów w tak zwany [pipeline](https://bit.ly/3cxZxBQ). Na razie nie potrzebujemy tego, więc tylko warto wiedzieć, że coś takiego jest.
# 
# ### Czy pamiętasz, co robiliśmy? 
# No właśnie chcemy przygotować cechy kategorialne, żeby móc wytrenować model. Model oczekuje liczb. To już wiemy, jak konwertować zmienne kategorialne, musimy tylko je znaleźć. Podpowiem Ci jeden `trick`, który zwykle działa dobrze w większości przypadków.

# In[29]:


train.info()


# Dość często to, co pokazuje się w `.info()` jako "object" jest zmienną kategorialną (choć czasem to może być data lub jakaś pokręcona liczba). Stąd wniosek jest taki, że żeby znaleźć wszystkie zmienne kategorialne trzeba odfiltrować po typie (zostawić tylko "object").

# In[30]:


train.select_dtypes(include=[np.object]).columns


# In[31]:


cat_feats = train.select_dtypes(include=[np.object]).columns

for cat_feat in cat_feats:
    train['{0}_cat'.format(cat_feat)] = pd.factorize( train[cat_feat] )[0]


# **Zapamiętaj**, jak transformujesz zmienną, to twórz lepiej nową kolumnę, nie nadpisuj (chyba że już było sprawdzone kilka razy), bo inaczej możesz stracić oryginalną wartość. 
# 
# *Inaczej mówiąc*, dodaję prefix `_cat`, i to nie chodzi o to, że lubię koty (chociaż to też prawda😺), to bardziej skrót od zmiennej kategorialnej (`category`)... aha i to moja własna konwencja, której się trzymam. Natomiast możesz wymyślić swoją, jeśli masz powody :).
# 
# To wróćmy do naszego liniowego modelu i dodajmy mu więcej cech (kategorialne i ilościowe).

# In[32]:


feats = train.select_dtypes(include=[np.int]).columns.values
feats


# **Target_cat** to jest zmienna docelowa, więc trzeba to usunąć z cech, a poza tym jest ok.

# In[33]:


feats = feats[:-1] #-1 oznacza ostatni element, który wycinamy
feats


# In[34]:


X = train[ feats ].values
y = train['Target_cat'].values


# In[35]:


train_and_predict(YOUR_CODE_HERE, X, y, test_size=0.33) ### podaj model (np. LogisticRegression())


# ## Zadania domowe
# 
# Kilka wskazówek jak lepiej badać dane. Dla zmiennych numerycznych używaj `.hist()`.

# In[36]:


train['Education-Num'].hist();


# Dla innych zmiennych (np. kategorialnych) to nie przejdzie i wtedy można zrobić zadanie w ten sposób:

# In[37]:


train['Education'].value_counts().plot(kind='bar');


# lub można użyć `seaborn`:

# In[38]:


plt.figure(figsize=(15, 5))
sns.countplot(x='Education', data=train);
plt.xticks(rotation=90);


# ## Zadanie 1.5.3
# 
# Spróbuj zbadać kolejne cechy, to może być przydatne, żeby zacząć tworzyć lepsze cechy.

# In[39]:


print(train.columns)
train.info()

plt.figure(figsize=(15, 5))
feats = train.select_dtypes(include=[np.object]).columns[:-1]
feats

for feat in feats:
    plt.figure(figsize=(15, 5))
    plt.title(feat)
    #train[feat].value_counts().plot(kind='bar')
    sns.countplot(x=feat, data=train);
    plt.xticks(rotation=90);
    plt.show();


# 
# 
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowiedź 👈 </summary>
# <p>
# 
# ```python
# print(train.columns)
# 
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Workclass', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Martial Status', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Relationship', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# ```
#  
# </p>
# </details>
# </p>
# </details> 

# Zobaczmy, jak wygląda rozkład danych. To pomoże Ci zobaczyć pewne zależności i zacząć tworzyć kolejne cechy (oparte na kombinacji istniejących cech).
# 
# Zaczniemy od zbadania: płci oraz edukacji. Przypominam, że na osi y, jest prawdopodobieństwo, że ta osoba zarobi więcej niż 50k rocznie (1.0 oznacza 100%).

# In[40]:


plt.figure(figsize=(15, 5))
sns.barplot(x="Education", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy teraz rasę i płeć.

# In[41]:


plt.figure(figsize=(15, 5))
sns.barplot(x="Race", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy kraj pochodzenia oraz płeć.

# In[42]:


plt.figure(figsize=(15, 5))
sns.barplot(x="Country", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy stan cywilny oraz płeć.

# In[43]:


plt.figure(figsize=(15, 5))
sns.barplot(x='Martial Status', y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy zawód oraz płeć.

# In[44]:


plt.figure(figsize=(15, 5))
sns.barplot(x='Occupation', y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Mam nadzieję, że już widać, które grupy wyróżniają się (a to brzmi jak cecha). Tu przy okazji zbadaliśmy temat niesprawiedliwości tego świata. Co jest ważne, model nic nie wie na temat dyskryminacji, jedynie uczy się z tego co jest "myśląc", że to jest normalne. Stąd właśnie pojawia się bias, zobacz ten [filmik](https://www.youtube.com/watch?v=59bMh59JQDo). Musisz na to uważać! Model staje się tym, czym go karmisz :). Podobnie jak nasz mózg (to co tam wrzucamy, wpływa na to, kim się stajemy później).
# 
# 
# Zbadaj teraz jeszcze inne kombinacje np. zamiast płci sprawdź rasę.

# In[45]:


plt.figure(figsize=(15, 5))
sns.barplot(x='Occupation', y="Target_cat", hue='Race', data=train)
plt.xticks(rotation=90);


# Zobacz jeszcze jedną wskazówkę co do wizualizacji. Możesz rozbić to na osobne wykresy.

# In[46]:


plt.figure(figsize=(20, 5))
g = sns.catplot(x="Occupation", y="Target_cat", col="Sex", data=train, kind="bar")

for ax in g.axes.flatten():
    plt.sca(ax)
    plt.xticks(rotation=90)


# Jeśli jest więcej niż 5, to można powiedzieć, ile ma być maksymalnie w jednym wierszu (używając `col_wrap`).

# In[47]:


plt.figure(figsize=(20, 5))
g = sns.catplot(x="Occupation", y="Target_cat", hue="Sex", col='Race', col_wrap=3, data=train, kind="bar")

for ax in g.axes.flatten():
    plt.sca(ax)
    plt.xticks(rotation=90)


# Zwróć uwagę na czarne pionowe kreski (w słupkach). Traktuj to jak rozrzut danych, jeśli ta kreska jest zbyt zmienna (np. zobacz ostatni wykres), to ciężko cokolwiek wnioskować, bo jest zbyt duża zmienność.
# 
# Już sporo wiesz, żeby dalej poruszać się samodzielnie. Spróbuj zrobić następujące rzeczy.

# ## Zadanie 1.5.4
# Dodaj kolejne cechy (*features*) bazując na tych, które już są (np. na podstawie dwóch cech `Relationship` i `Race`, można wykombinować kilka różnych cech, np. `White` + `Husband` lub `Black` + `Husband`).
# 
# Moje oczekiwania są takie, że spróbujesz stworzyć kilka czy kilkanaście nowych cech. Od razu powiem, przygotuj się, że często nowa cecha może być mało wartościowa. Natomiast wartością będzie, jeśli nauczysz się szybko iterować hipotezy (czyli odkrywać nowe cechy, które wnoszą wartość poprzez szybkie eksperymenty).
# 
# Ciekawostka: być może warto dodać zmienną waga? Zobacz ten artykuł: [People who are overweight get paid less, according to a new LinkedIn study](https://bit.ly/39njuch)

# Daj znać na Slacku, czy udało Ci się sprawdzić tę cechę :) 

# In[48]:


train['Black_Husband'] = train['Race'].map(lambda y: int(y == 'Black')) * train['Relationship'].map(lambda z: int(z == 'Husband'))

plt.figure(figsize=(10,4))
sns.barplot(x='Black_Husband', y="Target", data=train)
plt.xticks(rotation=90)


# In[1]:


train["relationship_race"] = train.apply(lambda x: "{}-{}".format(x["Relationship"], x["Race"]), axis=1)
train["relationship_race_cat"] = train["relationship_race"].factorize()[0]


# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć podpowiedź 👈 </summary>
# <p>
# Używając Relationship i Race utwórz nową kolumnę (atrybut).
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowiedź 👈 </summary>
# <p>
# 
# ```python
#     
# train["relationship_race"] = train.apply(lambda x: "{}-{}".format(x["Relationship"], x["Race"]), axis=1)
# train["relationship_race_cat"] = train["relationship_race"].factorize()[0]
# 
# #generate more ;)
# 
# ```
#  
# </p>
# </details>
# </p>
# </details> 

# ## Zadanie 1.5.5
# Zastosuj bardziej złożony model, np. [DecisionTreeClassifier](https://bit.ly/39qD4Vk). Dlaczego akurat ten? Bo jest relatywnie prosty, ale znacznie skuteczniejszy niż `Dummy` model. Więcej o drzewach decyzyjnych będzie w następnych modułach. Dlatego na razie możesz to potraktować jako czarne pudło.

# Koniecznie napisz na Slacku, jak już zrobisz to zadanie i podziel się wynikiem. Załącz screen! 

# In[4]:


from sklearn.tree import DecisionTreeClassifier
def train_and_predict_model(X_train, X_test, y_train, y_test, model, success_metric=accurancy_score):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random.state=2018)
lr = train_and_predict_model(X_train, X_test, y_train, y_test, LogisticRegression())
print('LogisticRegression: ', lr)
print('')
dtc = train_and_predict_model(X_train, X_test, y_train, y_test, DecisionTreeClassifier())
print('DecisionTreeClassifier: ', dtc)
train.columns


# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć podpowiedź  👈 </summary>
# <p>
# Użyj DecisionTreeClassifier, ale zwróć uwagę na max_depth
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowiedź  👈 </summary>
# <p>
# 
# ```python
# train_and_predict(DecisionTreeClassifier(max_depth=5), X, y, test_size=0.33)
# ```
# 
# </p>
# </details>
# </p>
# </details> 

# ### 🤝🗣️ Współpraca 💪 i komunikacja 💬
# 
# - 👉 [#pml_module1](https://practicalmlcourse.slack.com/archives/C045CNLNH89) - to jest miejsce, gdzie można szukać pomocy i dzielić się doświadczeniem - także pomagać innym 🥰. 
# 
# Jeśli masz pytanie, to staraj się jak najdokładniej je sprecyzować, najlepiej wrzuć screen z twoim kodem i błędem, który się pojawił ✔️
# 
# - 👉 [#pml_module1_done](https://practicalmlcourse.slack.com/archives/C045CP89KND) - to miejsce, gdzie możesz dzielić się swoimi przerobionymi zadaniami, wystarczy, że wrzucisz screen z #done i numerem lekcji np. *#1.2.1_done*, śmiało dodaj komentarz, jeśli czujesz taką potrzebę, a także rozmawiaj z innymi o ich rozwiązaniach 😊 
# 
# - 👉 [#pml_module1_ideas](https://practicalmlcourse.slack.com/archives/C044TFZLF1U)- tutaj możesz dzielić się swoimi pomysłami

# ### Podziel się ze światem swoimi nowymi umiejętnościami 👏
# 
# Za Tobą tydzień nauki i nowe umiejętności / doświadczenia. Pochwal się tym na swoim profilu LinkedIn ✔️
# 
# Dlaczego warto to zrobić? 🤔 
# 
# Przede wszystkim jest czym się pochwalić, bo wykonujesz właśnie w swoim życiu fajny krok w kierunku lepszej pracy i możliwości zawodowych, umiesz coś nowego, a więc Twoi potencjalni pracodawcy powinni o tym wiedzieć ;) 
# 
# Dodaj *#dataworkshop #newskills #practicalmachinelearning #datascience*
# 
# Oznacz nas! Będzie nam miło. Nasz profil znajdziesz [tutaj](https://bit.ly/2Py5eGK) 🥰 

# ## Przydatne linki:
# - [Managing Large Datasets with Python and HDF5](https://bit.ly/31t3HEE)
# - [Guide to Encoding Categorical Values in Python](https://bit.ly/2P6fHcJ)
# - [Who can earn more than 50K per year?](https://bit.ly/2O2LgDJ)
# 
# - [Qgrid is a Jupyter notebook widget](https://bit.ly/3d9BQPe)

# In[ ]:




