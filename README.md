# Rozpoznawanie Emocji i Sentymentu w Języku Polskim

## Opis projektu

Projekt realizowany w ramach przedmiotu Inżynieria Lingwistyczna, rozwiązujący zadanie z konkursu PolEval 2024 Task 2: Multi-level Sentiment Analysis. Celem zadania jest klasyfikacja wieloetykietowa emocji i sentymentu w polskich recenzjach restauracji na trzech poziomach:

- Całe recenzje
- Pojedyncze zdania
- Agregacja zdań do poziomu recenzji

## Struktura projektu

```
emotion-and-sentiment-recognition/
├── data/
│   ├── raw/              # Surowe dane z konkursu
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── clean/            # Przetworzone dane dla różnych modeli
│       ├── ml_*.csv      # Dane dla modeli klasycznych ML
│       ├── nn_*.csv      # Dane dla sieci neuronowych
│       └── tf_*.csv      # Dane dla transformerów
├── models/               # Wytrenowane modele
│   ├── machine_learning.pkl
│   ├── neural_network.keras
│   └── transformers_final/
├── src/                  # Kod źródłowy
│   ├── machine_learning.ipynb
│   ├── neural_network.ipynb
│   └── transformers.ipynb
└── README.md
```

## Zastosowane podejścia

### 1. Klasyczne uczenie maszynowe (`machine_learning.ipynb`)

- **Przetwarzanie tekstu**: TF-IDF + SVD (redukcja do 100 wymiarów)
- **Modele**: Random Forest, Gradient Boosting, SVM
- **Optymalizacja**: GridSearchCV dla każdej emocji osobno
- **Metryki**: Precision i Recall jako główne kryteria

### 2. Sieci neuronowe (`neural_network.ipynb`)

- **Przetwarzanie tekstu**:
  - Lematyzacja (spaCy)
  - Generalizacja słów (plWordNet)
  - Embeddingi (DistilRoBERTa dla języka polskiego)
- **Architektura**: Głęboka sieć neuronowa z warstwami Dense, BatchNormalization i Dropout
- **Optymalizacja**: Adam z early stopping i redukcją learning rate

### 3. Modele transformerowe (`transformers.ipynb`)

- **Model bazowy**: polish-roberta-base-v2
- **Przetwarzanie**: Specjalne tokeny [COMMENT_START] i [COMMENT_END] do oznaczania granic komentarzy
- **Fine-tuning**: 8 epok z early stopping
- **Framework**: Hugging Face Transformers

## Klasyfikowane etykiety

**Emocje** (8 klas):

- Joy, Trust, Anticipation, Surprise, Fear, Sadness, Disgust, Anger

**Sentyment** (3 klasy):

- Positive, Negative, Neutral

### Najlepsze wyniki F1-macro:

- **Klasyczne ML**: **0.672** (średnia harmoniczna: 0.691 dla tekstów, 0.654 dla zdań)
- **Sieci neuronowe**: 0.574 (średnia: 0.559 dla tekstów, 0.588 dla zdań)
- **Transformery**: 0.625(tylko F1-macro, bez podziału)

### Porównanie z konkursem:

Zwycięzcy konkursu PolEval 2024 Task 2 osiągnęli F1-macro w zakresie 0.75-0.81. Nasze najlepsze podejście (klasyczne ML) z wynikiem 0.672 osiąga około 83% wydajności zwycięskiego rozwiązania.

## Wnioski

## Wnioski

1. Najlepsze wyniki w naszym eksperymencie osiągnęły **modele klasyczne ML** (F1-macro 0.672), przewyższając transformery (0.625) i sieci neuronowe (0.574).
2. **Modele transformerowe** nadal są silnym podejściem i pokazują dobrą jakość, zwłaszcza biorąc pod uwagę potencjał dalszej optymalizacji.
3. **Kontekst ma znaczenie** - specjalne oznaczanie granic komentarzy poprawia wyniki.
4. **Agregacja predykcji** z poziomu zdań do recenzji pozostaje wyzwaniem.
5. **Pretrenowane modele językowe** dla języka polskiego (jak RoBERTa) stanowią solidną podstawę dla zadań NLP.
6. **Różnica wobec zwycięzców** (około 10 procentowych do najlepszych modeli ML w konkursie) sugeruje, że kluczowe mogą być:
   - Techniki ensemble'owania wielu modeli
   - Specjalizowane architektury dla różnych poziomów analizy (zdania vs. teksty)
   - Dodatkowa optymalizacja hiperparametrów
   - Augmentacja danych lub dodatkowe techniki regularyzacji

## Możliwe kierunki rozwoju

- Implementacja ensemble'u różnych modeli transformerowych
- Eksperymentacja z większymi modelami (np. polish-gpt2)
- Zastosowanie technik augmentacji danych specyficznych dla języka polskiego
- Optymalizacja osobnych modeli dla poziomu zdań i tekstów

## Wymagania

- Python 3.11+
- Biblioteki: transformers, torch, tensorflow, scikit-learn, pandas, numpy, spacy, plwordnet
- GPU rekomendowane dla trenowania transformerów

## Autorzy

Projekt realizowany w ramach przedmiotu Inżynieria Lingwistyczna przez Agatę Bogdanowicz, Michała Chojnę, Olgę Lewandowską i Kubę Nowakowskiego.
