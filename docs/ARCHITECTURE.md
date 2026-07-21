# Prism-Architektur

Dieses Dokument erklärt die geplante nächste Modellgeneration von Prism im
Detail. Es ist zugleich technische Referenz und Lerntext: Es beschreibt nicht
nur, aus welchen Modulen das Modell besteht, sondern auch, wie sich die Daten
vom extern gelieferten Token bis zur fertigen Vorhersage verändern.

Die hier beschriebene Transformer-Architektur ist der implementierte Kernpfad
für die erste Produktionsgeneration. Der trainierte Gold-only-Student bildet
die reproduzierbare Referenz für spätere Teacher-Distillation; historische
rekurrente Experimente gehören nicht mehr zur aktiven Architektur.

## Das wichtigste mentale Modell

Die vollständige Pipeline lässt sich auf vier Verantwortlichkeiten reduzieren:

```text
Tokenizer:
Text-Tokens -> Subword-IDs

Motor:
Subword-IDs -> kontextabhängige Sprachvektoren

Task-Heads:
Sprachvektoren -> konkrete Klassen und Logits

Decoder:
Klassen-IDs -> UPOS, Morphologie und Lemmas
```

Als Analogie:

```text
Motor = sehr gut ausgebildeter Sprachwissenschaftler
Heads = getrennte Prüfungsbögen mit konkreten Fragen
```

Der Motor besitzt allgemeines norwegisches Sprachwissen. Der UPOS-Head fragt
nach der Wortart, die Morphologie-Heads nach grammatischen Eigenschaften und
der Lemma-Head nach der Regel, die aus der Wortform das Lemma erzeugt.

Norwegisch ist dabei die erste konkrete Konfiguration. Allgemein besitzt der
Motor das Sprachwissen des jeweils geladenen Sprachprofils.

```plantuml
@startuml prism-overview
skinparam backgroundColor transparent
skinparam componentStyle rectangle
skinparam shadowing false

rectangle "Extern gelieferte Tokens\nmit stabiler Reihenfolge" as Tokens
rectangle "Tokenizer des Sprachprofils\nTokens -> Subwords -> IDs" as Tokenizer
rectangle "Kompakter Transformer-Motor\nkontextualisierte Subword-Vektoren" as Encoder
rectangle "Zuordnung zu Original-Tokens\nSubwords -> ein Vektor pro Token" as Alignment

rectangle "UPOS-Head" as Upos
rectangle "18 Morphologie-Heads" as Morph
rectangle "Lemma-Regel-Head" as Lemma

rectangle "Validierung und Decoding" as Decode
rectangle "UPOS, Morphologie,\nLemma und Konfidenzen" as Results

Tokens --> Tokenizer
Tokenizer --> Encoder
Encoder --> Alignment
Alignment --> Upos
Alignment --> Morph
Alignment --> Lemma
Upos --> Decode
Morph --> Decode
Lemma --> Decode
Decode --> Results
@enduml
```

## Sprachunabhängiger Kern und austauschbare Sprachprofile

Prism soll langfristig viele Sprachen unter derselben API bedienen. Deshalb
darf NorBERT4 nicht zum fest eingebauten Motor der gesamten Bibliothek werden.
NorBERT4 ist lediglich die erste norwegische Backbone-Konfiguration.

Die Architektur trennt zwei Ebenen:

### Sprachunabhängiger Prism-Kern

Der Kern kennt keine konkrete Sprache und kein konkretes vortrainiertes
Modell. Er stellt wiederverwendbare Mechanismen bereit:

- typisierte Token-, Batch-, Vorhersage- und Artefaktverträge;
- Subword-zu-Token-Alignment und Dokument-Batching;
- UPOS-, Morphologie-, Lemma- und Konfidenz-Head-Familien;
- Loss-Funktionen, Distillation und Kalibrierung;
- Evaluation, Export und native Runtime-Anbindung;
- einheitliche API-Semantik für Python, Swift, Java/Kotlin und C++.

### Sprachprofil

Ein Sprachprofil konfiguriert alle Entscheidungen, die zwischen Sprachen
ausgetauscht werden müssen:

- Sprach- und Locale-Kennung;
- Teacher- und Student-Backbone;
- Tokenizer, Leerzeichenbehandlung und Normalisierung;
- Dataset-Adapter und unterstützte Annotationsschemata;
- UPOS-, Morphologie- und Lemma-Regel-Inventare;
- sprachspezifisches Decoding, Provenienz, Lizenzen und Benchmarks.

Die Arten der Task-Heads bleiben gleich. Ihre konkrete Größe ist jedoch Teil
des Sprachschemas. Der Morphologie-Code kann beispielsweise für jede Sprache
einen Klassifikator pro Feature aufbauen, ohne die 18 norwegischen Features
fest einzuprogrammieren. Ebenso darf ein gemeinsamer Lemma-Head nicht an die
aktuell 1.059 norwegischen Editierregeln gekoppelt sein.

```plantuml
@startuml language-profiles
skinparam backgroundColor transparent
skinparam componentStyle rectangle
skinparam shadowing false

package "Sprachunabhängiger Prism-Kern" {
  rectangle "Token- und Batch-Verträge" as Contracts
  rectangle "Alignment und Dokument-Batching" as Batching
  rectangle "Wiederverwendbare Task-Heads" as Heads
  rectangle "Training, Distillation,\nEvaluation und Export" as Pipeline
  rectangle "Einheitliche native API" as API
}

package "Norwegisches Sprachprofil" {
  rectangle "NorBERT4 Teacher/Student\nund Tokenizer" as NorwegianBackbone
  rectangle "Norwegische Schemata,\nNormalisierung und Decoding" as NorwegianSchema
}

package "Weiteres Sprachprofil" {
  rectangle "Anderer Teacher/Student\nund Tokenizer" as OtherBackbone
  rectangle "Andere Schemata,\nNormalisierung und Decoding" as OtherSchema
}

NorwegianBackbone --> Contracts
NorwegianSchema --> Heads
OtherBackbone --> Contracts
OtherSchema --> Heads
Contracts --> Batching
Batching --> Heads
Heads --> Pipeline
Pipeline --> API
@enduml
```

Die Abhängigkeitsrichtung ist entscheidend: Ein Sprachprofil verwendet den
Prism-Kern. Der Prism-Kern importiert niemals ein konkretes Sprachprofil. So
kann später ein anderes Modell ausgewählt werden, ohne Batching, Heads,
Training, Export oder native Bibliotheken zu duplizieren.

## Was ist der Motor?

Der Motor ist ein vortrainierter Transformer-Encoder. Er liefert noch keine
UPOS-Tags, Morphologie-Werte oder Lemmas. Seine Aufgabe besteht darin, jedes
Subword in einen Zahlenvektor zu verwandeln, der möglichst viel Information
über die Bedeutung und grammatische Funktion dieses Subwords im konkreten Satz
enthält.

Der große Mehrwert ist der Kontext. Betrachten wir das norwegische Wort `så`:

```text
Jeg så filmen.      -> VERB, Tense=Past, Lemma=se
Det var så fint.    -> ADV, Lemma=så
```

Eine reine Wörterbuchsuche sieht zweimal dieselbe Zeichenfolge. Der
Transformer erzeugt dagegen zwei verschiedene Repräsentationen, weil `så` in
beiden Sätzen mit unterschiedlichen Nachbartokens und Satzstrukturen
interagiert.

Der Motor komprimiert also eine Aussage wie:

```text
"Das ist meine kontextabhängige Beschreibung dessen,
was dieses Token in diesem Satz wahrscheinlich bedeutet."
```

Die Task-Heads lernen anschließend, bestimmte Informationen aus dieser
Beschreibung herauszulesen.

## Warum wird der Motor nicht von null trainiert?

Der gepinnte UD-Trainingssplit enthält rund 244.000 Tokens. Das ist eine gute
Größe, um konkrete Aufgaben wie UPOS, Morphologie und Lemmas zu lernen. Es ist
aber viel zu wenig, um allgemeines norwegisches Sprachverständnis vollständig
von null aufzubauen.

Ein vortrainierter norwegischer Encoder hat bereits auf sehr großen Textmengen
Muster gelernt, zum Beispiel:

- welche Wörter häufig als Subjekt oder Objekt auftreten;
- welche Flexionsendungen typisch für Plural, Definitheit oder Verbformen sind;
- welche Wörter nach Präpositionen oder Hilfsverben vorkommen;
- wie norwegische Komposita und Wortbestandteile aufgebaut sind;
- welche Wortformen semantisch oder grammatisch verwandt sind;
- wie Satzreihenfolge und lokale Abhängigkeiten funktionieren.

Prism konstruiert daher das vollständige Aufgabenmodell selbst, verwendet aber
einen vortrainierten Encoder als sprachlichen Motor. Das ist kein
unverändertes Fremdmodell: Prism ergänzt die Token-Zuordnung, die
Multi-Task-Heads, die Loss-Funktionen, Distillation, Konfidenzkalibrierung,
Quantisierung, Decoding und den nativen Laufzeitvertrag.

## Teacher- und Student-Rollen

Die Teacher-Student-Architektur verwendet zwei Modelle für unterschiedliche
Ziele:

- Der Teacher ist groß und auf Qualität optimiert. Er wird nur beim Training
  und bei Experimenten verwendet.
- Der Student ist kompakt und auf lokale Inferenz, Exportierbarkeit und
  Dokumentdurchsatz optimiert. Nur er wird ausgeliefert.

Aktuelle Kandidaten sind:

- `ltg/norbert4-base` als erster Teacher;
- `ltg/norbert4-large` als späterer Teacher-Vergleich, falls der zusätzliche
  Aufwand einen messbaren Development-Gewinn bringt;
- `ltg/norbert4-xsmall` als erster Student-Kandidat.

NorBERT4-xsmall ist nur der vortrainierte Student-Backbone des ersten
norwegischen Sprachprofils, nicht das fertige Prism-Modell. Prism baut seine
sprachunabhängigen Eingabe-, Alignment- und Ausgabeschichten darum.

Die aktuelle xsmall-Konfiguration besitzt unter anderem:

- Hidden Size: 192;
- 16 Transformer-Layer;
- 3 Attention-Heads;
- Intermediate Size: 512;
- Vokabulargröße: 51.200.

NorBERT4 verwendet eigenen Modellcode und moderne Attention-Mechanismen.
Deshalb ist xsmall zunächst ein Kandidat. Prism führt früh einen Export-Spike
durch. Falls der Backbone nicht zuverlässig nach ExecuTorch exportierbar ist,
bleibt NorBERT4 Teacher und das Wissen wird in einen eigenen,
exportfreundlichen Standard-Transformer destilliert.

## Schritt 1: Externe Tokens

LexKeep besitzt bereits Tokenisierung und Quelltext-Offsets. Prism muss diese
Tokens deshalb akzeptieren können, ohne sie erneut auf Wortebene zu
segmentieren.

Beispiel:

```text
["Jeg", "så", "filmen", "."]
```

Die Reihenfolge und Anzahl dieser Original-Tokens bilden den öffentlichen
Vertrag. Prism muss später exakt ein Ergebnis pro Original-Token liefern.

## Schritt 2: Subword-Tokenisierung

Transformer arbeiten gewöhnlich nicht direkt mit vollständigen Wörtern. Ihr
Vokabular enthält häufige Wörter und Wortbestandteile, sogenannte Subwords.

Eine illustrative Zerlegung könnte so aussehen:

```text
Original-Tokens:
["Jeg", "så", "filmen", "."]

Subwords:
["<s>", "Jeg", "så", "film", "en", ".", "</s>"]
```

Die tatsächliche Zerlegung hängt vom konkreten Tokenizer ab. `filmen` kann als
ein einzelnes Vokabularelement existieren oder in mehrere Bestandteile
zerfallen.

Prism speichert die Zuordnung zwischen Subwords und Original-Tokens:

| Subword | Original-Token |
| --- | ---: |
| `<s>` | keines |
| `Jeg` | 0 |
| `så` | 1 |
| `film` | 2 |
| `en` | 2 |
| `.` | 3 |
| `</s>` | keines |

Diese Zuordnung ist nötig, weil der Transformer Subword-Vektoren erzeugt, die
öffentliche API aber Token-Ergebnisse zurückgeben muss.

## Schritt 3: IDs und Batch-Tensoren

Der Tokenizer ersetzt jedes Subword durch eine Vokabular-ID:

```text
["<s>", "Jeg", "så", "film", "en", ".", "</s>"]

-> illustrativ:

[1, 1842, 731, 9204, 318, 27, 2]
```

Ab diesem Punkt arbeitet das neuronale Netz nicht mehr mit Zeichenketten,
sondern mit Tensoren.

Für einen einzelnen Satz mit sieben Subwords:

```text
input_ids.shape = [1, 7]
```

Für einen Batch aus acht Sätzen mit höchstens 30 Subwords:

```text
input_ids.shape = [8, 30]
attention_mask.shape = [8, 30]
```

Kürzere Sätze werden mit Padding aufgefüllt. Die Attention-Maske markiert echte
Subwords mit `1` und Padding mit `0`, damit der Motor Padding nicht als
sprachlichen Inhalt behandelt.

Der zukünftige typisierte Batch-Vertrag wird mindestens enthalten:

- `input_ids`;
- `attention_mask`;
- die Zuordnung von Subwords zu Original-Tokens;
- die Anzahl echter Tokens pro Satz;
- UPOS-Ziele;
- Morphologie-Ziele;
- Lemma-Regel-Ziele;
- Masken für fehlende oder nicht repräsentierbare Annotationen.

## Schritt 4: Embeddings

Eine ID wie `731` ist nur eine Zahl. Der Motor besitzt deshalb eine große
Embedding-Tabelle. Jede Vokabular-ID wählt daraus einen Vektor aus.

Bei Hidden Size 192:

```text
Vokabular-ID
    -> Embedding-Tabelle
    -> Vektor mit 192 Fließkommazahlen
```

Illustrativ:

```text
[0.17, -0.42, 0.08, ..., 0.31]
```

Für einen Batch entsteht:

```text
embeddings.shape = [batch_size, subword_count, 192]
```

Diese anfänglichen Vektoren enthalten bereits vortrainierte lexikalische
Information. Ihre konkrete Satzfunktion entsteht jedoch erst durch die
Transformer-Layer.

## Schritt 5: Der Transformer-Block

Der xsmall-Kandidat verarbeitet die Repräsentationen in 16 aufeinanderfolgenden
Transformer-Blöcken. Jeder Block besteht vereinfacht aus:

1. Normalisierung;
2. Self-Attention;
3. Residual-Verbindung;
4. Feed-Forward-Netzwerk;
5. einer weiteren Residual-Verbindung und Normalisierung.

```plantuml
@startuml transformer-block
skinparam backgroundColor transparent
skinparam shadowing false

rectangle "Eingabe X\n[Batch, Subwords, 192]" as X
rectangle "Layer Normalization" as LN1
rectangle "Multi-Head Self-Attention\n3 Heads x 64 Dimensionen" as Attention
rectangle "Residual:\nX + Attention(X)" as Residual1
rectangle "Layer Normalization" as LN2
rectangle "Feed-Forward-Netz\n192 -> 512 -> 192" as FFN
rectangle "Residual:\nZwischenzustand + FFN" as Residual2
rectangle "Ausgabe\n[Batch, Subwords, 192]" as Output

X --> LN1
LN1 --> Attention
Attention --> Residual1
X --> Residual1
Residual1 --> LN2
LN2 --> FFN
FFN --> Residual2
Residual1 --> Residual2
Residual2 --> Output
@enduml
```

### Self-Attention

Self-Attention erlaubt jeder Position, Informationen aus anderen Positionen
des Satzes aufzunehmen.

Für jeden aktuellen Subword-Vektor `X` berechnet der Transformer drei gelernte
Transformationen:

```text
Q = X * Wq
K = X * Wk
V = X * Wv
```

- Query: Nach welcher Information sucht diese Position?
- Key: Welche Information bietet diese Position an?
- Value: Welche Information wird bei Aufmerksamkeit übertragen?

Die Query einer Position wird mit den Keys anderer Positionen verglichen.
Vereinfacht:

```text
score(i, j) = Query(i) · Key(j) / sqrt(head_dimension)
```

Eine Softmax-Funktion macht daraus normalisierte Attention-Gewichte. Für das
Token `så` in `Jeg så filmen` könnte eine illustrative Verteilung sein:

```text
Jeg        0,12
så         0,18
film       0,31
en         0,21
.          0,08
Spezialtokens 0,10
```

Die tatsächlichen Werte werden gelernt. Entscheidend ist, dass der Vektor von
`så` Information von `Jeg` und `filmen` aufnehmen kann. Im Satz `Det var så
fint` entstehen andere Gewichte und damit eine andere Repräsentation.

### Interne Attention-Heads

NorBERT4-xsmall besitzt drei interne Attention-Heads. Bei Hidden Size 192
arbeitet jeder Head mit 64 Dimensionen:

```text
192 / 3 = 64
```

Die drei Heads betrachten dieselbe Sequenz parallel, können aber verschiedene
Beziehungsmuster lernen. Ein Head kann beispielsweise stärker auf lokale
Wortbestandteile reagieren, ein anderer auf Satzstruktur und ein weiterer auf
weiter entfernte Beziehungen. Diese Rollen werden nicht programmiert, sondern
entstehen beim Vortraining und Fine-Tuning.

Die Head-Ausgaben werden wieder zusammengeführt:

```text
3 x 64 -> 192 Dimensionen
```

Diese Attention-Heads sind interne Bestandteile des Motors. Sie sind nicht
dasselbe wie die späteren UPOS-, Morphologie- und Lemma-Task-Heads.

### Positionsinformation

Ohne Positionen wären `Hund beißt Mann` und `Mann beißt Hund` für einen
Transformer schwer zu unterscheiden. NorBERT4 verwendet RoPE, Rotary Position
Embeddings. Positionen beeinflussen dabei die Query- und Key-Repräsentationen
der Attention.

Der Motor kann dadurch berücksichtigen:

- welches Token vorher oder nachher steht;
- wie weit zwei Positionen auseinanderliegen;
- welche Satzreihenfolge vorliegt.

### Feed-Forward-Netzwerk

Nach der Attention verarbeitet ein kleines neuronales Netz jede Position
separat:

```text
192 -> 512 -> 192
```

Die Attention sammelt Kontext aus der Sequenz. Das Feed-Forward-Netz kombiniert
und transformiert die gesammelte Information pro Position nichtlinear.

### Residual-Verbindungen

Ein Transformer-Block ersetzt den alten Zustand nicht vollständig. Er addiert
die berechnete Veränderung:

```text
neuer Zustand = alter Zustand + gelernte Veränderung
```

Residual-Verbindungen helfen tiefen Modellen, Information zu bewahren und
stabil trainierbar zu bleiben.

### Mehrere Layer

Die Repräsentationen durchlaufen den Block wiederholt. Als nützliches, aber
nicht strikt festgelegtes Denkmodell:

```text
frühe Layer:
Subwords, Zeichenmuster und lokale Beziehungen

mittlere Layer:
Wortformen, Flexion und Satzstruktur

späte Layer:
grammatische Rollen, Bedeutung und komplexer Kontext
```

Nach dem letzten Layer bleibt die Form erhalten:

```text
encoder_output.shape = [batch_size, subword_count, 192]
```

Der Inhalt der Vektoren ist nun kontextualisiert.

## Schritt 6: Subwords zurück zu Original-Tokens

Wenn `filmen` in `film` und `en` zerlegt wurde, liegen zwei Vektoren vor.
Prism benötigt einen Tokenvektor.

### Erster Subword-Vektor

```text
Tokenvektor("filmen") = Vektor("film")
```

Vorteile:

- geringer Aufwand;
- effizienter Gather-Operator;
- einfacher Export;
- etablierte Methode;
- der erste Vektor kann durch Self-Attention trotzdem die Endung sehen.

### Mittelwert der Subwords

```text
Tokenvektor("filmen")
    = Mittelwert(Vektor("film"), Vektor("en"))
```

Das kann Endungen direkter einbeziehen, benötigt aber zusätzliche
Aggregation.

Prism implementiert beide Varianten als typisierte
`TokenPoolingStrategy`. Der Tokenizer speichert für jedes Original-Token den
Startindex und den exklusiven Endindex seines zusammenhängenden
Subwordbereichs. First-Pooling sammelt den Startvektor ein. Mean-Pooling bildet
mit Präfixsummen den Mittelwert des gesamten Bereichs, ohne eine Python-Schleife
pro Token zu benötigen.

```plantuml
@startuml token-pooling
skinparam backgroundColor transparent
skinparam shadowing false

rectangle "Original-Token\nfilmen" as Token
rectangle "Subword-Spanne\n[film, en]" as Span
diamond "Checkpoint-Policy" as Policy
rectangle "first\nVektor(film)" as First
rectangle "mean\n(Vektor(film) + Vektor(en)) / 2" as Mean
rectangle "ein Tokenvektor\n192 Dimensionen" as Result

Token --> Span
Span --> Policy
Policy --> First : first
Policy --> Mean : mean
First --> Result
Mean --> Result
@enduml
```

Die Strategie wird im Checkpoint gespeichert und bei der Evaluation
automatisch wiederhergestellt. Format-3-Checkpoints ohne dieses neue Feld
werden aus Kompatibilitätsgründen eindeutig als `first` interpretiert. Die
kontrollierte Ablation hat Mean-Pooling als neuen Standard für norwegische
Student-Trainings ausgewählt: Es senkt den Development-Loss und verbessert
Lemma-Accuracy sowie Morphologie-Micro-F1 auf Bokmål und Nynorsk. First-Pooling
bleibt als explizite Vergleichs- und Kompatibilitätsstrategie verfügbar.

Danach entsteht:

```text
token_vectors.shape = [batch_size, original_token_count, 192]
```

Jedes Original-Token besitzt nun genau einen kontextualisierten Vektor.

## Schritt 7: Task-Heads

Ein Task-Head ist eine kleine spezialisierte Ausgabeschicht. Der Motor liefert
allgemeine Sprachinformation, der Head beantwortet eine konkrete Frage.

Ein einfacher Head ist eine lineare Transformation:

```text
logits = token_vector * W + b
```

Die Rohwerte heißen Logits. Sie werden erst danach durch Softmax oder Sigmoid
in Wahrscheinlichkeiten umgewandelt.

```plantuml
@startuml task-heads
skinparam backgroundColor transparent
skinparam shadowing false

rectangle "Tokenvektor\n192 Dimensionen" as TokenVector

rectangle "UPOS Linear\n192 -> 17" as Upos
rectangle "Lemma Linear\n192 -> Schema-Regeln" as Lemma

package "Morphologie" {
  rectangle "Case-Head" as Case
  rectangle "Gender-Head" as Gender
  rectangle "Number-Head" as Number
  rectangle "Tense-Head" as Tense
  rectangle "VerbForm-Head" as VerbForm
  rectangle "... insgesamt 18 Heads" as More
}

TokenVector --> Upos
TokenVector --> Lemma
TokenVector --> Case
TokenVector --> Gender
TokenVector --> Number
TokenVector --> Tense
TokenVector --> VerbForm
TokenVector --> More
@enduml
```

### UPOS-Head

Der UPOS-Head erhält 192 Werte und erzeugt 17 Logits:

```text
Linear(192 -> 17)
upos_logits.shape = [batch, tokens, 17]
```

Illustrativ:

```text
ADJ     -2,1
ADV      0,7
NOUN    -3,0
VERB     4,8
...
```

Softmax:

```text
VERB  0,94
ADV   0,04
ADJ   0,01
Rest  0,01
```

Der Head besitzt ungefähr:

```text
192 * 17 + 17 = 3.281 Parameter
```

### Lemma-Regel-Head

Das aktuelle gemeinsame norwegische Trainingsschema enthält 1.059
normalisierte Lemma-Regeln:

```text
Linear(192 -> 1.059)
lemma_logits.shape = [batch, tokens, 1.059]
```

Der Head erzeugt nicht direkt Zeichen. Er bewertet Regeln, die Präfix- und
Suffixteile entfernen oder ergänzen. Der Decoder wendet die gewählte Regel auf
das Original-Token an.

Die lineare Schicht besitzt ungefähr:

```text
192 * 1.059 + 1.059 = 204.387 Parameter
```

Eine Gold-Regel, die nicht im Trainingsschema vorkommt, wird als
`nicht repräsentierbar` markiert. Das Development-Set enthält aktuell 33 solche
Token. Sie dürfen nicht mit wirklich fehlenden Lemma-Annotationen verwechselt
werden.

### Morphologie-Heads: ein hybrider Vertrag

Prism verwendet einen separaten Head pro Feature:

```text
Abbr, Animacy, Case, Definite, Degree, Foreign,
Gender, Mood, NumType, Number, Person, Polarity,
Poss, PronType, Reflex, Tense, VerbForm, Voice
```

Nicht jedes morphologische Feature stellt dieselbe Art von Frage. Deshalb
verwendet Prism ab Checkpoint-Format 3 zwei Klassifikationsverträge. Das
Sprachschema entscheidet pro Feature, welcher Vertrag gilt; die generischen
Heads kennen keine fest einprogrammierten norwegischen Sonderfälle.

#### Exklusive Features: Softmax und Cross-Entropy

Bei einem exklusiven Feature ist genau eine Antwort richtig. `<NONE>` ist hier
eine echte Klasse neben den annotierten Werten. Beispiel `Tense`:

```text
<NONE>
Past
Pres
```

Der lineare Head erzeugt einen Logit pro vollständigem Label:

```text
Linear(192 -> 3)
    -> Softmax
    -> genau eine Vorhersage per Argmax
```

Für ein Token ohne Tense:

```text
<NONE>  0,99
Past    0,005
Pres    0,005
```

Trainiert wird dieser Head mit kategorialer Cross-Entropy. Optionale
Klassengewichte wirken auf die jeweilige Gold-Klasse. Damit konkurrieren
`<NONE>`, `Past` und `Pres` direkt miteinander, statt als voneinander
unabhängige Ja/Nein-Fragen behandelt zu werden.

#### Mehrwertige Features: Sigmoid und Binary Cross-Entropy

Bei einem genuin mehrwertigen Feature können mehrere reale Werte gleichzeitig
gelten. Beispiel `Case` mit `Acc,Dat`. Der Head erzeugt deshalb nur für die
realen Werte unabhängige Logits:

```text
Linear(192 -> Anzahl reale Werte)
    -> Sigmoid pro Wert
    -> alle Werte oberhalb des Schwellenwerts
```

`<NONE>` besitzt hier keinen eigenen trainierbaren Logit. Es wird exakt dann
abgeleitet, wenn kein realer Wert aktiv ist. Seine Wahrscheinlichkeit für
Evaluation und spätere Kalibrierung ergibt sich aus:

```text
P(<NONE>) = Produkt(1 - P(realer Wert))
```

Trainiert wird mit Binary Cross-Entropy pro realem Wert. Positive
Klassengewichte werden ebenfalls nur auf reale positive Ziele angewendet.
Dadurch kann das sehr häufige Fehlen eines Features keinen künstlichen
`<NONE>`-Ausgang dominieren.

Das aktuelle gemeinsame norwegische Schema erkennt aus den gepinnten
Trainingsdaten:

- 12 exklusive Features: `Abbr`, `Animacy`, `Degree`, `Foreign`, `Mood`,
  `NumType`, `Person`, `Polarity`, `Poss`, `Reflex`, `Tense`, `Voice`;
- 6 mehrwertige Features: `Case`, `Definite`, `Gender`, `Number`, `PronType`,
  `VerbForm`.

Die gespeicherten Targets und die öffentliche Ausgabe behalten für beide
Varianten den vollständigen Labelraum inklusive `<NONE>`. Nur der interne
Logit-Vertrag unterscheidet sich. Der Decoder validiert dabei unter anderem:

- mindestens ein aktives Label;
- `<NONE>` nie zusammen mit echten Werten;
- keine Mehrfachwerte bei einwertigen Features;
- korrekte Label-Anzahl pro Feature.

```plantuml
@startuml hybrid-morphology-heads
skinparam backgroundColor transparent
skinparam shadowing false

rectangle "Morphologie-Feature\naus dem Sprachschema" as Feature
diamond "Mehrfachwerte\nerlaubt?" as Multi

rectangle "Linearer Head\n<NONE> + reale Werte" as CategoricalHead
rectangle "Cross-Entropy\nSoftmax / Argmax" as CategoricalDecision

rectangle "Linearer Head\nnur reale Werte" as MultiHead
rectangle "Binary Cross-Entropy\nSigmoid / Schwellenwert" as MultiDecision
rectangle "<NONE> ableiten\nwenn kein Wert aktiv" as DerivedNone

Feature --> Multi
Multi --> CategoricalHead : nein
CategoricalHead --> CategoricalDecision
Multi --> MultiHead : ja
MultiHead --> MultiDecision
MultiDecision --> DerivedNone
@enduml
```

Die ausgewählte Kontrolle verwendet weiterhin direkt lineare Task-Heads. Als
nächste isolierte Ablation implementiert Prism zusätzlich einen gemeinsamen
residualen MLP-Projektionsblock vor allen Heads:

```text
normalisiert = LayerNorm(token_vector)
transformiert = GELU(Linear(192 -> 192)(normalisiert))
head_input = normalisiert + Dropout(transformiert)
```

Die Residualverbindung erhält den direkten Informationspfad vom Backbone. Der
MLP kann zugleich nichtlineare Kombinationen der 192 Merkmale lernen, bevor
UPOS, Morphologie und Lemma sie gemeinsam lesen. Eine einzige gemeinsame
Projektion vermeidet 20 separate MLPs, fügt bei Hidden Size 192 nur 37.056
Parameter hinzu und bleibt mit Linear, GELU, Dropout und Addition
exportfreundlich.

`TokenTaskHeadArchitecture` unterscheidet `linear` und `shared-mlp`. Die
Training-CLI wählt mit `--task-head-architecture`, Checkpoints speichern die
Auswahl, und Evaluation sowie Distillation rekonstruieren sie automatisch.
Format-3-Checkpoints ohne das Feld bleiben aus Kompatibilitätsgründen
eindeutig `linear`. Der kontrollierte Fünf-Epochen-Benchmark verbessert mit
`shared-mlp` auf Bokmål und Nynorsk jede berichtete Hauptmetrik. Daher ist der
gemeinsame residuale MLP der Standard für neue norwegische Trainingsläufe;
`linear` bleibt als explizite Ablations- und Kompatibilitätsoption erhalten.

## Konfidenz und Kalibrierung

Die erste Architektur benötigt wahrscheinlich keinen separaten
Konfidenz-Head. Die Konfidenz entsteht aus den Logits der jeweiligen Aufgabe.

Unkalibrierte neuronale Wahrscheinlichkeiten sind häufig zu selbstsicher.
Prism passt deshalb nach dem Training auf dem Development-Split
Kalibrierungsparameter an, beispielsweise eine Temperatur pro Task-Head.

```text
Logits
    -> Temperaturkalibrierung
    -> Wahrscheinlichkeiten
    -> Konfidenz oder Abstention
```

Ein Schwellenwert kann später bewirken, dass Prism eine Vorhersage als
unsicher markiert, statt sie in Lernsoftware als zuverlässig darzustellen.

## Multi-Task-Training

Alle Task-Heads lesen denselben Tokenvektor. Dadurch formen mehrere Aufgaben
den gemeinsamen Motor:

```text
Gesamt-Loss =
    Gewicht_UPOS * UPOS-Loss
  + Gewicht_Morphologie * Morphologie-Loss
  + Gewicht_Lemma * Lemma-Loss
```

UPOS kann beispielsweise helfen, Morphologie zu strukturieren:

- Verben tragen eher `Tense`, `Mood`, `VerbForm` oder `Voice`.
- Nomen und Adjektive tragen eher `Gender`, `Number`, `Definite` oder `Case`.
- Satzzeichen tragen meist `<NONE>`.

Die Aufgaben werden dennoch nicht hart voneinander abhängig gemacht. Jeder
Head liest die gemeinsame Repräsentation und trägt über seinen Loss zur
Optimierung bei.

Der vortrainierte Motor wird beim Fine-Tuning vorsichtig mitangepasst:

- kleinere Lernrate für den vortrainierten Encoder;
- größere Lernrate für neu initialisierte Task-Heads;
- Development-Auswahl statt wiederholter Testoptimierung;
- reproduzierbare Seeds und vollständige Checkpoint-Metadaten.

## Distillation vom Teacher zum Student

Teacher und Student besitzen denselben öffentlichen Aufgabenvertrag:

```text
UPOS
18 Morphologie-Features
Lemma-Regeln
```

Der Teacher wird zunächst auf Gold-Daten spezialisiert. Danach erzeugt er für
jedes Token Logits oder Wahrscheinlichkeitsverteilungen. Die Distillation
spiegelt denselben hybriden Vertrag: exklusive Morphologie-Features verwenden
kategoriale KL-Divergenz, mehrwertige Features binäre KL-Divergenz nur über
die realen Werte.

Gold:

```text
VERB
```

Teacher:

```text
VERB  0,86
ADV   0,12
Rest  0,02
```

Die Gold-Annotation sagt nur, welche Klasse richtig ist. Die
Teacher-Verteilung zeigt zusätzlich, welche Alternativen sprachlich ähnlich
oder plausibel waren.

```plantuml
@startuml teacher-student
skinparam backgroundColor transparent
skinparam shadowing false

database "UD Gold-Daten\nTraining Split" as Gold

rectangle "Großer Teacher\nNorBERT4-base Kandidat" as Teacher
rectangle "Teacher-Logits\npro Task und Token" as TeacherLogits

rectangle "Kompakter Student\nNorBERT4-xsmall Kandidat" as Student
rectangle "Student-Logits\npro Task und Token" as StudentLogits

rectangle "Supervised Loss\nStudent vs. Gold" as GoldLoss
rectangle "Distillation Loss\nStudent vs. Teacher" as DistillLoss
rectangle "Gewichteter Gesamt-Loss" as TotalLoss

Gold --> Teacher : Fine-Tuning
Teacher --> TeacherLogits

Gold --> Student
Student --> StudentLogits

Gold --> GoldLoss
StudentLogits --> GoldLoss

TeacherLogits --> DistillLoss
StudentLogits --> DistillLoss

GoldLoss --> TotalLoss
DistillLoss --> TotalLoss
TotalLoss --> Student : Backpropagation
@enduml
```

Vereinfacht:

```text
Student-Loss =
    Gold-Loss
  + alpha * Distillation-Loss
```

Der Teacher überträgt keine Gewichte direkt. Er liefert zusätzliche
Trainingssignale. Der Student wird gegen dieselbe Architektur ohne
Distillation verglichen. Nur dadurch lässt sich zeigen, dass der Teacher den
ausgelieferten Student tatsächlich verbessert.

Spätere Ablationen können zusätzlich prüfen:

- reine Logit-Distillation;
- Hidden-State-Distillation mit Projektionsschicht;
- unterschiedliche Temperaturen;
- unterschiedliche Loss-Gewichte;
- Teacher base gegen Teacher large.

## Dokument-Inferenz für LexKeep

Ein Dokument mit 6.000 Tokens wird nicht als eine einzige globale
6.000-Token-Sequenz behandelt. LexKeep liefert bereits Sätze und Tokens.

Prism:

1. übernimmt die externen Tokens;
2. gruppiert Sätze nach ähnlicher Länge;
3. bildet effiziente Batches;
4. führt den Student auf GPU oder einem geeigneten Backend aus;
5. stellt die ursprüngliche Dokumentreihenfolge wieder her.

```plantuml
@startuml document-inference
skinparam backgroundColor transparent
skinparam shadowing false

rectangle "LexKeep-Dokument\nca. 200 Sätze / 6.000 Tokens" as Document
rectangle "Extern gelieferte\nSätze und Tokens" as Sentences
rectangle "Längen bestimmen" as Lengths
rectangle "Bucket-Batching\nähnliche Satzlängen zusammen" as Buckets
rectangle "Tokenizer und Student\nGPU / Core ML / MPS / XNNPACK" as Runtime
rectangle "Ergebnisse entbatchen" as Unbatch
rectangle "Ursprüngliche\nDokumentreihenfolge" as Ordered

Document --> Sentences
Sentences --> Lengths
Lengths --> Buckets
Buckets --> Runtime
Runtime --> Unbatch
Unbatch --> Ordered
@enduml
```

Bucket-Batching reduziert unnötiges Padding. Viele Sätze können parallel
verarbeitet werden, ohne die quadratischen Kosten einer einzigen riesigen
globalen Attention-Sequenz zu bezahlen.

Die dokumentierten ersten Release-Grenzen sind:

- höchstens 1,0 Sekunde mediane warme Inferenz;
- höchstens 1,5 Sekunden p95 warme Inferenz;
- höchstens 3,0 Sekunden kalter Load plus Inferenz;
- höchstens 250 MiB zusätzlicher Peak-Speicher;
- höchstens 100 MiB für das quantisierte norwegische Paket.

## Export und native Laufzeit

Training bleibt in Python und PyTorch. Das ausgelieferte Sprachpaket enthält
keine Python-Umgebung und keinen rohen Trainingscheckpoint.

Zielstruktur:

```text
prism-no-bokmaal-<version>/
├── model.pte
├── manifest.json
├── vocabulary.json
├── labels.json
└── LICENSES/
```

`model.pte` ist das anfängliche ExecuTorch-Ziel. Das Manifest dokumentiert:

- Modell- und Schema-Version;
- Sprache und Aufgaben;
- Tensorformen und Padding-Vertrag;
- Tokenizer und Normalisierung;
- maximale unterstützte Formen;
- Quantisierung;
- Trainingsdaten-Provenienz;
- Modell- und Datenlizenzen;
- Benchmark-Identität.

Der öffentliche Swift-, Java/Kotlin- oder C++-Vertrag darf keine
ExecuTorch-Typen offenlegen. Native Bibliotheken übersetzen stabile
Prism-Typen in die jeweilige Runtime.

NorBERT4 verwendet eigenen Modellcode. Deshalb findet ein früher
Exportierbarkeits-Spike vor teurem Fine-Tuning statt:

```text
NorBERT4-xsmall exportierbar?

Ja:
    als Student-Kandidat weiter benchmarken

Nein:
    NorBERT4 als Teacher behalten
    und in exportfreundlichen Prism-Student destillieren
```

## Trainingsphasen

Der geplante Entwicklungsablauf ist:

1. Daten- und Output-Vertrag stabilisieren.
2. Den typisierten Sprachprofil- und Backbone-Vertrag festlegen.
3. Tokenizer-Alignment und Batch-Tensoren sprachunabhängig implementieren.
4. NorBERT4-xsmall über das norwegische Sprachprofil laden und einen
   Forward-Pass beweisen.
5. Früh die Exportierbarkeit untersuchen.
6. Sprachunabhängige Prism-Task-Heads und Loss-Funktionen implementieren.
7. Einen Student nur auf Gold-Daten trainieren.
8. Den Teacher auf denselben Aufgaben fine-tunen.
9. Den Student mit Distillation trainieren.
10. Student mit und ohne Distillation vergleichen.
11. Konfidenzen kalibrieren.
12. Quantisieren und PyTorch-zu-ExecuTorch-Parität prüfen.
13. Dokument-Inferenz messen.

## Was bereits implementiert ist

Der implementierte Daten- und Modellvertrag enthält aktuell:

- ein versioniertes UPOS-Schema;
- ein versioniertes Schema für 18 Morphologie-Features;
- atomare Multi-Value-Repräsentation;
- validierte Morphologie-Kodierung und -Dekodierung;
- 1.059 normalisierte Lemma-Regeln im gemeinsamen norwegischen Schema;
- stabile Klassen-IDs;
- Unterscheidung zwischen fehlendem Lemma und unbekannter Lemma-Regel;
- ein gebündeltes `TokenTaskSchema`;
- modellunabhängige Sätze und Corpora;
- Development-Abdeckungsmetriken;
- einen typisierten `TokenizedBatch`;
- einen gepinnten, typisierten Backbone-Vertrag mit NorBERT4-xsmall als erster
  norwegischer Konfiguration;
- einen generischen `LanguageProfileSpec` und ein separates
  `prism.languages.norwegian`-Paket, das die konkrete NorBERT4-Konfiguration
  besitzt;
- einen Fast-Tokenizer-Loader, der nur vom Backbone-Vertrag abhängt;
- Erhalt originaler Tokenabstände aus CoNLL-U `SpaceAfter=No`;
- Subword-zu-Token-Alignment und gepaddete Tokenmasken;
- einen sprachunabhängigen Adapter von `PretokenizedSentence` zum
  `TokenizedBatch`, der mit dem echten gepinnten NorBERT4-Tokenizer verifiziert
  wurde;
- NorBERT4-xsmall und NorBERT4-base als austauschbare Student- und
  Teacher-Backbones des norwegischen Sprachprofils;
- trainierbare UPOS-, Morphologie- und Lemma-Regel-Heads;
- Gold-only-Training, Distillation und getrennte Development-Evaluation für
  Bokmål und Nynorsk;
- den hybriden Morphologievertrag aus kategorialen exklusiven Features und
  binären mehrwertigen Features;
- Checkpoint-Format 3 als explizite Grenze für die geänderten
  Morphologie-Tensorformen.

## Aktuelle Modellgrenze

Die Hybridarchitektur ist als gemeinsamer norwegischer Format-3-Student
trainiert und getrennt auf Bokmål und Nynorsk Development ausgewertet. Sie
verbessert Morphologie-Micro-F1 und Macro Average Precision auf beiden
Schriftstandards, während UPOS und Lemma praktisch stabil bleiben. Die
Format-2-Benchmarks bleiben historische Kontrollen.

Die anschließende kontrollierte Token-Pooling-Ablation wählt Mean-Pooling als
neuen Student-Standard. Gegenüber dem ansonsten identischen First-Pooling-Modell
sinken die Development-Losses auf beiden Schriftstandards; Lemma-Accuracy und
Morphologie-Micro-F1 steigen ebenfalls auf beiden. Der ausgewählte Checkpoint
`runs/no-student-hybrid-mean-weighted/best.pt` bildet die lineare Kontrolle.
First-Pooling bleibt nur als explizite Ablations- und Kompatibilitätsoption
erhalten.

Die folgende Shared-MLP-Ablation verbessert gegenüber dieser linearen
Mean-Pooling-Kontrolle jede berichtete Hauptmetrik auf Bokmål und Nynorsk. Der
neue Student-Standard und ausgewählte Gold-only-Checkpoint ist
`runs/no-student-hybrid-mean-shared-mlp-weighted/best.pt`.

Checkpoint-Format 3 ist absichtlich nicht gewichtskompatibel mit Format 2:
Bei exklusiven Features ändern sich Loss, Interpretation und teilweise die
Anzahl der Head-Ausgänge. Ein alter State-Dict darf daher nicht stillschweigend
in das neue Modell geladen werden.

Der nächste kontrollierte Student-Schritt ist eine Acht-Epochen-Ablation mit
unveränderter Mean-Pooling- und Shared-MLP-Architektur. Erst nach Auswahl der
Trainingsdauer folgt das teure gemeinsame norwegische Teacher-Training mit
demselben Format-3-Vertrag.

## Quellen

- [NorBERT4-small model card](https://huggingface.co/ltg/norbert4-small)
- [NorBERT4-xsmall configuration](https://huggingface.co/ltg/norbert4-xsmall/blob/7483327d36a2daa5dbe936c68aa277149c6f9632/config.json)
- [Prism model strategy](model-strategy.md)
- [Confirmed project status](PROJECT_STATUS.md)
- [Benchmarks](benchmarks.md)
