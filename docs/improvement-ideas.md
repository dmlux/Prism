# Prism – Ideensammlung für Verbesserungen (Entwurf)

Status: Brainstorm zur Bewertung, **keine akzeptierte Strategie**.
Erstellt: 2026-07-20
Zuletzt überarbeitet: 2026-07-24 (Silber-Labeling-Policy-Entwurf; Einordnung
Teacher-Kaskade nach Forschungsstand)

Dieses Dokument sammelt konkrete, moderne Techniken, mit denen der ausgelieferte
Student **kleiner**, **schneller** und **besser** werden kann. Jeder Vorschlag
nennt: was er bewirkt, warum er zu Prism passt, sowie geschätzten Aufwand und
Risiko. Die Reihenfolge ist grob nach erwartetem Nutzen sortiert. Jede
Qualitätsbehauptung muss vor Übernahme mit einem Benchmark auf den gepinnten
Splits belegt werden – wie in `docs/model-strategy.md` gefordert.

Begriffe sind bewusst einfach erklärt, weil das Projekt auch ein Lerntext ist.

---

## Einordnung: Welcher Forschungsstrang gilt für Prism?

Bevor es um einzelne Techniken geht, ein wichtiger Rahmen. Vieles, was 2025/2026
als „Cutting Edge" der Distillation gilt – On-Policy- bzw. Generalized Knowledge
Distillation (GKD), Reasoning-Trace-Distillation, Speculative Decoding,
RLHF-Distillation – betrifft **generative, autoregressive Sprachmodelle**, die
Text Token für Token erzeugen. Prism ist etwas anderes: ein **Encoder für
Token-Klassifikation** (UPOS, Morphologie, Lemma). Es erzeugt keinen freien
Text, sondern liest pro Token eine feste Menge von Klassen aus.

Der für Prism relevante Forschungsstrang ist deshalb die **Encoder-Kompression**:
DistilBERT (2019) → TinyBERT (2019) → MiniLM/MiniLMv2 (2020/2021) → Decoupled
Knowledge Distillation (2022) und die aktuellen Prune-then-Distill-Rezepte
(2024/2025). Die generative LLM-Welle ist für Prisms Modellklasse **nicht**
anwendbar. Dieses Dokument bleibt daher bewusst in der Encoder-Tradition; das ist
keine Rückständigkeit, sondern die passende Wahl.

**Empirischer Realitätscheck.** `docs/benchmarks.md` zeigt, dass der distillierte
Student den gold-only Student bislang nur minimal schlägt (UPOS 98,64 % →
98,66 %). Das ist selbst der stärkste Beleg dafür, dass **reine globale
Logit-Distillation auf den ~244k Gold-Tokens ausgereizt ist**. Die drei größten
Hebel sind daher: mehr echtes Signal (1), ein besseres Distillations-Ziel (2)
und eine präzisere Verlust-Zerlegung statt blinder Gewichts-Sweeps (3).

**Aktueller Architekturstand.** Der unter Punkt 5 vorgeschlagene gemeinsame
MLP ist inzwischen kontrolliert getestet und ausgewählt. Die breite residuale
Variante `192 -> 384 -> 192` verbessert den Morphologie-Micro-F1 auf Bokmål
von 95,30 % auf 95,70 % und auf Nynorsk von 92,08 % auf 92,54 %. Die nächste
günstige Architekturklasse ist daher nicht weiteres blindes Verbreitern,
sondern die Nutzung mehrerer bereits berechneter Backbone-Schichten durch eine
kleine lernbare Schichtmischung.

---

## Kurzüberblick (nach Ziel)

| Ziel | Stärkste Hebel |
| --- | --- |
| Bessere Genauigkeit | Silber-Daten-Distillation (1), MiniLMv2-Attention-Distillation (2), DKD-basierte task-spezifische Policy (3), Multi-Teacher-Ensemble (4) |
| Kleineres Paket | Embedding quantisieren/Matryoshka (6), Vokabular-Beschneidung (7), MatQuant-Quantisierung (8) |
| Schnellere Inferenz | Prune-then-Distill (9), moderne Encoder-Effizienz (10), Bucket-Batching feintunen (11) |
| Vertrauenswürdige Ausgabe | Kalibrierung + konformale Mondrian-Abstention (12) |
| Glaubwürdiger Vergleich | Faire Benchmark-Matrix inkl. moderner Baselines (13) |
| Alternative Architekturen | Zeichenbewusster Kopf (15), strukturierter Decoder (16), Kontext-Granularität (18) |

---

## A. Genauigkeit (der Student soll den Teacher einholen)

### 1. Distillation auf großen Silber-Daten (größter Einzelhebel)

**Idee:** Der gepinnte UD-Trainingssplit hat nur ~244k Tokens pro Standard.
Distillation wirkt aber viel stärker, wenn der Student die *weichen*
Teacher-Vorhersagen auf **sehr viel unannotiertem Text** nachlernt. Man lässt
den fertigen NorBERT4-Base-Teacher über einen großen norwegischen Rohkorpus
laufen (z.B. Norwegian Colossal Corpus / öffentliche Web-/Wiki-Texte, Bokmål
und Nynorsk), speichert seine Logits als „Silber"-Ziele und trainiert den
Student auf Gold + Silber gemeinsam.

**Warum es hilft:** Der Flaschenhals eines kleinen Modells ist selten die
Architektur, sondern wie viel Signal es sieht. Mehr (wenn auch nicht perfekt)
gelabelte Daten schließen typischerweise den größten Teil der Lücke zwischen
Student und Teacher – oft mehr als jede Architektur-Feinheit. Die aktuelle
Kleinmodell-Forschung (z.B. die BabyLM-Reihe) bestätigt: Distillation von einem
starken Teacher kann klassisches Vortraining auf denselben Daten schlagen.

**Silber-Daten-*Selektion* (Verschärfung):** Nicht jeder Rohsatz trägt gleich
viel bei. Statt blind allen Text zu nehmen, filtert man nach Teacher-Konfidenz
oder -Entropie und bevorzugt informative, aber nicht triviale Sätze. Das
verhindert, dass der Student Teacher-Fehler auf uninformativem Text mitlernt –
der Unterschied zwischen „mehr Daten" und „mehr *Signal*".

**Aufwand:** mittel-hoch (Korpus beschaffen + lizenzieren, Teacher-Inferenz mit
Durchsatz, Silber-Cache verwalten).
**Risiko:** niedrig-mittel (Teacher-Fehler werden mitgelernt; Lizenz-/
Provenienz-Disziplin nötig, passt aber zu Prisms Artefakt-Regeln).
**Passung:** Nutzt den bereits akzeptierten Teacher, respektiert Offline-Ziel
(nur beim Training), verändert den ausgelieferten Student nicht in der Größe.

#### 1a. Warum ein „nur gleich guter" Teacher trotzdem reicht

Ein häufiger Einwand: Der Teacher ist kaum besser als der Student, also lernt
der Student vor allem Teacher-Fehler, und mehr als eine Annäherung an den
Teacher ist ohnehin nicht drin. Beides stimmt nur für das bisher gemessene
Regime – Logit-Distillation auf denselben Gold-Daten. Beim Silber-Labeling ist
der Teacher dagegen kein Wissens-Deckel, sondern ein **skalierbarer Annotator**,
und der Student kann ihn nachweislich übertreffen:

1. **Kontext-zu-Lexikon-Transfer.** Der Teacher labelt z.B. Genus dort richtig,
   wo die Syntax es verrät (*en bil*, *et hus*, Adjektiv-Kongruenz). Über einen
   großen Korpus erscheint fast jedes Nomen irgendwo in so einem verräterischen
   Kontext. Der Student sieht dasselbe Wort vielfach konsistent gelabelt und
   lernt die Eigenschaft **lexikalisch** (Char-CNN, Embeddings) – und wendet sie
   dann in Kontexten an, in denen der Teacher selbst scheitern würde. Das
   Korpus wirkt wie ein Ensemble tausender Teacher-Vorhersagen pro Wort; die
   Aggregation erzeugt Wissen, das kein einzelner Teacher-Forward-Pass hat.
   Genau das trifft Prisms gemessenes Fehlerprofil (Gender auf Rare/OOV-Nomen):
   „OOV" heißt nur „nicht in ~490k Gold-Tokens" – im Silber-Korpus sind diese
   Wörter häufig.
2. **Gelernt wird die beste Teilmenge, nicht der Durchschnitt.** Mit
   kalibrierter Konfidenz-Filterung und Teacher-Agreement liegt die
   Label-Qualität im behaltenen Teil weit über der Durchschnittsgenauigkeit
   des Teachers.
3. **Gold-Anker.** Gold + Silber werden gemeinsam trainiert; unsystematische
   Teacher-Fehler sind über den Korpus inkonsistent und schwer zu fitten.
4. **Empirisch belegt:** Born-Again Networks (Student = Teacher-Architektur,
   Student schlägt Teacher) und das Noisy-Student-Verfahren (iteriertes
   Pseudo-Labeling übertrifft den Ausgangs-Teacher) zeigen, dass ein
   überlegener Teacher **keine Voraussetzung** ist.

Die echte Gefahr sind **systematische** Teacher-Fehler – dagegen richten sich
Agreement-Filter, Gold-Anker und die getrennten Bokmål-/Nynorsk-Gates der
folgenden Policy.

#### 1b. Silber-Labeling-Policy (konkreter Entwurf, noch nicht akzeptiert)

Voraussetzungen in fester Reihenfolge, bevor Silber-Labels erzeugt werden:

1. **Architektur-gematchter Base-Teacher** (gleiche Student-Architektur auf
   `norbert4-base`) ist trainiert und schlägt auf beiden Schriftstandards
   kanonisch sowohl den korrigierten historischen Teacher-Kontrollpunkt als
   auch den aktuellen Student (UFeats, UPOS, Lemmas, Rare/OOV, Korrektur 1.0).
   Schlägt er den Student nicht klar, wird `norbert4-large` als **Labeler**
   autorisiert – nicht als Kaskadenglied (siehe 4).
2. **Kalibrierung vor Filterung:** Temperatur-Skalierung pro Task-Head des
   Teachers auf den Development-Splits. Ein unkalibrierter Konfidenz-Filter
   sortiert systematisch falsch; der dokumentierte Wide-MLP-NLL-Befund macht
   das konkret. Labeling verwendet ausschließlich **korrigierte** Logits
   (dokumentiertes Gate: unkorrigierte Teacher-Morphologie ist kein
   Pseudo-Gold).

Labeling- und Filter-Vertrag (sprachneutral, typisiert):

- **Konfidenz-Schwelle pro Task**, abgeleitet aus Development: Die Schwelle
  wird so gewählt, dass die Teacher-Genauigkeit auf den akzeptierten
  Dev-Tokens ein vordeklariertes Ziel erreicht (z.B. ≥ 99,5 % pro Task) –
  keine handgewählte feste Zahl, dadurch pro Sprache reproduzierbar ableitbar.
- **Zwei-Teacher-Agreement:** Der korrigierte historische Character-CNN-Teacher
  dient als Agreement-Kontrolle. Tokens mit Disagreement werden **maskiert**
  (die bestehenden Loss-Masken für fehlende Annotationen tragen das bereits),
  nicht stillschweigend übernommen.
- **Satz-Verwurf:** Sätze mit mehr als einem vordeklarierten Anteil maskierter
  Tokens (z.B. > 30 %) fallen ganz weg, damit der Student keine systematisch
  „durchlöcherten" Sätze lernt.
- Jedes Silber-Artefakt trägt Manifest mit Quelle, SHA-256, Teacher-Checkpoints,
  Kalibrierungs- und Schwellen-Policy – wie im bestehenden
  `prepare_silver_corpus`-Vertrag.

Gold/Silber-Mischung mit Nynorsk-Schutz:

- Gold bleibt in jeder Epoche vollständig enthalten; die NBdigital-Quelle ist
  **Bokmål-only**, daher wird der Nynorsk-Gold-Anteil pro Epoche konstant
  gehalten (Oversampling), damit Bokmål-Silber Nynorsk nicht verdrängt.
- Silber-Beispiele erhalten ein eigenes, vordeklariertes Loss-Gewicht
  (< Gold); erste Ablation als kleines Grid, nicht nachträglich getunt.
- **Nynorsk-Silberquelle (umgesetzt 2026-07-24):** Språkbanken
  `oai:nb.no:sbr-60` „Legal documents from Norwegian Nynorsk municipalities"
  – CC0, moderne Orthographie (kommunale Sakspapiere), Seiten bereits
  sprachklassifiziert. Die Vorbereitung ist implementiert
  (`prism.data.segmentation` + `prism.data.sakspapir`, Details in
  `PROJECT_STATUS.md`) und liefert 2.012.251 Sätze / 37,1 Mio. Tokens –
  vergleichbar mit den 50,4 Mio. Bokmål-Tokens, sodass die Mischung nicht
  dauerhaft nur über Oversampling geschützt werden muss. Alternative mit
  offener, aber nicht gemeinfreier Lizenz bleibt Målfrid 2021 (`sbr-69`,
  NLOD 2.0). Die gemeinfreien NBdigital-Bücher (`sbr-34`) enthalten zwar
  Nynorsk, aber überwiegend in **veralteter Orthographie** (Landsmål/frühe
  Reformen) – als Trainingsquelle für modernes Nynorsk riskant.

Pilot mit Dosis-Wirkungs-Kurve statt Vollkorpus:

- Deterministische Teilmengen von z.B. **1M → 5M → 10M Tokens** (dokumentierte
  Auswahlregel), erst danach Entscheidung über den 50M-Vollkorpus.
- Messpunkte nach jeder Dosis: kanonische Bokmål- und Nynorsk-Gates (UFeats,
  UPOS, Lemmas, jede Morphologie-Feature, Rare/OOV) – mit besonderem Blick auf
  **Gender/OOV**, den erklärten Zielfehler.

Abbruch- und Erfolgs-Kriterien (vordeklariert):

- Abbruch bzw. Policy-Revision, wenn ein Schriftstandard auf einem
  Gate-Aggregat regressiert, der dokumentierte Lemma-Guardrail verletzt wird
  oder Gender/OOV nach der 5M-Dosis keine messbare Verbesserung zeigt.
- Erfolg autorisiert die nächste Stufe: größere Dosis, danach optional eine
  **Noisy-Student-Iteration** (der verbesserte Student wird neuer Labeler) –
  unter exakt denselben Gates.

**Passung:** vollständig sprachunabhängig formulierbar (Schwellen aus Dev
abgeleitet, Masken- und Manifest-Verträge existieren); kein UDPipe-Einfluss auf
Training oder Schwellenwahl; Test-Splits bleiben unberührt.

### 2. MiniLMv2-artige Attention-Distillation (statt nur Logits)

**Idee:** Aktuell wird nur die *Ausgabe* (Logits) destilliert. MiniLM destilliert
stattdessen die **Selbst-Attention-Beziehungen der Teacher-Schicht** (die
Beziehungen zwischen Query, Key und Value). Die neuere Variante **MiniLMv2**
verallgemeinert das auf die Multi-Head-Relationen von Q-Q, K-K und V-V und darf
diese aus einer *frei wählbaren* Teacher-Schicht ziehen. Der Charme: Es braucht
**keine** Schicht-zu-Schicht-Zuordnung und funktioniert, obwohl Teacher und
Student unterschiedlich tief/breit sind (640 vs. 192 Hidden). Genau Prisms
Situation.

**Warum es hilft:** Der Student lernt *wie* der Teacher Kontext verknüpft, nicht
nur *was* am Ende herauskommt. Das überträgt mehr Wissen pro Trainingsschritt –
und greift genau dort, wo reine Logit-Distillation laut `benchmarks.md` schon
ausgereizt ist.

**Besonders günstig für Prism:** Teacher (`norbert4-base`) und Student
(`norbert4-xsmall`) teilen dieselbe NorBERT4-Architektur **und denselben
Tokenizer**. Die in der Literatur oft genannte „Cross-Tokenizer"-Hürde der
Attention-Distillation entfällt hier vollständig.

**Aufwand:** mittel (Zugriff auf Teacher-Attention, bei Value-Relationen keine
Projektionsschicht nötig).
**Risiko:** mittel (NorBERT4 nutzt eigenen Modellcode; interne Attention muss
zugänglich sein).
**Passung:** In `docs/ARCHITECTURE.md` bereits als mögliche Ablation genannt
(„Hidden-State-Distillation mit Projektionsschicht") – MiniLMv2 ist die
modernere, robustere Variante davon.

### 3. Task-spezifische Distillation-Policy auf DKD-Basis (bereits als nächster Schritt notiert)

**Umsetzungsstand:** Die Grundlage ist implementiert: UPOS, Morphologie und
Lemma besitzen getrennte Temperaturen und Loss-Gewichte. Der erste kontrollierte
Kandidat mit Gewichten 0,05/0,20/0,10 wurde gemessen und verworfen: Er verbessert
Nynorsk Rare/OOV-Morphologie, regressiert aber breitere Metriken auf beiden
Schriftstandards und verbessert Bokmål Rare/OOV-Morphologie nicht. Die echte
DKD-Zerlegung in TCKD und NCKD ist als optionale kategoriale
Trainingsstrategie implementiert und ihr erster kontrollierter Kandidat wurde
ausgewählt. Er senkt den Loss und verbessert Gesamt-UPOS, Lemma sowie
Rare/OOV-Lemma und -Morphologie auf beiden Schriftstandards. Mehrwertige
Morphologie bleibt bewusst beim binären KL-Loss, da dort keine einzelne
Zielklasse existiert.

**Idee:** Statt einer globalen Temperatur/Gewichtung je eine passende
Einstellung pro Kopf: Softmax-UPOS (17 Klassen), binäre Morphologie-Werte und
der 1.059-Wege-Lemma-Kopf reagieren völlig unterschiedlich auf Temperatur. Die
moderne, präzise Formulierung dieser Idee ist **Decoupled Knowledge Distillation
(DKD, 2022)** – der aktuelle Standard für Logit-Distillation. DKD zerlegt den
KD-Verlust in zwei getrennt gewichtbare Teile:

- **TCKD** (Target-Class): wie sicher der Teacher bei der *richtigen* Klasse ist;
- **NCKD** (Non-Target-Class): das „Dunkelwissen" über die *falschen* Klassen –
  gerade dieser Teil trägt den meisten Distillationsnutzen.

Ergänzend: **klassenbalancierte Morphologie-Distillation** (seltene Werte gezielt
gewichten) und optional eine **learnbare/dynamische Temperatur** pro Kopf
(CTKD/LSKD) statt fixer Werte.

**Warum es hilft:** In `docs/benchmarks.md` steht, dass eine globale Temperatur
2.0 die binären Morphologie-Ausgaben und die Lemma-Verteilung „über-glättet".
DKD behebt genau das, weil man die NCKD-Komponente (die für das Über-Glätten
verantwortliche „Dunkelwissen"-Verteilung) unabhängig von der Zielklasse steuern
kann.
**Aufwand:** niedrig-mittel (Trainingsschleife existiert bereits).
**Risiko:** niedrig. **Passung:** exakt der in `PROJECT_STATUS.md` genannte
unmittelbare nächste Schritt – ideal als erster Schritt.

### 4. Multi-Teacher-/Ensemble-Distillation

**Idee:** Prism hält `norbert4-base` als ersten Teacher und `norbert4-large` als
späteren Vergleich bereit. Statt sich für einen zu entscheiden, kann man **beide
als gemeinsamen Teacher** nutzen und ihre Ausgabeverteilungen mitteln (oder
konfidenzgewichtet mischen), bevor der Student daraus lernt. Aktuelle Arbeiten
zeigen, dass ein solches Teacher-Ensemble den Student zuverlässiger macht als
ein einzelner Teacher.

**Warum es hilft:** Verschiedene Teacher machen verschiedene Fehler. Ein
gemitteltes Ziel glättet idiosynkratische Teacher-Fehler und liefert ein
stabileres Distillationssignal – günstig, weil der `large`-Teacher ohnehin als
Kandidat vorgesehen ist.
**Aufwand:** niedrig-mittel (beide Teacher existieren bzw. sind geplant; nur
Verteilungen mischen). **Risiko:** niedrig. **Passung:** nutzt bereits geplante
Artefakte; nur übernehmen, wenn die Ablation einen Gewinn gegenüber dem besten
Einzel-Teacher zeigt.

**Einordnung Distillations-Kaskade (large → base → xsmall):** Die ursprünglich
angedachte Teacher-Assistant-Kette wird nach aktueller Forschungslage **nicht
empfohlen**. Die TAKD-Evidenz ist gemischt (Fehler-Akkumulation über die
Stufen, hohe Trainingskosten), und die *Distillation Scaling Laws* (2025)
zeigen, dass die optimale Teacher-Größe ungefähr linear mit der
Studentengröße wächst – ein sehr großer Teacher hilft einem 17M-Studenten per
Logit-Distillation kaum (Capacity-Gap). Der wertvolle Kern bleibt erhalten,
aber umformuliert: Beim Silber-Labeling zählt nur die **Label-Qualität**, denn
Labels sind Daten – die Capacity-Gap-Beschränkung gilt dort nicht. `large`
wird daher, falls der architektur-gematchte Base-Teacher den Student nicht
klar schlägt, direkt als **Silber-Labeler** eingesetzt (einmaliger
Offline-Lauf, keine Auswirkung auf das ausgelieferte Artefakt), optional mit
Base als Agreement-Partner (siehe 1b) – nicht als Zwischenstufe einer
Distillationskette.

### 5. Konsistenz- und Struktursignale für Morphologie

**Status:** Beide Schritte sind umgesetzt und ausgewählt. Der gemeinsame MLP
liefert die Tokenrepräsentation; der darauf aufbauende
`wide-shared-mlp-structured-morphology`-Decoder verfeinert die unabhängigen
Feature-Logits anhand weicher UPOS- und Morphologie-Verteilungen. Gegenüber dem
unabhängigen Kontrollmodell steigen Morphologie-Micro-F1 und Average Precision
auf Bokmål und Nynorsk bei nur rund 106 KB zusätzlicher Checkpoint-Größe.

**Idee:** UPOS und Morphologie hängen sprachlich zusammen (Verben tragen Tense/
Mood, Nomen tragen Gender/Number). Zwei Optionen: (a) Morphologie-Köpfe leicht
auf die UPOS-Repräsentation konditionieren; (b) einen kleinen gemeinsamen
Engpass-Layer (MLP) vor den linearen Köpfen einführen, statt direkt aus 192
Dimensionen zu projizieren.

**Warum es hilft:** Kann seltene, inkonsistente Fehler reduzieren (z.B.
Zeitform auf einem Nomen).
**Aufwand:** mittel. **Risiko:** mittel (fügt Parameter/Kopplung hinzu – gegen
die aktuelle saubere Entkopplung abzuwägen). **Passung:** nur wenn Ablation
echten Gewinn zeigt; sonst weglassen (Prisms Regel: keine Abstraktion ohne
belegten Nutzen).

### 5a. Lernbare Mischung mehrerer Backbone-Schichten

**Idee:** Prism verwendet derzeit ausschließlich `last_hidden_state`, also die
letzte NorBERT4-Schicht. Ein kleiner Scalar-Mix lernt stattdessen Softmax-
Gewichte über mehrere bereits berechnete Schichten und bildet daraus den
Tokenvektor für Pooling und Heads. Die erste Ablation mischt gemeinsam die
letzten vier Schichten; task-spezifische Mischungen bleiben eine getrennte
spätere Option.

**Warum es hilft:** Untersuchungen monolingualer und multilingualer BERT-
Modelle zeigen, dass POS-Information besonders in mittleren Schichten stark
ist und zusätzliche letzte Schichten nicht immer weitere nutzbare Information
beitragen. Ein lernbarer Mix kann diese Information zurückholen, ohne den
Backbone breiter oder tiefer zu machen.

**Kosten:** nur wenige trainierbare Skalare und praktisch keine zusätzliche
Artefaktgröße. Während Inferenz und Export müssen jedoch mehrere
Zwischenschichten verfügbar bleiben; Peak-Speicher und Exportierbarkeit sind
daher Teil der Ablation.

**Aufwand:** niedrig-mittel. **Risiko:** niedrig-mittel. **Passung:** sehr gut
als nächste isolierte Architekturablation nach dem breiten Shared-MLP.

Primärquelle: [What's so special about BERT's layers?](https://aclanthology.org/2020.findings-emnlp.389/)

---

## B. Modellgröße (Ziel: quantisiertes Paket ≤ 100 MiB)

### 6. Embedding-Tabelle verkleinern: erst quantisieren, dann Matryoshka

**Beobachtung:** Bei Vokabular 51.200 × Hidden 192 hat allein die
Embedding-Tabelle **~9,8 Mio. Parameter** – bei einem ~17-Mio.-Parameter-Modell
also grob **die Hälfte des gesamten Students**. Die eigentliche „Intelligenz"
(Attention + Feed-Forward) ist der kleinere Teil.

**Reihenfolge der Hebel (moderner Blick):**

1. **Embedding quantisieren** (INT8, ggf. INT4). Das ist 2025/2026 der direkteste
   Weg zum Größenziel und braucht keinen Architektureingriff. Da hier die meisten
   Bytes liegen, löst schon die Quantisierung der Embedding-Tabelle einen großen
   Teil des 100-MiB-Drucks (siehe auch 8).
2. **Vokabular beschneiden** (siehe 7), damit die Tabelle gar nicht erst so groß
   ist.
3. **Erst wenn das nicht reicht:** die Dimension reduzieren – aber nicht als
   starre ALBERT-Faktorisierung (51.200 → 128 → 192, Stand 2019), sondern als
   **Matryoshka Representation Learning**: geschachtelte „coarse-to-fine"-Prefixe
   in einer Repräsentation, sodass eine kleinere Dimension als eigenständige
   Repräsentation funktioniert. Das liefert eine einstellbare Größe/Qualität-
   Kurve statt einer einmaligen festen Wahl.

**Warum es hilft:** Direkt auf das 100-MiB-Ziel einzahlend, weil hier die meisten
Bytes liegen – und die Quantisierung erreicht das oft schon ohne den Backbone
umzubauen.
**Aufwand:** niedrig (nur Embedding quantisieren) bis mittel-hoch (Matryoshka
neu trainieren/anpassen). **Risiko:** niedrig-mittel. **Passung:** die
quantisierungszuerst-Reihenfolge passt besser zu „fertiges NorBERT4 laden" als
ein sofortiger Faktorisierungs-Umbau.

### 7. Vokabular auf Norwegisch beschneiden

**Idee:** NorBERT4 wurde laut Modellkarte auf Bokmål, Nynorsk **und Nordsamisch**
vortrainiert. Ein produktiver norwegischer Student braucht viele dieser
Subwords nie. Man misst die Vokabular-Nutzung auf norwegischem Text, behält die
tatsächlich genutzten Stücke und schrumpft die Embedding-Tabelle entsprechend.
Genau diese „Vokabular-Adaption" ist auch der Kern aktueller kompakter Encoder
(z.B. MrBERT-artige Vokabular-/Domänen-Anpassung).

**Warum es hilft:** Kleinere Embedding-Tabelle = kleineres Paket, gleiche
Genauigkeit auf realem norwegischem Input. Kombiniert sich mit (6).
**Aufwand:** mittel. **Risiko:** mittel (seltene Wörter/Fremdwörter dürfen nicht
brechen – über OOV-Rate messen). **Passung:** sehr gut zum Offline-/Klein-Ziel.

### 8. Quantisierung, idealerweise quantisierungs-bewusst (QAT) und geschachtelt (MatQuant)

**Idee:** Den Student in INT8 (ggf. teils INT4 für Embeddings) exportieren.
Statt nur nachträglich zu quantisieren, während der Distillation
**quantisierungs-bewusst** trainieren (QAT), damit der Student die INT8-Rundung
schon im Training „einplant". Der aktuelle Zusatzbaustein ist **MatQuant**
(2025): ein *geschachteltes* Quantisierungsschema, das INT8/INT4/INT2 in
denselben Gewichten trägt (kleinere Bit-Breiten liegen in den höchstwertigen
Bits des int8) und pro Deployment die passende Präzision wählt.

**Warum es hilft:** Ungefähr 4× kleiner und meist schneller auf CPU/mobil, bei
QAT mit minimalem Genauigkeitsverlust. MatQuant passt exakt zu Prisms
Multi-Backend-Manifest: ein Trainingslauf, aber verschiedene `.pte`-Artefakte
(Core ML, XNNPACK, …) können unterschiedliche Präzision fahren. Zahlt auf Größe
**und** Geschwindigkeit ein.
**Aufwand:** mittel-hoch (QAT + Parität PyTorch↔ExecuTorch prüfen; ExecuTorch
bevorzugt den PT2E-QAT-Fluss).
**Risiko:** mittel. **Passung:** direkt auf die dokumentierten Release-Gates
(≤100 MiB, ≤1,0 s median).

---

## C. Geschwindigkeit (Ziel: ≤1,0 s median auf 6.000 Tokens)

### 9. Prune-then-Distill: den Student aus dem Teacher herausschneiden

**Idee:** Latenz auf CPU/mobil hängt stark an der **Schichtzahl** (16 Layer sind
16 sequentielle Blöcke). Statt einen separat vortrainierten „xsmall"-Backbone zu
laden, erzeugt man den Student durch **strukturiertes Pruning des Teachers +
Distillation**: den starken Teacher gezielt auf weniger/schmalere Schichten
beschneiden und die verlorene Qualität allein durch Distillation
wiederherstellen. Zusätzlich als Ablation: weniger, dafür etwas breitere
Schichten bei gleicher Genauigkeit.

**Warum es hilft:** Dieses Rezept ist 2024/2025 der empirisch dominante Ansatz
für kompakte Modelle (bekannt aus NVIDIAs *Minitron* und aus *Sheared LLaMA*):
Aus einem starken Modell prunen und mit Distillation wiederherstellen schlägt
bei gleicher Größe fast immer ein from-scratch trainiertes Kleinmodell. Direkter
Latenz-Gewinn dort, wo er am meisten zählt.
**Aufwand:** hoch. **Risiko:** mittel-hoch. **Passung:** ambitioniert, aber
genau das „unkonventionell erlaubt"-Territorium – und der am besten belegte
Größen/Latenz-Hebel. Verdient höhere Priorität als bisher notiert; nur mit
sauberer Ablation.

### 10. Moderne Encoder-Effizienz für den Export-Student

**Idee:** Der Geschwindigkeitsplan setzt bisher fast nur auf Bucket-Batching.
Falls `norbert4-xsmall` das ≤1,0-s-Gate reißt oder sich – wie in
`ARCHITECTURE.md` vorgesehen – als nicht export-freundlich erweist, würde Prism
ohnehin in einen eigenen Standard-Transformer destillieren. Für diesen Fall
lohnt ein Blick auf **ModernBERT** (Dez. 2024) als Bauvorlage: alternierende
lokale/globale Attention, Unpadding und exportfreundliche Operatoren senken die
Latenz bei gleicher Genauigkeit deutlich.

**Warum es hilft:** Adressiert das Latenz-Gate architektonisch, nicht nur über
Batching. Ein export-freundlicher, effizienzoptimierter Student ist zugleich der
sicherere ExecuTorch-Pfad.
**Aufwand:** hoch (eigener Student-Backbone + Distillation). **Risiko:**
mittel-hoch. **Passung:** nur als Plan B, wenn der xsmall-Backbone das Gate oder
den Export nicht erfüllt; deckt sich mit dem bereits dokumentierten Fallback.

### 11. Bucket-Batching feintunen

**Idee:** `ARCHITECTURE.md` beschreibt bereits Bucket-Batching (Sätze nach Länge
gruppieren, Padding minimieren). Hier geht es um die **Feinabstimmung**:
Bucket-Grenzen, maximale Batch-Tokens pro Bucket und Sortierstrategie empirisch
auf dem 6.000-Token-Dokument-Fixture optimieren, statt feste Werte zu raten.

**Warum es hilft:** Padding ist verschwendete Rechenzeit. Gut gewählte Buckets
erhöhen den Durchsatz ohne Genauigkeitsverlust und ohne die quadratischen Kosten
einer einzigen globalen Attention-Sequenz.
**Aufwand:** niedrig. **Risiko:** niedrig. **Passung:** direkt auf das
dokumentierte Dokument-Inferenz-Gate; reine Messarbeit auf bestehender Pipeline.

---

## D. Vertrauenswürdige Ausgabe

### 12. Kalibrierung + konformale Mondrian-Abstention (Vertrauens-Qualität)

**Idee:** Nach dem Training pro Kopf Temperatur-Skalierung (schon in der Doku
vorgesehen). Darüber hinaus **konformale Vorhersage**: statt einer nackten
Wahrscheinlichkeit liefert das Modell eine Menge plausibler Tags mit einer
*garantierten* Trefferrate, oder markiert „unsicher". Für Token-Klassifikation
ist die präzise Variante **Split-Conformal pro Token mit Mondrian-
Konditionierung** – die Coverage-Garantie gilt dann *pro UPOS-Klasse bzw. pro
Morphologie-Feature*, nicht nur global gemittelt.

**Warum es hilft:** Prism-Vorhersagen erscheinen in Lernsoftware. Eine
mathematisch fundierte „ich bin hier unsicher"-Aussage ist wertvoller als eine
überoptimistische Einzelzahl – und ein echtes Alleinstellungsmerkmal gegenüber
UDPipe. Die Mondrian-Konditionierung ist gerade für Prism entscheidend, weil
häufige Klassen sonst die schwachen seltenen Morphologie-Labels „überdecken"
(genau die, die in `benchmarks.md` noch schwächeln). Der Rahmen ist aktuell gut
belegt (Conformal-Prediction-Survey für NLP, TACL 2024; Abstention-Arbeiten
2025).
**Aufwand:** niedrig-mittel. **Risiko:** niedrig. **Passung:** deckt die
AGENTS.md-Anforderung „kalibrierte Unsicherheit" modern ab.

---

## E. Rigorosität & Vergleichbarkeit

### 13. Faire Benchmark-Matrix gegen UDPipe und moderne Baselines

**Idee:** „Besser als UDPipe" ist nur haltbar bei identischer Datenrevision,
identischen Splits, gleicher Tokenisierungsbedingung (Gold-Tokens vs. Rohtext
nie mischen), gleichen Tasks und Metriken. Sinnvoll: neben UDPipe auch **Stanza,
spaCy und Trankit** als klassische Referenzpunkte – und, damit der Vergleich
nicht gegen einen veralteten Gegner läuft, **moderne mehrsprachige Encoder**
(z.B. mDeBERTa-v3, Glot500, die neuen Matryoshka-Encoder wie m3BERT/MrBERT) als
starke Oberkante. Gemessen werden nicht nur Accuracy, sondern auch
Tokens/Sekunde, Peak-Speicher und Paketgröße.

**Warum es hilft:** Macht die Kernbehauptung des Projekts überhaupt erst
belegbar – und glaubwürdig, weil auch gegen aktuelle Modelle verglichen wird.
**Aufwand:** mittel. **Risiko:** niedrig. **Passung:** bereits als Pflicht in
`docs/model-strategy.md` verankert, aber noch offen.

---

## F. Alternative Architekturen (Motor, Decoder, Kontext)

„BERT als Motor" und „Teacher-Student" sind **zwei getrennte Achsen**: Man kann
den Encoder tauschen, ohne das Distillations-Paradigma aufzugeben, und
umgekehrt. Jede Alternative muss aber drei harte Filter bestehen: (a) es gibt ein
**norwegisch vortrainiertes** Modell oder es entsteht bezahlbar; (b) es ist nach
**ExecuTorch exportierbar** und hält **≤100 MiB / ≤1 s**; (c) es passt zur
gewählten **Kontext-Granularität** (siehe 18). Zur Beruhigung vorab: NorBERT4
nutzt bereits RoPE und modernen Modellcode – der Motor ist **nicht veraltet**,
und sein norwegisches Vortraining ist zu wertvoll, um es für einen englischen/
mehrsprachigen Encoder (ModernBERT, NeoBERT, DeBERTaV3) aufzugeben. Diese dienen
höchstens als Bauvorlage für den Fallback-Student aus Idee 10.

### 14. Zeichen-/Byte-Ebene, tokenizer-frei (H-Net++, BLT, EvaByte)

**Idee:** Statt eines 51.200-Subword-Vokabulars liest der Motor direkt Zeichen
oder Bytes. Die moderne Generation (H-Net++ – *explizit für morphologisch reiche
Sprachen*, Byte Latent Transformer, EvaByte) lernt die Segmentierung dynamisch
und hält die Sequenzen kurz.

**Warum es zu Prism passt:** Trifft mehrere Schmerzpunkte an der Wurzel –
**eliminiert die ~9,8-Mio.-Parameter-Embedding-Tabelle** (Ideen 6/7), ist
**robust gegen Komposita und OOV** (Norwegisch ist kompositionsfreudig) und gibt
Morphologie-/Lemma-Köpfen **direkten Zugriff auf Zeichen** – genau die Ebene, auf
der Flexion und Editierregeln operieren.

**Datenbasis-Realität (wichtig):** Die ~490k UD-Tokens (Bokmål + Nynorsk) sind
drei bis vier Größenordnungen zu klein, um *irgendeinen* Encoder vorzutrainieren;
ein Byte-Modell ist eher **daten-hungriger** als ein Subword-Modell. Ein
Byte-Motor entsteht daher nur über (1) Vortraining auf einem großen Rohkorpus
(Norwegian Colossal Corpus – ohnehin für Idee 1 vorgesehen) oder (2)
**Distillation von NorBERT4 in einen Byte-Student** auf demselben Korpus
(nutzt den Teacher wieder, spart das volle MLM-Budget). Vortraining rein auf UD
ist ausgeschlossen.

**Aufwand:** hoch. **Risiko:** mittel-hoch (längere Sequenzen, Export der
dynamischen Chunking-Operatoren). **Passung:** die stärkste echte
Architektur-Alternative, aber teuer – erst nach den günstigeren Ideen und nur mit
klarer Ablation gegen den NorBERT4-Student.

### 15. Zeichenbewusster Lemma-/Morphologie-Kopf (pragmatischer Teil-Einstieg)

**Umsetzungsstatus:** Der kompakte Char-CNN-Zweig ist implementiert,
exportgetestet, auf zwölf Epochen trainiert und nach getrennten
Bokmål-/Nynorsk- sowie Rare/OOV-Auswertungen ausgewählt. Gegen die strukturierte
Kontrolle gewinnt Rare-Lemma end-to-end 2,67/2,42 Prozentpunkte und
Rare-Morphologie-Micro-F1 1,76/1,50 Punkte auf Bokmål/Nynorsk. Die OOV-Ziele
verbessern sich ebenfalls auf beiden Standards. Der neue Format-3-Teacher ist
daher mit genau dieser Architektur trainiert worden; seine getrennte
Bokmål-/Nynorsk-Auswertung bestätigt ihn als Distillationsquelle.

**Idee:** Nicht den ganzen Motor tauschen, sondern **Zeichenbewusstheit nur dort,
wo sie am meisten bringt**. Zwei Varianten: (a) ein kleines Char-CNN neben dem
Subword-Encoder (CharBERT-Stil), dessen Ausgabe in die Morphologie-/Lemma-Köpfe
fließt; (b) ein kleiner **Zeichen-Level-Transducer** als Lemma-Kopf statt des
1.059-Wege-Editierregel-Klassifikators – für Lemmatisierung der De-facto-Standard
(Stanza, UDPipe 2) und besonders stark bei **unbekannten Wörtern**.

**Warum es zu Prism passt:** Holt einen großen Teil des Byte-Vorteils, ist aber
**allein mit den UD-Daten trainierbar** (kein Vortraining nötig) und lässt den
bewährten NorBERT4-Motor unangetastet. Bestes Aufwand/Nutzen-Verhältnis unter den
Architektur-Ideen.
**Aufwand:** niedrig-mittel. **Risiko:** niedrig. **Passung:** sehr gut; direkter
Angriff auf die dokumentierte Schwäche bei unbekannten Lemmas und seltenen
Morphologie-Werten.

### 16. Leichter strukturierter Decoder (CRF oder Autoregression über Köpfe)

**Umsetzungsstatus:** Als exportfreundlicher paralleler Zwei-Pass-Decoder
implementiert. Der erste Pass erzeugt die bisherigen unabhängigen Logits; der
zweite liest weiche UPOS- und Morphologie-Verteilungen und lernt residuale
Korrekturen für alle Morphologie-Features. Damit gibt es weder eine harte
UPOS-Fehlerkaskade noch eine feste autoregressive Reihenfolge. Der kontrollierte
Bokmål-/Nynorsk-Benchmark wählt die Variante: Morphologie-Micro-F1 steigt auf
beiden Standards, ebenso die Average Precision, bei rund 106 KB zusätzlicher
Checkpoint-Größe. Sie ist daher der neue Gold-only-Standard und die Grundlage
für den geplanten zeichenbewussten Zweig.

**Idee:** Statt jeden Token unabhängig zu klassifizieren, koppelt ein leichter
strukturierter Decoder (klassisches CRF oder eine kleine Autoregression über die
Köpfe) benachbarte Entscheidungen.

**Warum es zu Prism passt:** Adressiert direkt die in `benchmarks.md` genannten
*inkonsistenten* seltenen Fehler (z.B. Tempus auf einem Nomen), ist billig und
exportierbar und ergänzt jeden Motor.
**Aufwand:** niedrig-mittel. **Risiko:** niedrig-mittel (CRF-Export nach
ExecuTorch prüfen). **Passung:** gut als Konsistenz-Ablation.

### 17. State-Space-Modelle / Mamba-2 (beobachten, nicht bauen)

**Idee:** Lineare, nicht-quadratische Sequenzmodelle (Mamba-2, bidirektionales
BiMamba-2 für MLM-artige Ziele) statt Attention.

**Warum eingeschränkt:** Der Hauptvorteil – lineare Kosten bei **langen**
Sequenzen – wird erst relevant, wenn Prism größere Kontexte am Stück verarbeitet
(siehe 18). Selbst dann bleiben: unreifer **ExecuTorch/Mobile-Export** und
**kein norwegisch vortrainiertes SSM**. Ohne beides ist es ein reines
Forschungswette.
**Aufwand:** hoch. **Risiko:** hoch. **Passung:** beobachten; erst wieder
bewerten, wenn Prism auf Kapitel-Kontext umstellt *und* der Export reift.

### 18. Kontext-Granularität: Satz, Fenster oder Kapitel

**Kontext (aktualisiert):** Die App hält den vollständigen Text vor; eine
satzweise Verarbeitung ist **nicht** zwingend. Ein Buchkapitel mit ~6.000 Wörtern
könnte auch als größere Einheit verarbeitet werden. Damit werden drei Optionen
vergleichbar:

- **(a) Satz** – der aktuelle Ansatz. Einfachster Export, kein satzübergreifender
  Kontext.
- **(b) Überlappende Fenster** (z.B. 256–512 Tokens mit Überlappung) – der
  „Sweet Spot": bringt satzübergreifenden Kontext **auf dem bestehenden
  NorBERT4-Motor**, ohne einen einzelnen 6.000-Token-Forward-Pass zu erzwingen.
  Ergebnisse aus dem überlappenden Bereich werden zusammengeführt.
- **(c) Ganzes Kapitel als eine Sequenz** – nur mit einem langkontext-fähigen
  Motor sinnvoll und begrenzt durch (zu prüfen) NorBERT4s trainierte
  Kontextlänge sowie die quadratischen Attention-Kosten. Genau hier würden
  Idee 14 (hierarchische Byte-Modelle) und 17 (SSM) ihren Vorteil ausspielen.

**Wichtig:** UD-Annotationen sind pro Satz definiert. Ob satzübergreifender
Kontext die Tagging-Qualität für UPOS/Morphologie/Lemma überhaupt messbar
verbessert, ist eine **empirische Frage** – vor einem Architekturwechsel erst mit
Option (b) auf dem bestehenden Motor prüfen.
**Aufwand:** niedrig (b) bis hoch (c). **Risiko:** niedrig (b). **Passung:**
Option (b) ist ein günstiges, sofort messbares Experiment und die Voraussetzung,
um überhaupt zu entscheiden, ob sich 14/17 lohnen.

---

## Empfohlene Reihenfolge zum Bewerten

1. **Silber-Daten-Distillation (1, Policy in 1b)** – wahrscheinlich größter
   nächster Genauigkeitssprung, nachdem DKD erfolgreich ausgewählt wurde;
   weil reine Logit-Distillation auf dem kleinen Gold-Set nachweislich ausgereizt
   ist. Feste Vorstufen: architektur-gematchter Base-Teacher, dann
   Teacher-Kalibrierung, dann Pilot mit Dosis-Wirkungs-Kurve.
2. **Embedding quantisieren + Vokabular-Beschneidung (6 + 7)** – größter
   Größensprung, zahlt aufs 100-MiB-Ziel ein, ohne den Backbone umzubauen.
3. **MatQuant/QAT-Quantisierung (8)** – Größe und Geschwindigkeit zugleich.
4. **MiniLMv2-Attention-Distillation (2)** – zusätzlicher Genauigkeitshebel, wenn
   Logit-Distillation und Silber-Daten ausgeschöpft sind.
5. **Prune-then-Distill (9)** – der am besten belegte Latenz-/Größenhebel, aber
   höherer Aufwand; erst nach den günstigeren Schritten.
6. **Konformale Mondrian-Abstention (12)** – Qualitäts-/Vertrauens-
   Differenzierung gegenüber UDPipe.

Ergänzende Architektur-Experimente (parallel bewertbar):

- **Kontext-Fenster-Ablation (18)** – günstig und sofort messbar; klärt, ob
  satzübergreifender Kontext überhaupt hilft, bevor man über Motorwechsel
  nachdenkt.
- **Zeichenbewusster Lemma-/Morphologie-Kopf (15)** – bestes Aufwand/Nutzen-
  Verhältnis unter den Architektur-Ideen, allein mit UD-Daten trainierbar.

Jeder Punkt wird gegen zwei Referenzen gemessen: den gold-only Student und den
gewählten DKD-Student (`t1.0/w0.1`, TCKD/NCKD `1.0/1.0`). Kein Testsplit wird angefasst,
bevor Modell und Kalibrierung fixiert sind.

---

## Quellen (Forschungsstand)

- DistilBERT (2019), TinyBERT (2019) – klassische Encoder-Distillation.
- MiniLM (2020) / MiniLMv2 (2021) – Self-Attention-Relations-Distillation ohne
  Schicht-Zuordnung.
- Decoupled Knowledge Distillation, DKD (2022) – aktueller Standard der
  Logit-Distillation (TCKD/NCKD); dynamische Temperatur: CTKD, LSKD (2023).
- Matryoshka Representation Learning (2022) und Matryoshka Quantization / MatQuant
  (2025) – geschachtelte Dimension bzw. Präzision.
- Minitron (2024) und Sheared LLaMA (2023) – Prune-then-Distill.
- ModernBERT (2024) – effiziente Encoder-Architektur.
- Conformal Prediction for NLP: A Survey (TACL 2024); Abstention-/CAP-Arbeiten
  (2025) – garantierte Unsicherheit.
- BabyLM-Reihe – Distillation kann Vortraining auf denselben Daten schlagen.
- Born-Again Networks (2018), Noisy Student (2020) – Student kann Teacher
  übertreffen; iteriertes Pseudo-Labeling mit Filterung.
- Distillation Scaling Laws (2025) – optimale Teacher-Größe wächst ~linear mit
  der Studentengröße; Beleg gegen große Kaskaden-Teacher bei Logit-Distillation.
- TAKD / Teacher-Assistant-KD (2020) und Folgearbeiten – gemischte Evidenz für
  Zwischenstufen-Kaskaden.
- ModernBERT (2024), NeoBERT (2025), DeBERTaV3 – modernisierte Encoder (nur als
  Fallback-Vorlage relevant, kein Ersatz für norwegisches Vortraining).
- CANINE, ByT5, Charformer, CharBERT sowie H-Net++, Byte Latent Transformer (BLT)
  und EvaByte (2024/2025) – tokenizer-freie Zeichen-/Byte-Modelle, teils
  hierarchisch für morphologisch reiche Sprachen.
- Mamba-2 / BiMamba-2 – lineare State-Space-Modelle (nur bei Kapitel-Kontext und
  gereiftem Export relevant).
- Stanza, UDPipe 2 – Zeichen-Level-Transducer als Referenz für Lemmatisierung.
