# Prism – Ideensammlung für Verbesserungen (Entwurf)

Status: Brainstorm zur Bewertung, **keine akzeptierte Strategie**.
Erstellt: 2026-07-20
Zuletzt überarbeitet: 2026-07-20 (Abgleich mit dem Forschungsstand 2025/2026)

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

### 5. Konsistenz- und Struktursignale für Morphologie

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

1. **DKD-basierte task-spezifische Distillation (3)** – geringster Aufwand, exakt
   der dokumentierte nächste Schritt, sofort messbar gegen die bestehende
   Referenz und die passende Antwort auf das „Über-Glätten" aus `benchmarks.md`.
2. **Silber-Daten-Distillation (1)** – wahrscheinlich größter Genauigkeitssprung,
   weil reine Logit-Distillation auf dem kleinen Gold-Set nachweislich ausgereizt
   ist.
3. **Embedding quantisieren + Vokabular-Beschneidung (6 + 7)** – größter
   Größensprung, zahlt aufs 100-MiB-Ziel ein, ohne den Backbone umzubauen.
4. **MatQuant/QAT-Quantisierung (8)** – Größe und Geschwindigkeit zugleich.
5. **MiniLMv2-Attention-Distillation (2)** – zusätzlicher Genauigkeitshebel, wenn
   Logit-Distillation und Silber-Daten ausgeschöpft sind.
6. **Prune-then-Distill (9)** – der am besten belegte Latenz-/Größenhebel, aber
   höherer Aufwand; erst nach den günstigeren Schritten.
7. **Konformale Mondrian-Abstention (12)** – Qualitäts-/Vertrauens-
   Differenzierung gegenüber UDPipe.

Ergänzende Architektur-Experimente (parallel bewertbar):

- **Kontext-Fenster-Ablation (18)** – günstig und sofort messbar; klärt, ob
  satzübergreifender Kontext überhaupt hilft, bevor man über Motorwechsel
  nachdenkt.
- **Zeichenbewusster Lemma-/Morphologie-Kopf (15)** – bestes Aufwand/Nutzen-
  Verhältnis unter den Architektur-Ideen, allein mit UD-Daten trainierbar.

Jeder Punkt wird gegen zwei Referenzen gemessen: den gold-only Student und den
gewählten distillierten Student (`t1.0/w0.1`). Kein Testsplit wird angefasst,
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
- ModernBERT (2024), NeoBERT (2025), DeBERTaV3 – modernisierte Encoder (nur als
  Fallback-Vorlage relevant, kein Ersatz für norwegisches Vortraining).
- CANINE, ByT5, Charformer, CharBERT sowie H-Net++, Byte Latent Transformer (BLT)
  und EvaByte (2024/2025) – tokenizer-freie Zeichen-/Byte-Modelle, teils
  hierarchisch für morphologisch reiche Sprachen.
- Mamba-2 / BiMamba-2 – lineare State-Space-Modelle (nur bei Kapitel-Kontext und
  gereiftem Export relevant).
- Stanza, UDPipe 2 – Zeichen-Level-Transducer als Referenz für Lemmatisierung.
