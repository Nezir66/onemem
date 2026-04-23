## 1. Projektverständnis

Du baust kein besseres Chat-Log und auch kein klassisches RAG-System, sondern eine **operative Gedächtnisschicht** für eine AI. Dieses Gedächtnis soll neue Informationen aufnehmen, in kleine sinnvolle Einheiten zerlegen, mit Provenienz und Zeitbezug versehen, mit bestehendem Wissen verknüpfen und später gezielt wieder in die Arbeit der AI zurückspielen. Es soll dabei zwei Dinge gleichzeitig leisten: **schnelle Aufnahme laufender Erfahrung** und **langfristig stabile Wissenskonsolidierung**. Genau diese Trennung zwischen flüchtiger Aufnahme und konsolidiertem Langzeitwissen wird in den Papers immer wieder als zentral sichtbar: im Survey als vollständiger Memory-Lifecycle, in GAM als Entkopplung von Encoding und Consolidation, in ByteRover als kuratierte Langzeitstruktur statt externer Black-Box-Memory.

Das Zielbild ist daher: eine AI mit **arbeitsfähigem, wachsendem, auditierbarem Langzeitgedächtnis**, das nicht alles im Kontextfenster halten muss, sondern relevante Dinge über semantische Ähnlichkeit, explizite Beziehungen, Zeitbezug und Wichtigkeit wiederfinden kann. Dieses Gedächtnis soll eher wie ein kuratiertes Wissensnetz funktionieren als wie eine unstrukturierte Vektormüllhalde. Es braucht dafür nicht viele exotische Subsysteme, sondern wenige starke Bausteine mit klaren Rollen.

---

## 2. Zentrale Architekturentscheidung

**Meine klare Empfehlung ist eine hybride Architektur mit nur einer Wahrheitsebene:**

**File-backed canonical memory + abgeleitete Indizes für Vector, Full-Text und Graph + separater Episodic Buffer vor der Konsolidierung.**

Das ist nicht die maximal fancy Lösung, sondern die mit dem besten Verhältnis aus Macht, Einfachheit und Erweiterbarkeit.

### Warum diese Kernarchitektur die richtige ist

Sie kombiniert die stärksten Ideen aus den Papers, ohne deren volle Komplexität zu übernehmen:

- **Von ByteRover** nimmst du die Idee eines **menschenlesbaren, dateibasierten Source of Truth** mit expliziten Relationen, Provenienz und Lifecycle-Metadaten. Das gibt dir Auditierbarkeit, Portabilität und klare Datenhoheit.
- **Von GAM** nimmst du die wichtigste strukturelle Entscheidung: **nicht direkt alles ins Langzeitgedächtnis schreiben**, sondern erst in einen episodischen Puffer und dann an semantischen Grenzen konsolidieren. Das verhindert Memory-Contamination und Drift.
- **Von A-MEM** nimmst du atomare Notizen, Link-Bildung und Evolution, aber nicht als unkontrollierte freie Mutation, sondern als kontrollierten Konsolidierungsprozess.
- **Von GAAMA** nimmst du die Erkenntnis, dass **Graph Retrieval semantische Suche ergänzen soll, nicht ersetzen**, und dass **Konzept-Knoten** bessere Traversal-Anker sind als rein entity-zentrierte Hubs.
- **Von Zep** nimmst du Zeitvalidität, Invalidation statt blindem Überschreiben und Duplicate Resolution als reale Produktionsprobleme ernst.

### Vergleich der Alternativen

#### Reines RAG

Für dein Ziel nicht ausreichend. Flat RAG verliert Beziehungen zwischen Ereignissen, Fakten und Themen; es ist gut für ähnliche Textfragmente, aber schlecht für langlebige, sich entwickelnde Erinnerung. Die Papers zu GAAMA und ByteRover kritisieren genau diese Relationlosigkeit bzw. semantische Drift über externe Pipelines.

#### Reines Graph-System

Auch nicht ideal. Ein reines Graph-System wird schnell zu starr, pflegeintensiv und retrieval-seitig schwerer als nötig. GAM zeigt, dass starre strukturierte Speicher in Echtzeitfluss und narrativer Dynamik schwächeln; GAAMA zeigt zusätzlich das Hub-Problem entity-zentrierter Graphen.

#### Reines File/Markdown-System

Viel besser als viele denken, aber für dein Ziel alleine zu schwach. ByteRover zeigt, dass file-only erstaunlich weit kommt. Aber du willst explizit semantisches Retrieval, Duplicate Handling und skalierbare Relevanzsuche. Dafür brauchst du abgeleitete Indizes. Files allein werden sonst irgendwann langsam und unscharf.

#### Hybride Architektur

Das ist hier die richtige Wahl, **wenn** sie sauber gebaut ist:

- **Files sind Source of Truth**
- **Vector ist Kandidatengenerator**
- **Graph ist Beziehungs- und Traversal-Layer**
- **Episodic Buffer schützt das Langzeitgedächtnis**
- **Konsolidierung ist der zentrale Intelligenzpunkt**

Das ist die stärkste und zugleich einfachste produktfähige Synthese aus den Papern.

---

## 3. Prinzipien der Architektur

1. **Eine Wahrheit, mehrere Sichten.**
   Canonical Memory liegt in lesbaren Dateien. Vector-, Full-Text- und Graph-Indizes sind nur abgeleitete Views und jederzeit rebuildbar. Das hält Source of Truth klar.

2. **Capture und Consolidate trennen.**
   Neue Information landet zuerst im Arbeits-/Episodenpuffer, nicht direkt im stabilen Langzeitwissen. Das ist die wichtigste Schutzmaßnahme gegen Memory-Spam und semantische Drift.

3. **Atomare Memory Units statt großer Blöcke.**
   Kleine, präzise Notizen/Fakten sind leichter zu verlinken, zu bewerten, zu aktualisieren und zu vergessen. A-MEM zeigt genau den Nutzen solcher atomaren Einheiten.

4. **Explizite Provenienz überall.**
   Jede konsolidierte Erinnerung muss sagen, woher sie kommt: Quelle, Zeit, Gespräch, Tool-Output, Beobachtung, Ableitung. Das ist essenziell für Vertrauen und Korrekturen. ByteRover und Zep betonen diesen Punkt stark.

5. **Zeit ist first-class.**
   Nicht nur `created_at`, sondern auch `valid_from`/`valid_to` bzw. Gültigkeitsfenster. Wissen darf nicht stillschweigend überschrieben werden. Zep zeigt, warum zeitliche Invalidation produktiv wichtig ist.

6. **Graph-Struktur ja, Graph-DB nein.**
   Du brauchst Beziehungen und Traversal, aber anfangs keinen spezialisierten Graph-Stack. Eine Edge-Tabelle als Sidecar reicht. Komplexität sparen. Das ist architektonisch viel sinnvoller für V1.

7. **Vector Search ist Kandidatengenerierung, nicht Wahrheit.**
   Embeddings finden grob Relevantes. Die eigentliche Präzision entsteht durch Provenienz, Graph-Nachbarn, Zeitfilter und Re-Ranking. GAAMA stützt genau diese mild-augmentierende Nutzung.

8. **LLM nur dort einsetzen, wo Semantik wirklich gebraucht wird.**
   Für Konsolidierung, Dedupe-Entscheidungen, Relationserzeugung, Zusammenfassung. Nicht für jeden simplen Lookup.

9. **Forgetting ist Teil des Designs, nicht späteres Tuning.**
   Das System braucht von Anfang an Abschwächung, Archivierung, Invalidation und TTL-Regeln. Der Survey und ByteRover machen klar, dass Memory Evolution immer auch Vergessen einschließt.

10. **Single-writer, deterministic apply.**
    AI darf Writes vorschlagen, aber ein kontrollierter Memory-Manager wendet sie an. Nicht beliebige freie Mutation im Speicher. Das spart viele spätere Probleme.

---

## 4. Empfohlene Zielarchitektur

### Kernidee

Eine **2-stufige Memory-Architektur**:

- **Stufe A: Episodic Working Memory**
  - hält rohe, frische, unvollständig verstandene Information
  - append-only
  - kurzlebig
  - noch nicht vertrauenswürdig

- **Stufe B: Consolidated Long-Term Memory**
  - hält atomare Fakten, Konzepte, Relationen, Hypothesen, Zusammenfassungen
  - wird nur über den Konsolidierer geschrieben
  - ist retrieval-optimiert und auditierbar

### Wichtigste Komponenten

#### 1. Agent Runtime

Die AI arbeitet normal, kann aber zwei Dinge tun:

- `retrieve_memory(query)`
- `propose_memory_write(observation, reason, source)`

Sie schreibt nicht direkt in die Langzeitstruktur.

#### 2. Episodic Buffer

Speichert neue Beobachtungen, Dialogausschnitte, Tool-Ergebnisse, Zwischenresultate, Fehler, Entscheidungen.
Inspiriert von GAM und Zep: rohe Episoden bleiben erhalten, bevor aus ihnen stabilere Einheiten werden.

#### 3. Consolidation Engine

Das ist der wichtigste Dienst.
Er macht aus Episoden:

- atomare Fakten
- Konzepte/Themen
- optionale Summaries
- typed relations
- Invalidation/Update/Merge-Entscheidungen

Hier fließen A-MEM, GAM, GAAMA, ByteRover und Zep zusammen.

#### 4. Canonical Memory Store

Dateibaum mit Markdown-Dateien als lesbare Wahrheitsebene.
Eine Datei pro Memory Node. Beziehungen und Metadaten im Frontmatter/Body. ByteRover liefert dafür die sauberste Referenz.

#### 5. Sidecar Index Layer

Abgeleitete Indizes:

- Full-Text Index
- Vector Index
- Edge/Adjacency Index
- Alias/Dedupe Register
- optional Retrieval Cache

Nicht kanonisch, nur beschleunigend.

#### 6. Retrieval Orchestrator

Orchestriert:

- semantische Kandidaten
- lexikalische Kandidaten
- zeitliche Filter
- Graph-Expansion
- Re-Ranking
- Context Packing

GAAMA und GAM geben hier die wichtigsten Ideen: top-down bzw. mild graph-augmented retrieval.

#### 7. Maintenance Worker

Läuft asynchron:

- decay
- promotion/demotion
- archive
- merge duplicates
- invalidate contradictions
- regenerate summaries

### Zusammenspiel

```text
Neue Information
  -> Episodic Buffer
  -> Boundary/Flush Trigger
  -> Consolidation Engine
  -> Canonical Files schreiben/aktualisieren
  -> Sidecar Indizes aktualisieren

Anfrage der AI
  -> Retrieval Orchestrator
  -> recent episodic + stable candidates
  -> graph expansion + rerank
  -> context packer
  -> AI Antwort / Handlung
```

Das ist stark genug für echtes Gedächtnis, aber deutlich einfacher als ein volles Forschungs-Ökosystem.

---

## 5. Minimaler Kernstack

### Die 4 Komponenten, die für ein starkes MVP reichen

1. **Canonical File Store**
   - Markdown-Dateien mit YAML-Frontmatter
   - Source of Truth

2. **SQLite Sidecar**
   - FTS5 für Volltext
   - Vector Extension (`sqlite-vec`) oder minimal FAISS
   - Tabellen für edges, aliases, metadata

3. **Memory Consolidator**
   - LLM-gestützte Extraktion, Merge, Relationserzeugung, Scoring

4. **Retrieval Orchestrator**
   - Candidate generation
   - Graph expansion
   - Re-ranking
   - Context assembly

### Absolut notwendig

- atomare Memory Nodes
- Provenienz
- Zeitfelder
- Importance/Decay
- explizite Relationen
- Vector + lexical retrieval
- ein episodischer Puffer vor Langzeitspeicherung

### Nice to have

- Graph-DB
- PPR
- Community detection
- RL-optimierte Retrieval Policies
- trainable graph weights
- multimodales Memory
- agent-native Sandboxed Code Curation à la ByteRover
- bi-temporal Vollmodell à la Zep

Für V1 sind diese Dinge nicht nötig. Einige davon sind spannend, aber klar V2/V3-Material.

---

## 6. Gedächtnismodell

Ich würde **ein einheitliches Node-Modell mit wenigen Typen** bauen, nicht viele Spezialobjekte.

### Node-Typen

#### A. Episode

Rohbeobachtung oder Rohinteraktion.

- flüchtig
- append-only
- hoher Zeitwert
- niedriger Stabilitätswert
- kann wörtliche Zitate enthalten

#### B. Fact

Atomare konsolidierte Aussage.

- aus Episoden extrahiert
- mit Provenienz
- mit Zeitgültigkeit
- mit Confidence

#### C. Concept

Themen-/Konzeptanker, nicht primär Personen-Hubs.

- verbindet verwandte Facts/Episodes
- dient als Traversal-Pivot
- reduziert Hub-Probleme im Vergleich zu entity-zentrierten Graphen.

#### D. Summary

Kompakte Verdichtung einer Session, eines Themas oder eines Clusters.

- dient Compaction
- nicht Wahrheitsebene über Facts, sondern Kontexthelfer

#### E. Hypothesis

Kein eigener Strukturtyp nötig; technisch ein Fact mit `status=hypothesis`.

- inferiert
- niedrigere Confidence
- kurze TTL
- darf nur mit Vorsicht retrieved werden

### Zusätzliche Flags

- `pinned=true` für user-markierte wichtige Erinnerungen
- `confidence`
- `salience`
- `stability_state`
- `valid_from`, `valid_to`
- `source_refs`
- `relations`

### Stabilitätszustände

- `ephemeral`
- `candidate`
- `stable`
- `core`
- `deprecated`

### Wie Informationen entstehen und sich verändern

- **Entstehen:** aus Episoden durch Konsolidierung
- **Bewertet:** anhand Quelle, Wiederholung, Nutzung, Explizitheit, User-Pin
- **Verstärkt:** bei Wiederauftreten, erfolgreicher Nutzung, bestätigender Evidenz
- **Abgeschwächt:** durch Zeit, Nicht-Nutzung, Widerspruch, niedrige Utility
- **Konsolidiert:** wenn mehrere Episoden dieselbe Aussage stützen oder eine semantische Einheit abgeschlossen ist
- **Vergessen:** durch TTL, Archivierung, Kompression oder Invalidation statt blindem Löschen

### Konkret empfohlene Logik

- Episode → nach Flush in Kandidaten zerlegt
- Kandidaten-Fact → `stable`, wenn er mindestens:
  - aus 2 unabhängigen Evidenzen kommt, oder
  - 1 hochwertige Evidenz + erfolgreiche spätere Nutzung hat

- `stable` → `core`, wenn häufig benutzt, gepinnt oder zentral verlinkt
- Hypothese verfällt automatisch, wenn sie nicht bestätigt wird
- Widersprochene Facts werden **nicht überschrieben**, sondern `deprecated` mit `valid_to`

Das folgt inhaltlich ByteRovers AKL, Zeps temporaler Invalidation und A-MEMs evolvierender Notizstruktur, aber in deutlich vereinfachter Form.

---

## 7. Datenfluss

### Wenn neue Information hereinkommt

1. Rohdaten kommen rein: User-Nachricht, Tool-Output, Beobachtung, Fehler, Entscheidung.
2. Sie werden als **Episode** gespeichert.
3. Ein Boundary Trigger entscheidet, ob konsolidiert wird:
   - Themenwechsel
   - Task abgeschlossen
   - Zeitfenster voll
   - manuelles Flush

4. Erst dann startet Konsolidierung.
   Das ist die praktische, einfache Version von GAMs Separation zwischen Buffering und Consolidation.

### Wenn die AI arbeitet

1. Anfrage oder Task wird formuliert.
2. Retrieval Orchestrator fragt zuerst:
   - recent episodic memory
   - stable facts
   - pinned memories
   - summaries

3. Kandidaten kommen aus:
   - Full-Text
   - Vector Search
   - optional Zeitfilter

4. Danach Graph Expansion:
   - 1–2 Hops über Relation- und Concept-Kanten

5. Danach Re-Ranking:
   - semantic relevance
   - lexical relevance
   - salience
   - confidence
   - recency
   - time validity

6. Context Packer baut kompakten Arbeitskontext.

### Wenn Wissen gespeichert wird

1. Consolidator extrahiert:
   - candidate facts
   - concepts
   - optional summary
   - relations

2. Dedupe Resolver sucht:
   - exact match
   - alias match
   - embedding-near match
   - same concept + similar wording

3. Ergebnis ist dann:
   - `ADD`
   - `UPDATE`
   - `MERGE`
   - `INVALIDATE`
   - `LINK`

ByteRovers Operationstypen sind hier ein guter Referenzpunkt, aber ich würde sie nicht frei durch den Agenten ausführen lassen, sondern durch einen kontrollierten Apply-Step.

### Wenn Wissen wiedergefunden wird

1. Query embedding + FTS
2. Kandidatenpool
3. Graph-/Concept-Expansion
4. Zeitfilter
5. Re-Rank
6. Pack nach Typbudget:
   - Facts zuerst
   - dann wenige Episoden
   - dann Summary
   - Hypothesen nur wenn explizit zugelassen

### Wenn Wissen konsolidiert oder vergessen wird

- alte Episoden werden in Summary verdichtet
- low-salience Episoden werden archiviert oder gelöscht
- stable Facts bleiben, aber ranken niedriger mit decay
- widersprochene Facts werden invalidiert
- selten genutzte Hypothesen verfallen schnell

Das entspricht dem Lifecycle Extraction → Storage → Retrieval → Evolution aus dem Survey.

---

## 8. Rolle von Files, Vector und Graph

### Text-/Markdown-Files

**Sollen tun:**

- Source of truth sein
- menschenlesbar und versionierbar sein
- Provenienz und Audit ermöglichen
- manuelle Inspektion/Korrektur erlauben

**Sollen bewusst nicht tun:**

- semantische Suche alleine tragen
- komplexe Traversals performant ausführen
- Dedupe/Ranking alleine lösen

ByteRover zeigt die Stärke von Files als canonical layer sehr klar.

### Vector Search

**Soll tun:**

- schnelle Kandidaten generieren
- semantisch ähnliche Facts/Episodes finden
- Dedupe-Vorschläge machen

**Soll bewusst nicht tun:**

- Wahrheit definieren
- Relationen modellieren
- Zeitlogik oder Widersprüche auflösen
- finale Auswahl alleine bestimmen

A-MEM zeigt, wie nützlich Embeddings für Linking und Retrieval sind; GAAMA zeigt, dass reine Similarity nicht reicht.

### Graph Memory

**Soll tun:**

- explizite Beziehungen modellieren
- Multi-hop-Zusammenhänge abbilden
- Provenienzpfade und Themenzusammenhang liefern
- semantische Suche strukturell verfeinern

**Soll bewusst nicht tun:**

- primäres Raw-Text-Archiv sein
- alleinige Retrieval-Engine sein
- von Anfang an als komplexe Graph-DB ausgerollt werden

GAAMA und GAM sprechen stark dafür, dass Graph-Struktur Retrieval verbessert, aber nicht den kompletten Recall ersetzen sollte.

**Kurzform:**
**Files = Wahrheit.**
**Vectors = Findbarkeit.**
**Graph = Zusammenhang.**

---

## 9. Umgang mit Problemen

### Memory-Spam

Lösung:

- erst in Episodic Buffer
- nur boundary-basiert konsolidieren
- salience threshold
- Kandidatenstatus statt sofort stable
- aggressive TTL für Episoden und Hypothesen

Das adressiert genau das Contamination-Problem aus GAM.

### Falsche Generalisierung

Lösung:

- Hypothesen separat markieren
- Confidence + Evidence Count
- keine stillschweigende Promotion
- Summary nie ohne zugrundeliegende Facts allein vertrauen

### Entitäten-Duplikate

Lösung:

- Alias-Register
- canonical IDs
- exact + lexical + vector duplicate detection
- manuelle/LLM-assisted merge queue
- keine voll entity-zentrierte Graphdominanz

Zep zeigt gute Entity-Resolution-Prinzipien; GAAMA warnt gleichzeitig vor entity-hubs.

### Zeitliche Fehler

Lösung:

- `valid_from` / `valid_to`
- Invalidation statt overwrite
- Query-time time filtering
- relative Zeiten immer an Referenzzeit auflösen

Zep ist hier der wichtigste Input.

### Vertrauensproblem

Lösung:

- Provenienz pro Fact
- observed vs inferred
- confidence score
- pinned / user-verified marker
- lokale Korrektur einzelner Knoten/Kanten

GAM betont, dass falsche Verknüpfungen lokal korrigierbar sein müssen.

### Unendliches Wachstum

Lösung:

- episodische TTL
- Kompression in summaries
- significance-based pruning
- archived/deprecated layer
- ranking-decay statt ewiger Gleichbehandlung

Survey und ByteRover stützen diesen Mechanismus.

### Retrieval-Fehler

Lösung:

- hybrid candidate generation
- graph expansion nur mild
- type budgets
- OOD / “weiß ich nicht”-Pfad
- final rerank mit Zeit + Confidence + Salience

ByteRover betont OOD Detection; GAAMA zeigt mild graph augmentation statt Dominanz.

---

## 10. Vereinfachung

### Was du anfangs weglassen solltest

- volle Graph-DB
- Personalized PageRank
- Community detection
- RL-gesteuerte Retrieval- oder Write-Policies
- trainierbare Graphgewichte
- multimodales Memory
- agent-native Code-Sandbox-Curation
- bi-temporales Vollmodell
- komplexe Meta-Cognition-Schichten
- automatische Selbst-Exploration

### Dinge, die schlau klingen, aber V1 unnötig kompliziert machen

- “Der Agent soll selbstständig seine Speicherstruktur neu erfinden”
- “Wir brauchen sofort ein knowledge graph engine backend”
- “Retrieval muss lernbar sein”
- “Wir sollten gleich episodic, semantic, reflection, strategy, community und meta-memory getrennt modellieren”
- “Wir brauchen ein differenzierbares Graph-Memory”

Das trainable graph memory paper ist spannend, aber klar V3-Material; es optimiert Gedächtnisgewichte per RL und injiziert Strategien in Trainingsschleifen. Das ist weit weg von einem robusten, baubaren Produkt-V1.

### Wo Overengineering droht

- beim Speicher-Backend
- bei zu vielen Node-Typen
- bei autonomer Graph-Reorganisation
- bei zu komplexer Boundary Detection
- bei zu “intelligenten” Forgetting-Policies
- bei Multi-Agent-Memory-Sharing vor Single-Agent-Stabilität

Die richtige V1 ist nicht “maximal kognitiv”, sondern **mechanisch klar und semantisch stark an genau wenigen Stellen**.

---

## 11. Konkreter Architekturvorschlag für V1

### Klare Empfehlung

**Baue V1 als file-backed memory system mit SQLite-Sidecar, episodischem Puffer und kontrolliertem Konsolidierer.**

### Komponentenliste

1. **Agent Runtime**
   - arbeitet normal
   - darf Retrieval auslösen
   - darf Write-Proposals erzeugen

2. **Episodic Buffer**
   - speichert rohe Einträge append-only
   - pro User/Projekt/Session namespace

3. **Memory Consolidator**
   - LLM-gestützte Extraktion von Facts, Concepts, Relations, Summaries
   - Merge / Invalidate / Promote / Demote

4. **Canonical File Store**
   - Markdown-Dateien mit YAML-Frontmatter

5. **SQLite Sidecar**
   - `nodes`
   - `edges`
   - `aliases`
   - `fts_index`
   - `embeddings`
   - `retrieval_cache`

6. **Maintenance Worker**
   - decay
   - archive
   - prune
   - reindex
   - summary refresh

### Speicherlogik

**Dateistruktur**

```text
memory/
  episodes/YYYY/MM/
  facts/<concept>/
  concepts/
  summaries/
  archive/
```

**Frontmatter pro Node**

```yaml
id:
kind: episode|fact|concept|summary
title:
status: ephemeral|candidate|stable|core|deprecated
confidence:
salience:
pinned:
created_at:
updated_at:
valid_from:
valid_to:
source_refs:
entity_refs:
concept_refs:
relations:
```

### Retrievallogik

1. recent episodic recall
2. FTS + vector candidate generation
3. temporal filtering
4. 1-hop / 2-hop graph expansion über:
   - explicit relations
   - concept links

5. additive rerank:
   - semantic similarity
   - lexical match
   - salience
   - confidence
   - recency
   - validity match

6. context packer mit Typbudgets:
   - 8–15 facts
   - 2–5 episodes
   - 1–3 summaries
   - hypotheses nur optional

### Konsolidierungslogik

Bei Flush:

- speichere rohe Episode
- extrahiere candidate facts
- ordne concepts zu
- prüfe Duplicates
- merge/update/invalidate
- verlinke zu ähnlichen Facts
- generiere/aktualisiere Topic- oder Session-Summary

### Forgetting-Mechanismus

- **Episodes:** TTL 7–30 Tage, wenn unreferenziert und low-salience
- **Hypotheses:** TTL 14 Tage, wenn unbestätigt
- **Candidate Facts:** löschen oder archivieren, wenn nie verstärkt
- **Stable Facts:** nicht hart löschen, sondern rank decay und bei Konflikt deprecate
- **Pinned/Core:** nie automatisch löschen

### Einfache Skalierungsstrategie

- Single writer queue pro namespace
- SQLite Sidecar lokal
- Rebuildbare Indizes
- Sharding später pro User/Projekt
- erst ab deutlicher Größenordnung Umstieg auf Postgres/pgvector oder Object Storage

Das ist klein, robust und direkt baubar. Es nutzt die stärksten Paper-Ideen, ohne ihre gesamte Forschungsmaschinerie mitzuschleppen.

---

## 12. Roadmap

### V1 — einfach, robust, baubar

- file-backed canonical memory
- episodic buffer
- LLM-Konsolidierer
- SQLite sidecar mit FTS + vector + edges
- salience/confidence/time fields
- basic forgetting
- simple graph expansion ohne PPR

**Ziel:** zuverlässig schreiben, wiederfinden, konsolidieren, vergessen.

### V2 — mehr Fähigkeiten

- bessere Boundary Detection
- bessere Dedupe-/Alias-Auflösung
- typed relation taxonomy
- summary layers pro Thema/Session
- OOD detection
- retrieval cache
- UI für inspect/edit/pin/delete
- stärkere Zeitlogik bei Queries

**Ziel:** Präzision, Korrekturfähigkeit, bessere Langzeitstabilität.

### V3 — fortgeschritten

- community/global summaries
- PPR oder fortgeschrittene graph reranking
- feedback-driven memory policy
- uncertainty-aware hypothesis handling
- optional RL-/utility-learned retrieval
- multimodale Episoden
- strategy/meta-memory nur falls dein Agent wirklich aus Task-Trajektorien lernen soll

**Ziel:** nicht nur erinnern, sondern strukturell aus Erfahrung besser werden.

---

## Endempfehlung in einem Satz

**Baue ein hybrides Memory-System mit dateibasiertem Source of Truth, getrenntem Episodic Buffer, LLM-basierter Konsolidierung und abgeleiteten Vector-/Graph-Indizes — aber halte Graph als Modell, nicht als frühe Infrastrukturpflicht, und Vector als Suchhilfe, nicht als Wahrheit.**

Das ist aus meiner Sicht die architektonisch beste Lösung für dein Ziel: stark, simpel, modular und realistisch baubar.
