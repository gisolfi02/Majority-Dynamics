# Majority Dynamics in Online Networks

Progetto per lo studio di un processo di **Influence Diffusion basato sulla Majority Dynamics** in una online social network.

L'obiettivo è selezionare, sotto un vincolo di budget, un insieme di nodi inizialmente attivi (_seed set_) capace di produrre la maggiore diffusione possibile nella rete e successivamente valutare quanto tale soluzione rimanga efficace quando la struttura della rete cambia nel tempo.

Il progetto comprende:

- utilizzo di una rete sociale reale;
- simulazione del processo di **Majority Cascade**;
- due differenti funzioni di costo associate ai nodi;
- due varianti dell'algoritmo `Cost-Seeds-Greedy`;
- un algoritmo aggiuntivo, `Cascade-Gain Greedy`, sviluppato appositamente per il progetto;
- esperimenti al variare del budget;
- esperimenti di robustezza rispetto alla rimozione casuale di archi;
- esperimenti di robustezza rispetto alla rimozione casuale di nodi;
- generazione di grafici, CSV e tabelle riassuntive.

---

# 1. Problema

Consideriamo una rete non orientata

$$
G=(V,E)
$$

dove:

- $V$ è l'insieme dei nodi;
- $E$ è l'insieme degli archi;
- ogni nodo rappresenta un utente;
- ogni arco rappresenta una relazione tra due utenti.

Dato un insieme iniziale di nodi attivi

$$
S\subseteq V,
$$

detto **seed set**, vogliamo studiare come l'attivazione si propaga nella rete secondo un processo di Majority Dynamics.

A ogni nodo viene inoltre associato un costo

$$
c:V\rightarrow\mathbb{N}.
$$

Il costo complessivo di un seed set è:

$$
c(S)=\sum_{u\in S}c(u).
$$

Dato un budget massimo $k$, cerchiamo quindi un seed set che soddisfi:

$$
c(S)\leq k
$$

e che produca una diffusione quanto più grande possibile.

L'obiettivo è quindi massimizzare:

$$
|Inf[G,S]|.
$$

Il problema di individuare il seed set ottimo è computazionalmente difficile; per questo motivo vengono utilizzate differenti **euristiche greedy**.

---

# 2. Dataset

## Facebook Social Network

Il progetto utilizza il dataset **Facebook Combined** della collezione SNAP.

Il file utilizzato è:

```text
data/facebook_combined.txt
```

La rete è rappresentata tramite una **edge list**:

```text
u v
```

dove ogni riga indica la presenza di un arco non orientato tra i nodi $u$ e $v$.

### Caratteristiche principali

La rete contiene:

- **4.039 nodi**
- **88.234 archi**
- una singola componente connessa principale contenente tutti i nodi

La rete viene caricata tramite `NetworkX` come:

```python
nx.Graph
```

poiché le relazioni di amicizia vengono considerate non orientate.

La scelta di Facebook è particolarmente adatta al progetto perché il modello di Majority Dynamics dipende direttamente da:

- grado dei nodi;
- vicinati;
- struttura locale della rete;
- densità delle connessioni sociali.

---

# 3. Analisi descrittiva della rete

Prima degli esperimenti vengono calcolate alcune statistiche strutturali della rete:

- numero di nodi;
- numero di archi;
- grado medio;
- grado minimo;
- grado massimo;
- densità;
- numero di componenti connesse;
- dimensione della componente principale;
- coefficiente medio di clustering.

Sono inoltre prodotti due istogrammi utilizzabili per descrivere il dataset:

```text
results/figures/degree_histogram.png
results/figures/local_clustering_histogram.png
```

## Distribuzione dei gradi

Per ogni nodo $v$ il grado è:

$$
d(v)=|N(v)|
$$

dove $N(v)$ rappresenta l'insieme dei suoi vicini.

Il grado medio della rete è:

$$
\bar d =
\frac{1}{|V|}
\sum_{v\in V}d(v)
=
\frac{2|E|}{|V|}.
$$

La distribuzione dei gradi permette di osservare la presenza di nodi con quantità differenti di connessioni.

Questa caratteristica è particolarmente importante perché sia il processo di Majority Cascade sia una delle funzioni di costo utilizzate dipendono direttamente da $d(v)$.

## Clustering locale

Per ogni nodo viene inoltre calcolato il coefficiente di clustering locale, che misura quanto i vicini di un nodo tendano a essere collegati tra loro.

L'istogramma della distribuzione del clustering permette quindi di descrivere la struttura locale tipica della social network.

---

# 4. Majority Cascade

Dato il grafo

$$
G=(V,E)
$$

e un seed set

$$
S\subseteq V,
$$

il processo di influenza viene definito come una sequenza:

$$
Inf[S,0],Inf[S,1],\ldots,Inf[S,r],\ldots
$$

con:

$$
Inf[S,0]=S.
$$

A ogni round un nodo inattivo $v$ diventa attivo se almeno metà dei propri vicini è già attiva:

$$
|N(v)\cap Inf[S,r-1]|
\geq
\frac{d(v)}{2}.
$$

Poiché il numero di vicini attivi è intero, nel codice la soglia viene rappresentata come:

$$
t(v)=
\left\lceil
\frac{d(v)}{2}
\right\rceil.
$$

Equivalentemente:

```python
threshold = (degree + 1) // 2
```

Il processo termina nel primo istante $t$ tale che:

$$
Inf[S,t]=Inf[S,t+1].
$$

L'insieme finale dei nodi influenzati è:

$$
Inf[G,S]=Inf[S,t].
$$

Una volta attivato, un nodo rimane attivo per tutti i round successivi.

## Implementazione

La simulazione è contenuta in:

```text
src/diffusion.py
```

e la funzione principale è:

```python
majority_cascade(graph, seeds)
```

che restituisce:

```text
active
rounds
```

dove:

- `active` rappresenta $Inf[G,S]$;
- `rounds` rappresenta il numero di round di propagazione necessari per raggiungere la stabilità.

L'implementazione utilizza una `frontier` contenente i nodi che si attivano nello stesso round, garantendo quindi una **dinamica sincrona**.

---

# 5. Funzioni di costo

Nel progetto vengono utilizzate due funzioni di costo.

---

## 5.1 Random Cost

A ogni nodo viene assegnato un costo casuale intero nel range:

$$
c_{random}(u)\in[1,10].
$$

Formalmente:

$$
c_{random}:V\rightarrow\{1,\ldots,10\}.
$$

Il generatore pseudo-casuale utilizza un **seed fissato**, in modo che l'assegnazione dei costi sia riproducibile.

È fondamentale utilizzare la stessa configurazione dei costi per tutti gli algoritmi, in modo da confrontarli sullo stesso problema.

Cambiare il seed produce una diversa istanza dei costi, ma durante una singola campagna sperimentale esso deve rimanere invariato.

---

## 5.2 Degree Cost

La seconda funzione dipende dal grado del nodo.

La funzione teorica prevista è:

$$
c(u)=\frac{d(u)}{2}.
$$

Poiché nel progetto i costi appartengono a $\mathbb{N}$, nell'implementazione viene utilizzata:

$$
\boxed{
c_{degree}(u)=
\left\lceil
\frac{d(u)}{2}
\right\rceil
}
$$

e quindi:

```python
cost = (graph.degree[node] + 1) // 2
```

Questa funzione rende i nodi fortemente connessi più costosi.

Di conseguenza un nodo potenzialmente molto influente può richiedere una quota significativa del budget.

---

# 6. Costo del seed set

Indipendentemente dalla funzione di costo utilizzata:

$$
c(S)=
\sum_{u\in S}c(u).
$$

Ogni algoritmo deve quindi produrre una soluzione compatibile con:

$$
c(S)\leq k.
$$

---

# 7. Algoritmi

Nel progetto vengono confrontati tre algoritmi:

1. `Cost-Seeds-Greedy` con $f_1$;
2. `Cost-Seeds-Greedy` con $f_2$;
3. `Cascade-Gain Greedy`.

---

# 8. Cost-Seeds-Greedy

La struttura generale di `Cost-Seeds-Greedy` parte da:

$$
S=\emptyset
$$

e seleziona iterativamente un nodo $u$ sulla base del rapporto:

$$
\frac{\Delta_u f_i(S)}{c(u)}
$$

dove:

$$
\Delta_u f_i(S)
=
f_i(S\cup\{u\})-f_i(S)
$$

rappresenta il **guadagno marginale** ottenuto inserendo il nodo $u$ nel seed set corrente.

In termini intuitivi, l'algoritmo cerca il nodo che fornisce il maggiore beneficio per unità di costo.

---

# 9. Cost-Seeds-Greedy con $f_1$

La prima funzione obiettivo è:

$$
f_1(S)=
\sum_{v\in V}
\min
\left\{
|N(v)\cap S|,
\left\lceil\frac{d(v)}2\right\rceil
\right\}.
$$

Per ogni nodo $v$, $f_1$ misura quanti suoi vicini appartengono al seed set, limitando però il contributo alla soglia majority.

Se:

$$
t(v)=
\left\lceil\frac{d(v)}2\right\rceil,
$$

allora, una volta raggiunti $t(v)$ vicini appartenenti a $S$, ulteriori seed nel vicinato di $v$ non aumentano $f_1$.

## Guadagno marginale

Quando viene valutato un candidato $u$, soltanto i suoi vicini possono modificare il proprio contributo.

Il guadagno può quindi essere calcolato come:

$$
\Delta_u f_1(S)
=
\left|
\left\{
v\in N(u):
|N(v)\cap S|<t(v)
\right\}
\right|.
$$

Il candidato viene valutato attraverso:

$$
score_{f_1}(u,S)=
\frac{\Delta_u f_1(S)}{c(u)}.
$$

---

# 10. Cost-Seeds-Greedy con $f_2$

La seconda funzione è:

$$
f_2(S)=
\sum_{v\in V}
\sum_{i=1}^{|N(v)\cap S|}
\max
\left\{
\left\lceil\frac{d(v)}2\right\rceil-i+1,
0
\right\}.
$$

Se definiamo:

$$
t(v)=
\left\lceil\frac{d(v)}2\right\rceil
$$

e:

$$
s(v)=|N(v)\cap S|,
$$

l'aggiunta di un candidato $u$ produce un incremento:

$$
\boxed{
\Delta_u f_2(S)
=
\sum_{v\in N(u)}
\max
\left\{
t(v)-s(v),
0
\right\}
}
$$

e lo score utilizzato dal greedy è:

$$
score_{f_2}(u,S)=
\frac{\Delta_u f_2(S)}{c(u)}.
$$

A differenza di $f_1$, il contributo non è sempre unitario.

Un nodo molto lontano dalla propria soglia può contribuire maggiormente al valore di $f_2$.

---

# 11. Cascade-Gain Greedy

Il terzo algoritmo è stato sviluppato appositamente per il progetto.

A differenza di $f_1$ e $f_2$, che utilizzano funzioni surrogate per stimare l'utilità di un candidato, **Cascade-Gain Greedy (CGG)** valuta direttamente quanto aumenterebbe la vera Majority Cascade.

Dato il seed set corrente $S$, per ogni candidato:

$$
u\notin S
$$

viene calcolato:

$$
Inf[G,S\cup\{u\}].
$$

Il guadagno marginale reale è:

$$
\boxed{
\Delta_u Inf(S)
=
|Inf[G,S\cup\{u\}]|
-
|Inf[G,S]|
}
$$

e lo score è:

$$
\boxed{
score_{CGG}(u,S)
=
\frac{
|Inf[G,S\cup\{u\}]|
-
|Inf[G,S]|
}{
c(u)
}
}
$$

L'algoritmo seleziona quindi il nodo che produce il maggiore incremento reale della diffusione per unità di costo.

## Caratteristiche

CGG:

- considera direttamente il processo Majority;
- considera le attivazioni indirette;
- valuta l'effetto della cascade completa;
- non utilizza esclusivamente grado o vicinato immediato;
- ottimizza direttamente la quantità di interesse:

$$
|Inf[G,S]|.
$$

Se un nodo è già attivo nella cascade corrente, non viene considerato come nuovo seed perché trasformarlo in seed non aumenterebbe la chiusura finale.

CGG considera inoltre solamente candidati ancora compatibili con il budget residuo.

Se tutta la rete risulta già attiva:

$$
|Inf[G,S]|=|V|,
$$

l'algoritmo può terminare anticipatamente, perché nessun seed aggiuntivo potrebbe migliorare ulteriormente l'obiettivo.

---

# 12. Ottimizzazione di Cascade-Gain Greedy

Una versione diretta di CGG richiederebbe di calcolare da zero:

$$
Inf[G,S\cup\{u\}]
$$

per ogni candidato e a ogni iterazione.

Questo è computazionalmente costoso.

L'implementazione utilizzata sfrutta invece il fatto che la cascade corrente:

$$
A=Inf[G,S]
$$

è già conosciuta.

Per valutare un nuovo candidato $u$, la simulazione può partire da:

$$
A\cup\{u\}
$$

propagando solamente le **nuove attivazioni** generate da $u$.

L'ottimizzazione non modifica il criterio dell'algoritmo e produce lo stesso seed set della versione diretta, riducendo significativamente il tempo di esecuzione.

---

# 13. Budget

Gli esperimenti non utilizzano gli stessi valori assoluti di $k$ per entrambe le funzioni di costo.

Questo sarebbe poco significativo perché le distribuzioni dei costi Random e Degree sono molto diverse.

Il budget viene quindi definito come **percentuale del costo totale della rete**.

Se:

$$
C_{tot}=
\sum_{u\in V}c(u),
$$

per una percentuale $p$ viene utilizzato:

$$
\boxed{
k_p=
\left\lfloor
p\cdot C_{tot}
\right\rfloor
}
$$

Le percentuali considerate sono:

$$
\boxed{
1\%,\ 2\%,\ 5\%,\ 10\%,\ 20\%
}
$$

Questo permette di confrontare gli algoritmi utilizzando la stessa quantità relativa di risorse.

---

# 14. Configurazioni baseline

Per evitare di calcolare ripetutamente gli stessi seed set, tutte le configurazioni vengono costruite una sola volta sul grafo originale.

Sono presenti:

- 2 funzioni di costo;
- 5 budget;
- 3 algoritmi.

Il numero totale di configurazioni è quindi:

$$
2\times5\times3=30.
$$

Per ogni configurazione vengono memorizzati:

- funzione di costo;
- budget;
- algoritmo;
- seed set $S$;
- numero di seed;
- costo effettivo $c(S)$;
- $|Inf[G,S]|$;
- percentuale di rete influenzata;
- numero di nodi attivati dalla cascade oltre ai seed;
- numero di round;
- tempo di esecuzione.

Le stesse configurazioni vengono successivamente riutilizzate negli esperimenti di perturbazione.

In questo modo il seed set viene sempre determinato sul grafo originale e non viene ricalcolato dopo le modifiche della rete.

---

# 15. Esperimento 1 - Variazione del budget

Il primo esperimento studia la relazione tra budget e capacità di diffusione.

Per ogni:

- funzione di costo;
- algoritmo;
- livello di budget;

viene determinato un seed set $S$ e calcolato:

$$
|Inf[G,S]|.
$$

La metrica principale è quindi:

$$
\boxed{|Inf[G,S]|}
$$

mentre viene inoltre calcolata la percentuale di rete influenzata:

$$
\boxed{
InfluencePercentage=
\frac{|Inf[G,S]|}{|V|}
\cdot100
}
$$

I risultati vengono salvati in:

```text
results/budget_experiments.csv
```

e rappresentati graficamente separatamente per:

- Random Cost;
- Degree Cost.

---

# 16. Experiment 2 - Edge Removal

Il secondo esperimento valuta la robustezza del seed set quando cambiano le relazioni della rete.

Partendo da $G$, viene rimossa casualmente una percentuale di archi, producendo:

$$
G'.
$$

Il seed set rimane quello determinato sulla rete originale:

$$
S.
$$

Non viene quindi rieseguito l'algoritmo di selezione dopo la perturbazione.

Viene confrontato:

$$
|Inf[G,S]|
$$

con:

$$
|Inf[G',S]|.
$$

## Percentuali di archi rimosse

Vengono utilizzati:

$$
\boxed{
1\%,\ 5\%,\ 10\%,\ 20\%
}
$$

degli archi.

Per ogni configurazione e percentuale vengono effettuate:

$$
\boxed{20}
$$

ripetizioni indipendenti.

Questo riduce la dipendenza dei risultati da una singola rimozione casuale.

---

# 17. Effetto della rimozione degli archi sulla Majority

La rimozione di archi non implica necessariamente una riduzione della diffusione.

Infatti, rimuovendo archi, il grado può diminuire:

$$
d_G(v)>d_{G'}(v).
$$

Di conseguenza può diminuire anche la soglia:

$$
\left\lceil
\frac{d_G(v)}2
\right\rceil
>
\left\lceil
\frac{d_{G'}(v)}2
\right\rceil.
$$

La rimozione degli archi produce quindi due effetti opposti:

1. può interrompere percorsi di diffusione;
2. può abbassare le soglie majority e rendere alcuni nodi più facili da attivare.

Per questo motivo è possibile osservare anche:

$$
|Inf[G',S]|>|Inf[G,S]|.
$$

---

# 18. Retention

Per gli esperimenti di robustezza viene calcolata anche la percentuale di efficacia mantenuta:

$$
\boxed{
Retention=
\frac{|Inf[G',S]|}{|Inf[G,S]|}
\cdot100
}
$$

Interpretazione:

- `100%` → stessa diffusione della rete originale;
- `< 100%` → perdita di efficacia;
- `> 100%` → aumento della diffusione dopo la perturbazione.

I risultati completi vengono salvati in:

```text
results/edge_removal_experiments.csv
```

mentre i dati aggregati sono disponibili in:

```text
results/edge_removal_aggregated.csv
```

---

# 19. Esperimento 3 - Node Removal

Il terzo esperimento simula l'abbandono della rete da parte degli utenti.

Viene rimossa casualmente una percentuale dei nodi di $G$, insieme a tutti gli archi incidenti.

Si ottiene quindi:

$$
G'=G[V-R]
$$

dove $R$ è l'insieme dei nodi rimossi.

## Percentuali utilizzate

Vengono eliminate:

$$
\boxed{
1\%,\ 5\%,\ 10\%,\ 20\%
}
$$

dei nodi.

Anche in questo caso vengono effettuate:

$$
\boxed{20}
$$

ripetizioni per configurazione.

---

# 20. Seed rimossi dalla rete

A differenza dell'edge removal, durante il node removal può essere eliminato anche un nodo appartenente al seed set originale.

I seed non vengono protetti dalla perturbazione.

Dato il seed set originale:

$$
S
$$

e il nuovo insieme di nodi:

$$
V(G'),
$$

il seed set ancora disponibile è:

$$
\boxed{
S'=S\cap V(G')
}
$$

La cascade viene quindi calcolata come:

$$
Inf[G',S'].
$$

Per ogni esperimento vengono registrati anche:

- numero originale di seed;
- numero di seed sopravvissuti;
- numero di seed rimossi.

Questo permette di distinguere la perdita di efficacia dovuta:

1. alla modifica della topologia;
2. alla rimozione diretta di nodi appartenenti al seed set.

I risultati sono salvati in:

```text
results/node_removal_experiments.csv
```

e quelli aggregati in:

```text
results/node_removal_aggregated.csv
```

---

# 21. Numero di esperimenti di robustezza

Per ciascuna famiglia di perturbazioni abbiamo:

$$
2
\text{ costi}
\times
5
\text{ budget}
\times
3
\text{ algoritmi}
\times
4
\text{ livelli di rimozione}
\times
20
\text{ ripetizioni}.
$$

Quindi:

$$
\boxed{2400}
$$

osservazioni per l'edge removal e:

$$
\boxed{2400}
$$

osservazioni per il node removal.

---

# 22. Grafici

## Dataset

```text
degree_histogram.png
local_clustering_histogram.png
```

## Budget

Per ciascuna funzione di costo viene rappresentato:

$$
|Inf[G,S]|
$$

al variare del budget per i tre algoritmi.

```text
budget_influence_random.png
budget_influence_degree.png
```

## Edge Removal

Per evitare un numero eccessivo di figure vengono utilizzati grafici compositi.

Ogni figura contiene tre pannelli:

- CSG-f1;
- CSG-f2;
- CGG.

In ogni pannello vengono rappresentati i cinque budget:

$$
1\%,2\%,5\%,10\%,20\%.
$$

L'asse X contiene:

$$
0\%,1\%,5\%,10\%,20\%
$$

di archi rimossi.

Il punto `0%` corrisponde alla baseline:

$$
|Inf[G,S]|.
$$

Gli altri punti corrispondono alla media di:

$$
|Inf[G',S]|
$$

nelle 20 ripetizioni.

Le barre di errore rappresentano la deviazione standard.

```text
edge_removal_random_composite.png
edge_removal_degree_composite.png
```

## Node Removal

La stessa struttura viene utilizzata per la rimozione dei nodi:

```text
node_removal_random_composite.png
node_removal_degree_composite.png
```

---

# 23. Tabelle

Vengono generate tabelle riassuntive per:

- percentuale di influenza al variare del budget;
- retention dopo edge removal;
- retention dopo node removal.

I file sono salvati in:

```text
results/tables/
```

e comprendono:

```text
budget_random.csv
budget_degree.csv

edge_removal_random.csv
edge_removal_degree.csv

node_removal_random.csv
node_removal_degree.csv
```

Le tabelle di budget utilizzano:

$$
\frac{|Inf[G,S]|}{|V|}\cdot100
$$

mentre le tabelle di robustezza utilizzano:

$$
\frac{|Inf[G',S]|}{|Inf[G,S]|}\cdot100.
$$

---

# 24. Struttura del progetto

```text
MajorityDynamics/
│
├── data/
│   └── facebook_combined.txt
│
├── src/
│   ├── graph.py
│   ├── diffusion.py
│   ├── costs.py
│   ├── algorithms.py
│   ├── perturbations.py
│   ├── experiments.py
│   └── plots.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── csv/
│       ├── budget_experiments.csv
│       ├── edge_removal_experiments.csv
│       ├── edge_removal_aggregated.csv
│       ├── node_removal_experiments.csv
│       └── node_removal_aggregated.csv
│
├── main.py
├── generate_plots.py
├── generate_tables.py
├── requirements.txt
└── README.md
```

---

# 25. Moduli

## `graph.py`

Gestisce:

- caricamento del dataset;
- creazione del grafo NetworkX;
- statistiche descrittive.

## `diffusion.py`

Contiene:

- soglia majority;
- simulazione della Majority Cascade.

## `costs.py`

Contiene:

- Random Cost;
- Degree Cost;
- calcolo di $c(S)$.

## `algorithms.py`

Contiene:

- Cost-Seeds-Greedy con $f_1$;
- Cost-Seeds-Greedy con $f_2$;
- Cascade-Gain Greedy.

## `perturbations.py`

Contiene:

- rimozione casuale degli archi;
- rimozione casuale dei nodi.

## `experiments.py`

Gestisce:

- configurazioni baseline;
- esperimenti sui budget;
- edge removal;
- node removal.

## `plots.py`

Gestisce:

- istogrammi del dataset;
- grafici sui budget;
- aggregazione dei risultati;
- grafici compositi di robustezza.

---

# 26. Installazione

È consigliato utilizzare un ambiente virtuale Python.

Su Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Installare quindi le dipendenze:

```powershell
pip install -r requirements.txt
```

Le librerie principali utilizzate sono:

- `networkx`
- `numpy`
- `pandas`
- `matplotlib`

---

# 27. Esecuzione

L'esecuzione principale può essere avviata con:

```powershell
python main.py
```

Gli script di generazione dei grafici utilizzano direttamente i CSV già prodotti, evitando di rieseguire gli algoritmi:

```powershell
python generate_plots.py
```

Per generare le tabelle:

```powershell
python generate_tables.py
```

---

# 28. Riproducibilità

Gli esperimenti casuali utilizzano seed pseudo-casuali controllati.

Questo riguarda:

- generazione dei Random Cost;
- rimozione casuale degli archi;
- rimozione casuale dei nodi.

Il seed utilizzato per i costi deve rimanere invariato tra gli algoritmi, in modo che tutti vengano confrontati sulla stessa istanza.

Per le perturbazioni vengono utilizzati seed differenti tra le repliche, ma deterministici, così da rendere possibile la riproduzione delle stesse reti perturbate.

Inoltre, all'interno della stessa ripetizione, tutti gli algoritmi vengono valutati sulla stessa perturbazione.

---

# 29. Principio di confronto

Tutti i confronti sperimentali vengono effettuati mantenendo invariati:

- dataset;
- funzione di costo;
- budget;
- configurazione casuale;
- rete perturbata.

L'unico elemento che varia nel confronto è quindi l'algoritmo utilizzato per costruire il seed set.

Questo permette di confrontare in maniera equa:

$$
CSG-f_1,
\qquad
CSG-f_2,
\qquad
CGG.
$$

---

# 30. Obiettivo finale dell'analisi

L'obiettivo non è solamente individuare quale algoritmo produca il valore massimo di:

$$
|Inf[G,S]|.
$$

Gli esperimenti permettono anche di studiare:

- effetto della funzione di costo;
- effetto del budget;
- capacità del seed set di generare vere cascade;
- profondità della propagazione;
- robustezza rispetto alla perdita di relazioni;
- robustezza rispetto all'abbandono di utenti;
- compromesso tra efficacia e stabilità;
- costo computazionale degli algoritmi.

In questo modo il progetto analizza sia la **qualità iniziale del seed set** sia la sua **capacità di mantenere efficacia in una online network dinamica**.
