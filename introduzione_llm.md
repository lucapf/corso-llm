# LLM corso introduttivo

Il seguente documento, inspirato dalle lezioni di Andrej Karpathy "Neural Networks: Zero to Hero" descrive un percorso di apprendimento che descrive tutti i dettagli che permettono di generare una rete neurale.

---

## Indice

- [Guida al corso](#guida-al-corso)
- **Parte I --- Le fondamenta**
  - [Lezione 1 --- micrograd: derivata e backpropagation](#lezione-1-----micrograd-derivata-e-backpropagation)
  - [Lezione 2 --- Il modello bigramma](#lezione-2-----il-modello-bigramma)
  - [Lezione 3 --- MLP ed embedding](#lezione-3-----mlp-ed-embedding)
- **Parte II --- Far funzionare l'addestramento**
  - [Lezione 4 --- Attivazioni, gradienti, BatchNorm](#lezione-4-----attivazioni-gradienti-batchnorm)
  - [Lezione 5 --- Backprop a mano, senza autograd](#lezione-5-----backprop-a-mano-senza-autograd)
  - [Lezione 6 --- WaveNet e la gerarchia](#lezione-6-----wavenet-e-la-gerarchia)
- **Parte III --- Il Transformer**
  - [Lezione 7 --- Costruiamo GPT da zero](#lezione-7-----costruiamo-gpt-da-zero)
  - [Lezione 8 --- Il tokenizer: BPE da zero](#lezione-8-----il-tokenizer-bpe-da-zero)
- **Parte IV --- Dalla teoria al modello reale**
  - [Lezione 9 --- Riprodurre GPT-2 (124M)](#lezione-9-----riprodurre-gpt-2-124m)
  - [Lezione 10 --- Da modello base ad assistente](#lezione-10-----da-modello-base-ad-assistente)
- **Appendici**
  - [A --- Da nanoGPT a llama.cpp](#appendice-a-----da-nanogpt-a-llamacpp)
  - [B --- Quantizzazione e GGUF](#appendice-b-----quantizzazione-e-gguf)
  - [C --- Fine-tuning con LoRA e QLoRA](#appendice-c-----fine-tuning-con-lora-e-qlora)
  - [D --- Cosa è cambiato dopo GPT-2 (2019--2026)](#appendice-d-----cosa-è-cambiato-dopo-gpt-2-2019--2026)
  - [Glossario minimo](#glossario-minimo)

---

## Guida al corso

### L'obiettivo

Capire davvero come funziona un LLM costruendone uno (estremamente semplice) da zero.

Al termine di questo percorso otterremo un Large Language Model semplice ma nella sua architettura simile ad un moderno modello AI. Stile GPT.
Faremo questo partendo da semplici classi scritte in Python e poi evolvere quanto fatto, comprendere come queste primitive siano già parte di librerie specializzate. Adotteremo, quindi  PyTorch per completare  problemi più complessi fino ad arrivare al prodotto completo.

Il percorso formativo segue i seguenti, fondamentali passaggi tutti equamente importanti.

```text
derivata -> backprop -> neurone -> MLP -> embedding -> attenzione
        -> Transformer -> tokenizer -> scala -> post-training
```

### Come le lezioni sono strutturate

Ogni lezione segue, a grandi linee, lo stesso percorso:

1. **Il problema**: cosa non sappiamo ancora fare.
2. **L'idea**: il concetto centrale, prima del codice.
3. **Il codice**: scritto da zero, commentato.
4. **La verifica**: un controllo numerico che dimostra che funziona.
5. **Esercizi**: tre (quattro nella Lezione 1), dal più semplice al più impegnativo.
6. **Verso gli LLM**: dove ritrovi questo pezzo dentro GPT.
7. **Riepilogo**: i punti da ricordare.

Nelle lezioni più sperimentali (4, 6, 9 e 10) la parte centrale è divisa per argomento invece che fra idea e codice.

Tre segni ricorrono nel testo:

- **[Attenzione]** un bug tipico, che spesso non dà alcun errore;
- **[Verifica]** un controllo da fare sempre;
- **[Nuovo]** una parte aggiunta in questa revisione (aggiornamenti e integrazioni).

Le durate indicate all'inizio di ogni lezione sono **tempi di lettura**. Scrivere il codice e fare gli esercizi richiede da due a quattro volte tanto.

### Il programma

| #   | Lezione                 | Concetto chiave                                                                            |
| --- | ----------------------- | ------------------------------------------------------------------------------------------ |
| 1   | micrograd               | Il gradiente dice di quanto cambia la loss se muovi di poco un parametro.                  |
| 2   | Bigramma                | Un modello di linguaggio stima $P(x_t \mid x_{<t})$.                                       |
| 3   | MLP ed embedding i      | Gli embedding sono lo spazio in cui il modello rappresenta i simboli.                      |
| 4   | Attivazioni e BatchNorm | Addestrare significa tenere sane le statistiche dei segnali, strato per strato.            |
| 5   | Backprop a mano         | Chi sa scrivere il backward sa trovare i bug di un training che non converge.              |
| 6   | WaveNet                 | L'architettura decide come l'informazione del contesto viene fusa.                         |
| 7   | GPT da zero             | L'attenzione è una comunicazione fra token con pesi appresi.                               |
| 8   | Tokenizer BPE           | Molti "difetti" degli LLM sono in realtà difetti del tokenizer.                            |
| 9   | GPT-2 (124M)            | Fra un modello didattico e uno vero la differenza è quasi tutta ingegneria.                |
| 10  | SFT, RLHF, DPO          | Un assistente è un modello base con un secondo addestramento, su dati e obiettivi diversi. |

### Prerequisiti

- Python di base.
- Un po' di algebra lineare: il prodotto fra matrice e vettore.
- L'idea di derivata (la ripassiamo nella Lezione 1).

PyTorch **non** è un prerequisito: costruiamo noi le sue parti essenziali e poi le riconosciamo.

### Come usarlo

Il corso va fatto **scrivendo il codice**, non solo leggendolo.
E' una buona pratica cercare di ripercorrere almeno mentalmente il capitolo che si è terminato prima di passare al successivo.

**Nota sui numeri.** Molti risultati (loss, percentuali) dipendono dal seme casuale e dalla versione delle librerie. Se ottieni valori un po' diversi, controlla l'ordine di grandezza e la tendenza, non la terza cifra decimale.

**Software.** La Lezione 1 usa solo Python. Dalla Lezione 2 serve PyTorch (`pip install torch`); per i grafici degli esercizi, `matplotlib`. La Lezione 9 richiede una GPU per essere eseguita davvero.

**Dati.** Le Lezioni 2--6 usano un file `nomi.txt` con 399 nomi propri italiani (uno per riga). Le Lezioni 7, 8 e 10 usano un racconto, `racconto.txt` (circa 22 500 caratteri). Se non li hai, va bene qualunque lista di nomi e qualunque testo in prosa di dimensione simile: i numeri cambieranno, i fenomeni no.

---

# PARTE I --- LE FONDAMENTA

## Lezione 1 --- micrograd: derivata e backpropagation

*Lettura: circa 45 minuti.* Alla fine avrai scritto da zero un motore di differenziazione automatica (circa 100 righe) e ci avrai addestrato sopra una piccola rete neurale. Senza PyTorch.

### 1. Il problema

Una rete neurale è una funzione con milioni di "manopole", i **parametri** $\theta = (\theta_1, \dots, \theta_n)$. Le regoliamo per ridurre un solo numero, la **loss** $L(\theta)$: quanto stiamo sbagliando.

La domanda da porsi è sempre la stessa:

> Se muovo di pochissimo questa manopola, la loss sale o scende? E di quanto?

La risposta è la **derivata parziale** $\partial L / \partial \theta_i$. Il **gradiente** $\nabla L$ è la lista di tutte queste risposte, una per parametro.

Il problema è la scala. Con $10^8$ parametri non possiamo muovere una manopola alla volta e rimisurare: servirebbero $n+1$ valutazioni della rete per **ogni** passo. Ci serve un metodo che calcoli tutte le derivate in una sola passata. Quel metodo è la **backpropagation**.

### 2. L'idea

#### 2.1 La derivata è una sensibilità

Definizione operativa: $f'(x) \approx \dfrac{f(x+h) - f(x)}{h}$ con $h$ piccolo.

```python
def f(x):
    return 3*x**2 - 4*x + 5

x, h = 3.0, 0.0001
print((f(x + h) - f(x)) / h)   # circa 14.0  (analiticamente f'(x) = 6x - 4 = 14)
```

Il 14 significa: se aumento $x$ di poco, $f$ aumenta di circa 14 volte quel poco. È una **sensibilità**, niente di più.

Con più variabili, per $d = ab + c$ si ha $\partial d/\partial a = b$, $\partial d/\partial b = a$, $\partial d/\partial c = 1$. Il senso: $b$ entra nel risultato moltiplicata per $a$, quindi la sua influenza è amplificata di un fattore $a$.

#### 2.2 La regola della catena

Se $c$ influenza $b$ e $b$ influenza $L$, le sensibilità **si moltiplicano**:

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial b}\cdot\frac{\partial b}{\partial c}$$

(Analogia di Karpathy: se l'auto va 2 volte più veloce della bici e la bici 4 volte più veloce di chi cammina, l'auto va 8 volte più veloce di chi cammina.)

Se un nodo influenza la loss attraverso **più strade**, i contributi **si sommano**:

$$\frac{\partial L}{\partial x} = \sum_j \frac{\partial L}{\partial u_j}\cdot\frac{\partial u_j}{\partial x}$$

Questa somma è il motivo per cui nel codice scriveremo `+=` e non `=`.

#### 2.3 Il grafo computazionale

Ogni espressione è un grafo. Prendiamo $L = (a\cdot b + c)\cdot f$ con $a=2$, $b=-3$, $c=10$, $f=-2$.

- **Forward** (da sinistra a destra): calcolo i valori. $e = ab = -6$, $d = e + c = 4$, $L = d\cdot f = -8$.
- **Backward** (da destra a sinistra): parto da $\partial L/\partial L = 1$ e, a ogni nodo, moltiplico il gradiente che arriva per la derivata locale dell'operazione.

Indichiamo con $\bar x$ il gradiente $\partial L/\partial x$ accumulato sul nodo $x$ (nel codice sarà `x.grad`). Per un nodo $z$ calcolato da $x$ e $y$ bastano tre regole locali:

| Operazione      | Regola nel backward                                                    | In parole                            |
| --------------- | ---------------------------------------------------------------------- | ------------------------------------ |
| $z = x + y$     | $\bar x \mathrel{+}= \bar z$, $\quad \bar y \mathrel{+}= \bar z$       | il **+ distribuisce**                |
| $z = x \cdot y$ | $\bar x \mathrel{+}= y\,\bar z$, $\quad \bar y \mathrel{+}= x\,\bar z$ | il **$\boldsymbol{\times}$ scambia** |
| $z = \tanh(x)$  | $\bar x \mathrel{+}= (1 - z^2)\,\bar z$                                | la **tanh scala**                    |

Applicandole al nostro esempio, partendo da $\bar L = 1$:

| Nodo           | Regola           | Risultato                                                   |
| -------------- | ---------------- | ----------------------------------------------------------- |
| $L = d\cdot f$ | $\times$ scambia | $\bar d = f = -2$, $\bar f = d = 4$                         |
| $d = e + c$    | + distribuisce   | $\bar e = -2$, $\bar c = -2$                                |
| $e = a\cdot b$ | $\times$ scambia | $\bar a = \bar e\cdot b = 6$, $\bar b = \bar e\cdot a = -4$ |

Quindi $\nabla L = (6, -4, -2, 4)$ rispetto a $(a, b, c, f)$.

Come si leggono questi numeri:

- **Il segno dà la direzione.** $\bar b = -4 < 0$: per far scendere $L$ devo far crescere $b$.
- **Il modulo dà l'importanza.** $|\bar b| = 4$ contro $|\bar c| = 2$: muovere $b$ conta il doppio.
- **Conta il grafo intero.** $c$ entra con coefficiente 1, eppure il suo gradiente è $-2$: dipende da cosa succede dopo (viene moltiplicato per $f$).

**Un passo di discesa.** Spostiamo ogni variabile di $-\eta$ volte il suo gradiente, con $\eta = 0{,}01$: $a \to 1{,}94$, $b \to -2{,}96$, $c \to 10{,}02$, $f \to -2{,}04$. La loss passa da $-8$ a circa $-8{,}73$. Al primo ordine la teoria prevede una variazione di $-\eta\,\lVert\nabla L\rVert^2 = -0{,}01 \cdot (36 + 16 + 4 + 16) = -0{,}72$; la piccola differenza ($-0{,}73$ contro $-0{,}72$) è dovuta alla curvatura. Questo è, letteralmente, ciò che succede a ogni passo dell'addestramento di un LLM, con $10^8$ o più numeri invece di quattro.

#### 2.4 Perché all'indietro?

La regola della catena si può percorrere in due direzioni:

- **In avanti** (*forward mode*): una passata dà la sensibilità rispetto a **un** input. Per $n$ parametri servono $n$ passate.
- **All'indietro** (*reverse mode*): una passata dà la sensibilità di **un** output rispetto a **tutti** gli input.

Il deep learning ha milioni di input (i parametri) e un solo output (la loss): è il caso ideale per il reverse mode. Il backward costa più o meno quanto un paio di forward. È anche il primo motivo per cui la loss deve essere **un unico numero**.

#### 2.5 Tre proprietà del backward di ogni nodo

1. **È locale.** Ogni nodo conosce solo la propria operazione e i propri input; tutta l'informazione globale arriva nel gradiente che riceve.
2. **Accumula, non assegna.** Se un valore viene usato più volte, i contributi si sommano (`+=`).
3. **Ricorda il forward.** Per derivare deve conservare dei valori del forward (per esempio, nel prodotto $a \cdot b$, il valore di $b$ per derivare rispetto ad $a$). Per questo l'addestramento consuma molta più memoria dell'inferenza: le attivazioni intermedie vanno tenute fino al backward.

> **Terminologia.** In PyTorch la funzione di backward di un nodo è `grad_fn` (`AddBackward0`, `MulBackward0`, ...): è il corrispettivo del nostro `Value.op`. Il nome formale dell'operazione è *vector--Jacobian product* (VJP). Attenzione al falso amico: *gradient accumulation* nel gergo pratico indica un'altra cosa, cioè sommare i gradienti di più micro-batch prima di aggiornare i pesi (Lezione 9).

### 3. Il codice: le classi `Value` e `Op`

Due concetti, due classi. Un `Value` è un numero **che si ricorda da dove viene**: contiene il risultato, il suo gradiente e, se non è una foglia, l'operazione che l'ha prodotto. Un `Op` è quell'operazione: calcola il forward e sa come distribuire il gradiente ai suoi input.

```python
import math

class Op:
    """Un nodo-operazione. Ogni sottoclasse, nel suo __init__, calcola il
    forward (una volta sola) e imposta due attributi:
      self.inputs: tupla di Value, gli input (servono al backward);
      self.out:    il Value risultato, creato con op=self."""

    def _to_value(self, x):
        """Se x è un numero lo avvolge in un Value, altrimenti lo lascia."""
        return x if isinstance(x, Value) else Value(x)

    def backward(self, g):
        """Distribuisce il gradiente dell'output ai gradienti degli input
        (regola della catena).
        Input:  g, un float: dL/d(out), il gradiente che arriva dal nodo
                a valle.
        Output: una tupla di float, uno per ogni input e nello stesso
                ordine di self.inputs: dL/d(input_i) = derivata locale * g.
                Non modifica nessun grad: li accumula Value.backward.
        Da implementare nelle sottoclassi."""
        raise NotImplementedError

class Add(Op):
    def __init__(self, x, y):
        x, y = self._to_value(x), self._to_value(y)
        self.inputs = (x, y)
        self.out = Value(x.data + y.data, op=self)
    def backward(self, g):
        return g, g                                 # il + distribuisce

class Mul(Op):
    def __init__(self, x, y):
        x, y = self._to_value(x), self._to_value(y)
        self.inputs = (x, y)
        self.out = Value(x.data * y.data, op=self)
    def backward(self, g):
        x, y = self.inputs
        return y.data * g, x.data * g               # il * scambia

class Pow(Op):
    def __init__(self, x, k):
        x = self._to_value(x)
        self.k = k                    # k è un numero, non un input
        self.inputs = (x,)
        self.out = Value(x.data ** k, op=self)
    def backward(self, g):
        (x,) = self.inputs
        return (self.k * x.data ** (self.k - 1) * g,)

class Tanh(Op):
    def __init__(self, x):
        x = self._to_value(x)
        self.inputs = (x,)
        self.out = Value(math.tanh(x.data), op=self)
    def backward(self, g):
        return ((1 - self.out.data**2) * g,)        # la tanh scala

class Exp(Op):
    def __init__(self, x):
        x = self._to_value(x)
        self.inputs = (x,)
        self.out = Value(math.exp(x.data), op=self)
    def backward(self, g):
        return (self.out.data * g,)                 # d/dx e^x = e^x

class Value:
    """Un numero, il suo gradiente, e (se non è una foglia) l'Op che l'ha prodotto."""
    def __init__(self, data, op=None):
        self.data = data
        self.grad = 0.0                 # dL/dself, all'inizio zero
        self.op = op                    # None per input e parametri

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other): return Add(self, other).out
    def __mul__(self, other): return Mul(self, other).out
    def __pow__(self, k):     return Pow(self, k).out
    def tanh(self):           return Tanh(self).out
    def exp(self):            return Exp(self).out

    # Operazioni derivate: composizioni di quelle sopra, il backward è gratis
    def __neg__(self):            return self * -1
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return (-self) + other
    def __radd__(self, other):    return self + other
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        self.grad = 1.0                 # dL/dL = 1, il seme
        for node in reversed(_topo_order(self, set(), [])):
            if node.op:                 # le foglie non hanno nulla da propagare
                grads = node.op.backward(node.grad)
                for inp, g in zip(node.op.inputs, grads):
                    inp.grad += g       # accumula: l'unico += di tutto il codice

def _topo_order(v, visited, topo):
    """Ordine topologico: ogni nodo dopo tutti i nodi da cui dipende."""
    if v not in visited:
        visited.add(v) # Perché e' un grafo aciclico non un albero.
                       # posso raggiungere lo stesso nodo  usando tante strade
        if v.op:
            for inp in v.op.inputs:
                _topo_order(inp, visited, topo) # parto dalla fine!
        topo.append(v)
    return topo
```

> **Gli operatori sono metodi.** I metodi come `__add__` e `__radd__` sono associati direttamente all'operazione che descrivono: quando Python incontra un operatore, chiama il metodo corrispondente. Se scriviamo `Value(3) + Value(5)`, Python lo esegue come `Value(3).__add__(Value(5))`. Il nostro `__add__` non restituisce un numero: costruisce un nodo `Add` e ne restituisce l'`.out`. Così basta scrivere la formula perché il grafo si costruisca da solo.
>
> | Scriviamo             | Python esegue                     |
> |-----------------------|-----------------------------------|
> | `Value(3) + Value(5)` | `Value(3).__add__(Value(5))`      |
> | `Value(3) * Value(5)` | `Value(3).__mul__(Value(5))`      |
> | `Value(3) ** 2`       | `Value(3).__pow__(2)`             |
> | `-Value(3)`           | `Value(3).__neg__()`              |
> | `Value(3) - Value(5)` | `Value(3).__sub__(Value(5))`      |
> | `Value(3) / Value(5)` | `Value(3).__truediv__(Value(5))`  |
> | `2 + Value(5)`        | `Value(5).__radd__(2)`            |
> | `2 * Value(5)`        | `Value(5).__rmul__(2)`            |
> | `2 - Value(5)`        | `Value(5).__rsub__(2)`            |
>
> I metodi con la `r` (*reflected*, riflessi) servono quando il `Value` sta a **destra** dell'operatore. In `2 + Value(5)` Python prova prima `(2).__add__(Value(5))`, ma un `int` non sa sommare un `Value` e rinuncia. Allora prova il metodo riflesso del secondo operando, `Value(5).__radd__(2)`. Senza `__radd__` e `__rmul__`, espressioni come `2 * x` darebbero `TypeError`.
>
> I metodi con il doppio trattino basso (*dunder*) sono **metodi speciali** con nomi riservati dal linguaggio; ridefinirli si chiama **overloading degli operatori**. `tanh` ed `exp` non hanno un operatore: si chiamano esplicitamente, `x.tanh()`.

Cinque punti da capire bene:

- **Il forward si calcola una volta sola**, nell'`__init__` di ogni operazione, quando viene creata. L'`Op` non lo rifà mai: conserva gli input (e, se serve, l'output) solo per poter derivare nel backward (la proprietà 3 del §2.5).
- **Il `backward` di ogni `Op` è locale e puro**: riceve $\bar z$ e restituisce i gradienti dei suoi input, senza toccare nient'altro. È la tabella del §2.3 scritta in Python.
- **`+=` e non `=`.** L'accumulo sta in un solo punto, in `Value.backward`. Con `b = a + a`, `a` riceve gradiente da due strade. Con `=` la seconda cancellerebbe la prima: nessun errore, solo un training che non converge.
- **`reversed(...)` dell'ordine topologico** garantisce che quando chiamiamo `node.op.backward(node.grad)` il gradiente di quel nodo sia già completo.
- **La tanh è un nodo unico** solo perché conosciamo la sua derivata in forma chiusa. Potremmo spezzarla in operazioni elementari e funzionerebbe lo stesso. Quanto "grandi" fare i nodi è una scelta di ingegneria (è l'idea dei *kernel fusi*), non di matematica.

### 4. La verifica: un neurone, contro PyTorch

Un neurone calcola $o = \tanh\big(\sum_i w_i x_i + b\big)$.

```python
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(7.0)
o = (x1*w1 + x2*w2 + b).tanh()
o.backward()
print(x1.grad, w1.grad)   # circa -1.2599 e 0.8399
print(x2.grad, w2.grad)   # circa  0.4200 e 0.0
```

Lo stesso calcolo in PyTorch:

```python
import torch
x1 = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
w1 = torch.tensor([-3.0], dtype=torch.double, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
b  = torch.tensor([7.0], dtype=torch.double, requires_grad=True)
o = torch.tanh(x1*w1 + x2*w2 + b)
o.backward()
print(x1.grad.item(), w1.grad.item())   # circa -1.2599 e 0.8399
print(x2.grad.item(), w2.grad.item())   # circa  0.4200 e 0.0
```

Gli stessi numeri. Nota: $\partial o/\partial w_2 = 0$ perché $x_2 = 0$. **Un input nullo rende invisibile il suo peso**, che quindi non verrà aggiornato: è uno dei motivi per cui i dati si normalizzano.

La differenza vera fra `Value` e `torch.Tensor` è di ingegneria, non di concetto: PyTorch applica le stesse regole a tensori interi, in parallelo, su GPU.

### 5. La rete neurale

```python
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def forward(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def forward(self, x):
        outs = [n.forward(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Il cuore è `Neuron.forward`: prodotto scalare, bias, non-linearità. Il resto è organizzazione.

#### 5.1 La loss: la bussola

La loss comprime in **un solo numero** la distanza fra predizioni e valori desiderati. Deve avere tre proprietà:

1. **Essere uno scalare.** Serve al reverse mode, ma soprattutto stabilisce come "barattare" un errore contro un altro: è il contratto che definisce cosa vuol dire migliorare.
2. **Essere derivabile.** "Quante risposte sono giuste" è una funzione a gradini: derivata zero quasi ovunque, nessun segnale. Per questo si ottimizza sempre una versione morbida dell'obiettivo.
3. **Avere il minimo nel posto giusto**, cioè quando le predizioni coincidono con i target.

Per una rete che produce numeri, la scelta classica è l'**errore quadratico**: $L = \sum_i (\hat y_i - y_i)^2$. Il quadrato rende ogni termine positivo, è simmetrico, e punisce di più gli errori grandi. La sua derivata, $2(\hat y_i - y_i)$, è **l'errore stesso** (raddoppiato): predizione troppo alta $\to$ gradiente positivo $\to$ la discesa la abbassa.

> La rete diventa brava esattamente in ciò che la loss misura, non in ciò che avevi in mente tu.

#### 5.2 Il ciclo di addestramento

```python
random.seed(42)
n = MLP(3, [4, 4, 1])           # 3 input -> 4 -> 4 -> 1 output (41 parametri)
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]     # target desiderati

for k in range(100):
    # forward
    ypred = [n.forward(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    # backward
    n.zero_grad()               # MAI dimenticarlo
    loss.backward()
    # update: discesa del gradiente
    for p in n.parameters():
        p.data += -0.05 * p.grad
```

La loss scende da circa 5,3 al primo passo a circa 0,01 dopo 100 passi; le predizioni passano da valori quasi uguali (la rete dice "sì" a tutto) a circa $(0{,}97, -0{,}94, -0{,}96, 0{,}94)$ contro i target $(1, -1, -1, 1)$.

Le tre fasi **forward $\boldsymbol{\to}$ backward $\boldsymbol{\to}$ update** sono le stesse dell'addestramento di GPT-4. Cambia solo la scala.

Tre dettagli che valgono la lezione:

- **Il segno meno.** Il gradiente punta dove la loss cresce; noi scendiamo, quindi $-\nabla L$. Togli il meno e guarda la loss esplodere.
- **`zero_grad()`.** I gradienti si accumulano (`+=`). Se non li azzeri, a ogni passo il gradiente usato è la somma di tutti quelli dei passi precedenti: gli aggiornamenti crescono senza controllo e l'addestramento diverge. È il bug numero uno dei principianti in PyTorch.
- **Il learning rate.** Troppo basso: impari lentissimo. Troppo alto: superi il minimo e oscilli o esplodi. Si trova sperimentando (Lezione 3).

### 6. Esercizi

1. **Aggiungi la ReLU.** Implementa una `class ReLU(Op)` e il metodo `Value.relu()`, con derivata 1 per $x>0$ e 0 altrimenti. Sostituiscila alla tanh e confronta la convergenza. Cosa succede a un neurone che riceve sempre input negativi? (cerca *dying ReLU*)
2. **Il bug del riuso.** Calcola il gradiente di `b = a + a` con `a = Value(3.0)`: deve venire 2. Ora sostituisci `inp.grad += g` con `inp.grad = g` in `Value.backward` e rifai. Cosa ottieni e perché?
3. **Verifica numerica (gradient checking).** Scrivi una funzione che confronti `.grad` con la stima a differenze centrali $\frac{L(x+h) - L(x-h)}{2h}$. È lo strumento con cui si controllano davvero le implementazioni del backward.
4. *(Ponte verso la Lezione 2)* **Softmax e cross-entropy.** Implementa $p_i = e^{z_i}/\sum_j e^{z_j}$ su una lista di `Value` e la loss $L = -\log p_y$. Verifica contro PyTorch il risultato notevole $\partial L/\partial z_i = p_i - \mathbf{1}[i=y]$.

### 7. Verso gli LLM

Un LLM è, letteralmente, un grafo computazionale come quello di micrograd: miliardi di volte più grande, con le stesse operazioni e un `backward()` che parte da $\partial L/\partial L = 1$.

- **Già definitivo:** il meccanismo di apprendimento, il ciclo forward $\to$ backward $\to$ update, l'accumulo dei gradienti.
- **Manca ancora:** l'efficienza (i tensori, dalla Lezione 2; le GPU, nella Lezione 9), la rappresentazione del testo (Lezioni 2 e 8), l'architettura (l'attenzione, Lezione 7), la scala (Lezione 9).

### Riepilogo

1. La derivata è una sensibilità: di quanto cambia l'output se muovo questo numero.
2. La regola della catena moltiplica le sensibilità lungo il grafo e le somma quando le strade sono più di una.
3. Il backward parte da 1 e percorre il grafo in ordine topologico inverso: una passata dà tutti i gradienti.
4. Il + distribuisce, il $\times$ scambia, la tanh scala per $1 - t^2$.
5. I gradienti si accumulano: azzerali a ogni passo.
6. La loss è un unico scalare derivabile; per l'errore quadratico il gradiente in ingresso è l'errore stesso.

---

## Lezione 2 --- Il modello bigramma

*Lettura: circa 35 minuti.* Costruiamo il primo vero modello di linguaggio contando le coppie di lettere. Poi lo riscriviamo come rete neurale, e scopriamo che le due strade arrivano allo stesso identico risultato.

### 1. Il problema

Nella Lezione 1 il compito era finto (quattro esempi inventati). Ora il compito è vero:

> dato un testo, imparare a generarne altro che gli somigli.

Useremo `nomi.txt`, 399 nomi propri italiani. L'obiettivo è generare nomi nuovi che *suonino* italiani. Da qui il nome del progetto di Karpathy: **makemore**, "fanne ancora". È un giocattolo, ma il compito è lo stesso di GPT: cambiano solo la lunghezza dei testi e la dimensione dell'alfabeto.

#### 1.1 Il compito, in formule

Un modello di linguaggio assegna una probabilità a una sequenza. Per una parola $x_1 x_2 \dots x_T$:

$$P(x_1 \dots x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

Il modello deve stimare il fattore $P(x_t \mid x_{<t})$: dato tutto ciò che è venuto prima, quanto è probabile ogni possibile carattere successivo? **È esattamente ciò che fa GPT-4.** Cambia solo quanto contesto $x_{<t}$ riesce a usare.

### 2. L'idea: guardare solo il carattere precedente

Partiamo dall'ipotesi più brutale: $P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-1})$. Solo l'ultimo carattere conta. Sappiamo già che è falso, ma partiamo da qui e misuriamo quanto è falso.

Aggiungiamo un **token speciale `.`** che segna inizio e fine parola: `anna` diventa `.anna.`. Così il modello impara anche **quando fermarsi**.

```python
words = open('nomi.txt', encoding='utf-8').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i+1 for i, s in enumerate(chars)}   # 'a'->1, 'b'->2, ...
stoi['.'] = 0                                   # il token speciale prende lo 0
itos = {i: s for s, i in stoi.items()}
V = len(itos)                                   # 23: 22 lettere + '.'
```

Le lettere sono 22, non 26: nei nomi italiani non compaiono j, w, x, y. Il vocabolario **emerge dai dati**, non è deciso a priori (ci torneremo con il tokenizer, Lezione 8).

### 3. Il modello a conteggi

#### 3.1 Contare

```python
import torch
N = torch.zeros((V, V), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):          # coppie consecutive
        N[stoi[ch1], stoi[ch2]] += 1
# N.sum() = 3073 bigrammi in totale
```

La matrice $N$ ($23 \times 23$) **è** il modello: la riga $i$ dice cosa segue il carattere $i$. Già i conteggi contengono conoscenza della lingua: i bigrammi più frequenti sono `a.` (196) e `o.` (160), cioè i nomi che finiscono in -a e in -o.

#### 3.2 Da conteggi a probabilità

```python
P = N.float()
P /= P.sum(1, keepdim=True)     # ogni riga ora somma a 1
```

> **[Attenzione] Il bug del broadcasting.** Senza `keepdim=True` la somma ha forma `(23,)`, che il broadcasting allinea a destra e tratta come `(1, 23)`: ogni **colonna** $j$ viene divisa per la somma della **riga** $j$. Il risultato non è normalizzato né per righe né per colonne, e PyTorch non dà alcun errore: il modello sembra funzionare e dà risultati sbagliati. Regola pratica: dopo ogni normalizzazione controlla che `P.sum(1)` dia tutti 1.

#### 3.3 Generare

Parti da `.`, estrai un carattere dalla riga corrente, ripeti finché non esce di nuovo `.`.

```python
g = torch.Generator().manual_seed(7)
for _ in range(10):
    ix, out = 0, []
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
```

Risultato tipico: `mona, s, ro, eldilizplvbdo, alito, ia, cla, ara, sirio, lisena`. Non sono buoni, ma nemmeno casuali (confronta con lettere estratte a caso: `uchsvkcinqhfnt...`). Il modello ha imparato lunghezze plausibili, alternanza di consonanti e vocali, desinenze. Con un solo carattere di memoria, di più non può fare.

### 4. Valutare: la loss

"Non sono buoni" non è una misura. Ci serve un numero.

La qualità di un modello è la probabilità che assegna ai dati veri (la **verosimiglianza**): il prodotto delle probabilità di tutti i bigrammi osservati. Ma il prodotto di 3073 numeri minori di 1 è circa $10^{-2800}$, che in un computer vale zero. Passiamo ai logaritmi (i prodotti diventano somme), cambiamo segno (per convenzione si minimizza) e facciamo la media:

$$L = -\frac{1}{n}\sum_{i=1}^{n} \log P(b_i)$$

dove $b_i$ è l'$i$-esimo bigramma osservato e $n = 3073$. Questa è la **negative log likelihood media**, chiamata anche **cross-entropy**: la loss di ogni modello di linguaggio, GPT compreso.

```python
log_likelihood, n = 0.0, 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
        n += 1
print(-log_likelihood / n)      # circa 2.089
```

#### 4.1 Come leggere quel 2,089

| Riferimento                              | Loss                | Significato                                    |
| ---------------------------------------- | ------------------- | ---------------------------------------------- |
| Modello uniforme (1/23 a ogni carattere) | $\log 23 = 3{,}135$ | il modello che non sa niente                   |
| Bigramma                                 | 2,089               |                                                |
| Modello perfetto                         | 0                   | irraggiungibile: il testo non è deterministico |

Due modi intuitivi di leggerlo:

- $e^{-2{,}089} \approx 0{,}124$: in media (geometrica) il modello dà circa **1 probabilità su 8** al carattere giusto.
- $e^{2{,}089} \approx 8$ si chiama **perplessità**: il modello è indeciso come se scegliesse fra 8 opzioni equiprobabili invece che fra 23.

#### 4.2 Il problema degli zeri e lo smoothing

Un bigramma mai visto (per esempio `qz`) ha probabilità 0, quindi $-\log 0 = +\infty$: il modello dichiara impossibile ciò che semplicemente non ha visto. La soluzione è lo **smoothing**: aggiungere un conteggio finto $\alpha$ a tutte le coppie.

```python
P = (N + 1).float()             # alpha = 1
P /= P.sum(1, keepdim=True)
```

La loss sui dati di addestramento peggiora un po' (da 2,089 a 2,159), in cambio di robustezza su dati nuovi. Più $\alpha$ è grande, più la distribuzione si appiattisce verso l'uniforme. Tienilo a mente: tra poco lo ritroveremo sotto un altro nome.

### 5. La stessa cosa, come rete neurale

Ora buttiamo via i conteggi e usiamo lo strumento della Lezione 1: una rete addestrata con il gradiente. Non otterremo un modello migliore, ma **lo stesso modello**. Il punto è che questa strada, a differenza dei conteggi, si generalizza a contesti lunghi (Lezione 3).

#### 5.1 Dati, input, architettura

```python
import torch.nn.functional as F

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1]); ys.append(stoi[ch2])
xs, ys = torch.tensor(xs), torch.tensor(ys)
n = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g) * 0.01   # pesi piccoli: all'inizio i logit sono quasi uguali
W.requires_grad = True
```

- **One-hot.** Non diamo alla rete direttamente l'indice (per esempio 13): farebbe aritmetica su quel numero, come se la lettera 13 fosse "più grande" della 2. Ogni carattere diventa invece un vettore di 23 numeri, tutti zero tranne un 1 nella sua posizione: `xenc = F.one_hot(xs, num_classes=V).float()`, di forma (3073, 23). Così tutti i caratteri sono equidistanti.
- **Un solo strato lineare**, senza bias: `logits = xenc @ W`.

> **Un fatto che tornerà per tutto il corso.** Moltiplicare un vettore one-hot per una matrice **seleziona una riga**: `xenc @ W` è identico a `W[xs]`. Il primo strato di un modello di linguaggio non è un calcolo, è una **tabella di consultazione**. Nella Lezione 3 si chiamerà *embedding*.

#### 5.2 Dai logit alle probabilità: la softmax

I logit sono numeri reali qualsiasi. Per ottenere probabilità (positive e a somma 1) si esponenzia e si normalizza:

$$p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Il collegamento con i conteggi: $e^{W_{ij}}$ fa la parte di $N_{ij}$, quindi la rete impara $W_{ij} \approx \log N_{ij}$ (a meno di una costante per riga, che la softmax ignora). **La rete e i conteggi sono due modi di scrivere lo stesso modello.**

#### 5.3 Addestramento

```python
for k in range(6000):
    # forward
    xenc = F.one_hot(xs, num_classes=V).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)          # softmax
    loss = -probs[torch.arange(n), ys].log().mean()       # cross-entropy
    # backward
    W.grad = None                                         # equivale a zero_grad()
    loss.backward()
    # update
    lr = 30.0 if k < 1000 else (10.0 if k < 3000 else 3.0)
    W.data += -lr * W.grad
```

La loss parte da 3,135 e arriva a circa 2,091. Il valore iniziale non è un caso: con pesi piccoli tutti i logit sono quasi uguali, la softmax è quasi uniforme e la loss è quella del modello che non sa niente, $\log 23$.

> **[Verifica] Controllo gratuito, da fare sempre.** All'inizio la loss deve valere circa $\log(\text{numero di classi})$, qui $\log 23 = 3{,}135$. Se parte più alta, la rete è "sicura di sé e sbagliata": l'inizializzazione è troppo grande. (Senza il fattore 0,01, cioè con `torch.randn` a scala piena, qui partirebbe da circa 3,6; nella Lezione 4 vedremo un caso molto più grave.) Se parte molto più bassa, quasi certamente l'input contiene di nascosto la risposta.

Il learning rate decrescente serve perché con un passo fisso e alto la loss arriva vicino al minimo e poi oscilla scavalcandolo. È una versione artigianale dello *scheduling* (Lezione 9).

### 6. La verifica: le due strade coincidono

| Metodo                  | Loss   |
| ----------------------- | ------ |
| Conteggi ($\alpha = 0$) | 2,0890 |
| Rete neurale            | 2,0907 |

Non è una coincidenza. Imponendo gradiente nullo alla cross-entropy si ottiene proprio $p_{ij} = N_{ij} / \sum_k N_{ik}$, cioè i conteggi normalizzati: il modello a conteggi è **l'unico minimo** della rete (in termini di probabilità). La piccola differenza è solo convergenza incompleta sulle righe rare.

#### 6.1 Regolarizzazione = smoothing

Aggiungiamo alla loss una penalità sui pesi grandi (**regolarizzazione L2**):

```python
loss = -probs[torch.arange(n), ys].log().mean() + 0.01 * (W**2).mean()
```

La penalità spinge $W$ verso 0, cioè tutti i logit verso valori uguali, cioè la softmax verso l'uniforme: **lo stesso effetto dello smoothing**. Due nomi per lo stesso meccanismo: un po' di precisione sui dati di addestramento in cambio di un modello meno sicuro di sé.

### 7. Esercizi

1. **Trigrammi.** Estendi il modello a $P(x_t \mid x_{t-2}, x_{t-1})$, sia con i conteggi (tensore $23 \times 23 \times 23$) sia con la rete (input one-hot di dimensione 46). Di quanto scende la loss?
2. **Train/dev/test.** Dividi i nomi in 80/10/10 e valuta la loss sulle tre parti. Per il bigramma i valori saranno simili; per il trigramma comincerai a vedere un divario. Quel divario è l'**overfitting**, tema della Lezione 3.
3. **Temperatura.** Prima della softmax dividi i logit per $\tau$: $p_j \propto \exp(z_j/\tau)$. Genera nomi con $\tau = 0{,}5,\ 1,\ 2$. È il parametro *temperature* delle API di ogni LLM: basso = scelte più prevedibili, alto = più varie e più strane.

### 8. Verso gli LLM

Hai costruito in miniatura l'impianto completo di un modello di linguaggio moderno:

- **il compito**, $P(x_t \mid x_{<t})$;
- **la loss**, la cross-entropy, che non cambierà più fino alla fine del corso;
- **la softmax**, l'ultimo strato di ogni LLM;
- **la generazione autoregressiva**: predici un token, aggiungilo al contesto, ripeti (è ciò che vedi quando ChatGPT scrive parola per parola);
- **la tabella di consultazione**, futuro embedding;
- **la temperatura**.

Il limite non è la rete, è il **contesto**. Allungarlo contando esplode: con $k$ caratteri di contesto la tabella ha $23^{k+1}$ celle, quasi tutte vuote. La rete neurale ci salverà, perché può comprimere il contesto in un vettore e generalizzare a combinazioni mai viste.

### Riepilogo

1. Un modello di linguaggio stima $P(x_t \mid x_{<t})$; il bigramma lo approssima con $P(x_t \mid x_{t-1})$.
2. La loss è la negative log likelihood media (cross-entropy); $e^L$ è la perplessità.
3. Contare i bigrammi e addestrare uno strato lineare con softmax danno lo stesso modello: la rete impara $W \approx \log N$.
4. La strada del gradiente è più lenta, ma è l'unica che si estende a contesti lunghi.
5. Regolarizzazione L2 e smoothing sono lo stesso meccanismo.

---

## Lezione 3 --- MLP ed embedding

*Lettura: circa 30 minuti.* Allunghiamo il contesto da uno a tre caratteri seguendo il modello di Bengio (2003). Per farlo nasce l'oggetto più importante del corso: l'**embedding**.

### 1. Il problema: la maledizione della dimensionalità

Con i conteggi, ogni carattere di contesto in più moltiplica per 23 la dimensione della tabella:

| Contesto     | Celle della tabella       |
| ------------ | ------------------------- |
| 1 carattere  | $23^2 = 529$              |
| 3 caratteri  | $23^4 \approx 280\,000$   |
| 10 caratteri | $23^{11} \approx 10^{15}$ |

Con 3 caratteri abbiamo 280 000 celle e solo 3073 esempi: la tabella è quasi tutta vuota. Le combinazioni crescono in modo esponenziale, i dati no. È il muro contro cui i modelli a n-grammi si sono fermati storicamente attorno a $n = 5$.

C'è anche un problema più profondo: per una tabella "a" ed "e" sono lontane quanto "a" e "q". Non esiste somiglianza. Se il modello ha visto molte volte `mar` ma mai `mer`, sul secondo non sa dire nulla.

### 2. L'idea: rappresentare i simboli con vettori appresi

L'idea di Bengio: rappresentare ogni carattere con un **vettore di numeri reali** e impararlo insieme al resto della rete. Caratteri che si comportano in modo simile finiranno vicini, e ciò che il modello impara su uno si trasferisce all'altro. Si chiama **rappresentazione distribuita**.

Ricorda la Lezione 2: one-hot per matrice = selezione di una riga. Quindi smettiamo di fingere che sia una moltiplicazione: la matrice $C$ (di forma $V \times d$) è la **tabella degli embedding**, e `C[i]` è il vettore del carattere $i$. Con 23 caratteri e $d = 10$, $C$ ha 230 numeri, appresi dal gradiente come tutti gli altri.

|                        | One-hot               | Embedding                          |
| ---------------------- | --------------------- | ---------------------------------- |
| Come è fatto           | fisso, un 1 e tanti 0 | denso, numeri reali                |
| Distanze fra caratteri | tutte uguali          | decise dal modello in base ai dati |

**L'architettura di Bengio** (contesto di 3 caratteri):

1. i 3 caratteri diventano 3 embedding tramite $C$;
2. i 3 vettori vengono concatenati ($3 \times 10 = 30$ numeri);
3. uno strato nascosto con tanh;
4. uno strato lineare produce 23 logit;
5. softmax e cross-entropy, come sempre.

$$h = \tanh\big([c_{x_1}; c_{x_2}; c_{x_3}]\,W_1 + b_1\big), \qquad \text{logits} = h\,W_2 + b_2$$

dove $c_x = C[x]$ è l'embedding del carattere $x$ e $[\,\cdot\,;\,\cdot\,]$ indica la concatenazione.

### 3. Il codice

#### 3.1 Dataset a finestra scorrevole

```python
block_size = 3   # quanti caratteri di contesto

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size              # riempito con '.'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]        # scorri la finestra
    return torch.tensor(X), torch.tensor(Y)
```

Da `anna` nascono cinque esempi: `...` $\to$ `a`, `..a` $\to$ `n`, `.an` $\to$ `n`, `ann` $\to$ `a`, `nna` $\to$ `.`.

#### 3.2 Parametri e forward

```python
n_emb, n_hidden = 10, 200
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((V, n_emb), generator=g)
W1 = torch.randn((block_size * n_emb, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, V), generator=g)
b2 = torch.randn(V, generator=g)
parameters = [C, W1, b1, W2, b2]          # 11 053 parametri
for p in parameters:
    p.requires_grad = True
# Nota: qui inizializziamo tutto con torch.randn a scala piena, senza il
# fattore 0.01 della Lezione 2. Nella Lezione 4 vedremo quanto costa.

emb = C[X]                                 # (N, 3, 10): la consultazione
h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)   # (N, 30) -> (N, 200)
logits = h @ W2 + b2                       # (N, 23)
loss = F.cross_entropy(logits, Y)
```

- `C[X]` indicizza una tabella con un intero tensore di indici: è l'operazione con cui, in ogni LLM, i token diventano vettori.
- `.view(N, -1)` concatena **senza copiare**: cambia solo il modo di leggere la memoria.
- `F.cross_entropy` fa in un colpo ciò che nella Lezione 2 scrivevamo a mano, ed è numericamente stabile (sottrae il massimo dei logit prima di esponenziare, così non va in overflow). Da qui in poi usiamo sempre questa.

### 4. Far funzionare l'addestramento

#### 4.1 Minibatch

Calcolare la loss su tutti gli esempi a ogni passo è impossibile con dati veri. Si usa un **minibatch**: un piccolo campione casuale. (Per ora usiamo tutti i dati `X, Y`; nella sezione 5 addestreremo solo su una parte.)

```python
for i in range(30000):
    ix = torch.randint(0, X.shape[0], (64,))   # 64 esempi a caso
    Xb, Yb = X[ix], Y[ix]
    emb = C[Xb]
    h = torch.tanh(emb.view(-1, block_size * n_emb) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 20000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

Il gradiente su un minibatch è una stima **rumorosa ma non distorta**: in media punta nella direzione giusta. Conviene fare molti passi approssimati piuttosto che pochi passi esatti.

#### 4.2 Trovare il learning rate

Invece di indovinarlo: un breve addestramento in cui il learning rate cresce in modo esponenziale (da 0,001 a 1), registrando la loss. A sinistra la loss scende pigra, a destra esplode, in mezzo c'è la valle. Qui il minimo è attorno a $\eta \approx 0{,}1$; alla fine lo si abbassa a 0,01 per rifinire.

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre        # nel ciclo: lr = lrs[i], e registra (lre[i], loss)
```

### 5. Il problema centrale: l'overfitting

Una loss bassa sui dati di addestramento non basta. Il modello ha 11 000 parametri, più manopole che esempi: potrebbe **memorizzare** i dati invece di imparare la struttura della lingua. Per accorgercene dobbiamo misurarlo su dati che non ha mai visto. Dividiamo quindi i nomi in tre parti:

```python
import random
random.seed(42); random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%: si addestra qui
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%: si scelgono gli iperparametri
Xte,  Yte  = build_dataset(words[n2:])     # 10%: giudizio finale, una volta sola
```

- **train**: si calcolano i gradienti e si muovono i pesi
- **dev** (o *validation*): si scelgono dimensioni, learning rate, regolarizzazione
- **test**: si guarda **una volta sola**, alla fine; ogni volta che lo guardi e cambi qualcosa, smette di essere una misura onesta

Addestriamo con il ciclo della sezione 4, usando `Xtr, Ytr` al posto di `X, Y` (circa 2 450 esempi). La loss di train scende a **0,98**, contro il 2,16 del bigramma con smoothing: sembra un trionfo. Ma sul dev la loss è **2,81**: su dati nuovi il modello è **peggio del bigramma**. Ha imparato a memoria questi nomi.

> **Il divario train--dev è il termometro dell'overfitting.** Se train e dev sono vicine, il modello generalizza (o è troppo piccolo: *underfitting*). Se train è molto più bassa di dev, sta memorizzando. Conta sempre la coppia, mai un numero da solo.

Le leve contro l'overfitting: **più dati** (la cura migliore, e il motivo per cui gli LLM si addestrano su terabyte di testo), **modello più piccolo**, **regolarizzazione**. Aggiungiamo alla loss la penalità L2 della Lezione 2 (detta anche *weight decay*), con $\lambda = 0{,}01$:

| Modello                            | train | dev      |
| ---------------------------------- | ----- | -------- |
| MLP, nessuna regolarizzazione      | 0,98  | 2,81     |
| MLP + weight decay 0,01            | 1,57  | **2,00** |
| Bigramma con smoothing (confronto) | 2,16  | 2,16     |

La loss di train peggiora (il modello non può più memorizzare), quella di dev migliora e scende sotto il bigramma: ora il modello impara davvero la struttura su tre caratteri.

Il test resta da parte: si misura una volta sola, alla fine, sul modello scelto.

### 6. Cosa ha imparato: gli embedding

Addestrando con embedding a 2 dimensioni si possono disegnare su un piano. Partiti da rumore casuale, senza che nessuno abbia detto al modello cosa sia una vocale, **le vocali si raggruppano in una zona e le consonanti in un'altra**. Il modello ha scoperto la distinzione perché serve a predire il carattere successivo.

È il cuore della rappresentazione distribuita: caratteri con comportamento simile finiscono vicini, e il modello può generalizzare per vicinanza ("a" ed "e" sono vicine, quindi ciò che ha imparato dopo `mar` aiuta anche dopo `mer`). In un LLM la tabella ha decine di migliaia di righe (una per token) e migliaia di dimensioni, ma è lo stesso oggetto.

### 7. Generare nomi

```python
context, out = [0] * block_size, []
while True:
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    probs = F.softmax(h @ W2 + b2, dim=1)
    ix = torch.multinomial(probs, num_samples=1).item()
    if ix == 0:
        break
    out.append(itos[ix])
    context = context[1:] + [ix]
print(''.join(out))
```

Undici nomi dal modello regolarizzato: `pestavio, giulia, lauro, vita, alioca, geeranco, errella, arnenie, billo, gemo, alessina`. Il salto rispetto al bigramma è netto: quasi tutti sono pronunciabili e suonano italiani.

### 8. Esercizi

1. **Sposta il collo di bottiglia.** Aumenta uno alla volta `n_emb`, `n_hidden`, `block_size` guardando la loss di dev. Quale aiuta di più? Oltre una certa dimensione la dev smette di migliorare: il limite si è spostato altrove (sui dati).
2. **Disegna gli embedding** con $d = 2$. Ritrovi la separazione vocali/consonanti? Prova anche con solo 10 passi di addestramento.
3. **Overfitting a comando.** Addestra un modello grande, senza regolarizzazione, su soli 50 nomi, e traccia train e dev nel tempo. La dev a un certo punto risale: lì il modello smette di imparare e comincia a memorizzare. Fermarsi lì si chiama *early stopping*.

### 9. Verso gli LLM

- **L'embedding**: la prima cosa che fa GPT con i token è trasformarli in vettori con una tabella come $C$.
- **La finestra di contesto**: il nostro `block_size = 3` è l'antenato del *context length*. "128k token di contesto" è questo numero, portato a 128 000.
- **Il metodo sperimentale**: train/dev/test, divario come termometro, ricerca del learning rate, regolarizzazione. È il mestiere quotidiano di chi addestra modelli da miliardi di parametri.

Restano due limiti: le reti profonde diventano difficili da addestrare (Lezione 4), e concatenare gli embedding è un modo rozzo di fondere il contesto, che non scala (la soluzione sarà l'attenzione, Lezione 7).

### Riepilogo

1. I conteggi esplodono con il contesto ($23^{k+1}$ celle): è la maledizione della dimensionalità.
2. L'embedding rappresenta ogni carattere con un vettore denso appreso; `C[x]` sostituisce il one-hot.
3. Il modello di Bengio concatena gli embedding, li passa in un MLP con tanh e produce una softmax.
4. Train/dev/test rivela l'overfitting (train 0,98, dev 2,81); la regolarizzazione lo riduce (dev 2,00).
5. Gli embedding appresi raggruppano da soli i caratteri simili (vocali e consonanti).

---

# PARTE II --- FAR FUNZIONARE L'ADDESTRAMENTO

## Lezione 4 --- Attivazioni, gradienti, BatchNorm

*Lettura: circa 35 minuti.* La rete della Lezione 3 funzionava, ma per fortuna. Qui impariamo a misurare cosa succede dentro una rete mentre si addestra, e a renderla robusta.

### 1. Il problema

Una rete in addestramento è un **sistema dinamico**. Nel forward i segnali attraversano gli strati, nel backward li attraversano i gradienti. Se a ogni strato il segnale si amplifica un po', dopo molti strati esplode; se si attenua un po', svanisce. L'addestramento funziona solo se medie e deviazioni standard restano "sane" strato dopo strato, in entrambe le direzioni.

Vedremo che con l'inizializzazione ingenua (`torch.randn` a scala piena):

1. anche la rete della Lezione 3 sprecava migliaia di passi;
2. una rete appena più profonda non impara quasi nulla.

### 2. La loss iniziale che tradisce

Al passo zero la rete non sa nulla, quindi dovrebbe assegnare 1/23 a ogni carattere: loss $\log 23 = 3{,}135$. Sulla rete della Lezione 3, inizializzata con `torch.randn` a scala piena, misuriamo invece **25,4**: otto volte peggio dell'ignoranza. La rete non è ignorante, è **sicura di sé e sbagliata**. I logit hanno deviazione standard 13,6: la softmax si concentra su pochi caratteri scelti a caso.

Nella curva della loss si vede la classica "mazza da hockey": un crollo verticale iniziale (la rete che si "sgonfia") e poi la discesa vera. Quei primi passi sono sprecati.

La cura è gratuita: logit piccoli al passo zero, con lo stesso fattore 0,01 usato nella Lezione 2.

```python
W2 = torch.randn((n_hidden, V), generator=g) * 0.01
b2 = torch.zeros(V)
# loss iniziale: 3.135 = log 23
```

> **Perché $\boldsymbol{\times}$ 0,01 e non $\boldsymbol{\times}$ 0?** Per l'ultimo strato anche 0 funzionerebbe. Ma azzerare **tutti** i pesi è un errore fatale: neuroni identici ricevono gradienti identici e restano identici per sempre. Serve casualità per rompere la simmetria. Regola: **mai zero negli strati interni, quasi zero nell'ultimo.**

### 3. Le tanh sature

Ora guardiamo il primo strato. Le preattivazioni entrano nella tanh con deviazione standard 6,75, e la tanh schiaccia tutto in $(-1, 1)$: il **67%** delle attivazioni ha $|h| > 0{,}99$.

Perché è grave? Dalla Lezione 1: il backward della tanh moltiplica per $(1 - t^2)$, che vale **zero** quando $t = \pm 1$. Il gradiente muore attraversando una tanh satura. Un neurone saturo su **tutti** gli esempi è un **neurone morto**: non verrà mai più aggiornato. Lo stesso vale per la sigmoide, e la ReLU ha la sua versione (il *dying ReLU*: gradiente zero per ogni input negativo).

### 4. L'inizializzazione di Kaiming

La domanda giusta non è "che numero magico metto davanti a `randn`?", ma: **quale scala dei pesi conserva le statistiche del segnale da uno strato al successivo?**

Per una preattivazione $z = \sum_{i=1}^{n} w_i x_i$ con pesi e input indipendenti a media zero:

$$\text{Var}(z) = n \cdot \text{Var}(w) \cdot \text{Var}(x)$$

Perché la varianza non cresca né si spenga serve $n\,\text{Var}(w) = 1$, cioè $\sigma_w = 1/\sqrt{n}$, dove $n$ è il numero di input del neurone (il *fan-in*, $n_{in}$). Poi la non-linearità comprime il segnale, e si compensa con un **guadagno** $g$:

$$\sigma_w = \frac{g}{\sqrt{n_{in}}}, \qquad g = \begin{cases} 1 & \text{lineare} \\ 5/3 & \text{tanh} \\ \sqrt{2} & \text{ReLU} \end{cases}$$

È l'**inizializzazione di Kaiming** (He e colleghi, 2015); la variante di Glorot/Xavier (2010) bilancia fan-in e fan-out. I valori di $g$ sono quelli suggeriti da PyTorch (`torch.nn.init.calculate_gain`); l'inizializzazione si applica con `torch.nn.init.kaiming_normal_`.

```python
fan_in = block_size * n_emb      # 30 input per ogni neurone del primo strato
W1 = torch.randn((fan_in, n_hidden), generator=g) * (5/3) / fan_in**0.5
b1 = torch.zeros(n_hidden)
# saturazione: dal 67% al 16%
```

**L'esperimento in profondità** (5 strati $200 \to 200$ con tanh, deviazione standard delle attivazioni):

| Scala                            | Strato 1 | Strato 3 | Strato 5 | Diagnosi                           |
| -------------------------------- | -------- | -------- | -------- | ---------------------------------- |
| $\sigma_w = 1$ (ingenua)         | 0,97     | 0,97     | 0,97     | sembra sana, ma è saturata all'85% |
| $\sigma_w = 1/\sqrt n$           | 0,63     | 0,41     | 0,32     | il segnale svanisce                |
| Kaiming ($\tfrac{5}{3}/\sqrt n$) | 0,76     | 0,67     | 0,65     | stabile: il "corridoio"            |

Prima morale della diagnostica: **la deviazione standard da sola non basta**, va guardata insieme alla saturazione.

### 5. La BatchNorm

L'inizializzazione sistema il passo zero, ma durante l'addestramento i pesi si muovono. L'idea della **Batch Normalization** (Ioffe e Szegedy, 2015): invece di sperare che le preattivazioni abbiano media 0 e varianza 1, **imponiamolo**, a ogni strato e a ogni passo.

Per ogni neurone $j$, sul batch:

$$\hat z_{ij} = \frac{z_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}, \qquad y_{ij} = \gamma_j\,\hat z_{ij} + \beta_j$$

$\gamma$ (scala) e $\beta$ (spostamento) sono parametri appresi che restituiscono alla rete la libertà di spostarsi dove le serve.

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.01):
        self.eps, self.momentum, self.training = eps, momentum, True
        self.gamma = torch.ones(dim)       # parametri appresi
        self.beta  = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)   # statistiche per l'inferenza
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            mean = x.mean(0)   # forma (dim,): una media per neurone
            var  = x.var(0)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

Qui `mean` e `var` hanno forma `(dim,)` senza `keepdim`, ed è corretto: il broadcasting allinea a destra, quindi a ogni riga di `x` (forma `(B, dim)`) viene sottratta la media del suo neurone. È proprio l'allineamento che nella Lezione 2 era sbagliato e qui è quello voluto.

La BatchNorm si mette fra lo strato lineare e la non-linearità: `h = torch.tanh(bn(x @ W1))`.

**Tre dettagli che mordono:**

- **Il bias diventa inutile.** Qualunque costante aggiunga, la sottrazione della media la cancella. Il suo gradiente è esattamente zero (lo dimostreremo nella Lezione 5). Quindi lo strato lineare che precede una BatchNorm va scritto senza bias (in PyTorch: `nn.Linear(..., bias=False)`).
- **Gli esempi del batch si accoppiano.** La predizione su "anna" dipende da chi le è capitato accanto nel minibatch. È strano, ma il rumore che ne risulta fa da regolarizzatore gratuito.
- **All'inferenza** il batch è di un solo esempio: per questo si usano le medie mobili `running_mean` e `running_var`. È l'origine di `model.train()` / `model.eval()` in PyTorch, e del bug classico di dimenticare `model.eval()`.

**Tre addestramenti a confronto.** Rete più profonda: 30 input, 6 strati nascosti da 100 neuroni, 23 uscite; 20 000 passi, batch 64, learning rate 0,1 e poi 0,01. In tutte e tre le configurazioni l'ultimo strato parte già $\times\,0{,}01$, così il confronto riguarda solo gli strati nascosti.

| Configurazione           | loss a 5 000 passi | train finale | dev  |
| ------------------------ | ------------------ | ------------ | ---- |
| Init ingenua             | 2,37               | 1,38         | 2,22 |
| Kaiming                  | 1,02               | 0,97         | 3,29 |
| Init ingenua + BatchNorm | 1,12               | 0,99         | 2,80 |

Cosa leggere:

1. **La profondità trasforma il difetto in malattia.** Con l'init ingenua, al passo zero il gradiente del primo strato è circa 100 volte quello del sesto. Con un unico learning rate per tutta la rete non esiste un valore giusto per tutti. Con Kaiming i gradienti degli strati restano entro un fattore 2.
2. **La BatchNorm rende la rete robusta all'inizializzazione**: con pesi a scala ingenua arriva al livello di Kaiming. È il motivo per cui dopo il 2015 le reti profonde hanno smesso di essere fragili.
3. **L'effetto regolarizzante è reale**: a parità di train, la BatchNorm ha una dev migliore di Kaiming (2,80 contro 3,29). La dev di 2,22 della prima riga non è una vittoria: con train a 1,38 il modello sta solo imparando poco (*underfitting*).

### 6. La diagnostica: tre pannelli da guardare sempre

1. **Statistiche delle attivazioni, strato per strato**: deviazione standard stabile con la profondità, saturazione sotto il 5% circa. Se cresce strato dopo strato $\to$ esplosione; se si spegne $\to$ svanimento.
2. **Statistiche dei gradienti, strato per strato**: devono essere comparabili fra gli strati. Due ordini di grandezza di differenza = nessun learning rate va bene per tutti.
3. **Il rapporto aggiornamento/valore**: a ogni passo, per ogni tensore di parametri,

$$\log_{10} \frac{\text{std}(\eta \cdot \nabla W)}{\text{std}(W)}$$

cioè quanto è grande l'aggiornamento rispetto al valore che modifica. Regola empirica di Karpathy: attorno a $-3$ (cioè $10^{-3}$). Molto sotto $-4$: i parametri si muovono di briciole. Sopra $-2{,}5$ in modo stabile: passi troppo grandi, rischio di instabilità.

Con questi pannelli, "il training non converge" diventa una diagnosi: saturazione? gradienti squilibrati? learning rate fuori scala? Ognuno ha la sua firma.

### 7. Esercizi

1. **Radiografia delle attivazioni.** Visualizza `plt.imshow(h.abs() > 0.99, cmap='gray')` (esempi $\times$ neuroni, bianco = saturo). Le colonne tutte bianche sono neuroni morti. Con quale scala di inizializzazione compaiono?
2. **Il gain è un termostato.** Sulla rete a 5 strati prova $g \in \{0{,}5,\ 1,\ 5/3,\ 3\}$: sotto il valore giusto il segnale svanisce, sopra satura.
3. **Inventa la LayerNorm.** Modifica la BatchNorm perché normalizzi **per riga** (i neuroni di ogni singolo esempio) invece che per colonna: `mean(1, keepdim=True)`. Verifica che il risultato non dipenda più dal batch e che train ed eval coincidano senza medie mobili. È la normalizzazione che useremo nel Transformer.

### 8. Verso gli LLM

- **La LayerNorm** (esercizio 3) sostituisce la BatchNorm nei Transformer: niente accoppiamento fra esempi, niente doppia modalità train/eval.
- **L'inizializzazione conta ancora.** GPT-2 usa $\sigma = 0{,}02$ e riscala gli strati che scrivono nel flusso residuo di $1/\sqrt{2\,n_{layer}}$ (lo vedremo nella Lezione 9).
- **Le connessioni residue**: una terza soluzione al problema della profondità, che qui non abbiamo usato. Dare al gradiente una "autostrada" che salta gli strati: $x + f(x)$ invece di $f(x)$. Le introduciamo nella Lezione 7.
- **A grande scala questa lezione è il lavoro quotidiano**: picchi della loss, norme dei gradienti, precisione numerica. La differenza fra un run che converge e uno che brucia una settimana di GPU è spesso in un grafico come questi, guardato in tempo.

> **[Nuovo] Aggiornamento.** Nei modelli recenti (Llama, Mistral, Qwen, DeepSeek...) la LayerNorm è quasi sempre sostituita dalla **RMSNorm**, che divide solo per la radice della media dei quadrati, senza sottrarre la media: è più semplice e veloce e funziona altrettanto bene. Vedi l'[Appendice D](#appendice-d-----cosa-è-cambiato-dopo-gpt-2-2019--2026).

### Riepilogo

1. Al passo zero la loss deve valere $\log V$; se è più alta la rete è sicura di sé e sbagliata: si corregge con l'ultimo strato $\times\,0{,}01$.
2. Le tanh sature bloccano il gradiente ($1 - t^2 \to 0$).
3. Kaiming ($\sigma_w = g/\sqrt{n_{in}}$, $g = 5/3$ per la tanh) mantiene stabili le statistiche in profondità.
4. La BatchNorm impone media 0 e varianza 1 a ogni strato: rende la rete robusta e regolarizza, ma accoppia il batch e separa train ed eval.
5. In profondità un'inizializzazione sbagliata squilibra i gradienti di 100 volte fra gli strati.
6. Diagnostica sempre: attivazioni, gradienti, rapporto aggiornamento/valore $\approx 10^{-3}$.

---

## Lezione 5 --- Backprop a mano, senza autograd

*Lettura: circa 35 minuti.* Spegniamo `loss.backward()` e scriviamo noi ogni gradiente della rete della Lezione 4, verificandolo contro PyTorch. Dopo, i gradienti non saranno più magia.

### 1. Il problema

Karpathy chiama la backpropagation un'**astrazione che perde** (*leaky abstraction*): puoi usarla senza capirla finché funziona. Ma tutti i problemi visti finora vivono nel backward: tanh sature, bias fantasma, gradienti squilibrati, il bug del `keepdim`. Se il backward è una scatola nera, questi problemi si subiscono; se lo sai scrivere, li sai leggere.

Nella Lezione 1 l'abbiamo scritto per gli scalari. Sui tensori entrano in gioco le **forme** e il **broadcasting**: è la competenza che separa chi usa una libreria da chi potrebbe scriverla.

### 2. Due regole che fanno quasi tutto il lavoro

**Regola 1 --- Il gradiente ha la stessa forma del tensore.** Se $W$ è (30, 200), anche $\partial L/\partial W$ è (30, 200). Spesso c'è un solo modo di combinare i pezzi disponibili perché le forme tornino: la derivazione diventa un puzzle a incastro.

**Regola 2 --- Broadcasting e somma sono duali.**

| Nel forward       | Nel backward |
| ----------------- | ------------ |
| copia (broadcast) | **somma**    |
| somma             | **copia**    |

È il `+=` della Lezione 1 in versione tensoriale: un valore copiato in $k$ posti influenza la loss per $k$ strade, e i contributi si sommano. Se nel forward un tensore (1, H) è stato copiato su (n, H), nel backward il suo gradiente torna (1, H) con `.sum(0, keepdim=True)`. **Le forme ti dicono dove sommare.**

#### 2.1 Il backward del prodotto matriciale

Per $Z = XW + b$, con $X$ di forma $(n, d)$, $W$ di forma $(d, m)$ e $\bar Z = \partial L/\partial Z$ già noto (la notazione $\bar x$ è quella della Lezione 1):

$$\bar X = \bar Z\,W^\top, \qquad \bar W = X^\top \bar Z, \qquad \bar b = \textstyle\sum_i \bar Z_{i,:}$$

Il trucco per non doverle ricordare: $\bar X$ deve essere $(n, d)$; hai in mano $\bar Z$ $(n, m)$ e $W$ $(d, m)$; l'unico prodotto che dà $(n, d)$ è $\bar Z W^\top$. Idem per gli altri.

### 3. Il forward esploso in passi elementari

Riprendiamo la rete a un solo strato nascosto delle Lezioni 3--4, con la BatchNorm (embedding $\to$ lineare $\to$ BatchNorm $\to$ tanh $\to$ lineare $\to$ cross-entropy), e scriviamo ogni operazione come una riga separata. Per tenere piccoli i tensori usiamo 64 neuroni nascosti invece di 200. Teniamo apposta il bias `b1` prima della BatchNorm, per dimostrare che è inutile.

```python
# parametri: C (23, 10), W1 (30, 64), b1 (64,), W2 (64, 23), b2 (23,), bngain e bnbias (1, 64)
# n = 32 esempi nel batch; a destra le forme
emb = C[Xb]                                         # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)                 # (32, 30)
hprebn = embcat @ W1 + b1                           # (32, 64)
# BatchNorm
bnmeani = 1/n * hprebn.sum(0, keepdim=True)         # (1, 64)
bndiff = hprebn - bnmeani                           # (32, 64)  broadcast
bndiff2 = bndiff**2                                 # (32, 64)
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)      # (1, 64)   correzione di Bessel
bnvar_inv = (bnvar + 1e-5)**-0.5                    # (1, 64)
bnraw = bndiff * bnvar_inv                          # (32, 64)  broadcast
hpreact = bngain * bnraw + bnbias                   # (32, 64)  broadcast
h = torch.tanh(hpreact)                             # (32, 64)
logits = h @ W2 + b2                                # (32, 23)
# cross-entropy
logit_maxes = logits.max(1, keepdim=True).values    # (32, 1)
norm_logits = logits - logit_maxes                  # (32, 23)  broadcast
counts = norm_logits.exp()                          # (32, 23)
counts_sum = counts.sum(1, keepdim=True)            # (32, 1)
counts_sum_inv = counts_sum**-1                     # (32, 1)
probs = counts * counts_sum_inv                     # (32, 23)  broadcast
logprobs = probs.log()                              # (32, 23)
loss = -logprobs[range(n), Yb].mean()               # scalare
```

Ogni riga marcata *broadcast*, nel backward, diventerà una somma. Per confrontare con PyTorch:

```python
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | esatto: {str(ex):5s} | approssimato: {str(app):5s} | maxdiff: {maxdiff}')
```

### 4. Il backward, dal fondo

```python
# loss e probabilità
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs          # derivata di log è 1/x

# counts è usato due volte: i contributi si sommano
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)   # broadcast -> somma
dcounts = counts_sum_inv * dprobs                          # strada 1
dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum           # strada 2: +=

dnorm_logits = counts * dcounts             # derivata di exp è se stessa
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits = dnorm_logits.clone()
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

# strati lineari e tanh: le formule del matmul
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
dhpreact = (1.0 - h**2) * dh

# BatchNorm: la dualità applicata più volte
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbndiff = bnvar_inv * dbnraw                                # strada 1
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv
dbndiff2 = 1.0/(n-1) * torch.ones_like(bndiff2) * dbnvar    # somma -> copia
dbndiff += 2 * bndiff * dbndiff2                            # strada 2: +=
dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0, keepdim=True)
dhprebn += 1.0/n * torch.ones_like(hprebn) * dbnmeani

# primo strato ed embedding
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
dC.index_add_(0, Xb.view(-1), demb.view(-1, emb.shape[2]))  # copia -> somma per indice
```

Tre momenti da notare:

- **`1/p` amplifica gli esempi sorprendenti.** Se il modello dava probabilità bassa al carattere giusto, il gradiente è più grande: gli esempi che sorprendono il modello sono quelli che gli insegnano di più.
- **Un gradiente che deve essere zero.** Sottrarre il massimo dai logit non cambia la softmax, quindi `dlogit_maxes` deve essere nullo. Misurato in doppia precisione: circa $10^{-17}$, cioè zero a meno degli arrotondamenti. Quella riga esiste solo per la stabilità numerica. **Quando un gradiente che dovrebbe essere zero non lo è, hai trovato un bug.**
- **L'embedding accumula.** Ogni riga di `C` è stata copiata in tutte le posizioni in cui compare il suo carattere: all'indietro, i contributi si sommano.

**Verifica.** Con `cmp` tutti i gradienti risultano uguali a quelli dell'autograd di PyTorch, esattamente o a meno di arrotondamenti. Un controllo indipendente con le differenze finite in doppia precisione (Lezione 1, esercizio 3) dà scarti dell'ordine di $10^{-10}$, il rumore tipico del metodo. E `db1` è nullo a meno degli arrotondamenti (circa $10^{-17}$ in doppia precisione): il bias prima della BatchNorm ha gradiente **esattamente** zero.

### 5. Due scorciatoie in forma chiusa

In pratica due blocchi si derivano una volta per tutte e si scrivono in una riga: più veloci, più stabili, più chiari.

**Cross-entropy.** Per un esempio con target $y$: $\ell = -\log p_y$, e

$$\frac{\partial \ell}{\partial z_j} = p_j - \mathbf{1}[j = y]$$

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

Sette passi (max, exp, somma, inverso, prodotto, log, indicizzazione) collassano in tre righe. Il significato: **il logit del carattere giusto viene spinto in su con forza $1 - p_y$, ogni altro logit in giù con forza $p_j$, cioè in proporzione a quanto il modello ci credeva**. Ogni riga somma a zero: la softmax può solo spostare probabilità, non crearne. È la terza volta che lo incontriamo: nella Lezione 1 il gradiente in ingresso era l'errore $2(\hat y - y)$; qui è l'errore in versione probabilistica, $p - \text{target}$. **Il segnale di apprendimento di ogni modello, GPT compreso, è la differenza fra ciò che ha predetto e ciò che era vero.**

**BatchNorm.** Il gradiente arriva a ogni $z_i$ per tre strade (diretta, attraverso la media, attraverso la varianza). Semplificando:

```python
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0)
    - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)
)
```

I tre termini: il gradiente che passa; meno la media del gradiente sul batch; meno la componente allineata con $\hat z$. Il gradiente di ogni esempio contiene quelli degli altri (l'accoppiamento visto nella Lezione 4). Sommando sulle righe i termini si annullano: $\sum_i \bar z_i = 0$, ed ecco la dimostrazione che il bias a monte ha gradiente nullo.

### 6. Addestrare senza autograd

Con questi gradienti scritti a mano (le due scorciatoie, le formule del matmul, la tanh, l'accumulo sull'embedding) la stessa rete (questa volta con 200 neuroni nascosti, come nella Lezione 4, e senza `b1`) si addestra per 20 000 passi senza mai chiamare `loss.backward()`. Risultati: loss iniziale 3,144 $\approx \log 23$, train 0,99, dev 2,50, in linea con la Lezione 4. Nomi generati: `ciriana, tizio, corrado, liberto, geltrude, dana, siliana, achiara, ascania, sara`.

### 7. Esercizi

1. **Il max che non serve (finché serve).** Togli la sottrazione del massimo e riaddestra in float32 con logit grandi. A quale valore `exp` va in overflow? (circa 88,7 in float32)
2. **Le forme non mentono.** Deriva $\bar W_2$ elemento per elemento e verifica che coincida con quanto ottieni dall'incastro delle forme. Perché $\bar X = \bar Z W^\top$ non può essere $W \bar Z^\top$?
3. **Il gradiente come debugger.** Reintroduci il bug della Lezione 2 (`counts_sum = counts.sum(1)` senza `keepdim=True`). Il forward gira senza errori. Quale `cmp` fallisce per primo?

### 8. Verso gli LLM

- **Il backward dell'attenzione è già nelle tue mani**: tre prodotti matriciali, una softmax e una maschera.
- **La memoria è il backward.** Tutte le attivazioni salvate (`h`, `bnraw`, `counts`...) sono il motivo per cui addestrare costa molta più memoria che usare il modello. Da qui nascono l'*activation checkpointing* (non salvare, ricalcolare nel backward) e FlashAttention (Lezione 9).
- **I kernel fusi sono questa lezione, industrializzata.** Chi scrive kernel CUDA o Triton per i Transformer fa ciò che hai fatto tu: prende un pezzo di grafo, ne deriva il backward in forma chiusa e lo implementa a mano.
- **Il debug a scala.** Quando un run su migliaia di GPU produce un NaN, qualcuno apre esattamente queste formule: varianza con batch degeneri? `exp` senza il max?

### Riepilogo

1. Il gradiente di un tensore ha la sua stessa forma: le formule si ricostruiscono dagli incastri.
2. Ciò che il forward copia, il backward somma (e viceversa).
3. Matmul: $\bar X = \bar Z W^\top$, $\bar W = X^\top \bar Z$, $\bar b = \sum_i \bar Z_{i,:}$.
4. Cross-entropy: $\bar z = (p - \text{onehot}(y))/n$, il gradiente è l'errore.
5. BatchNorm in forma chiusa: tre termini che accoppiano il batch e sommano a zero.
6. Un gradiente che dovrebbe essere zero e non lo è segnala un bug.

---

## Lezione 6 --- WaveNet e la gerarchia

*Lettura: circa 25 minuti.* Il contesto passa da 3 a 8 caratteri, e invece di schiacciarlo tutto insieme lo fondiamo a coppie, come WaveNet. Nel farlo costruiamo un piccolo `torch.nn` nostro.

### 1. Il problema

L'MLP di Bengio concatena tutto il contesto e lo schiaccia in un unico strato. Con 8 caratteri questo:

- **cresce male**: la matrice del primo strato cresce in proporzione al contesto;
- **fonde male**: tutti i caratteri vengono mescolati subito, alla pari, senza l'idea che i caratteri vicini abbiano più da dirsi di quelli lontani.

La proposta, presa da **WaveNet** (DeepMind, 2016, nato per generare audio): fondere il contesto **a coppie, in più livelli**. Prima coppie di caratteri, poi coppie di coppie, e così via: $8 \to 4 \to 2 \to 1$. L'informazione vicina si incontra subito, quella lontana più tardi, e a ogni livello c'è una non-linearità che permette rappresentazioni via via più astratte.

C'è un dettaglio importante: a ogni livello si usa **la stessa matrice** per tutte le coppie. La regola per fondere due caratteri adiacenti non dipende da dove si trova la coppia nella parola. Un'operazione identica fatta scorrere lungo una sequenza è una **convoluzione**; la nostra, che a ogni livello salta il doppio delle posizioni, è una **convoluzione dilatata causale**, il mattone di WaveNet.

### 2. Il nostro `torch.nn`

Moduli con la stessa interfaccia (`__call__` per il forward, `parameters()` per i pesi), componibili.

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0):
        self.weight = torch.randn((fan_in, fan_out)) * gain / fan_in**0.5   # Kaiming
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x); return self.out
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]; return self.out
    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    """(B, T, C) -> (B, T//k, k*C): affianca k posizioni consecutive."""
    def __init__(self, k):
        self.k = k
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.k, self.k * C)
        if x.shape[1] == 1:
            x = x.squeeze(1)          # ultimo livello: via la dimensione del tempo
        self.out = x; return self.out
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x; return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`FlattenConsecutive` non calcola nulla: affianca due posizioni (è una `view`). La fusione vera la fa il `Linear` che segue. E qui c'è un regalo: con `x` di forma $(B, T, C_{in})$, `x @ W` applica **la stessa matrice a ogni posizione** in parallelo. I pesi condivisi sono gratis.

Come BatchNorm usiamo la classe `BatchNorm1d` della Lezione 4, con un ritocco che vediamo fra poco.

#### 2.1 Il modello in poche righe

```python
vocab_size = V          # 23, come nelle lezioni precedenti
block_size = 8          # contesto di 8 caratteri: va usato anche in build_dataset
n_emb, n_hidden = 10, 68
model = Sequential([
    Embedding(vocab_size, n_emb),
    FlattenConsecutive(2), Linear(2*n_emb,    n_hidden, bias=False, gain=5/3), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False, gain=5/3), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False, gain=5/3), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, gain=0.01),
])
```

Tutta la Lezione 4 è dentro queste righe: niente bias prima della BatchNorm, gain 5/3 per la tanh, ultimo strato piccolo. Il ciclo di addestramento non sa nulla dell'architettura: **cambiare rete = cambiare la lista**.

#### 2.2 La passeggiata delle forme

Prima di addestrare: un batch finto attraverso il modello, stampando le forme strato per strato (BatchNorm e Tanh non cambiano la forma e qui sono omesse).

```text
input               (32, 8)
Embedding        -> (32, 8, 10)
Flatten(2)       -> (32, 4, 20)    Linear -> (32, 4, 68)
Flatten(2)       -> (32, 2, 136)   Linear -> (32, 2, 68)
Flatten(2)       -> (32, 136)      Linear -> (32, 68)
Linear finale    -> (32, 23)
```

Si legge l'albero sull'asse di mezzo: $8 \to 4 \to 2 \to$ sparito. **Prima le forme, poi l'addestramento.**

### 3. Il bug della BatchNorm

Facendo la passeggiata delle forme si scopre un problema. La BatchNorm della Lezione 4 calcola media e varianza con `x.mean(0)`: corretto su input 2D. Ma ai livelli intermedi l'input è 3D, (32, 4, 68), e `mean(0)` media solo sui 32 esempi, producendo statistiche di forma **(4, 68)** invece che **(68,)**.

La versione sbagliata tiene una media separata per ogni posizione, rompendo proprio la condivisione fra posizioni che è il senso dell'architettura, e stima ogni statistica su 32 campioni invece di 128. La correzione: normalizzare su tutte le dimensioni tranne l'ultima.

```python
# dentro BatchNorm1d.__call__, al posto delle due righe che calcolano mean e var
xs = x.reshape(-1, x.shape[-1])   # (B, C) oppure (B*T, C): ogni posizione conta come un esempio
mean = xs.mean(0)                 # (C,)
var  = xs.var(0)                  # (C,)
```

Il resto della classe non cambia: `mean` e `var` di forma `(C,)` funzionano sia con input (B, C) sia con input (B, T, C), e anche le medie mobili restano di forma `(C,)`.

Quanto costa il bug sulla loss? Quasi niente: in questo esperimento la versione sbagliata ha dev 2,16 contro 2,26 della corretta (a questa scala il rumore in più regolarizza per caso). **È proprio questo il punto: la loss non ti avvisa.** L'unico allarme è una forma, (4, 68) dove doveva esserci (68,).

> **[Attenzione] Convenzione di PyTorch.** `torch.nn.BatchNorm1d` su input 3D vuole i canali in seconda posizione, (B, C, T); il nostro layout è (B, T, C). Un motivo in più per sapere cosa fa un modulo invece di fidarsi del nome.

### 4. L'esperimento onesto: la gerarchia conviene?

Tre modelli principali, stessa ricetta (Kaiming, BatchNorm, ultimo strato piccolo): piatto a contesto 3, piatto a contesto 8, gerarchico a contesto 8. Gli ultimi due hanno un numero di parametri simile, circa 22 000.

**Senza regolarizzazione** (quanto bene ottimizzano): train 1,00 (piatto, contesto 3), 0,78 (piatto, contesto 8), **0,76** (gerarchico, contesto 8). A parità di parametri il gerarchico scende più in basso: usa meglio la capacità.

**Con il weight decay migliore per ciascuno** (quanto bene generalizzano):

| Modello                                  | train | dev   |
| ---------------------------------------- | ----- | ----- |
| Piatto, contesto 3                       | 1,50  | 2,049 |
| Piatto, contesto 8                       | 1,67  | 2,047 |
| Gerarchico, contesto 8                   | 1,10  | 2,259 |
| Gerarchico piccolo (34 neuroni nascosti) | 1,91  | 2,203 |

Due cose che non ci aspettavamo:

1. **Il contesto 8 non compra quasi nulla** (2,049 contro 2,047). Con 399 nomi non ci sono abbastanza esempi per imparare cosa dicono i caratteri lontani. Il collo di bottiglia è il dataset.
2. **Il gerarchico qui generalizza peggio**: ottimizza meglio, ma usa la capacità in più per memorizzare.

La conclusione giusta non è "la gerarchia non funziona", ma **"i guadagni di architettura si pagano con i dati"**. Karpathy, su un dataset di 32 000 nomi (80 volte il nostro), guadagnò con il gerarchico pochi centesimi di loss: reali ma modesti. WaveNet ripaga il suo albero su ore di audio; il Transformer ripagherà l'attenzione su terabyte di testo.

### 5. Esercizi

1. **Verifica i pesi condivisi.** Passa un contesto in cui due coppie sono identiche e verifica che le righe corrispondenti dell'output del primo livello coincidano.
2. **Un livello in più.** Porta il contesto a 16 caratteri: basta portare `block_size` a 16 e aggiungere alla lista una riga di quattro moduli (`FlattenConsecutive`, `Linear`, `BatchNorm1d`, `Tanh`); il ciclo di addestramento non si tocca. Il limite dei dati si è alzato?
3. **Il divario dipende dai dati.** Addestra piatto-8 e gerarchico-8 su 100, 200 e 399 nomi e traccia le dev. Come si muove il divario? A quanti nomi ti aspetti il sorpasso?

### 6. Verso gli LLM

- **Fusione fissa contro fusione appresa.** WaveNet decide a priori chi parla con chi (coppie adiacenti). Il Transformer fa il passo successivo: ogni token può guardare ogni token precedente, e quanto guardarlo lo decide una quantità appresa, l'**attenzione**. Inoltre nell'albero due caratteri lontani si incontrano dopo $\log_2 T$ livelli; con l'attenzione in uno.
- **I pesi condivisi restano**: lo stesso blocco, con gli stessi pesi, viene applicato a ogni posizione.
- **`torch.nn` non è più una scatola**: ogni modulo ha un gemello che hai scritto tu.
- **"I guadagni di architettura si pagano con i dati"** è, in miniatura, l'idea delle **leggi di scala** (vedi Lezione 9): modello, dati e calcolo vanno cresciuti insieme.

### Riepilogo

1. Il contesto lungo si fonde a livelli ($8 \to 4 \to 2 \to 1$), con una non-linearità a ogni livello.
2. Lo stesso `Linear` serve tutte le posizioni di un livello: è una convoluzione dilatata causale.
3. I moduli separano l'architettura dal ciclo di addestramento.
4. Prima di addestrare, controlla le forme: trovano bug che la loss nasconde.
5. Su 399 nomi il gerarchico ottimizza meglio ma generalizza peggio: il limite è il dataset.
6. I guadagni di architettura si pagano con i dati.

---

# PARTE III --- IL TRANSFORMER

## Lezione 7 --- Costruiamo GPT da zero

*Lettura: circa 45 minuti.* Il cuore del corso. Ricaviamo l'attenzione da un'idea semplicissima, una media che scorre, la componiamo in un Transformer e addestriamo un piccolo GPT su un racconto italiano.

### 1. Il problema

WaveNet ha risolto un problema e ne ha lasciato un altro: la fusione è **fissa e cieca**. Ogni coppia viene fusa sempre allo stesso modo, qualunque cosa contenga. Ci serve un meccanismo in cui:

1. ogni posizione possa raccogliere informazione da **qualunque** posizione precedente;
2. quanto una posizione ne influenza un'altra dipenda dal **contenuto**, e sia **appreso**;
3. due posizioni lontane comunichino in un solo passo.

Questo meccanismo è la **self-attention**: l'unica idea architetturale davvero nuova del corso. Tutto il resto lo abbiamo già.

### 2. L'idea

#### 2.1 Una media che scorre, con un prodotto matriciale

Far comunicare le posizioni nel modo più semplice: ogni posizione prende la **media di tutte le precedenti** (e di se stessa). Si ottiene moltiplicando per una matrice triangolare inferiore normalizzata per riga:

```python
import torch
import torch.nn.functional as F

a = torch.tril(torch.ones(4, 4))
a = a / a.sum(1, keepdim=True)
# [[1.00, 0.00, 0.00, 0.00],
#  [0.50, 0.50, 0.00, 0.00],
#  [0.33, 0.33, 0.33, 0.00],
#  [0.25, 0.25, 0.25, 0.25]]
b = torch.tensor([[2., 4.], [6., 2.], [1., 5.], [3., 3.]])   # 4 "token" con 2 canali
c = a @ b
# [[2.00, 4.00],
#  [4.00, 3.00],
#  [3.00, 3.67],
#  [3.00, 3.50]]   ogni riga è la media delle righe di b fino a quel punto
```

Si può ottenere la stessa matrice in un altro modo, che è quello che useremo: partire da zeri, mettere $-\infty$ sopra la diagonale, applicare la softmax per riga.

```python
wei = torch.zeros(4, 4)
wei = wei.masked_fill(torch.triu(torch.ones(4, 4), diagonal=1) == 1, float('-inf'))
wei = F.softmax(wei, dim=1)       # identica ad a
```

Perché la strada più lunga? Perché la softmax accetta numeri qualsiasi, non solo zeri: è lì che entreranno i **pesi appresi**.

#### 2.2 Query, key, value

Ogni posizione emette tre vettori, ottenuti con tre proiezioni lineari apprese dello stesso input $X$:

- **query** $Q = XW_Q$: "cosa sto cercando";
- **key** $K = XW_K$: "cosa ho da offrire";
- **value** $V = XW_V$: "cosa comunico se mi ascolti".

(Attenzione: da qui in poi $V$ indica i *value*, non più la dimensione del vocabolario delle Lezioni 2--6.)

L'affinità fra la query di una posizione e la key di un'altra è un prodotto scalare, e diventa il peso con cui la seconda influenza la prima:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

È la stessa struttura della media che scorre, `softmax(qualcosa di mascherato) @ V`, con una differenza: prima i pesi erano zero (softmax uniforme), ora sono $QK^\top$. **Il contenuto decide i pesi.**

#### 2.3 Perché dividere per $\sqrt{d_k}$

Con $q$ e $k$ a varianza 1, il prodotto scalare è una somma di $d_k$ termini e ha varianza $d_k$. Logit con varianza grande fanno **saturare la softmax**: si concentra su un solo elemento e il gradiente verso gli altri si spegne (la stessa patologia della tanh satura, Lezione 4). Dividendo per $\sqrt{d_k}$ la varianza torna circa 1. Esempio su 8 posizioni: senza scala il peso massimo è 0,59; con la scala 0,28, e gli altri pesi restano vivi.

#### 2.4 La maschera causale

$M$ vale 0 dove una posizione può guardare e $-\infty$ dove non può. Un modello che genera testo un carattere alla volta non deve vedere il futuro: $M$ è $-\infty$ sopra la diagonale, la stessa maschera della media che scorre. Questo rende GPT un modello **decoder-only**. Un *encoder* (usato per esempio per classificare o capire un testo già completo, come BERT) omette la maschera: ogni posizione vede tutte le altre.

#### 2.5 Multi-head: più canali di comunicazione

Invece di una testa di attenzione con dimensione $C$, se ne usano $n_h$ più piccole ($d_k = C / n_h$), in parallelo, e si concatenano i risultati. Il numero di parametri non cambia; cambia il numero di "canali" indipendenti. Una testa può occuparsi dei caratteri immediatamente precedenti, un'altra di relazioni più lontane.

### 3. Il codice: il blocco Transformer

#### 3.1 Attenzione causale multi-testa

```python
import torch, torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)    # Q, K, V in un'unica moltiplicazione
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)                 # (B, T, C) ciascuno
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / hs**0.5                  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v                                            # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)     # riunisce le teste
        return self.c_proj(out)
```

`view(...).transpose(1, 2)` porta le teste in una dimensione di batch: il prodotto matriciale si applica identico a ogni testa, in parallelo (la stessa mossa della Lezione 6).

**Verifica della causalità.** Cambiando solo l'ultimo carattere dell'input, i logit di tutte le posizioni precedenti restano **identici** (differenza 0,0 esatta); cambiano solo quelli dell'ultima. E dove la maschera mette $-\infty$, la softmax produce 0 e il suo backward si annulla da solo: non serve trattare la maschera a parte nel gradiente.

#### 3.2 Feed-forward: comunicazione, poi calcolo

L'attenzione fa comunicare le posizioni. Poi ogni posizione deve **elaborare** per conto suo ciò che ha raccolto: un MLP identico per ogni posizione, senza comunicazione.

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)
```

L'espansione $\times 4$ dà spazio per un calcolo più ricco. La **GELU** è una versione "morbida" della ReLU: vale circa $x$ per $x$ grande e positivo, circa 0 per $x$ molto negativo, con una transizione liscia ($\text{GELU}(x) = x\,\Phi(x)$, dove $\Phi$ è la funzione di ripartizione della normale). GPT-2 ne usa un'approssimazione che contiene una tanh, ma la funzione si comporta come una ReLU, non come una tanh.

#### 3.3 Il blocco: due "autostrade" per il gradiente

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # comunicazione
        x = x + self.ffn(self.ln2(x))      # calcolo
        return x
```

Due scelte ormai definitive:

- **Connessioni residue: `x = x + sublayer(x)`.** La derivata di una somma distribuisce il gradiente identico a entrambi gli addendi (Lezione 1). Quindi il gradiente raggiunge ogni blocco anche per la via diretta, senza attraversare attenzione e feed-forward. Senza residui un Transformer di più di qualche strato non si addestrerebbe.
- **LayerNorm, non BatchNorm**, e messa **prima** del sotto-strato (*pre-norm*, da GPT-2 in poi). Il percorso residuo resta una somma pulita, e l'addestramento di reti profonde è più stabile. Il Transformer originale del 2017 normalizzava dopo (*post-norm*).

#### 3.4 Il modello completo

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)                                    # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))     # (T, C)
        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))                        # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss
```

**Perché serve `pos_emb`?** Il prodotto $QK^\top$ e la softmax non sanno nulla dell'ordine: confrontano contenuti, e lo stesso carattere produce la stessa query e la stessa key in qualunque posizione si trovi. (La maschera causale dà solo un'informazione indiretta: quanti token ci sono prima.) Gli **embedding di posizione** aggiungono a ogni token l'informazione "sono in posizione $t$".

Forme sul nostro modello (C = 64, 4 teste, 4 strati, contesto 64): input (32, 64) $\to$ embedding (32, 64, 64) $\to$ dopo i 4 blocchi **(32, 64, 64)** $\to$ logit (32, 64, 56). **Un blocco non cambia la forma**: per questo se ne possono impilare quanti se ne vogliono.

Nessun pezzo è nuovo tranne `CausalSelfAttention`: l'embedding viene dalla Lezione 3, la LayerNorm dalla 4, `Sequential` dalla 6, la cross-entropy dalla 2, la feed-forward è l'MLP delle Lezioni 1 e 3. I residui sono una semplice somma.

### 4. Il corpus: un racconto

L'attenzione ha senso su testo continuo, dove ciò che serve può stare trenta posizioni indietro. Usiamo un racconto italiano originale scritto per il corso, *La lanterna di Pietrarsa* (circa 22 500 caratteri).

```python
text = open('racconto.txt', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)                                   # 56 simboli
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # x spostato di uno
    return x, y
```

### 5. L'addestramento

Configurazione: $n_{embd} = 64$, 4 teste, 4 strati, contesto 64, batch 32: **211 384 parametri**. GPT-2 piccolo ha la stessa architettura con 768 dimensioni, 12 teste, 12 strati, contesto 1 024 e un vocabolario di 50 257 token: circa 590 volte più grande.

#### 5.1 AdamW, in breve

Finora abbiamo usato la discesa del gradiente semplice. Per il Transformer passiamo ad **AdamW**, l'ottimizzatore standard per questi modelli. L'idea: ogni parametro riceve un passo adattato alla propria storia, usando una media mobile del gradiente ($m$, il "momento") e una media mobile del suo quadrato ($v$, la scala tipica); $\hat m$ e $\hat v$ sono le stesse medie, corrette perché all'inizio partono da zero:

$$\theta \leftarrow \theta - \eta\left(\frac{\hat m}{\sqrt{\hat v} + \varepsilon} + \lambda\,\theta\right)$$

Perché conta: nella Lezione 4 abbiamo visto gradienti che differiscono di 100 volte fra gli strati; un passo per parametro compensa. Il termine $\lambda\theta$ è il **weight decay** applicato direttamente ai pesi, separato dal gradiente (da cui la "W" di AdamW): la stessa regolarizzazione L2 della Lezione 2, in una forma che si comporta meglio con la scala adattiva.

```python
model = GPT(vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
for step in range(1500):
    xb, yb = get_batch(train_data, block_size=64, batch_size=32)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

(Nell'esperimento si usa anche un breve riscaldamento del learning rate e poi un decadimento a coseno: li spieghiamo nella Lezione 9.)

#### 5.2 La curva di addestramento

| Passo               | 1    | 100  | 400               | 1000 | 1500 |
| ------------------- | ---- | ---- | ----------------- | ---- | ---- |
| loss di batch       | 4,00 | 2,33 | 1,27              | 0,33 | 0,26 |
| loss di validazione | 3,98 | 2,33 | **1,56** (minimo) | 2,16 | 2,56 |

La loss iniziale (4,00) è vicina a $\log 56 = 4{,}03$: partenza onesta. La validazione scende fino al passo 400, poi **risale** mentre il training continua a scendere: overfitting, come nella Lezione 3. Con 211 000 parametri e 20 000 caratteri, memorizzare è facile.

### 6. Generare testo: due modelli, due storie

```python
@torch.no_grad()
def generate(model, start_text, n_new, temperature=1.0):
    idx = torch.tensor([[stoi[c] for c in start_text]])
    for _ in range(n_new):
        idx_cond = idx[:, -model.block_size:]              # solo gli ultimi block_size caratteri
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return ''.join(itos[i.item()] for i in idx[0])
```

- **Modello fermato al passo 400** (*early stopping*): testo con l'aspetto dell'italiano, parole spesso inventate.
- **Modello al passo 1500** (overfit): frasi più fluide, ma alcune sono **citazioni letterali** del racconto, per esempio "--- So accendere il fuoco, --- disse Elia, quando".

Misurato: la frazione di frammenti di 8 caratteri copiati alla lettera dal testo di addestramento è **13%** per il modello fermato in tempo e **64%** per quello in overfit. Il secondo non genera più: recita a memoria.

### 7. Dentro l'attenzione

Guardando i pesi di attenzione del primo strato sulla frase "la lanterna non si spegne mai":

- la struttura è esattamente triangolare: i pesi sopra la diagonale sommano a 0,0;
- teste diverse hanno imparato comportamenti diversi: una guarda quasi solo i caratteri immediatamente precedenti (una specie di bigramma appreso), altre distribuiscono il peso più indietro;
- sulla "i" finale di "mai", circa l'82% dell'attenzione va allo spazio prima di "mai", alla "m", alla "a" e alla "i" stessa. **Senza che nessuno gli abbia detto cosa sia una parola**, il modello si concentra sulla parola che sta completando.

### 8. Esercizi

1. **Encoder contro decoder.** Togli la maschera. Su che tipo di compito avrebbe senso? Perché per generare testo sarebbe un errore che rende la loss bassa ma finta?
2. **Senza posizione.** Elimina `pos_emb` e riaddestra. La loss peggiora poco o tanto? Perché?
3. **Predici, poi misura.** Con un racconto dieci volte più lungo, il minimo della validazione arriverà prima o dopo il passo 400? Il divario train--validazione al passo 1500 sarà più grande o più piccolo? Scrivi la previsione, poi verifica.

### 9. Verso gli LLM

- **Questo è GPT-2 in miniatura**, con gli stessi nomi degli iperparametri (`n_embd`, `n_head`, `n_layer`, `block_size`). Cambiando quei numeri, non la struttura, si arriva a GPT-2 piccolo (124M) e a GPT-2 XL (1,5 miliardi).
- **Il costo nascosto: $O(T^2)$.** Ogni testa calcola una matrice $T \times T$: raddoppiare il contesto quadruplica quel lavoro. Da qui nascono ottimizzazioni come FlashAttention (Lezione 9).
- **La KV-cache.** Il nostro `generate` ricalcola tutto il contesto a ogni nuovo carattere. I sistemi veri tengono in memoria le key e i value già calcolati e, a ogni passo, calcolano solo quelli del **nuovo token**: stessa matematica, molto meno lavoro.
- **Cosa manca:** un tokenizer vero (Lezione 8), la scala (Lezione 9), e la capacità di rispondere invece di continuare (Lezione 10).

> **[Nuovo] Aggiornamento.** I modelli recenti hanno cambiato alcuni pezzi di questo blocco, lasciando intatta l'idea: posizioni codificate con **RoPE** invece di `pos_emb`, **RMSNorm** invece di LayerNorm, MLP di tipo **SwiGLU**, attenzione con key/value condivise fra teste (**GQA/MQA**) per ridurre la KV-cache. Dettagli nell'[Appendice D](#appendice-d-----cosa-è-cambiato-dopo-gpt-2-2019--2026).

### Riepilogo

1. L'attenzione nasce da una media che scorre; renderla appresa significa sostituire pesi uniformi con $\text{softmax}(QK^\top)$.
2. Q, K, V sono tre proiezioni lineari apprese; la scala $1/\sqrt{d_k}$ evita che la softmax saturi.
3. La maschera causale ($-\infty$ sul futuro) rende GPT autoregressivo.
4. Il multi-head divide lo stesso budget in più canali paralleli, a parità di parametri.
5. Un blocco è comunicazione (attenzione) più calcolo (feed-forward), ciascuno con LayerNorm e residuo: `x = x + sublayer(ln(x))`.
6. Con 211 000 parametri su 20 000 caratteri il modello va in overfitting, e lo si vede nel testo: 13% contro 64% di frammenti copiati.
7. L'unica idea davvero nuova è l'attenzione; tutto il resto viene dalle lezioni precedenti.

---

## Lezione 8 --- Il tokenizer: BPE da zero

*Lettura: circa 30 minuti.* Costruiamo il pezzo che trasforma il testo in interi, e scopriamo che molte stranezze degli LLM nascono qui.

### 1. Il problema

- **A carattere** le sequenze sono lunghe: "lanterna" occupa 8 posizioni. Con un costo dell'attenzione che cresce come $T^2$, sequenze due volte più lunghe costano quattro volte tanto. Nessun GPT vero tokenizza a carattere.
- **A parola** il vocabolario esplode (flessioni, errori, nomi propri) e ogni parola mai vista diventa un buco.

Serve una via di mezzo: unità più grandi di un carattere e più piccole di una parola, **imparate dai dati**. L'algoritmo è il **Byte Pair Encoding (BPE)**.

### 2. L'idea

**Unicode non basta.** `ord(c)` dà un numero a ogni carattere, ma i caratteri sono oltre 150 000 e lo standard continua a crescere: una tabella di embedding enorme e instabile.

**UTF-8 sì.** Rappresenta ogni carattere come 1--4 byte, ciascuno fra 0 e 255. Qualunque testo, in qualunque lingua, diventa una sequenza in un **alfabeto fisso di 256 simboli**. Le lettere accentate italiane usano 2 byte (`è` $\to$ `[195, 168]`), la lineetta (U+2014) ne usa 3, un ideogramma cinese 3, un'emoji fino a 4.

**Ma i byte da soli allungano le sequenze.** Serve comprimere: trovare le coppie di simboli più frequenti e fonderle in un simbolo nuovo, ripetutamente. È un algoritmo di compressione del 1994 (Philip Gage), adattato alla tokenizzazione da Sennrich e colleghi nel 2015 e diventato lo standard con GPT-2.

**L'algoritmo in tre frasi:**

1. Parti dai byte UTF-8: vocabolario iniziale di 256 simboli.
2. Conta tutte le coppie adiacenti, prendi la più frequente, sostituiscila con un simbolo nuovo (256, poi 257, ...).
3. Ripeti sul testo aggiornato tante volte quanti merge vuoi. Vocabolario finale: 256 + numero di merge.

Esempio su `"abababcababa"`: la coppia più frequente è `ab` (5 volte); diventa il simbolo 256 e la sequenza passa da 12 a 7 simboli: `[256, 256, 256, 99, 256, 256, 97]` (99 è `c`, 97 è `a`). Il giro successivo lavora sui nuovi simboli (per esempio `256 256` = `abab`): così nascono token lunghi più lettere.

### 3. Il codice

```python
def get_stats(ids, counts=None):
    """Conta le coppie di simboli adiacenti."""
    counts = {} if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Sostituisce ogni occorrenza non sovrapposta di pair con new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(new_id); i += 2       # salta entrambi: niente sovrapposizioni
        else:
            out.append(ids[i]); i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode('utf-8'))
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)     # la coppia più frequente
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id                # l'ordine di apprendimento conta
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]   # i byte che il token rappresenta
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode('utf-8'))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))   # il merge imparato PRIMA
        if pair not in merges:
            break                            # nessuna coppia rimasta è fondibile
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')
```

> **[Attenzione] Il dettaglio facile da sbagliare.** `encode` deve applicare i merge **nell'ordine in cui sono stati imparati**, non scegliere la coppia più frequente nel testo nuovo. Altrimenti produce una tokenizzazione diversa da quella su cui il modello è stato addestrato.

#### 3.1 Sul nostro racconto

300 merge (vocabolario 556): da 22 847 byte a **8 459 token**, cioè 2,7 byte per token.

- I primi merge sono "vocale + spazio" (`a `, `e `, `o `, `i `): l'italiano finisce quasi ogni parola in vocale. L'algoritmo non sa nulla di grammatica, sa solo contare. <!-- markdownlint-disable-line MD038 -->
- Più avanti nascono token lunghi legati al racconto: `Pietrarsa`, `lanterna`, `\u2014 disse` (la lineetta del dialogo seguita da `disse`), perfino `.\n\nCapitolo ` (12 byte). <!-- markdownlint-disable-line MD038 -->
- Su una frase nuova, "La lanterna di Pietrarsa non si è mai spenta, nemmeno una notte.", si ottengono 19 token, e `decode(encode(s)) == s`.

**Token speciali.** I tokenizer veri riservano id per marcatori che non vengono dal testo, come `<|endoftext|>` (fine di un documento e inizio del successivo). È la stessa idea del `.` della Lezione 2, applicata a documenti interi. Si gestiscono a parte: si divide il testo sul marcatore, si codificano i pezzi e si inserisce l'id riservato.

### 4. I rendimenti decrescenti

| Vocabolario | Merge | Token nel racconto | Byte/token |
| ----------- | ----- | ------------------ | ---------- |
| 256         | 0     | 22 847             | 1,00       |
| 356         | 100   | 12 031             | 1,90       |
| 556         | 300   | 8 459              | 2,70       |
| 806         | 550   | 6 725              | 3,40       |
| 1 056       | 800   | 5 768              | 3,96       |

I primi 100 merge quasi raddoppiano la compressione; i successivi rincorrono schemi sempre più rari. GPT-2 si ferma a 50 257 token, un compromesso fra sequenze corte e una tabella di embedding non enorme.

### 5. Le stranezze che il tokenizer spiega

- **Contare le lettere.** `"Pietrarsa "` è **un solo token**. Chiedere al modello quante lettere contiene significa chiedergli di guardare dentro un simbolo che vede come un intero. Non è che "non sa contare": non vede le lettere come noi.
- **L'aritmetica.** I numeri vengono spezzati in base alla frequenza, non al valore posizionale: "127" può essere un token unico e "128" due. Il modello vede frammenti di lunghezza variabile, non colonne di unità, decine e centinaia. (Per questo diversi tokenizer recenti dividono i numeri in cifre singole o in gruppi di al massimo tre cifre.)
- **Spazi e maiuscole.** `"lanterna"`, `" lanterna"` e `"Lanterna"` hanno tokenizzazioni completamente diverse: per il modello sono quasi tre parole distinte, e ciò che impara su una si trasferisce solo in parte alle altre. Da qui la sensibilità agli spazi nei prompt.
- **L'italiano paga di più.** Il nostro tokenizer, addestrato sul racconto, comprime 2,92 byte/token su testo simile al racconto, 1,72 su italiano tecnico, 1,26 su inglese. Specularmente, i tokenizer dei grandi modelli sono addestrati su corpus a maggioranza inglese: la stessa frase in italiano occupa più token della sua traduzione. Più token significa meno testo nella stessa finestra di contesto e un costo più alto.

### 6. Esercizi

1. **Il bug del rank.** Modifica `encode` per scegliere la coppia più frequente invece di quella imparata per prima. Trova un testo su cui le due versioni producono token diversi. Perché la versione sbagliata decodifica comunque bene, ma è incompatibile con il modello?
2. **Input ostili.** Verifica `decode(encode(x)) == x` con emoji, caratteri cinesi e stringa vuota. Costruisci una sequenza di id che costringe `decode` a produrre il carattere di sostituzione U+FFFD, e spiega perché può succedere.
3. **Il costo dell'italiano.** Con la libreria `tiktoken`, confronta i token di una frase italiana e della sua traduzione inglese con `cl100k_base` (GPT-4) e `o200k_base` (GPT-4o). Il divario è più piccolo o più grande di quello del nostro tokenizer?

### 7. Verso gli LLM

- **Il nostro BPE è lo stesso di GPT-2, tranne un dettaglio.** GPT-2 prima divide il testo con un'espressione regolare (lettere, numeri, punteggiatura, spazi separati) e applica i merge solo dentro i pezzi: così non nascono token come `cane.` che mescolano parola e punteggiatura. La libreria vera di OpenAI è `tiktoken`.
- **SentencePiece**, usata da Llama 1 e 2, dalle prime versioni di Mistral, da Gemma e da altri, tratta il testo grezzo (spazi inclusi) senza regole di pre-divisione specifiche per una lingua.
- **La dimensione del vocabolario è un compromesso.** Vocabolario grande $\to$ sequenze più corte, ma tabella di embedding e strato di output più grandi. Vocabolario piccolo $\to$ l'opposto.

> **[Nuovo] Aggiornamento.** La tendenza recente è verso vocabolari più grandi e più multilingue: circa 100 000 token per GPT-4 (`cl100k_base`), circa 200 000 per GPT-4o (`o200k_base`), 128 000 per Llama 3 (che è passato da SentencePiece a un BPE basato su `tiktoken`), 256 000 per Gemma. Vocabolari più grandi riducono il divario di costo per lingue come l'italiano. In ricerca si studiano anche modelli che lavorano direttamente sui byte, senza tokenizer fisso (per esempio la *Byte Latent Transformer* di Meta, 2024), ma non sono ancora lo standard.

### Riepilogo

1. Nessun LLM vero tokenizza a carattere: sequenze troppo lunghe per un'attenzione che costa $O(T^2)$.
2. Il testo diventa byte UTF-8 (256 simboli, qualunque lingua); il BPE fonde ripetutamente la coppia più frequente.
3. `encode` riapplica i merge nell'ordine di apprendimento; `decode` concatena i byte e decodifica.
4. Più merge, più compressione, con rendimenti decrescenti.
5. Contare lettere, sbagliare l'aritmetica, sensibilità agli spazi: sono effetti del tokenizer.
6. Un tokenizer comprime bene ciò che somiglia al suo corpus: le lingue meno rappresentate pagano più token.

---

# PARTE IV --- DALLA TEORIA AL MODELLO REALE

## Lezione 9 --- Riprodurre GPT-2 (124M)

*Lettura: circa 35 minuti.* Nessuna nuova idea di machine learning: l'architettura è quella della Lezione 7. Cambia tutto ciò che sta intorno: come inizializzarla, come farla girare in fretta, come addestrarla bene, con quali dati, e come giudicarla.

> **Nota.** Questa lezione richiede una GPU per essere eseguita davvero. Dove non è possibile verificare in locale, riportiamo i valori pubblici della riproduzione di Karpathy (repository `build-nanogpt` e `llm.c`).

### 1. Il problema

Il nostro GPT ha 211 000 parametri e si addestra in pochi minuti su CPU. GPT-2 (124M) ne ha **124 439 808**, circa 590 volte di più, ed è addestrato su miliardi di token. Stessa architettura. Cosa cambia in mezzo? Quasi solo ingegneria, in quattro gruppi:

| Gruppo              | Tecniche                                                             |
| ------------------- | -------------------------------------------------------------------- |
| Ritocchi al modello | pesi condivisi, inizializzazione scalata                             |
| Velocità            | precisione mista, `torch.compile`, FlashAttention                    |
| Addestramento       | gradient accumulation, schedule del learning rate, gradient clipping |
| Dati e valutazione  | FineWeb, HellaSwag                                                   |

### 2. I ritocchi al modello

#### 2.1 Pesi condivisi (*weight tying*)

La tabella degli embedding (`wte`: token $\to$ vettore) e l'ultimo strato (`lm_head`: vettore $\to$ logit sui token) fanno lavori speculari. GPT-2 usa **la stessa matrice** per entrambi (idea di Press e Wolf, 2016). Oltre a risparmiare parametri, costringe le due mappe a condividere la stessa geometria.

```python
model.lm_head.weight = model.tok_emb.weight    # stesso tensore, non una copia
```

Su GPT-2 124M (768 dimensioni, 12 teste, 12 strati, vocabolario 50 257, contesto 1 024, `lm_head` senza bias): senza tying 163 037 184 parametri, con tying **124 439 808**, esattamente il "124M" del nome. Il risparmio è di 38 597 376 parametri: la sola tabella `wte` vale circa il 31% del modello. (Il GPT della Lezione 7 ha invece un bias in `lm_head`, che il tying non elimina; per GPT-2 si usa `nn.Linear(n_embd, vocab_size, bias=False)`.)

#### 2.2 Inizializzazione scalata del flusso residuo

Ogni blocco scrive due volte nel flusso residuo (`x = x + attn(...)` e `x = x + ffn(...)`): con $L$ strati, $2L$ contributi che si sommano. Se i contributi sono indipendenti e ciascuno ha varianza circa 1, la somma ha varianza circa $2L$: a 12 strati la deviazione standard è $\sqrt{24} \approx 4{,}9$. Il flusso cresce con la profondità.

La cura è la logica di Kaiming applicata al flusso residuo: scalare ogni contributo di $1/\sqrt{2L}$, così la somma torna a varianza circa 1. Una simulazione a 12 strati lo conferma: deviazione standard 4,90 senza scala, 1,00 con la scala. In codice si applica solo alle proiezioni che scrivono nel residuo:

```python
import math
std = 0.02 / math.sqrt(2 * n_layer)
# vale per c_proj dell'attenzione e per l'ultimo Linear della feed-forward (net[2] nella Lezione 7);
# tutti gli altri pesi usano std = 0.02
nn.init.normal_(module.c_proj.weight, mean=0.0, std=std)
```

### 3. Farlo girare in fretta

#### 3.1 Precisione mista: bf16

Finora tutti i numeri erano **float32** (23 bit di mantissa). Il **bfloat16** usa 16 bit in tutto: lo stesso esponente di float32 (stesso intervallo, niente overflow a sorpresa) ma solo 7 bit di mantissa. Sulle GPU NVIDIA da Ampere in poi i prodotti matriciali in bf16 sono molto più veloci.

Quanta precisione si perde? Sui singoli pesi poco: l'errore relativo di arrotondamento è al massimo circa 0,4% (e in media più piccolo). Il problema è **l'accumulo**: sommando 100 000 volte 0,0001 in float32 si ottiene 10,0; facendolo in bf16 la somma a un certo punto **smette di crescere**, perché l'incremento diventa più piccolo della risoluzione di bf16 attorno al valore corrente.

Per questo si parla di precisione **mista**: i calcoli pesanti (prodotti matriciali) in bf16, mentre i pesi "maestri", gli stati dell'ottimizzatore e alcune operazioni delicate (riduzioni, softmax, loss) restano in float32. In PyTorch: `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.

#### 3.2 `torch.compile`

Analizza il grafo del modello e fa **kernel fusion**: invece di eseguire una catena di operazioni elementari (per esempio le varie operazioni dentro la GELU), ciascuna delle quali legge e scrive la memoria della GPU, le fonde in un solo kernel che legge una volta e scrive una volta. È la stessa idea della Lezione 1 (un nodo `tanh` invece di tanti nodi elementari) e della Lezione 5 (`F.cross_entropy`), applicata automaticamente a tutto il modello.

#### 3.3 FlashAttention

La nostra attenzione scrive in memoria l'intera matrice `att` di forma $(B, n_h, T, T)$, ne calcola la softmax e la rilegge. Per contesti lunghi la matrice è enorme. **FlashAttention** (Dao e colleghi, 2022) calcola **lo stesso identico risultato** (non è un'approssimazione) senza mai scrivere la matrice intera: la divide in blocchi che stanno nella memoria veloce del chip e combina i risultati parziali con una "softmax online". Meno traffico di memoria, non meno calcolo. In PyTorch:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # sostituisce 4 righe della Lezione 7
```

### 4. Addestrarlo bene

#### 4.1 Gradient accumulation

La riproduzione di GPT-2 usa batch di **524 288 token** per passo ($2^{19}$). Non stanno in memoria tutti insieme. Soluzione: dividere il batch in micro-batch, fare forward e backward su ciascuno **accumulando** i gradienti, e aggiornare i pesi una volta sola.

```python
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    xb, yb = get_batch(...)
    logits, loss = model(xb, yb)
    loss = loss / grad_accum_steps     # la loss è una media: va divisa
    loss.backward()                    # i .grad si accumulano, come nella Lezione 1
optimizer.step()
```

Verificato sul GPT della Lezione 7: il gradiente accumulato su 4 micro-batch da 8 coincide con quello di un batch da 32 a meno degli arrotondamenti (differenza circa $10^{-17}$ in doppia precisione). È il `+=` della Lezione 1, usato di proposito.

#### 4.2 Warmup e decadimento a coseno

- **Warmup.** Nei primi passi i pesi sono ancora casuali e le medie mobili di AdamW si basano su pochissimi campioni: un learning rate pieno rischia un passo enorme nella direzione sbagliata. Si sale linearmente da quasi zero al massimo in qualche centinaio di passi.
- **Decadimento a coseno.** Poi il learning rate scende seguendo mezza onda di coseno fino a un minimo (tipicamente il 10% del massimo): una discesa liscia, senza i salti a gradini usati nelle Lezioni 2 e 3.

```python
def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

Valori della riproduzione: $\eta_{max} = 6 \cdot 10^{-4}$, 715 passi di warmup, 19 073 passi totali. I numeri si incastrano: $19\,073 \times 524\,288 \approx 10$ miliardi di token, la dimensione del dataset usato.

#### 4.3 Gradient clipping

Ogni tanto un batch produce un gradiente enorme, e un singolo passo enorme può rovinare ore di addestramento. Rimedio: calcolare la norma di tutto il gradiente (tutti i parametri come un unico vettore) e, se supera una soglia, riscalarlo.

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Un gradiente sano (norma 0,9) non viene toccato. Uno con un elemento esploso a 850 viene moltiplicato per circa 0,0012 e torna a norma 1. Il prezzo: il fattore è lo stesso per tutti i parametri, quindi anche i gradienti sani vengono schiacciati. Tenere d'occhio la norma restituita è anche un ottimo strumento diagnostico: picchi improvvisi segnalano instabilità.

### 5. Nutrirlo: FineWeb

Il nostro racconto non ha duplicati né testo di bassa qualità. Un corpus vero è l'opposto. **FineWeb** (Hugging Face, 2024) contiene circa **15 000 miliardi di token** estratti da 96 raccolte di Common Crawl (2013--2024), con due passaggi fondamentali:

- **Deduplicazione**: il web si ripete moltissimo; senza togliere i duplicati il modello spreca capacità a memorizzare copie.
- **Filtri di qualità**: euristiche e classificatori rimuovono spam, boilerplate, testo degenere.

**FineWeb-Edu** è un sottoinsieme di circa 1 300 miliardi di token, filtrato con un classificatore addestrato su giudizi di "valore educativo" generati da Llama-3-70B-Instruct. Pur essendo dieci volte più piccolo, un modello addestrato su FineWeb-Edu batte uno addestrato su FineWeb completo nei test di conoscenza e ragionamento: **oltre una certa soglia, la qualità conta più della quantità**.

La riproduzione di Karpathy usa un campione di **10 miliardi di token di FineWeb-Edu** (`sample-10BT`).

### 6. Giudicarlo: HellaSwag

"Loss di validazione 3,2" è difficile da interpretare. **HellaSwag** dà un numero più leggibile: dato un contesto, il modello deve scegliere fra quattro continuazioni quella sensata (le altre tre sono costruite per sembrare plausibili in superficie). Non serve addestramento speciale: si calcola la cross-entropy media di ciascuna continuazione e si sceglie quella con loss più bassa. A caso si ottiene il 25%.

| Modello                                               | Accuratezza HellaSwag        |
| ----------------------------------------------------- | ---------------------------- |
| Caso (4 scelte)                                       | 25,0%                        |
| GPT-2 124M originale (OpenAI, 2019)                   | $\approx$ 29,4%              |
| Riproduzione 124M su 10B token FineWeb-Edu (Karpathy) | $\approx$ 30% (supera GPT-2) |
| GPT-3 Small 125M (paper di GPT-3, 300B token)         | 33,7%                        |
| GPT-2 XL 1,5B                                         | $\approx$ 48,9%              |

GPT-3 Small, praticamente la stessa architettura e taglia di GPT-2 124M, ottiene circa 4 punti in più grazie alla ricetta di addestramento (molti più token, batch più grandi). E la riproduzione supera GPT-2 addestrandosi su soli 10 miliardi di token, ma di qualità alta (FineWeb-Edu). **La ricetta e i dati contano quanto l'architettura.**

HellaSwag oggi è quasi saturo per i modelli di frontiera (oltre il 95%), ma per un modello della taglia di GPT-2 resta un segnale pulito durante l'addestramento.

### 7. Le leggi di scala

> **[Nuovo] Integrazione.** Il corso originale rimanda più volte alle "leggi di scala della Lezione 9" senza poi trattarle. Ecco l'essenziale.

La loss di un modello di linguaggio scende in modo **prevedibile** (come una legge di potenza) aumentando tre cose: **parametri** $N$, **token di addestramento** $D$, **calcolo** $C$ (approssimativamente $C \approx 6ND$ operazioni). Due lavori fondamentali:

- **Kaplan e colleghi (OpenAI, 2020)**: la loss migliora con regolarità su molti ordini di grandezza; la forma dell'architettura conta poco rispetto alla scala.
- **Chinchilla (Hoffmann e colleghi, DeepMind, 2022)**: a parità di calcolo conviene crescere parametri e dati **insieme**. Il rapporto ottimale è circa **20 token per parametro**. Molti modelli dell'epoca (incluso GPT-3, 175 miliardi di parametri su 300 miliardi di token) erano "sotto-addestrati".

Nella pratica recente si va **ben oltre** i 20 token per parametro (Llama 3 8B è stato addestrato su circa 15 000 miliardi di token, quasi 2 000 per parametro): a parità di qualità, un modello più piccolo addestrato più a lungo richiede più calcolo in addestramento rispetto all'ottimo di Chinchilla, ma costa molto meno ogni volta che viene usato.

È la versione formale della morale della Lezione 6: **i guadagni di architettura si pagano con i dati**.

### 8. Esercizi

1. **Il tying al variare del vocabolario.** Calcola la percentuale di parametri risparmiata con vocabolari di 1 000, 10 000 e 50 257 token, tenendo fisso il resto di GPT-2. Cresce o cala? Perché?
2. **Il muro di bf16.** Ripeti la somma di 100 000 incrementi con incrementi di 0,01, 0,001, 0,0001. A che valore la somma smette di crescere in ciascun caso? Perché dipende dal rapporto fra incremento e somma corrente?
3. **Lo schedule al contrario.** Scrivi una funzione che, dati token totali e token per batch, calcoli `max_steps` e un `warmup_steps` ragionevole (1--2% del totale). Applicala al racconto della Lezione 7: ha senso uno schedule così elaborato per 22 000 caratteri?

### 9. Verso gli LLM

Ognuna di queste tecniche esiste per un motivo misurabile: senza tying un terzo dei parametri sarebbe ridondante; senza la scala $1/\sqrt{2L}$ il segnale cresce con la profondità; senza gradient accumulation i batch da mezzo milione di token sarebbero irraggiungibili; senza warmup Adam parte con stime rumorose.

Cosa resta: abbiamo un modello che **continua** il testo. Non risponde a domande, non segue istruzioni, non rifiuta richieste pericolose: ha solo imparato a prevedere il token successivo di internet. È l'argomento della Lezione 10.

> **[Nuovo] Per proseguire.** Nell'ottobre 2025 Karpathy ha pubblicato **nanochat**, il seguito naturale di questa lezione: un unico repository che copre tokenizer, pre-training, fine-tuning, una fase di reinforcement learning, inferenza con KV-cache e un'interfaccia di chat, addestrabile in poche ore su un nodo con 8 GPU. Usa un'architettura aggiornata (RoPE, RMSNorm senza parametri, normalizzazione di query e key, MLP con ReLU$^2$, embedding e `lm_head` non condivisi) e l'ottimizzatore **Muon** insieme ad AdamW. Vedi l'[Appendice D](#appendice-d-----cosa-è-cambiato-dopo-gpt-2-2019--2026).

### Riepilogo

1. Fra il GPT giocattolo e GPT-2 124M l'architettura non cambia: cambiano inizializzazione, precisione, dati, valutazione.
2. Il weight tying condivide `wte` e `lm_head`: su GPT-2 risparmia circa 38,6 milioni di parametri.
3. Scalare l'init delle proiezioni residue di $1/\sqrt{2L}$ tiene la varianza del flusso residuo vicina a 1.
4. bf16 accelera i calcoli ma può annullare piccoli incrementi accumulati: pesi maestri e ottimizzatore restano in float32.
5. Gradient accumulation dà lo stesso gradiente di un batch grande; warmup e coseno stabilizzano; il clipping protegge dai picchi.
6. FineWeb e FineWeb-Edu mostrano che deduplicazione e qualità contano quanto la quantità.
7. Le leggi di scala dicono come crescere insieme parametri, dati e calcolo.

---

## Lezione 10 --- Da modello base ad assistente

*Lettura: circa 30 minuti.* Nessuna nuova architettura: il modello resta quello della Lezione 7. Cambiano i dati e l'obiettivo dell'addestramento. È la differenza fra un modello che completa internet e un assistente che risponde.

### 1. Il problema, visto prima di spiegarlo

Diamo al nostro GPT della Lezione 7 un prompt in forma di domanda:

```text
PROMPT:        "Domanda: dove si trova Pietrarsa? Risposta:"
CONTINUAZIONE: " racco quella lare un accendo la fiama, con le ragazio..."
```

Il modello continua a scrivere narrativa. Non ha rifiutato di rispondere: **non sa nemmeno che gli è stata fatta una domanda**. Fa ciò per cui è stato addestrato, prevedere il carattere più plausibile nello stile di ciò che ha visto, e in 22 000 caratteri di narrativa non c'è nemmeno un esempio di domanda e risposta.

Il nostro modello è troppo piccolo per dire qualcosa sulla qualità, ma il fenomeno è identico a qualunque scala: un **modello base** è un modello statistico del testo su cui è stato addestrato, non di "come rispondere a una persona". La differenza fra GPT-2 e ChatGPT non è nell'architettura: è tutta in ciò che viene **dopo** il pre-training.

### 2. Due obiettivi diversi

Il **pre-training** (Lezioni 2--9) ottimizza una cosa sola: la probabilità del token successivo su un corpus enorme e non etichettato. Le etichette sono gratis (sono il testo stesso), ma l'obiettivo non sa nulla di "utile", "onesto" o "conversazione": se il prompt somiglia a un forum sconclusionato, il modello continua come un forum sconclusionato.

Il **post-training** è l'insieme di fasi che trasformano il modello base in assistente. Sorpresa: cambia pochissimo rispetto a quanto già costruito.

### 3. SFT: la stessa loss, dati diversi

Il **supervised fine-tuning** riprende il modello base e continua ad addestrarlo con la **stessa cross-entropy**, ma su un corpus diverso: decine o centinaia di migliaia di esempi curati di (istruzione, risposta ideale), scritti da persone o da altri modelli.

I turni della conversazione sono marcati con **token speciali** (la stessa idea di `<|endoftext|>`, Lezione 8):

```text
<|system|> Sei un assistente utile e onesto.
<|user|> Dove si trova Pietrarsa?
<|assistant|> Pietrarsa è un paese di montagna immaginario, ambientato in un racconto...<|end|>
```

**La loss si calcola solo sui token dell'assistente**, non sul prompt: altrimenti il modello sprecherebbe capacità a imparare a generare domande. Si usa una maschera sulla loss, concettualmente simile alla maschera causale:

```python
loss_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float)   # 0 = prompt, 1 = risposta
loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()
```

Verifica: cambiando un logit su un token del prompt la loss resta identica; cambiandolo su un token della risposta, cambia.

Dopo l'SFT il modello sa **imitare** la forma di una buona risposta. Ma per molte richieste non esiste un'unica risposta giusta da copiare: una spiegazione può essere più o meno chiara. L'SFT insegna a imitare, non a distinguere il meglio dal peggio.

### 4. RLHF: imparare dalle preferenze

**Reinforcement Learning from Human Feedback** parte da un'osservazione pratica: per una persona è più facile **confrontare** due risposte che scriverne una perfetta. Tre fasi:

1. **Raccogliere preferenze.** Per lo stesso prompt $x$ si generano più risposte; una persona indica quale preferisce. Si ottengono terne $(x, y_w, y_l)$: prompt, risposta vincente, risposta perdente.

2. **Addestrare un reward model**, un modello **separato** (di solito inizializzato dal modello SFT) in cui l'ultimo strato produce un solo numero, il punteggio di qualità $r(x, y)$, invece della softmax sul vocabolario. Si usa il modello di **Bradley--Terry**:

   $$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad L_{RM} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

   dove $\sigma$ è la sigmoide. Esempio: con $r(y_w) = 2{,}3$ e $r(y_l) = 0{,}4$, $P = \sigma(1{,}9) = 0{,}87$ e $L_{RM} = 0{,}14$ (il reward model giudica bene). Con i punteggi invertiti $L_{RM} = 2{,}04$, quasi quindici volte di più. È una cross-entropy binaria: sigmoide e logaritmo, nessuna idea nuova.

3. **Ottimizzare il modello contro il reward model, con PPO.** Qui l'addestramento smette di essere un semplice backward. Il modello genera risposte **campionando** token (`torch.multinomial`, Lezione 2), e il campionamento non è derivabile. Serve il reinforcement learning: **PPO** (*Proximal Policy Optimization*) stima come cambiare i pesi per rendere più probabili le risposte premiate. Si aggiunge una penalità (**divergenza KL**) che impedisce al modello di allontanarsi troppo dal modello SFT, per evitare che impari a "ingannare" il reward model con risposte degeneri ma ben valutate (*reward hacking*).

### 5. DPO: saltare la parte complicata

**Direct Preference Optimization** (Rafailov e colleghi, 2023) mostra che, con lo stesso modello di Bradley--Terry, il problema RLHF ha una soluzione in forma chiusa. Sostituendola, il reward model sparisce e resta una loss calcolabile direttamente sulle probabilità del modello:

$$L_{DPO} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

- $\pi_\theta$: il modello che stiamo addestrando;
- $\pi_{ref}$: una copia congelata del modello SFT, il punto di partenza;
- $\beta$: quanto pesa il vincolo di non allontanarsi dal riferimento.

Niente reward model separato, niente campionamento durante l'addestramento, niente PPO: un forward e un backward, come ogni altra loss del corso.

**Esempio numerico** (log-probabilità sommate sui token della risposta, $\beta = 0{,}1$):

|                   | $\log \pi_\theta$ | $\log \pi_{ref}$ | "reward implicita" $\beta(\log\pi_\theta - \log\pi_{ref})$ |
| ----------------- | ----------------- | ---------------- | ---------------------------------------------------------- |
| Risposta vincente | $-18{,}2$         | $-19{,}5$        | $+0{,}13$                                                  |
| Risposta perdente | $-25{,}1$         | $-20{,}3$        | $-0{,}48$                                                  |

Il modello ha reso la risposta buona più probabile del riferimento e quella cattiva meno probabile: la direzione giusta. Loss: $-\log\sigma(0{,}13 + 0{,}48) = 0{,}434$. Le derivate rispetto alle due log-probabilità valgono $-0{,}035$ e $+0{,}035$: uguali e opposte. La loss spinge su la vincente e giù la perdente con lo stesso peso.

### 6. Perché un modello allucina

La previsione del token successivo non contiene alcun incentivo a dire "non lo so": il modello produce sempre una distribuzione sul token successivo, e la continuazione più plausibile nello stile di una risposta, vera o no. Se i dati di post-training non contengono abbastanza esempi di incertezza espressa onestamente, il modello non impara quel comportamento.

> **[Nuovo] Aggiornamento (2025).** Uno studio di OpenAI (*Why Language Models Hallucinate*, Kalai e colleghi, settembre 2025) aggiunge un punto importante: **anche il modo in cui si valutano i modelli incentiva le allucinazioni**. La maggior parte dei benchmark dà punti per la risposta giusta e zero sia per quella sbagliata sia per "non lo so". Come uno studente a un test a crocette, al modello conviene tirare a indovinare. Gli autori propongono di cambiare i criteri dei benchmark esistenti in modo da penalizzare gli errori sicuri più dell'astensione. Nel loro esempio, un modello che si astiene spesso ha accuratezza simile a uno che non si astiene quasi mai, ma un tasso di errore molto più basso.

### 7. Strumenti e ragionamento

**Uso di strumenti** (ricerca web, esecuzione di codice, calcolatrice): nessun meccanismo nuovo. Con esempi di SFT si insegna al modello a emettere una sequenza di token che rappresenta una "chiamata" a uno strumento. Il sistema attorno al modello intercetta la chiamata, esegue l'azione e reinserisce il risultato nel contesto. Il modello resta un predittore di token; è il protocollo intorno a cambiare.

**Modelli di ragionamento.** Sono modelli che, prima della risposta finale, producono una lunga catena di passaggi intermedi.

> **[Nuovo] Aggiornamento (2024--2026).** Questa è stata la novità più importante dopo la scrittura del corso.
>
> - **o1 (OpenAI, settembre 2024)** ha mostrato che si può migliorare la qualità facendo "pensare" il modello più a lungo al momento della risposta (*test-time compute*), invece che solo ingrandendolo.
> - **DeepSeek-R1 (gennaio 2025)** ha reso pubblica una ricetta: reinforcement learning con **reward verificabili** (*RLVR*). Invece di un reward model appreso dalle preferenze, la reward è una regola: la risposta matematica coincide con quella corretta? il codice passa i test? C'è anche un piccolo premio per il formato (ragionamento fra tag `<think>`, risposta separata). Il ragionamento lungo, le verifiche e le correzioni "emergono" dall'addestramento senza essere scritti a mano.
> - L'algoritmo usato è **GRPO** (*Group Relative Policy Optimization*, introdotto con DeepSeekMath, 2024): per ogni prompt si generano più risposte, e ciascuna viene premiata o penalizzata **rispetto alla media del suo gruppo**. Così non serve il secondo modello (il *critic*) che PPO usa per stimare il valore atteso: meno memoria, più semplicità.
> - La pipeline di R1 alterna fasi: un piccolo SFT iniziale per rendere il ragionamento leggibile, RL sul ragionamento, un nuovo SFT su risposte selezionate, un RL finale. I modelli piccoli si ottengono poi per **distillazione**: si addestrano con SFT sulle catene di ragionamento del modello grande.
>
> Concettualmente è lo stesso RLHF di questa lezione, con la reward umana sostituita da una reward automatica e verificabile.

### 8. Il quadro d'insieme

In pseudocodice (le funzioni `train`, `clone`, `ppo_train`, `grpo_train` e `frozen_copy` non esistono: indicano le fasi):

```python
# Fase 0 -- pre-training (Lezioni 2-9): previsione del token successivo su testo grezzo
model = GPT(...)
train(model, dataset=fineweb, loss=cross_entropy)                 # -> modello base

# Fase 1 -- SFT: stessa loss, dati curati, mascherata sulle risposte
train(model, dataset=instruction_pairs, loss=masked_cross_entropy)

# Fase 2, strada A -- RLHF
reward_model = clone(model)
reward_model.lm_head = nn.Linear(n_embd, 1)                       # un punteggio, non una softmax
train(reward_model, dataset=preference_pairs, loss=bradley_terry_loss)
ppo_train(model, reward_model, reference_model=frozen_copy(model))

# Fase 2, strada B -- DPO: niente reward model, niente campionamento
train(model, dataset=preference_pairs, reference_model=frozen_copy(model), loss=dpo_loss)

# Fase 3 (modelli di ragionamento) -- RL con reward verificabili, per esempio GRPO
grpo_train(model, problems=math_and_code, reward=check_answer)
```

Il pre-training resta di gran lunga la fase più costosa (settimane su migliaia di GPU). Ma sono le fasi successive, molto più piccole, a decidere se il risultato si comporta da assistente o da generatore di testo qualunque. (Nei modelli più recenti anche le fasi di RL sono diventate costose, perché richiedono di generare moltissime risposte.)

### 9. Esercizi

1. **Prova il modello base.** Prova altri prompt in forma di istruzione ("Riassumi la storia in una frase:", "Elenca i personaggi:") sul modello della Lezione 7. Segue mai il formato? Se puoi, confronta con GPT-2 originale tramite la libreria `transformers`.
2. **Il ruolo di $\boldsymbol{\beta}$.** Ricalcola la loss DPO dell'esempio con $\beta \in \{0{,}01,\ 0{,}1,\ 1\}$. Cosa succede per $\beta \to 0$? E per $\beta$ grande? Collega la risposta al vincolo di non allontanarsi dal riferimento.
3. **Etichette invertite.** Scambia $y_w$ e $y_l$ nell'esempio DPO. La loss sale o scende? Le derivate cambiano segno? Che cosa dice questo sull'importanza della qualità dei dati di preferenza?

### 10. Fine del corso

Tutto ciò che abbiamo costruito è la ricetta dietro InstructGPT, ChatGPT, Claude e gli altri assistenti: lo stesso SFT, lo stesso reward model o la stessa DPO, la stessa RL con reward verificabili, applicati a scala industriale.

Guardando indietro: nella Lezione 1 una derivata era "di quanto cambia l'output se muovo questo numero". Nella 2 un modello di linguaggio stimava $P(x_t \mid x_{<t})$. Nella 5 il gradiente della cross-entropy era $p - \text{onehot}(y)$. Nella 7 lo stesso principio, applicato a un Transformer, produceva testo. Qui, lo stesso principio, probabilità del modello, una loss, un gradiente, viene piegato a preferire le risposte che una persona approverebbe o che si possono verificare.

**Non c'è mai stata magia: solo la regola della catena, applicata con pazienza, un pezzo alla volta.**

### Riepilogo

1. Un modello base continua il testo nello stile che ha visto; non risponde se non ha visto esempi di risposte.
2. SFT: stessa cross-entropy, dati curati di (istruzione, risposta), loss calcolata solo sulle risposte.
3. RLHF: un reward model separato, addestrato con Bradley--Terry sulle preferenze, poi PPO con penalità KL (serve perché il campionamento non è derivabile).
4. DPO elimina reward model e campionamento: una loss calcolabile in un forward e un backward.
5. Le allucinazioni nascono dall'obiettivo stesso e sono incentivate da benchmark che premiano il tirare a indovinare.
6. Strumenti: token speciali e un protocollo. Ragionamento: RL con reward verificabili (RLVR, GRPO).
7. Il pre-training resta la fase più costosa; il post-training decide il comportamento.

---

# APPENDICI

> **[Nuovo] Nota.** Il programma del corso originale prometteva tre appendici pratiche (A, B, C), ma il PDF non le contiene: si chiude rimandando a esse. Le versioni qui sotto sono **sintesi introduttive** scritte per questa revisione: spiegano i concetti e indicano dove approfondire, ma non hanno la profondità né le verifiche numeriche delle lezioni. L'Appendice D è nuova.

## Appendice A --- Da nanoGPT a llama.cpp

**L'idea.** Il modello della Lezione 7 è, alla fine, una sequenza fissa di operazioni: consultazione di tabelle, prodotti matriciali, softmax, normalizzazioni, somme. Per **usarlo** (inferenza) non serve PyTorch né l'autograd: basta un programma che legga i pesi da un file ed esegua quelle operazioni. È ciò che fanno `llama.cpp` (Georgi Gerganov, 2023) e, in versione didattica, `llama2.c` di Karpathy (circa 700 righe di C per eseguire un modello Llama 2).

**Cosa serve davvero per generare un token:**

1. un file con i pesi e gli iperparametri (dimensioni, numero di strati e teste, vocabolario);
2. il forward di un blocco: normalizzazione, attenzione, feed-forward, residui;
3. la **KV-cache** (Lezione 7): per ogni strato, le key e i value dei token già visti, così a ogni passo si calcola solo il nuovo token;
4. il campionamento: softmax con temperatura e `multinomial` (Lezione 2), spesso con filtri *top-k* (solo i $k$ token più probabili) o *top-p* (i token più probabili fino a una probabilità cumulata $p$);
5. il tokenizer (Lezione 8) per tradurre testo in id e viceversa.

**Perché funziona su un portatile.** Nella generazione token per token il collo di bottiglia non è il calcolo ma **leggere i pesi dalla memoria** a ogni passo. Per questo contano due cose: pesi più piccoli (Appendice B) e codice che sfrutta bene la CPU (istruzioni vettoriali) o la GPU integrata. Regola pratica: la velocità di generazione è circa la banda di memoria divisa per la dimensione del modello in byte.

**Esercizio.** Esporta i pesi del GPT della Lezione 7 in un file binario e scrivi in Python puro (senza PyTorch, solo `numpy`) il forward e un ciclo di generazione con KV-cache. Verifica che i logit coincidano con quelli di PyTorch.

**Per approfondire:** i repository `karpathy/llama2.c` e `ggml-org/llama.cpp`.

## Appendice B --- Quantizzazione e GGUF

**L'idea.** Nella Lezione 9 abbiamo visto bf16: 16 bit per numero invece di 32. La **quantizzazione** spinge oltre: 8, 4 o anche meno bit per peso. Un modello da 7 miliardi di parametri occupa circa 14 GB in bf16 e circa 4 GB a 4 bit: entra nella memoria di un portatile.

**Come funziona, nella forma più semplice.** Si dividono i pesi in **blocchi** (per esempio di 32 numeri). Per ogni blocco si salva una **scala** $s$ (in alta precisione) e ogni peso come intero piccolo $q$:

$$w \approx s \cdot q, \qquad s = \frac{\max|w_{blocco}|}{2^{b-1} - 1}, \qquad q = \text{round}(w/s)$$

Con $b = 4$ bit, $q$ va da $-7$ a $7$ (si rinuncia al valore $-8$ per avere un intervallo simmetrico). L'errore è l'arrotondamento di $q$: piccolo se nel blocco non ci sono valori anomali. Dividere in blocchi serve proprio a questo: un valore anomalo rovina solo il suo blocco.

```python
import torch
def quantize_q4(w, block=32):
    w = w.reshape(-1, block)          # il numero di pesi deve essere multiplo di block
    s = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 7   # clamp: niente divisioni per zero
    q = torch.clamp(torch.round(w / s), -7, 7).to(torch.int8)       # i formati veri mettono 2 valori in un byte
    return q, s

def dequantize(q, s):
    return (q.float() * s).view(-1)
```

**Cosa si perde.** A 8 bit la qualità resta praticamente invariata. A 4 bit di solito la perdita è piccola, soprattutto con metodi più raffinati (scale diverse per sotto-blocchi, scelta dei pesi più importanti). Sotto i 3--4 bit la qualità cala in modo visibile.

**GGUF** è il formato di file di `llama.cpp`: un unico file che contiene pesi (anche quantizzati), iperparametri e tokenizer. I nomi tipo `Q4_K_M` o `Q8_0` indicano lo schema di quantizzazione (numero di bit e variante).

**Esercizio.** Quantizza a 8 e a 4 bit i pesi del GPT della Lezione 7, ricaricali e misura la loss di validazione. Di quanto peggiora? Prova blocchi di 16, 32, 128 numeri.

## Appendice C --- Fine-tuning con LoRA e QLoRA

**Il problema.** Fare fine-tuning completo (Lezione 10) di un modello da miliardi di parametri richiede memoria per i pesi, per i gradienti e per gli stati di AdamW (due numeri in più per ogni parametro): diverse volte la memoria occupata dai soli pesi.

**L'idea di LoRA** (*Low-Rank Adaptation*, Hu e colleghi, 2021). Si congelano i pesi originali $W$ (forma $d \times k$) e si impara solo una **correzione di rango basso**:

$$W' = W + \frac{\alpha}{r}\,B A, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll d, k$$

Con $d = k = 4096$ e $r = 8$, la correzione ha circa 65 000 parametri invece di 16,7 milioni: lo 0,4%. Si inizializza $B = 0$, così all'inizio il modello è identico all'originale.

```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                       # pesi originali congelati
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # parte da zero
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```

Perché funziona: l'adattamento a un compito specifico sembra richiedere cambiamenti "semplici" (di rango basso) rispetto alla conoscenza generale già nei pesi. Dopo l'addestramento si può sommare $BA$ a $W$ e non resta alcun costo in inferenza.

**QLoRA** (Dettmers e colleghi, 2023) combina LoRA con la quantizzazione dell'Appendice B: i pesi congelati sono tenuti a **4 bit** (formato NF4), mentre le piccole matrici $A$ e $B$ sono addestrate in alta precisione. Così si può fare fine-tuning di un modello da decine di miliardi di parametri su una sola GPU.

**Esercizio.** Applica `LoRALinear` alle proiezioni `c_attn` del GPT della Lezione 7 e fai fine-tuning su un testo diverso (per esempio un secondo racconto). Confronta numero di parametri addestrati, memoria e loss con un fine-tuning completo.

**Per approfondire:** le librerie `peft` (Hugging Face) e `bitsandbytes`.

## Appendice D --- Cosa è cambiato dopo GPT-2 (2019--2026)

Il corso costruisce GPT-2, un'architettura del 2019. L'idea di fondo è la stessa nei modelli di oggi, ma molti pezzi sono stati sostituiti. Questa appendice è una mappa, non una trattazione completa.

### D.1 L'architettura

- Embedding di posizione appresi
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: **RoPE** (*rotary position embedding*): query e key vengono "ruotate" di un angolo che dipende dalla posizione
  - *Perché*: il prodotto $q \cdot k$ dipende dalla **distanza** fra le posizioni; si estende meglio a contesti più lunghi
- LayerNorm
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: **RMSNorm**: divide per la radice della media dei quadrati, senza sottrarre la media
  - *Perché*: più semplice e veloce, risultati equivalenti
- MLP con GELU, espansione $\times 4$
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: **SwiGLU**: due proiezioni, una "fa da cancello" all'altra
  - *Perché*: migliore qualità a parità di parametri
- Bias in tutti i `Linear`
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: Nessun bias
  - *Perché*: meno parametri, nessuna perdita
- Ogni testa ha le sue K e V
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: **GQA / MQA**: più teste di query condividono le stesse K e V
  - *Perché*: la KV-cache diventa molte volte più piccola: contesti lunghi e inferenza più economici
- Tutto il modello lavora per ogni token
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: **Mixture of Experts (MoE)**: il feed-forward è diviso in tanti "esperti" e un router ne attiva solo pochi per token
  - *Perché*: molti parametri totali, poco calcolo per token (Mixtral, DeepSeek-V3, Qwen3-MoE...)
- Contesto 1 024 token
  - *Modelli recenti (Llama, Mistral, Qwen, DeepSeek, Gemma...)*: Contesti da 128 000 a oltre 1 milione di token
  - *Perché*: RoPE estesa, FlashAttention, attenzione locale a finestra in alcuni strati

Alcuni modelli aggiungono anche la **normalizzazione di query e key** (QK-norm) prima del prodotto scalare, che stabilizza l'addestramento: è la stessa preoccupazione della Lezione 7 (non far saturare la softmax).

### D.2 L'addestramento

- **Ottimizzatori.** AdamW resta il riferimento, ma **Muon** (2024), che "ortogonalizza" l'aggiornamento delle matrici, è stato adottato in progetti come `modded-nanogpt` e `nanochat` e in alcuni modelli di produzione, con addestramenti più rapidi a parità di risultato.
- **Dati.** Più importanza a qualità, filtraggio e **dati sintetici** generati da altri modelli. Il rapporto fra token e parametri è ben oltre i 20 di Chinchilla (Lezione 9).
- **Precisione.** Oltre a bf16, l'addestramento in **FP8** su hardware recente (per esempio DeepSeek-V3).

### D.3 Il post-training

- Oltre a RLHF con PPO e DPO: **RLAIF** e *Constitutional AI* (preferenze fornite da un modello guidato da principi scritti), varianti di DPO, e soprattutto **RL con reward verificabili** e **GRPO** per i modelli di ragionamento (Lezione 10).
- **Distillazione**: modelli piccoli addestrati sulle risposte e sui ragionamenti di modelli grandi.

### D.4 L'uso

- **Inferenza**: KV-cache con GQA, quantizzazione (Appendice B), *speculative decoding* (un modello piccolo propone più token e quello grande li verifica in un'unica passata).
- **Agenti**: modelli che usano strumenti in molti passi, con protocolli standard per collegarli a servizi esterni (per esempio il *Model Context Protocol*).

### D.5 Dove proseguire

- **nanochat** (Karpathy, ottobre 2025): la pipeline completa, da tokenizer a chat, in un unico repository leggibile. È il seguito diretto di questo corso.
- **Deep Dive into LLMs like ChatGPT** (Karpathy, video, febbraio 2025): visione d'insieme di pre-training, post-training e ragionamento, senza codice.
- **modded-nanogpt** (Keller Jordan e altri): la "gara di velocità" per addestrare GPT-2 124M, dove si vedono in azione molte delle tecniche di questa appendice.

---

## Glossario minimo

- **Attenzione (self-attention)**: Meccanismo con cui ogni token raccoglie informazione dagli altri, con pesi decisi dal contenuto (Lezione 7).
- **Backpropagation**: Calcolo di tutti i gradienti con una passata all'indietro sul grafo (Lezione 1).
- **Batch / minibatch**: Gruppo di esempi usato per stimare il gradiente in un passo (Lezione 3).
- **BatchNorm / LayerNorm / RMSNorm**: Normalizzazioni delle attivazioni: sul batch, sul singolo esempio, sul singolo esempio senza sottrarre la media (Lezione 4, Appendice D).
- **Cross-entropy**: La loss dei modelli di linguaggio: logaritmo negativo della probabilità del token giusto (Lezione 2).
- **Embedding**: Vettore appreso che rappresenta un token (Lezione 3).
- **Fine-tuning**: Continuare ad addestrare un modello già addestrato su dati specifici (Lezione 10, Appendice C).
- **Gradiente**: Vettore delle derivate della loss rispetto a tutti i parametri (Lezione 1).
- **KV-cache**: Memoria delle key e dei value già calcolati, per generare un token alla volta senza ricalcolare tutto (Lezione 7).
- **Learning rate**: Ampiezza del passo di discesa del gradiente (Lezioni 1, 3, 9).
- **Logit**: Punteggio grezzo prima della softmax (Lezione 2).
- **Loss**: Numero che misura quanto il modello sbaglia; l'addestramento la minimizza.
- **Overfitting**: Il modello memorizza i dati di addestramento invece di generalizzare: train bassa, dev alta (Lezione 3).
- **Perplessità**: $e^{\text{loss}}$: fra quante opzioni equiprobabili il modello è "indeciso" (Lezione 2).
- **Post-training**: Le fasi dopo il pre-training: SFT, RLHF/DPO, RL con reward verificabili (Lezione 10).
- **Pre-training**: Addestramento sulla previsione del token successivo su un grande corpus (Lezioni 2--9).
- **Residuo (connessione residua)**: $x + f(x)$: una "autostrada" per il gradiente attraverso la profondità (Lezione 7).
- **Softmax**: Trasforma logit in probabilità: esponenziale e normalizzazione (Lezione 2).
- **Temperatura**: Divisore dei logit prima della softmax: bassa = scelte prevedibili, alta = più varie (Lezione 2).
- **Token / tokenizer**: Unità di testo su cui lavora il modello / il componente che divide il testo in token (Lezione 8).
- **Weight decay**: Regolarizzazione che spinge i pesi verso zero (Lezioni 2, 3, 7).
