# **Introduction**

## I want to keep my notes in one page

## It's a mess

## Content table sucks, search sucks

## I try graph and tree

## anchor links are connected with headings, making a graph

## To create a node in graph, use markdown `# heading` and `**strong**`

## To keep a paragraph in the margin, try clicking the hovering `M` right beside.


# **Machine Learning**

## Examples of Learning Tasks

- Text

- Language

- Speech

- Image

- Games

- Unassisted control of vehicles

- Medical diagnosis, fraud detection, network intrusion

## some broad tasks

- classification

- regression

- Ranking

- Clustering

- Dimensionality reduction

## General Objective

- Theoretical questions:
  
  - what can be learned, under what conditions?
  
  - are there learning guarantees?

  - analysis of learning algorithms.

- Algorithms:
  - more efficient and more accurate algorithms

  - deal with large-scale problmes

  - handle a variety of different learning problems

## Topics

- [Convex Optimization](#convex)

- Probability tools, concentration inequalities.

- [PAC](#pac) learning model, [Rademacher complexity](#rademacher), [VC-dimension](#vcdimension).

- [SVMs](#SVM), margin bounds, [kernel methods](#kernel-methods).

- ensemble methods, [boosting](#boosting).

- Logistic regression and conditional maximum entropy models.

- On-line learning, weighted majority algorithm, Perceptron algorithm, mistake bounds.

- Regression, generalization, algorithms.

- Ranking, generalization, algorithms.

- [Reinforcement learning](#reinforcement), [MDPs](#mdp), [bandit](#bandit) problems and algorithms.


# **Convex Optimization**

- [Convexity](#convexity)

# **Convexity**

- definition: $X \subset \Bbb R^N $ is said to be convex if for any two points $x, y \in \mathrm X$ the segment $\lbrack x, y \rbrack$ lies in X:

$$ \lbrace \alpha x + (1-a)y, 0 \le \alpha \le 1 \rbrace \subseteq \mathrm X.$$

- Definition: let X be a convex set. A function $\mathcal f: X \to \Bbb R$ is said to be convex if for all $x, y \in \mathrm X$ and $\alpha \in \lbrack 0,1 \rbrack$
$$f(\alpha x + (1 - \alpha)y) \le \alpha f(x) + (1 - \alpha)f(y)$$

  - with a strict inequality, $f$ is said to be strictly convex.

  - $f$ is said to be concave when $-f$ is convex.



# **feature spaces**

### Very large feature spaces have two potential issues:

- 1. [Overfitting](#overfitting)

- 2. Memory and computational costs

> Overfitting we handle with regularization.

> “[Kernel methods](#kernel-methods)” can (sometimes) help with memory and computational costs.



# **Kernel Methods**

## Definition

A method is kernelized if every feature vector ψ(x) only appears inside an inner product with another feature vector ψ(x′). In particular, this applies to both the optimization problem and the prediction function.



# **Kernel Functions**

### The Kernel Function

- Input space: $\mathbf{X}$

- Feature space: $\mathbf{H}$

- Feature map: $\mathbf{\psi} : \mathbf{X} \rightarrow \mathbf{H}$

> The kernel function corresponding to $\psi$ is

$$k\left(x,x'\right) =  \langle\psi(x),\psi(x')\rangle$$

where $\langle \cdot, \cdot \rangle$ is the inner product associated with $\mathbf{H}$.

What are the Benefits of Kernelization?

- 1. Computational (e.g. when feature space dimension d larger than sample size n).

- 2. Can sometimes avoid any O(d) operations, allows access to infinite-dimensional feature spaces.

- 3. Allows thinking in terms of “similarity” rather than features.


# **Similarity Scores**

It is often useful to think of the kernel function as a similarity score. But this is not a mathematically precise statement.

# **Overfitting**

# **MDPs**

# **VC dimension**

# **Boosting**

# **Weak Learning**

# **NMF**
r(n+d)
NP-hard
reformulated

# **LDA**
Gamma function
Dirichlet distribution

# **HMM**
Viterbi algorithm (dp with max)

# **VC dimension**

vapnik chervonenkis
growth function: expression power of hypothesis space
dichotomy
shatter
Given a set S of examples and a concept class H, we say that S is shattered by H if for every A ⊆ S there exists some h ∈ H that labels all examples in A as positive and all examples in S \ A as negative.

The VC-dimension of H is the size of the largest set shattered by H.

## **shatter function**

Given a set S of examples and a concept class H, let H[S] = {h ∩ S : h ∈ H}. That is, H[S] is the concept class H restricted to the set of points S. For integer n and class H, let H[n] = max|S|=n |H[S]|; this is called the growth function of H. 

# **CRF**

https://spaces.ac.cn/archives/4695/

# **EM**

estimation, maximization
likelihood is usually defined on exponential function, thus use ln() to iterate EM
to get latent variables,  compute its expectation

# **regularization**

why L1 get sparsity?
Ridge

# **PAC learnable**

efficient
properly
sample complexity m >= poly(,,,)

# **max entropy**

uncertainty should be equally distributed
conditional entropy

### Definition:

concept class C is weakly [PAC-learnable](#) if there exists a (weak) learning algorithm L and   > 0 such that:

- for all $\delta > 0$, for all $c \in C$ and all distributions D,
$$ {Pr \atop {S \sim D}} \lbrack R(h(s)) \leq \frac{1}{2} - \gamma \rbrack \ge 1 - \delta, $$

- for sample S of size $m = poly(\frac{1}{\delta})$ for a fixed polymonial.

## **Hoeffding’s inequality**


# **computation theory**

### **NP-hardness**

> non-deterministic polynomial-time hardness

a problem H is NP-hard when every problem L in NP can be reduced in polynomial time to H; that is, assuming a solution for H takes 1 unit time, we can use H‎'s solution to solve L in polynomial time.

### **P ≠ NP**

> If P ≠ NP, then NP-hard problems cannot be solved in polynomial time.

![nphard](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/P_np_np-complete_np-hard.svg/800px-P_np_np-complete_np-hard.svg.png?1521031605731)

# **numpy example**

###  np api
```
np.repeat(n, 3)
np.arange(n)
np.full((3, 3), 1, dtype=bool)
out = np.where(arr % 2 == 1, -1, arr)
np.reshape()
np.repeat(a, 3)
np.tile(a, 3)
np.concatenate
np.vstack
np.hstack
np.r_[a, b]
np.intersect1d(a, b)
```

- From 'a' remove all of 'b'
```
np.setdiff1d(a,b)
```


# **DS-GA3001Graphs and Networks**

## Detailed Syllabi for lectures:

Jan 25: Introduction to graph theory, approximation algorithm, Max-Cut approximation. Chapter 8 on Lecture Notes. 

Feb 01: Max-Cut approximation. Lifting / SDP relaxations technique in mathematical signal processing, phase retrieval and k-means SDP.

Feb 08: Unique Games Conjecture, Sum-of-Squares interpretation of SDP relaxation. Chapter 8 of Lecture Notes.

Feb 15: Shannon Capacity, Lovasz Theta Function. Section 7.3.1. on Lecture Notes and "On the Shannon Capacity of a Graph" by Laszlo Lovasz. See also Section 6.5.3.

Feb 22: Stochastic Block Model and Phase Transitions on graphs. Chapter 9 of Lecture Notes

Mar 01: Recovery in the Stochastic Block Model with Semidefinite relaxations. Chapter 9 of Lecture Notes

## Detailed Syllabi for labs:
Jan 24: review of linear algebra and probability

Jan 31: discussion of homework 1

Feb 07:  graph Laplacian and Cheeger's inequality

Feb 14:  pseudo distribution for maxcut,  derivation of primal and dual program for Maxcut, SOS4

Feb 21:  introduction to Grothendieck inequality and a proof of an upper bound of Grothendieck constant (Krivine�s bound)

Feb 28:  calculate the Lovasz theta function for n-cycle and discuss connection with Grothendieck constant on graph

## **Ramsey number**

A natural question is whether it is possible to have arbitrarily large graphs without cliques (and without its complement having cliques), Ramsey answer this question in the negative in 1928 [Ram28]

> We say an event happens with high probability if its probability is ≥ 1 − n−Ω(1)

### Spencer 94

> “Erd ̋os asks us to imagine an alien force, vastly more powerful than us, landing on Earth and demanding the value of R(5) or they will destroy our planet. In that case, he claims, we should marshal all our computers and all our mathematicians and attempt to find the value. But suppose, instead, that they ask for R(6). In that case, he believes, we should attempt to destroy the aliens.”

## **Clique**

A clique of a graph G is a subset S of its nodes such that the subgraph corresponding to it is complete. In other words S is a clique if all pairs of vertices in S share an edge. The clique number c(G) of G is the size of the largest clique of G.

## **Independent set**

An independence set of a graph G is a subset S of its nodes such that no two nodes in S share an edge. Equivalently it is a clique of the complement graph Gc := (V, Ec). The independence number of G is simply the clique number of Sc.

## Erdos-Hajnal Conjecture
For any finite graph H, there exists a constant $\delta H > 0$ such that any graph on n nodes that does
not contain H as a subgraph (is a H-free graph) must have

$$r(G) \geq n^{\delta^H}$$

## max-cut problem

> to design polynomial algorithms that, in any instance, produce guaranteed approximate solutions.

Given a graph G = (V, E) with non-negative weights wij on the edges, find a set S ⊂ V for which cut(S) is maximal.

Goemans and Williamson [GW95] introduced an approximation algorithm that runs in polynomial time and has a randomized component to it, and is able to obtain a cut whose expected value is guaranteed to be no smaller than a particular constant αGW times the optimum cut. The constant αGW is referred to as the approximation ratio.

### cut

> a cut is a partition of the vertices of a graph into two disjoint subsets. Any cut determines a cut-set, the set of edges that have one endpoint in each subset of the partition.

- max-cut
- min-cut
- sparse-cut

  - The sparsest cut problem is to bipartition the vertices so as to minimize the ratio of the number of edges across the cut divided by the number of vertices in the smaller half of the partition. NP-hard, best known approximation algorithm is an O({\sqrt {\log n))) approximation due to Arora, Rao & Vazirani (2009)

- cut-space

## **Unique Game Problem**

> Given a graph and a set
of k colors, and, for each edge, a matching between the colors, the goal in the unique games problem
is to color the vertices as to agree with as high of a fraction of the edge matchings as possible.

### conjecture 
For any ε > 0, the problem of distinguishing whether an instance of the Unique Games Problem is such that it is possible to agree with a ≥ 1 − ε fraction of the constraints or it is not possible to even agree with a ε fraction of them, is NP-hard.

# **fast algorithm**

## **N-body problem**

### **FFT**

$\sim 5NlogN$
$$u_j = \sum_{k=1}^N e^{2\pi ijk/N}\omega_k$$

## **FMM**
- Laplace equation
- fast Gauss transform
- Helmholtz equation

### **degenerate kernel**
