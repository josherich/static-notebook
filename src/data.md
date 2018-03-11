# **Reading List**

- foundation of data science
- Sutton reinforcement learning draft
- matrix calculas
- matrix cookbook
- math4ml
- J&M speech and language processong, 2nd, 3rd
- cho notes: Natural Language Understanding with Distributed Representation
- derivatives, gradient, Jacobian
- review-differential-calculus
- Linguistic Fundamentals for NLP
- neural network methods for NLP
- granda notes
- the four fundamental subspaces 4 lines/starting with two matrix
- Foundations of Machine Learning (math review)
- understanding machine learning algo
- principle of scientific conputing
- integral equations and fast algorithm 2017 notes
- convolution arithmetic for deep learning
- convex opt bubeck
- TenLecturesFortyTwoProblems.pdf
- geometric linear algebra

# **Rolling Question List**

- Open set
- Lagrangian multiplier
- gamma density
- is MLP multi layer perceptron equal to feature extraction(mapping function)?

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

- [SVMs](#SVM), margin bounds, [kernel methods](#kernelmethods).

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

> “[Kernel methods](#kernelmethods)” can (sometimes) help with memory and computational costs.



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

### Definition:

concept class C is weakly [PAC-learnable](#) if there exists a (weak) learning algorithm L and   > 0 such that:

- for all $\delta > 0$, for all $c \in C$ and all distributions D,
$$ {Pr \atop {S \sim D}} \lbrack R(h(s)) \leq \frac{1}{2} - \gamma \rbrack \ge 1 - \delta, $$

- for sample S of size $m = poly(\frac{1}{\delta})$ for a fixed polymonial.
