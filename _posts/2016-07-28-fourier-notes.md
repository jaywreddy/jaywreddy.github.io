---
layout: post
title: Fourier Analysis Notes
category: "Fourier Analysis"
---

# Chapter 0: Inner Product Spaces

## Inner Product

$$\langle X, Y \rangle = \sum_{j=1}^n x_j y_j, \qquad X,Y \in \mathbb{R}^n$$

$$\langle Z, W \rangle = \sum_{j=1}^n z_j \overline{w_j}, \qquad Z,W \in \mathbb{C}^n$$

Bilinearity:

$$\langle X+Y,Z \rangle = \langle X,Z \rangle + \langle Y, Z \rangle$$ and $$\langle X,Y +Z \rangle = \langle X,Y \rangle + \langle X, Z \rangle$$

### Properties of the Inner Product

$$\langle .,. \rangle: V \times V \to C$$

- Positivity: $$\langle v,v \rangle > 0$$
- Conjugate summetry: $$\overline{\langle v,w \rangle} = \langle w,v \rangle$$
- Homogeneity: $$\langle cv,w \rangle = c \langle v,w \rangle$$
- Linearity: $$\langle u+v,w \rangle = \langle u,w \rangle + \langle v,w \rangle$$

## Spaces $$L^2$$ and $$l^2$$

$$L^2([a,b])$$ is the set of all square integrable functions defined on [a,b].

$$L^2([a,b]) = { f: [a,b] \to C; \int_a^b \vert f(t)\vert ^2dt < \infty} $$

### $$L^2$$ Inner Product

$$ \langle f,g \rangle_{L^2} = \int_a^b f(t) \overline{g(t)} dt $$

### $$l^2$$ Space - Discretized $$L^2$$

$$\langle X,Y\rangle_{l^2} = \sum_{n=-\infty}^\infty x_n \overline{y_n} $$

- Relative error = $$\frac{\vert \vert f-g\vert \vert _{L^2}}{\vert \vert f\vert \vert _{L^2}}$$

### Convergence in $$L^2$$ Versus Uniform Convergence

Convergence in the Mean: $$\vert \vert f_n - f \vert \vert _{L^2} \to 0$$

Convergence pointwise: for t in [a,b]$$n \geq N \implies \vert f_{n}(t) - f(t)\vert  < \epsilon$$

Uniform Convergence: $$\forall t \in [a,b], \qquad n \geq N \implies \vert f_n(t) - f(t)\vert < \epsilon$$

## Schwarz and Triangle Inequalities

- Schwarz Inequality: $$\vert \langle X,Y \rangle\vert  \leq \vert \vert X\vert \vert  \ \vert \vert Y\vert \vert $$
- Triangle Inequality: $$\vert \vert  X+Y\vert \vert \leq \vert \vert X\vert \vert +\vert \vert Y\vert \vert $$

## Orthogonality

Law of Cosines: $$\langle X,Y \rangle = \vert \vert X\vert \vert  \ \vert \vert Y\vert \vert  \text{cos}(\theta)$$

Orthogonal: $$\langle X,Y \rangle = 0$$ (90 degrees)

### Orthonormal Basis of $$L^2$$
$$\{ \frac{\text{cos}(nt)}{\sqrt{\pi}}, \frac{\text{sin}(nt)}{\sqrt{\pi}}, \qquad n= 1,2,\cdots\} \text{ is orthonormal in } L^2([-\pi,\pi])$$

### Orthonormal Decomposition

$$ v = \sum_{j=1}^N \langle v, e_j \rangle e_j$$

#### Orthogonal projection
For subspace $$V_0 = \text{span } \{e_j\}$$:

$$ \vert \vert v-v_0\vert \vert  = \text{min}_{w \in V_0} \vert \vert v-w\vert \vert $$

$$v_0 = \sum_{j=1}^N \langle v, e_j \rangle e_j $$

#### Orthogonal Complement
$$ V_0^\perp = \{v \in V; \ \langle v,w\rangle =0 \forall w \in V_0\}$$

$$ V = V_0 \oplus V_0^\perp$$

### Gram-Schmidt Orthogonalization

Given $$\{ v_i\}$$, a basis of $$V_0$$

$$ e_1 = \frac{v_1}{\vert \vert v_1\vert \vert } $$

$$ e_i = \text{normalized}(v_j - \sum_1^{j-1} \langle v_j, e_i \rangle e_i) $$

## Linear Operators and Their Adjoints

### Linear Operators
Linear Operator: $$T:V \to W$$ s.t. $$T(\alpha v + \beta w) = \alpha T(v) + \beta T(w)$$

$$\mathcal{M}(T), \{v_i\},\{w_i\}$$:
$$ T(v_j)= \sum_{i=1}^m a_{ij}w_i$$

### Adjoints

$$ \langle T(v),w \rangle w = \langle v,T^* (w)\rangle v $$

Where $$T^* : W \to V$$

$$a^{*}_{ij}= \overline{a_ij} \qquad \text{(Conjugate Transpose)}$$

$$(T_2 \circ T_1)^* = T_1^* \circ T_2^* $$

If $$P$$ is an orthogonal projection on to $$V_0$$, $$P^* = P$$

## Least Squares and Linear Predictive Coding

Best Fit Line minimizes MSE:
$$ E = \sum_{i=1}^{N}\vert y_i - (mx_i + b)\vert ^2 $$

### Matrix formulation

$$X = \{ x_1, x_2, \cdots, x_N \}$$
$$Y = \{y_1, y_2, \cdots, y_N \}$$

$$ Z = \begin{pmatrix}
  x_1 & 1 \\
  \vdots & \vdots \\
  x_N & 1
  \end{pmatrix} $$

Solution found by solving:

$$ Z^TY = Z^TZ \begin{pmatrix} m \\ b \end{pmatrix} $$

Yielding:

$$ m = \frac{\langle Y,X \rangle - N \overline{x} \overline{y}}{\sigma_x} \text{ and } b= \frac{\overline{y}(\sum_i x_i^2) - \overline{x} \langle X,Y \rangle}{\sigma_x} $$

$$\sigma_x = \sum_i (x_i - \overline{x})^2$$

### General Least Squares

With a general matrix $$Z$$ and $$Y$$ vector, there is a unique vector $$V$$ s.t. $$ZV$$ is closest to $$Y$$, given by soln to :

$$Z^* Y = Z^* ZV$$

If components of $$Z$$ are real, the adjoint becomes transpose.

### Linear Predictive Coding
Steps:

1. Sender cuts the data into blocks

$$\{x_1, \cdots, x_N\},\{x_{N+1},\cdots, x_{2N}\},\cdots $$

where each block has some near repetitive pattern. Then choose $$p$$ close to the length of the repetitive pattern for the first block.

2. For $$1 \leq i \leq p$$, form the vectors

$$ z_i = \begin{pmatrix} x_i \\ \vdots \\ X_{N+i-p-1} \end{pmatrix} $$

3. Sender solves the system of equations

$$ \begin{pmatrix}
\langle Z_P, Y \rangle \\
\vdots \\
\langle Z_1, Y \rangle
\end{pmatrix} =
\begin{pmatrix}
\cdots & Z_p^T & \cdots\\
& \vdots & \\
\cdots & Z_1^T  & \cdots
\end{pmatrix}
\begin{pmatrix}
\vdots & & \vdots\\
Z_p & \cdots & Z_1 \\
\vdots & & \vdots
\end{pmatrix}
\begin{pmatrix}
a_1\\
\vdots \\
a_p
\end{pmatrix}
$$

$$ Z =
\begin{pmatrix}
x_p & x_p-1 & \cdots &x_1 \\
x_{p+1} & x_p & \cdots & x_2 \\
\vdots & & \ddots & \vdots \\
x_{N-1} & x_{N-2} & \cdots & x_{N-p}
\end{pmatrix} $$

 for the coefficients $$a_1, \cdots, a_p$$ and transmits to the receiver both $$a_1, \cdots, a_p$$ and $$x_1, \cdots, x_p$$

4. The receiver then reconstructs $$x_{p+1}, \cdots, x_N$$ via the equations
$$ x_n = a_1 x_{n-1} + \cdots + a_p x_{n-p} $$
for those $$x_n$$ where the corresponding errors are smaller than a specified tolerance. If the error exceeds the tolerance, then the sender must transmit $$x_n$$.

# Chapter 1: Fourier Series

## Computation of Fourier Series

### Trigonometric Expansion:

$$ f(t) = a_0 + \sum_k a_k \text{cos}(kx) + b_k \text{sin}(kx) $$

### Trig Integral Relations:

$$\frac{1}{\pi} \int_{- \pi}^{\pi} \text{cos}(nx)\text{cos}(kx)dx =  \begin{cases} 1 & \text{if } n=k \geq 1 \\    2 & \text{if } n=k=0 \\ 0 & \text{otherwise} \end{cases} $$

$$\frac{1}{\pi} \int_{- \pi}^{\pi} \text{sin}(nx)\text{sin}(kx)dx = \begin{cases} 1 & \text{if } n=k \geq 1 \\ 0 & \text{otherwise} \end{cases} $$

$$\frac{1}{\pi} \int_{- \pi}^{\pi} \text{cos}(nx)\text{cos}(kx)dx = 0 \qquad \text{for all integers } n,k $$

### Fourier Coefficients:

$$a_0 = \frac{1}{2\pi}\int_{-\pi}^\pi f(x)dx$$

$$a_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\text{cos}(nx)dx$$

$$b_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\text{sin}(nx)dx$$

#### Other Intervals of Length $$2 \pi$$:

$$\int_{-\pi +c}^{\pi +c}F(x)dx = \int_{- \pi}^{\pi}F(x)dx$$

#### Intervals of length $$2a$$:

By change of variables we can scale any interval to $$[-\pi,\pi]$$

$$\frac{1}{\pi} \int_{-\pi}^\pi F(x)dx = \frac{1}{a}\int_{-a}^a F(\frac{\pi t}{a})dt$$

Yielding the following fourier coefficients:
$$f(x) = a_0 + \sum_{k=1}^\infty a_k \text{cos}(\frac{k \pi x}{a}) + b_k \text{sin}(\frac{k \pi x}{a})$$

$$a_0 = \frac{1}{2a} \int_{-a}^a f(t) dt $$
$$a_n = \frac{1}{a} \int_{-a}^a f(t)\text{cos}(\frac{n \pi t}{a}) dt $$

 $$b_n = \frac{1}{a} \int_{-a}^a f(t)\text{sin}(\frac{n \pi t}{a}) dt $$

### Cosine and Sine Expansions:

- Even: $$f(-x) = f(x)$$

  $$\int_{-a}^{a} f(x) dx = 2 \int_0^a f(x) dx$$

Fourier series will only involve cosines:

$$f(x) = a_0 + \sum_{k=1}^\infty a_k \text{cos}(\frac{k \pi x}{a})$$

$$a_0 = \frac{1}{a} \int_0^a f(x) dx$$

$$a_k = \frac{2}{a} \int_0^a f(x) \text{cos}(\frac{k \pi x}{a})dx$$

- Odd: $$f(-x) = -f(x)$$

  $$\int_{-a}^a f(x) dx = 0$$

Fourier series will only involve sines

$$f(x) = \sum_{k=1}^\infty b_k \text{sin}(\frac{k \pi x}{a})$$

$$b_k =\frac{2}{a} \int_0^a f(x) \text{sin}(\frac{k \pi x}{a})dx $$

#### Fourier Cosine and Sine series on a Half Intervals

For $$f$$ defined on $$[0,a]$$, we can expand $$f$$ into a cosine or sine series by considering even or odd expansions

### Complex Form of Fourier Series

#### Complex Exponential:

$$ e^{it} = \text{cos}(t) + i \text{sin}(t)$$

Properties:

$$e^{i(t + 2\pi)} = e^{it}$$
$$\vert e^{it}\vert  = 1 $$
$$\overline{e^{it}} = e^{-it}$$
$$e^{it} e^{is} = e^{i (s+t)} $$
 $$\frac{e^{it}}{e^{is}} = e^{i(t-s)} $$
 $$ \frac{d}{dt}e^{it} = ie^{it} $$

- {$$\frac{e^{int}}{\sqrt{2\pi}}, n = \cdots, -1,0,1,\cdots$$} are orthonormal in $$L^2([-\pi,\pi])$$

#### Complex Fourier Coefficients

$$f(t) = \sum_{n=-\infty}^\infty \alpha_n e^{int}$$

$$ \alpha_n = \frac{1}{2\pi} \int_{- \pi}^\pi f(t) e^{- int} dt$$

##### Arbitrary Intervals:

{$$\frac{1}{\sqrt{2a}}e^{\frac{in\pi t}{a}}$$} is orthonormal in $$L^2([-a,a])$$ $$f(t) = \sum_{n=-\infty}^\infty \alpha_n e^{\frac{i n \pi t}{a}}$$

$$\alpha_n = \frac{1}{2a} \int_{-a}^a f(t) e^{\frac{-i n \pi t}{a}}dt  $$

#### Relation between Real and Complex Representations

$$ \alpha_0 = a_0 $$

$$ \alpha_n = \begin{cases} \frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{-int} dt \\ \frac{1}{2} (a_n - i b_n) \end{cases} $$

## Convergence Theorems for Fourier Series

### The Riemann-Lebesgue Lemma

The size of the coefficients $$a_k, b_k$$ converge to 0 if $$f$$ is a piecewise-continuous function on the interval $$a \leq x \leq b$$

### Convergence of Higher Order Fourier Components

$$ \lim\limits_{k \to \infty} \int_a^b f(x) \text{cos}(kx) dx = \lim\limits_{k \to \infty} \int_a^b f(x) \text{sin}(kx) dx = 0 $$

### Convergence at a Point of Continuity

If $$f$$ is a continuous and 2$$\pi$$ periodic function, then for each point where the derivative of $$f$$ is defined the Fourier series of $$f$$ at $$x$$ converges to $$f(x)$$

Note: Review proof

### Convergence at a Point of Discontinuity

At points of discontinuity, the Fourier series converges to the average of the left and right limits of $$f$$.

### Uniform Convergence

The Fourier series of a continuous, piecewise smooth 2$$\pi$$ periodic function $$f(x)$$ converges uniformly.

Continuous, piecewise smooth: $$f$$ is piecewise-continuous and $$f'$$ is piecewise-continuous. $$f'$$ contains only jump discontinuities

$$\vert f(x) - S_N(x)\vert  < \epsilon \qquad \text{for } N > N_0 $$

### Convergence in the Mean

If $$f$$ is not continuous, then the Fourier series does not converge to $$f$$ at points of discontinuity.

But it can converge in $$V = L^2([- \pi, \pi])$$ for all square integrable functions ($$f: \int_{- \pi}^{\pi} \vert f(x)\vert ^2 dx < \infty$$)

$$V$$ has inner product:

$$\langle f,g \rangle = \int_{- \pi}^{\pi} f(x) \overline{g(x)} dx$$

and Norm:

$$\rVert f\vert \vert ^2 = \int_{-\pi}^\pi \vert f(x)\vert ^2 dx$$

$$V_n = \text{span} { 1, \text{cos}(kx), \text{sin}(kx), k= 1,\cdots,N }$$

#### Real Fourier Series

Partial Fourier Series:  

$$f_N(x) = a_0 + \sum_{k=1}^N a_k \text{cos}(kx) + b_k \text{sin}(kx)  \in V_N$$

Partial Fourier Series is the projection of $$f$$ onto $$V_N$$:

$$\vert \vert f- f_N\vert \vert _{L^2} = \text{min}_{g \in V_N} \vert \vert f-g\vert \vert _{L^2} $$

If $$f$$ in $$L^2([-\pi, \pi])$$ then $$f_N$$ converges to $$f$$ in $$L^2([-\pi,\pi])$$ then $$\vert \vert f_N - f\vert \vert _{L^2} \to 0$$ as $$N \to \infty$$

#### Complex Fourier Series

$$ \alpha_n = \frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{-int} dt $$

Partial Fourier Series: $$ f_N(t) = \sum_{-N}^N \alpha_k e^{ikt}$$ $$f_N \to f$$ in the $$L^2([-\pi,\pi])$$ norm as $$N \to \infty$$

A function in $$L^2([-\pi, \pi])$$ can be approximated arbitrarily closely by a smooth, $$2\pi$$-periodic function.

### Parseval's Equation

#### Real Version

$$ \frac{1}{\pi} \int_{-\pi}^\pi \vert f(x)\vert ^2 dx = 2\vert a_0\vert ^2 + \sum_{k=1}^\infty \vert a_k\vert ^2 + \vert b_k\vert ^2$$

#### Complex Version

$$ \frac{1}{2 \pi} \vert \vert f\vert \vert ^2 = \frac{1}{2 \pi} \int_{-\pi}^\pi \vert f(x)\vert ^2 dx = \sum \vert a_k\vert ^2$$

$$ \frac{1}{2\pi} \langle f,g \rangle = \frac{1}{2 \pi} \int_{-\pi}^\pi f(t) \overline{g(t)}dt = \sum_{n = - \infty}^\infty \alpha_n \overline{\beta_n}$$

$$L^2$$ norm is the system energy.

# Hermite Functions and the Fourier Transform

## Hermite Functions

### Definition

$$h_n = \frac{(-1)^n}{n!} e^{x^2/2}D^n e^{-x^2}$$

Fourier Transform:

$$ \hat{f}(k) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty f(x) e^{-ikx} dx$$

### Properties

$$h_n$$ provides an orthogonal basis for $$L^2(\mathbb{R})$$

$$h_n(x) = \frac{1}{n!} H_n(x)e^{-x^2/2} $$

### Ladder Function Properties

$$ h_n'(x) = xh_n(x) - (n+1) h_{n+1}(x)$$

$$h_n'(x) =-xh_n(x) + 2 h_{n-1}(x) $$

$$h_n''(x)-x^2 h_n(x) = -(2n+1)h_n(x)$$

## Eigenfunctions of Fourier Transform

Hermite functions are eigenfunctions of the fourier transform ,as taking their transforms results in only a scaling.

$$\hat{h_n}(k) = (-i)^n h_n(k) $$

$$f(x) = \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle h_n(x) $$

$$\hat{f}(k) =\sum_{n=0}^\infty \frac{(-i)^n}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle h_n(k) $$

$$ \frac{1}{\sqrt{2}}e^{-ikx} = \sum_{n=0}^\infty \frac{(-i)^n}{\vert \vert h_n\vert \vert ^2} h_n(k) h_n(x) $$

### Norm of hermite functions
$$\vert \vert h_n\vert \vert ^2 = \frac{\sqrt{\pi}}{n!} 2^n $$

# Poisson Summation formulas

If $$f$$ is a nice smooth function and $$\hat{f}$$ is its Fourier Transform:

## Standard formula

$$ \sqrt{2\pi} \sum_{-\infty}^\infty f(2\pi n) = \sum_{-\infty}^\infty \hat{f}(n) $$

## General Formula

$$\sum_{-\infty}^\infty f(x+kT) = \frac{\sqrt{2\pi}}{T} \sum_{-\infty}^\infty \hat{f}(\frac{2\pi n}{T}) e^{2 \pi i n x /T} $$

# Chapter 3: Discrete Fourier Analysis

## The Discreet Fourier Transform

### Definitions

$$\hat{y}_k = \sum\limits_{j=0}^{n-1} y_k \bar{w}^{jk} \quad \text{with  } w=e^{2 \pi i / n} $$

$$ \implies \hat{y}_k = \sum\limits_{j=0}^{n-1} y_j e^{-2\pi i k j /n}$$

### Matrix Formulation

$$\mathcal{F}_n(y) = \hat{y} = (\bar{F}_n) \dot (y) $$

where

$$ F_n =  \begin{pmatrix}
1 & 1 & 1 &  \cdots & 1 \\
1 & w & w^2 &  \cdots & w^{n-1} \\
1 & w^2 & w^4 &  \cdots & w^{2(n-1)} \\
\vdots & 1 & \ddots & 1 & \vdots \\
1 & w^{n-1} & w^{2(n-1)} &  \cdots & w^{(n-1)^2} \\
   \end{pmatrix}$$

 The DCT operates on elements of $$S_n$$, where $$S_n$$ is the set of $$n$$-periodic sequences of complex numbers. For elements of $$S_n$$, $$y_{n+k} = y_k$$ for all $$k$$.

 Related to fourier coefficients by $$\alpha_k \approx \frac{1}{n} \hat{y}_k$$ for $$k$$ small relative to $$n$$.

## Properties of the Discrete Fourier Transform

### Inverse DFT

With $$w = e^{2 \pi i /m}$$, the inverse DFT is computed via

$$y_j = \frac{1}{n} \sum\limits_{k=0}^{n-1} \hat{y}_k w^{jk}$$

In matrix formulation,

$$ y = \mathcal{F}_n^{-1} \hat{y} = \frac{1}{n}(F_n) \cdot (\hat{y}) $$

An Important property of this is that the multiplication of the two matrices is the identity.

$$I_N = \Big( \frac{F_n}{\sqrt{n}} \Big) \Big( \frac{\bar{F_n}}{\sqrt{n}} \Big) $$

### Properties of the Discreet Fourier Transform

#### Shifts and Translations
If $$y \in S_n$$ and $$z_k = y_{k+1}$$ then $$\mathcal{F}[z]_ j = w^j \mathcal{F}[y]_ j$$

#### Convolutions
If $$y,z \in S_n$$, then the convolution is defined

$$[y*z]_ k = \sum\limits_{j=0}^{n-1} y_j z_{k-j} $$

$$y*z$$ is also in $$S_n$$ and the fourier transform is given:

$$ \mathcal{F}[y*z]_ k = \mathcal{F}[y]_ k \mathcal{F}[z]_ k$$

#### Conjugates
If $$y\in S_n$$ is a sequence of real numbers, then

$$\mathcal{F}[y]_ {n-k} = \bar{\mathcal{F}[y]_ k} $$

or

$$ \hat{y}_{n-k} = \bar{\hat{y_k}} $$

This implies that only $$\hat{y}_0 \cdots \hat{y}_{n/2-1}$$ need to be calculated.  

## The Fast Fourier Transform
TODO

## Discrete Signals

Consider applications to Discrete Signals: $$x = (\cdots x_{-2}, x_{-1}, x_{0}, x_1, \cdots)$$ with index $$k$$: $$x_k$$.

### Time-Invariant, Discrete Linear Filters
Define the time-shift operator $$T_p$$:

$$[T_p(x)]_k = x_{k-p}$$

It takes the time series and shifts the indices by $$k$$ units to the right.

#### Time-Invariant Linear Operators
A linear operator $$F: x \to y$$ that takes a sequence $$x$$ into another sequence $$y$$ is time-invariant if $$F(T_p(x)) - T_p(F(x))$$.

Any linear operator $$F$$ can be completely determined by what it does to the unit sequences $$e^n$$ where

$$e_k^n = \begin{cases} 0, & k \neq n, \\
1, & k = n \end{cases} $$

Since any sequence $$x$$ can be written as the linear combination of the unit sequences. Thus, the linearity of $$F$$ implies that it could be mapped to each of the unit sequences and applied.

#### Discrete Convolution
Given sequences $$x,y$$ the convolution is defined:

$$(x * y )_k = \sum\limits_{n \in \mathbb{Z}} x_{k-n}y_n $$

If $$F$$ is a time-invariant linear operator acting on sequences, it can be expressed equivalently as a convolution with a sequence $$f$$:

$$F(x) = f * x $$

In this context, $$f$$ is the impulse response.

### Z-Transform and Transfer Functions
The Z-Transform is a generalization of the DFT to infinite sequences in $$l^2$$.

Recall that $$l^2$$ is the set of absolutely summable sequences: $$\sum_n \vert x_n\vert ^2 < \infty$$

#### Definition of Z-Transform
The Z-Transform of an infinite sequence $$x$$ is the function $$\hat{x}:[-\pi,\pi] \to \mathbb{C}$$:

$$\hat{x}(\phi) - \sum\limits_{j = - \infty}^\infty x_j e^{-ij \phi} $$

If we use $$z=e^{i \phi}$$, we get:

$$\hat{x}(z) = \sum_{-\infty}^\infty x_j z^{-j} $$

Modifying the Z-Transform to finite sequences by substituting $$w = e^{\frac{2\pi i }{n}}$$ yields the DFT.

#### Connection with Fourier Series
Recall that for a function $$f\in L^2[-\pi,\pi]$$, the Fourier series representation is:

$$f(\phi) = \sum_{n = - \infty}^\infty x_n e^{in \phi} $$

with fourier coefficients $$x_n$$ given by ::

$$x_n  = \frac{1}{2\pi} \int_{-\pi}^\pi f(\phi) e^{-in\phi} d\phi $$

Compare this to the Z-Transform:

$$\hat{x}(\phi) = \sum_{n=-\infty}^\infty x_n e^{=in\phi}= f(-\phi) $$

Thus, Fourier series expansion takes a function $$f\in L^2[-\pi,\pi]$$ to a sequence $$\{x_n\} \in l^2$$.

Meanwhile, the Z-transform takes a sequence $$\{x_n\} \in l^2$$ to a function $$f \in L^2[-\pi,\pi]$$.  

##### Isometry Between L-squared and l-squared

Via Parseval's Theorem, we can derive for the Fourier Transform (and then modify for the z-transform using $$\phi \to -\phi$$):

$$(1/2\pi) \langle \hat{x}, \hat{y} \rangle_{L^2[-\pi,\pi]} = \langle x, y \rangle_l^2 $$

Thus, the Z-transform is an isometry between $$l^2$$ and $$L^2[-\pi,\pi]$$ (it preserves the inner products up to a scalar factor).

##### Convolution Properties
Consider $$f = \{f_n\}, x= \{x_n\}$$ are sequences in $$l^2$$. then

$$(\widehat{f * x})(\phi) = \hat{f}(\phi)\hat{x}(\phi) $$

The function $$\hat{f}(\phi)$$ is the Z-transform of the sequence $$f$$, also called the transfer function of the operator $$F$$, where $$F(z) = f * x$$.

###### Adjoint of Convolution Operators
Recall that the adjoint of an operator $$F:l^2 \to l^2$$ is the operator $$F^*: l^2 \to l^2$$ defined:

$$\langle F(x),y \rangle = \langle x, F^*(y) \rangle ,\quad x,y \in l^2 $$

Then if $$F$$ is the convolution operator associated with the sequence $$f_n$$, then $$F^*$$ is the convolution operator associated with the sequence $$f^*_n = \bar{f}_{-n}$$. The transfer function for $$F^*$$ is $$\bar{\hat{F}}(\phi)$$.
