---
layout: post
title:  "Complex Analysis Notes"
category: "Complex Analysis"
---

# Chapter 1: The Complex Plane and Elementary Functions

## 1.1 - Complex Numbers

Complex Number: $$z = x + iy$$

* $$(x + iy) + (u + vi) = (x+u) + (y+v)i$$
* $$\vert  z\vert   = \sqrt{x^2 + y^2}$$ 
* Triangle Inequal: $$\vert  z+w\vert   \leq \vert  z\vert   + \vert  w\vert  , \qquad  z,w \in  \mathbb{C}$$
* Multiplication through distributive property
* Multiplicative Inverse: $$\frac{1}{z} = \frac{x-iy}{x^2 + y^2}$$
* Complex conjugate: $$\bar{z} = x - iy$$
* $$\frac{1}{z} = \frac{\bar{z}}{\vert  z\vert  ^2}$$
* $$\text{Re } z = (z + \bar{z})/2x$$
* $$\text{Im } z = (z - \bar{z})/2i$$

Complex Polynomial of degree $$n$$: $$p(z) = \sum_{i=0}^n a_n z^n$$

### Fundamental Theorem of Algebra:

Every complex polynomial p(z) of degree $$n \geq 1$$ has a factorization
$$ p(z) = c(z-z_1)^m_1 \cdots (z-z_k)^m_k$$
where the $$z_j$$'s are distinct and $$m)j \geq 1$$. This factorization is unique, up to a permutation of the factors.

## 1.2 - Polar Representation
z:=

* $$r,\theta$$ where $$r = \sqrt{x^2 + y^2}$$
* $$\theta = \text{ arg }z$$

$$\begin{cases}
  x = r \text{ cos } \theta \\
  y = r \text{ sin } \theta
  \end{cases}
 $$

![polar](/images/complex_analysis/notes/polar.png)

Arg $$z$$: Principle values, $$\theta \in \{-\pi, \pi\}$$

$$e^{i \theta} = \text{cos } \theta + i \text{ sin } \theta$$

### Useful Identities:
$$\vert  e^{i \theta}\vert   = 1 $$
$$\overline{e^{i \theta}} = e^{-i \theta}$$
$$ \frac{1}{e^{i \theta}} = e^{-i \theta}$$
$$ e^{i(\theta + \varphi)} = e^{i \theta}e^{i \varphi} $$

$$n$$th root: $$z^n =w$$: zeros of poly $$z^n-w$$
Concretely: $$r^n e^{in \theta} = p e^{i \varphi}$$
$$r = p^{1/n} $$
$$ \theta = \frac{\varphi}{n} + \frac{2 \pi k}{n}, \qquad k = 0,1,2,\cdots n-1$$

## 1.3 -  Stereographic Projection

Extended Complex Plane: $$\mathbb{C}^* = \mathbb{C}\cup \{\infty\}$$

Stereographic Projection: map from $$\mathbb{R}^3$$ unit sphere to $$\mathbb{C}^{* }$$

* $$N = (0,0,1)$$, and we draw a line to $$P$$, a point on the unit circle. $$z$$ is intersection of line and coordinate plane. $$P=N \rightarrow \infty$$

![stereographic projection](/images/complex_analysis/notes/stereographic.png)

Coordinate transformations:
$$\begin{cases}
  x = X/(1-Z) \\
  y = Y/(1-Z)
  \end{cases}$$

$$ \begin{cases}
   X = 2x/(\vert  z\vert  ^2 +1) \\
   Y = 2y/(\vert  z\vert  ^2 +1) \\
   Z = (\vert  z\vert  ^2 -1)/(\vert  z\vert  ^2 +1)
   \end{cases}$$

Circles on the sphere correspond to circles or straight lines in the plane: longitudinal:straight, latitudinal:circle

## 1.4 - The Square and Square Root Functions

### Graphing Complex Functions

A real function can be graphed in $$\mathbb{R^2}$$, but a complex function would require $$\mathbb{R}^4$$.

Solution 1: Graph real and imaginary parts separately

#### W,Z graph
z-plane: domain
w-plane: range

$$ w = f(z)$$
We can consider any path in $$z$$, and plot the result in $$w$$.

### The Square Function
$$ w = z^2 $$

$$\vert  w\vert   = \vert  z\vert  ^2$$

$$\text{arg}(w) = 2\text{arg}(z) $$

Some properties of the w,z graph:
* If a ray in z traces at a set angle from 0 to $$\infty$$, the ray in z will travel at constant speed, while the ray in $$w$$ will start slowly and accelerate.

* If a ray sweeps across angles at a fixed magnitude, it travels twice as far in $$w$$ as in $$z$$

### Square Root Function

Trying to find an inverse, we notice that each point in $$w$$ except zero is hit by two points in $$z$$, $$\pm \sqrt{w}$$.

To solve this problem, we draw a branch cut between the positive and negative values: $$\{\text{Re}(z) > 0\}$$ and $$\{\text{Re}(z) < 0\}$$.

The positive one of these values is defined as the principle branch.

![Square Root Branches](/images/complex_analysis/notes/sqrtbranches.png)

### Riemann surface
Thinking of the two branch cuts as surfaces, we can glue them together over $$w$$ to get a 3D surface where each side represents the value of $$z$$ corresponding to each branch.

## 1.5 - The Exponential Function

### Complex Exponential

$$ e^z = e^x \text{cos}(y) + i e^x \text{sin}(y), \qquad z=x+iy \in \mathbb{C}$$

$$ e^iy = \text{cos} (y) + i \text{sin}(y) $$
equivalently,

$$e^z = e^x e^{iy}, \qquad z = x+iy$$

These identities correspond to the polar representation of $$e^x$$
$$ \vert  e^z\vert   = e^x$$

$$ \text{arg}(e^z) = y $$

Note: remember that the exponential is $$2\pi$$-periodic

## 1.6 The Logarithm Function

$$ \text{log} z = \text{log} \vert  z\vert   + i \text{arg} z $$

Note, $$e^x$$ is $$2\pi$$-periodic, so many args will map to the same value. To solve this, we use Arg, definied to be $$-\pi<\theta\leq \pi$$:

$$\text{log} z = \text{log} \vert  z\vert   + i \text{Arg}(z+2\pi i m) \qquad m=0,\pm1,\pm2,\cdots$$

This forms the principle value of log.

$$ \text{Log} z = \text{log} z + i \text{Arg} z $$

Riemman Surface of log forms a spiral ribbon

## 1.7 - Power Functions and Phase Factors



## 1.8 - Trigonometric and Hyperbolic Functions



# Chapter 2: Analytic Functions

## 2.1 - Review of Basic Analysis

### Sequence Convergence

A sequence of complex numbers $$\{s_n\}$$ converges to $$s$$ if for any $$\epsilon > 0$$, there is an integer $$N \geq 1$$ such that $$\vert  s_n -s\vert   < \epsilon$$ for all $$n \geq N$$.
Written $$s_n \to s$$

### Properties of Limits
Addition, subtraction, multiplication, and division hold as usual.

### In between theorem
If $$r_n \leq s_n \leq t_n$$, and $$r_n \to L,$$ and $$t_n \to L$$, $$s_n \to L$$

### Bounded monotone sequences converge

### Convergence of Complex Sequences
A sequence of complex numbers converges iff the corresponding sequences of real and imaginary parts converge.

### Cauchy Sequence:
$$\vert  s_n - s_m\vert   \to 0$$ as $$n,m \to \infty$$

#### Complex sequences converge iff it is a cauchy sequence.

### Convergence of Functions

$$f(z) \to L$$ if $$f(z_n) \to L$$ for any sequence $$z_n \to z_0$$

#### Function limits obey +-/*  

### Closed/Open Sets
For open sets, any $$z\in U$$ has disk around it in $$U$$.
i.e.: $$\vert  z-z_0\vert  <p$$ is open, $$\vert  z-z_0\vert   \leq p$$ is closed

Closed sets contain the limit of any sequence in the set.

Boundary - points z such that any disk contains points inside and outside the set. Closed sets contain boundary, open do not.

Any closed, bounded set is said to be compact.

#### Compact Set Thm
A continuous real-valued function on a compact set attains its maximum.

### Domain
A domain is connected. i.e. any two points can be connected by a series of line segments

$$\mathbb{C}\\ \mathbb{R}$$ is not a domain, since we can't get from upper half to lower half without crossing the real line.

### Convexity
Any 2 points can be connected by a single line

### Star-Shaped
Star shaped w.r.t. $$z_0$$, any point can be connected via a line to $$z_0$$, not necessarily any point to each other.

## 2.2 - Analytic Functions

### Differentiability

A function $$f(z)$$ is differentiable at $$z_0$$ if

$$\frac{f(z)-f(z_0)}{z-z_0} $$

has a limit as $$z \to z_0$$

#### Differentiability implies continuity

#### Derivative Rules

$$(fg)'(z) = f(z)g'(z) + f'(z)g(z) $$

$$(f/g)'(z) = \frac{g(z)f'(z)-f(z)g'(z)}{g(z)^2} $$

#### Chain rule
$$(f \circ g)'(z_0)= f'(g(z_0))g'(z_0) $$

### Analytic Functions
A function $$f(z)$$ is analytic on the open set $$U$$ if $$f(z)$$ is differentiable at each point of $$U$$ and the complex derivative is continuous on $$U$$.

Sums and products of analytic functions are analytic, and quotients are as well so long as teh denominator does not vanish.

## 2.3 - The Cauchy-Riemann equations

### Motivation

For $$f = u + iv$$, the derivative can depend on how you approach the number in the complex plane.

$$f'(z) = \frac{\partial u}{\partial x}(x,y) + i \frac{\partial v}{\partial x}(x,y) $$

$$f'(z) = \frac{\partial v}{\partial y}(x,y) - i \frac{\partial u}{\partial y}(x,y) $$

Leading to the

### Cauchy-Riemann Equations

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \qquad \frac{\partial u}{\partial y}= -\frac{\partial v}{\partial x} $$

### Analyticity Thm
Let $$f = u+iv$$ be defined on a domain $$D$$ int the complex plane, where $$u$$ and $$v$$ are real-valued. then $$f(z)$$ is analytic on $$D$$ if and only if $$u(x,y)$$ and $$v(x,y)$$ have continuous first-order partial derivatives that satisfy the Cauchy-Riemann equations.

### Derivative Identity Theorems
1. If $$f(z)$$ is analytic on a domain $$D$$, and if $$f'(z) =0$$ on $$D$$, then $$f(z)$$ is constant.

2. If $$f(z)$$ is analytic and real-valued on a domain $$D$$, then $$f(z)$$ is constant.

The second follows because in this case, $$v=0$$.

## 2.4 - Inverse Mappings and the Jacobian

### Def of Jacobian

Let $$f=u+iv$$ be analytic on a domain $$D$$. We can regard $$f$$ as a map from $$D$$ to $$\mathbb{R}^2$$ with components $$(u(x,y),v(x,y))$$ The Jacobian matrix is then:

$$
J_f = \begin{pmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \\
\end{pmatrix}
$$

With determinant:

$$\text{det} J_f = (\frac{\partial u}{\partial x})^2 + (\frac{\partial v}{}\partial x)^2 = \Big\vert   \frac{\partial u}{\partial x} + i \frac{\partial v }{\partial x}\Big\vert  ^2$$

Thus,

### Jacobian of Analytic Function

$$\text{det} J_f = \vert  f'(z)\vert  ^2 $$

### Inverse of Analytic Function
Suppose $$f(z)$$ is analytic on a domain $$D$$, $$z_0 \in D$$, and $$f'(z_0) \neq 0$$. Then there is a small disk $$U \subset D$$ containing $$z_0$$ such that $$f(z)$$ is one-to-one on $$U$$, the image $$V=f(U)$$ of $$U$$ is open, and the inverse function

$$f^{-1}: V \to U $$

is analytic and satisfies

$$(f^{-1})'(f(z)) = 1/f'(z), \qquad z\in U $$

## 2.5 - Harmonic Functions

### Laplace's Equation

$$ \Delta =\frac{\partial^2 }{\partial x_1^2} + \cdots + \frac{\partial^2 }{\partial x^2_n} $$

Is the Laplacian.

$$\frac{\partial^2 u}{\partial x_1^2} + \cdots + \frac{\partial^2 u }{\partial x^2_n} = 0 $$
$$\Delta u = 0 $$
Is Laplace's equation

### Harmonic Function Def
Harmonic Functions are smooth functions that satisfy laplace's equation

Thus a function is harmonic if its first and second partial derivatives exist and sum to 0.

### Connection to Analyticity

If $$f=u+iv$$ is analytic, and the functions $$u$$ and $$v$$ have continuous second-order partial derivatives, then $$u$$ and $$v$$ are harmonic. (As a consequence of the cauchy-riemann equations)


### Harmonic Conjugates
If $$u$$ is harmonic on $$D$$, and $$v$$ is a harmonic function such that $$u+iv$$ is analytic, $$v$$ is the harmonic conjugate of $$u$$, which is unique up to the addition of a constant.

This can be solved with explicit equation for any fixed point $$x_0, y_0$$ on $$D$$

$$v(x,y) = \int_{y_0}^y \frac{\partial u}{\partial x}(x,t)dt - \int_{x_0}^x\frac{\partial u}{\partial y}(s,y_0)ds + C $$

### Existence of Harmonic Conjugates
Let $$D$$ be an open disk, or an open rectangle with sides parallel to the axes, and let $$u(x,y)$$ be a harmonic function on $$D$$. Then there is a harmonic function $$v(x,y)$$ on $$D$$ such that $$u+iv$$ is analytic on $$D$$. The harmonic conjugate $$v$$ is unique, up to adding a constant.

## 2.6 - Conformal Mappings

### Tangent Vectors on Curves

Let $$\gamma(t) = x(t) +iy(t)$$, $$0 \leq t \leq 1$$, be a smooth parameterized curve terminating at $$z_0 = \gamma(0)$$ We refer to

$$\gamma'(0) = \lim\limits_{t \to 0} \frac{\gamma(t) - \gamma(0)}{t} = x'(0) + i y'(0) $$

as the tangent vector to the curve $$\gamma$$ at $$z_0$$. It is the complex representation of the usual tangent vector. We definte the angle between two curves at $$z_0$$ to be the ange between their tangent vectors at $$z_0$$.

### Tangents of Parameterized Functions
If $$\gamma(t_, - \leq t \leq 1$$, is a smooth parameterized curve terminating at $$z_0 = \gamma(0)$$, and $$f(z)$$ is analytic at $$z_0$$, then the tangent to the curve $$f(\gamma(t))$$ terminating at $$f(z_0)$$ is
$$(f \circ \gamma)'(0)  = f'(z_0)\gamma'(0) $$

This follows from the chain rule.

### Conformal Def
A function is conformal if it preserves angles.

I.E. A function $$g(z)$$ is conformal at $$z_0$$ if whenever $$\gamma_0,\gamma_1$$ are curves terminating at $$z_0$$ with nonzero tangents, then $$g \circ \gamma_1, g\circ \gamma_2$$ have nonzero tangents at $$g(z_0)$$ and the angle from $$(g\circ \gamma_0)'(z_0)$$ to
$$(g \circ \gamma_1)'(z_0)$$ is the same as from $$\gamma' _ 0 (z_0) $$ to $$\gamma'_ 1(z_0)$$.

A conformal mapping of domain $$D$$ onto $$V$$ is a continuously differentiable function that is conformal at each point of $$D$$ and that maps $$D$$ one-to-one onto $$V$$.

### Analytic and Conformal Thm
If $$f(z)$$ is analytic at $$(z_0)$$ and $$f'(z_0) \neq 0$$, then $$f(z)$$ is conformal at $$z_0$$.

## 2.7 - Fractional Linear Transformations

### Def of Fractional Linear Transformation

A fractional linear transformation (also called Mobius transformations) is a function of the form

$$w = f(z) = \frac{az+b}{cz+e}, $$

where $$a,b,c,d$$ are complex constants satisfying $$ad-bc \neq 0$$

$$f'(z) = \frac{ad - bc}{(cz +d)^2} $$,

the condition $$ad-bc \neq 0$$ makes sure $$f$$ is not constant.

#### Affine transformations
 Functions $$f(z) = az + b$$ are called affine transformations. These are simply FLT's with $$c=0$$, special cases are translations: $$z \to z + b$$, and dilations $$z \to az$$.

#### Inverse properties
The inverse of an FLT is an FLT, so each FLT can be considered a one-to-one function from the extended complex plane to itself.

#### Composition
The composition of two FLTs is an FLT. This corresponds to matrix multiplication of the constants. In this context, the nonzero constraint becomes the determinant of the matrix is nonzero, ensuring that the matrix is invertible.

### Uniqueness at 3 points
Given any three distinct points $$z_0, z_1, z_2$$ in the extended complex plane, and given any three distinct values $$w_0,w_1,w_2$$ in the extended complex plane, there is a unique fractional linear transformation $$w = w(z)$$ that maps the pairs.

#### 3 Complex Parameters
Although the FLT is defined in terms of 4 complex parameters, one can be changed freely, and the FLT will remain the same by scaling the other 4 by the same nonzero constant.

#### Formula for a Mapping

We can get a unique formula for mapping three points $$z_0,z_1,z_2$$ to $$0,1,\infty$$ respectively via:

$$ f(z) = \frac{z-z_0}{z-z_2}\frac{z_1-z_2}{z_1 -z_0}$$

(slight modifications needed if one of the z's is $$\infty$$).

Define a similar function $$g$$ mapping the $$w$$'s to the same values, and you can form the map matrix by $$g^{-1} \circ f$$

### FLTs are composed of basic operations
Every FLT is a composition of dilations, translations, and inversions.

#### A FLT maps circles in the extended complex plane to circles.

# Chapters 1-3 Supplement: Handout 1

## Complex Numbers
Normal formulas for operations on complex numbers

Fundamental Theorem of Algebra

### Branches of Square Root Function
There does not exist a continuous function such that $$f(z)^2 = z$$

### Roots of Unity:
$$z = e^{i 2 \pi k/ n}\qquad k= 0,1,\cdots n-1 $$


### C is a metric space
Triangle ineq: $$\vert  w+z\vert   \leq \vert  z\vert   + \vert  w\vert  $$

### The Riemann sphere

In Complex Analysis, there is no distinction of positive and negative infinities.

#### Extended Complex Plane
$$\mathbb{C}^* = \mathbb{C} \cup \{\infty\}$$

Also referred to as the Riemann sphere. To see this, use stereographic projection. Thus, only one $$\infty$$.

## Complex Functions
Repeated stuff from the book for the most part.

Review Worksheet again after finishing taking notes in the book.

# Chapter 3: Line Integrals and Harmonic Functions

## 3.1 - Line Integrals and Green's Theorem

### Types of Paths
A path is a function $$\gamma(t)$$ defined on interval $$a \leq t \leq b$$ with endpoints $$\gamma(a)=A, \gamma(b)=B$$.

A simple path never crosses over itself, so there is a one-to-one relationship between the interval and the location in the plane.

A closed path has connected endpoints.

Smooth path: Has form $$\gamma(t)= (x(t),y(t)$$, which have defined derivatives.


#### Re-parameterizing Paths
If $$\gamma(t), a \leq t \leq b$$, is a path from $$A$$ to $$B$$, and $$\phi(s), \alpha \leq s \leq \beta$$ is a strictly increasing continuous function with $$\phi(\alpha)= a, \phi(\beta)= b$$, then $$(\gamma \dot \phi)$$ is also a path from A to B, a reparameterization of $$\gamma$$.

### Line Integral:
$$\int_\gamma Pdx + Qdy = \int_a^b P(x(t),y(t))\frac{dx}{dt}dt + \int_a^b Q(x(t),y(t))\frac{dy}{dt}dt$$

Note that the line integral over $$\gamma$$ is independent of its parameterization.

#### Domains defined by paths
A domain $$D$$ has a piecewise smooth boundary if the boundary of $$D$$ can be decomposed into a finite number of smooth curves meeting at their endpoints.

$$\partial D$$ is the boundary of $$D$$
Default orientation is that the interior of $$D$$ lies on the left of $$\partial D$$ as we traverse $$\partial D$$ in the positive direction.

### Green's Theorem
Let $$D$$ be a bounded domain in the plane whose boundary $$\partial D$$ consists of af inite number of disjoint piecewise smooth closed curves. Let $$P$$ and $$Q$$ be continuously differentiable functions on $$D \cup \partial D$$. Then

$$\int_{\partial D} P dx + Q dy = \iint_D (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y})dx dy $$


## 3.2 - Independence of Path

### Fundamental Theorem of Calculus
1. If $$F(t)$$ is an antiderivative for the continuous function $$f(t)$$,then

$$\int_a^b f(t)dt = F(b) - F(a) $$

2. If $$f(t)$$ is a continuous function on $$[a,b]$$, then the indefinite integral:

$$F(t)= \int_a^t f(s) ds, \qquad a \leq t $$
is an antiderivative for $$f(t)$$. Further, each antiderivative for $$f(t)$$ differs from $$F(t)$$ by a constant.

### Path Independence Thm
If $$\gamma$$ is a piecewise smooth curve from $$A$$ to $$B$$, and if $$h(x,y)$$ is continuously differentiable on $$\gamma$$, then
$$ \int_\gamma dh = h(B) - h(A)$$

#### Path Independence Requirements Lemma
Let $$P,Q$$ be continuous complex-valued functions on a domain $$D$$. Then $$\int Pdx +Qdy$$ is independent of path in D iff $$Pdx+Qdy$$ is exact, namely, there is a continuously differentiable function $$h(x,y)$$ s..t $$dh = Pdx+Qdy$$. Then this function $$h$$ is unique, up to addinga  constant.

### Closed/Conservative Functions:
Let $$P,Q$$ be continuously differentiable complex-valued functions on a domain $$D$$. Then $$Pdx + Qdy$$ is closed on $$D$$ if

$$ \frac{\partial P }{\partial y} = \frac{\partial Q}{\partial x}$$
Notice, this is the condition for the integrand of Green's theorem to be 0. Thus, $$\int_{\partial U}=0$$ for $$U \subset D$$.

Aside: Integral of a closed path on a conservative field is 0. Remember physics?

#### Exact Differentials are Closed
If $$Pdx + Qdy = dh$$ is exact, then
$$\frac{\partial P}{\partial y} = \frac{\partial}{\partial y} \frac{\partial h}{\partial x} = \frac{\partial }{\partial x} = \frac{\partial Q }{\partial x} $$
So it is closed as well. Note the converse is not necessarily true.

#### Closed Differentials on Simple/Star-shaped Domains
Let $$P,Q$$ be continuously differentiable complex-valued functions on a domain $$D$$. Suppose
(1) $$D$$ is a star-shaped domain (superset of simple), and
(2) the differential $$Pdx + Qdy$$ is closed on $$D$$.
Then $$Pdx+Qdy$$ is exact on $$D$$.

#### Closed Differentials on Continuously Deformed Paths
Let $$D$$ be a domain, and let $$\gamma_0(t)$$ and $$\gamma_1(t)$$, $$a \leq t \leq b$$, be two paths in $$D$$ from A to B. Suppose that $$\gamma_0$$ can be continuously deformed to $$\gamma_1$$, in the sense that for $$0\leq s \leq 1$$ there are paths $$\gamma_s(t)$$ from A to B such that $$\gamma_s(t)$$ depends continuously on $$s$$ and $$t$$. Then

$$\int_{\gamma_0} Pdx + Qdy = \int_{\gamma_1} Pdx + Qdy $$
for any closed differential $$Pdx + Qdy$$ on $$D$$.

### Summary:
All domains:

$$\text{independent of path} \iff \text{exact} \implies \text{closed} $$

Star Shaped Domains:
$$\text{independent of path} \iff \text{exact} \iff \text{closed} $$

## 3.3 - Harmonic Conjugates

### Harmonic differentials are closed
If $$u(x,y)$$ is harmonic, then the differential
$$-\frac{\partial u}{\partial y}dx + \frac{\partial u}{\partial x}dy $$
is closed.

This condition is equivalent to Laplace's equation.

Rederivation of the harmonic conjugate.

## 3.4 - The Mean Value Property

### Average Value:
Average value on a circle $$\vert  z-z_0\vert  =r$$ of a continuous real-valued function on a domain $$D$$ is defined:
$$ A(r) = \int_0^{2\pi} h(z_0 + re^{i \theta})\frac{d \theta}{2\pi}$$

### Average Value for Harmonic Functions
On harmonic functions, the average value of the boundary circle of any disk is the value of the center point.

If $$u(z)$$ is a harmonic function on a domain $$D$$, and if the disk $$\vert  z-z_0\vert  <p$$ is contained in D, then:

$$u(z_0) = \int_0^{2\pi} u(z_0 + re^{i\theta}\frac{d\theta}{2\pi}), \qquad 0 < r <p $$

The Mean Value Property means that for each point $$z_0 \in D$$, $$h(z_0)$$ is the average of the values over a circle centered at $$z_0$$.

## 3.5 - The Maximum Principle

### Strict Maximum Principle (Real Version)
Let $$u(z)$$ be a real-valued harmonic function on a domain $$D$$ such that $$u(z)\leq M$$ for all $$z\in D$$. If $$u(z_0)=M$$ for some $$z_0 \in D$$, then $$u(z)=M$$ for all $$z\in D$$.

#### Reminder about harmonic functions
A complex valued function is harmonic if its real and imaginary parts are harmonic. Any analytic function is harmonic.

### Strict Maximum Principle (Complex Version)
Let $$h$$ be a boudned complex-valued harmonic function on a domain $$D$$. if $$\vert  h(z)\vert   \leq M$$ for all $$z \in D$$, and $$\vert  h(z_0)\vert  =M$$ for some $$z_0 \in D$$, then $$h(z)$$ is constant on $$D$$.

## Maximum Principle
Let $$h(z)$$ be a complex-valued harmonic function on a bounded domain $$D$$ such that $$h(z)$$ extends continuously to the boundary $$\partial D$$ of $$D$$. If $$\vert  h(z)\vert  \leq M$$ for all $$z\in \partial D$$, then $$\vert  h(z)\vert   \leq M$$ for all $$z \in D$$.

# Chapter 4: Complex Integration and Analyticity

## 4.1 - Complex Line Integrals

### Basic Definitions

In Complex Analysis, we define $$dz = dx + idy$$. If $$h(z)$$ is a complex-valued function on a curve $$\gamma$$, then
$$\int_\gamma h(z) dz = \int_\gamma h(z) dx + u \int_\gamma h(z) dy $$



### ML-Estimate
Suppose $$\gamma$$ is a piecewise smooth curve. If $$h(z)$$ is a continuous function on $$\gamma$$, then

$$\Bigg\vert   \int_\gamma h(z)dz\Bigg\vert   \leq \int_\gamma \vert  h(z)\vert  \vert  dz\vert   $$

Further, if $$\gamma$$ has length $$L$$, and $$\vert  h(z)\vert  \leq M$$ on $$\gamma$$, then

$$\Bigg\vert   \int_\gamma h(z) dz \Bigg\vert   \leq ML $$

A sharp estimate is an $$ML$$ estimate where equality holds.

## 4.2 - Fundamental Theorem of Calculus for Analytic Functions

### Complex Primitive
Let $$f(z)$$ be a continuous function on a domain $$D$$. A function $$F(z)$$ on $$D$$ is a complex primitive for $$f(z)$$ if $$F(z)$$ is analytic and $$F'(z)=f(z)$$.

### FTC Pt 1
If $$f(z)$$ is continuous on a domain $$D$$, and if $$F(z)$$ is a primitive for $$f(z)$$, then

$$\int_A^B f(z)dz = F(B) - F(A) $$
where the integral can be taken over any path in $$D$$ from $$A$$ to $$B$$.

### FTC Pt 2
Let $$D$$ be a star-shaped domain, and let $$f(z)$$ be analytic on $$D$$. Then $$f(z)$$ has a primitive on $$D$$, and the primitive is unique up to adding a constant. A primitive for $$f(z)$$ is given explicitly by
$$ F(z) = \int_{z_0}^z f(\varsigma)d\varsigma, \qquad z\in D$$
where $$z_0$$ is any fixed point of $$D$$, and where the integral can be taken along any path in $$D$$ from $$z_0$$ to $$z$$.


## 4.3 - Cauchy's Theorem

### Introductory Definitions
All the below information should be included in other sections of the notes, but I repeat here for better readability.

Assume smooth complex-valued function $$f(z) = u + iv$$, then
$$ f(z)dz= (u+iv)(dx +idy) = (u+iv)dx + (-v+iu)dy$$

The condition that $$f(z)dz$$ be a closed differential is

$$\frac{\partial}{\partial y}(u+iv) = \frac{\partial}{\partial x}(-v+iu) $$

Comparing the real and imaginary parts, we get the Cauchy-Riemann equations

$$\frac{\partial u}{\partial y} = - \frac{\partial v}{\partial x},\qquad \frac{\partial v}{\partial y} = \frac{\partial u }{\partial x} $$

### OG Morera's Theorem
A continuously differentiable function $$f(z)$$ on $$D$$ is analytic iff the differential $$f(z)dz$$ is closed.

### Cauchy's Theorem
Let $$D$$ be a bounded domain with piecewise smooth boundary. If $$f(z)$$ is an analytic function on $$D$$ that extends smoothly to $$\partial D$$, then

$$\int_{\partial D} f(z)dz = 0 $$

## 4.4 - The Cauchy Integral Formula

### Cauchy Integral Formula
Let $$D$$ be a bounded domain with piecewise smooth boundary. If $$f(z)$$ is analytic on $$D$$, and $$f(z)$$ extends smoothly to the boundary of $$D$$, then

$$f(z) = \frac{1}{2\pi i} \int_{\partial D} \frac{f(w)}{w-z}dw, \qquad z \in D $$

### Complex Derivative Formula
Let $$D$$ be a bounded domain with piecewise smooth boundary. If $$f(z)$$ is an analytic function on $$D$$ that extends smoothly to the boundary of $$D$$, then $$f(z)$$ has complex derivatives of all orders on $$D$$, which are given by

$$ f^{(m)}(z) = \frac{m!}{2\pi i} \int_{\partial D} \frac{f(w)}{(w-z)^{m+1}},\qquad z\in D, m\geq 0$$

### Analytic Derivatives
If $$f(z)$$ is analytic on domain $$D$$, then $$f(z)$$ is infinitely differentiable, and the successive complex derivatives $$f'(z), f''(z), \cdots$$ are all analytic on $$D$$.


## 4.5 - Liouville's Theorem

### Cauchy Estimates
Suppose $$f(z)$$ is analytic for $$\vert  z-z_0\vert  \leq p$$. If $$\vert  f(z)\vert   \leq M$$ for $$\vert  z-z_0\vert  =p$$, then

$$\Big\vert   f^{(m)}(z_0)\Big\vert   \leq \frac{m!}{p^m}M, \qquad m \geq 0 $$

### Liouville's Theorem
Let $$f(z)$$ be an analytic function on the complex plane. If $$f(z)$$ is bounded, then $$f(z)$$ is constant.

### Entire Functions
An entire function is a function that is analytic on the entire complex plane. Polynomials, $$e^z$$, cos $$z$$, sin $$z$$, cosh $$z$$, and sinh $$z$$ are all examples of entire functions.

Linear combinations and products of entire functions are entire.

Examples of non-entire functions: $$1/z$$, log $$z$$, $$\sqrt{z}$$.

Liouville's theorem can be restated: A bounded entire funciton is constant.

# Chapter 5: Power series

## 5.1 - Infinite series

Convergence of Infinite Series

$$S_k = \sum_{k=0}^K  $$

The series converges to $$S$$ if the partial sums $$S_k$$ converge to $$S$$.

Any statement about series can be interpreted as a statement about sequences by examining the sequence of partial sums.

Since a series of positive numbers is monotonically increasing, it converges if the series is bounded.

### Comparison Test
If $$0 \leq a_k \leq r_k$$, and if $$\sum r_k$$ converges, then $$\sum a_k$$ converges and $$\sum a_k \leq \sum r_k$$

If $$\sum a_k$$ converges, then $$a_k \to 0$$ as $$k \to \infty$$

#### Geometric Series
$$\sum_{k=0}^\infty z^k = \frac{1}{1-z}, \qquad \vert  z\vert  <1 $$

### Absolute Convergence

If $$\sum a_k$$ converges absolutely, then $$\sum a_k$$ converges, and

$$ \vert  \sum_{k=0}^\infty a_k \vert   \leq \sum_{k=0}^\infty \vert  a_k\vert  $$

## 5.2 - Sequences and Series of Functions

Let $$\{f_j\}$$ denote a sequence of complex-valued functions defined on some set $$E$$.

### Pointwise Convergence

$$\{f_j\}$$ converges pointwise on $$E$$ if for every $$x \in E$$ the sequence of complex numbers $$\{ f_j(x)\}$$ converges. The limit $$f(x)$$ is then a complex valued function on $$E$$.

### Uniform Convergence

Sequence $$\{f_j\}$$ converges uniformly to $$f$$ on $$E$$ if $$\vert  f_j(x) -f(x)\vert   \leq \epsilon_j$$ for all $$x \in E$$, where $$\epsilon_j \to 0$$ as $$j \to \infty$$. $$\epsilon_j$$ is a worst-case estimator: i.e.

$$e_j = \text{sup}\ (x \in E)\ \vert  f_j(x) - f(x)\vert  $$

Uniform Convergence implies pointwise convergence

### Continuity

Let $$\{f_j\}$$ be a sequence of complex-valued functions on $$E$$. If each $$f_j$$ is continuous on $$E$$ and if $$\{f_j\}$$ converges uniformly to $$f$$ on $$E$$, then $$f$$ is continuous on $$E$$.

### Integration

Let $$\gamma$$ be a piecewise smooth curve in the complex plane. If $$\{f_j \}$$ is a sequence of continuous complex-valued functions on $$\gamma$$, and if $$\{ f_j\}$$ converges uniformly to $$f$$ on $$\gamma$$, then $$\int_\gamma f_j(z)dz$$ converges to $$\int_\gamma f(z) dz$$

### Series of Functions

Let $$\sum g_j(x)$$ be a series of complex valued functions defined on a set $$E$$. The partial sums of the series are the functions:


$$S_n(x) = \sum_{k=0}^n g_j(x)$$

The series converges pointwise if the sequence of partial sums converges pointwise.

The series converges uniformly if the sequence of partial sums converges uniformly.

### Weierstrass M-Test

Suppose $$M_k \geq 0$$ and $$\sum M_k$$ converges. If $$g_k(x)$$ are complex-valued functions on a set $$E$$ such that $$\vert  g_k(x)\vert   \leq M_k$$ for all $$x \in E$$, then $$\sum g_k(x)$$ converges uniformly on $$E$$.

### Analytic Limits

If $$\{f_k(z) \}$$ is a sequence of analytics functions on a domain $$D$$ that converges uniformly to $$f(z)$$ on $$D$$, then $$f(z)$$ is analytic on $$D$$.

### Convergence of Derivatives

Suppose that $$f_k(z)$$ is analytic for $$\vert  z - z_0\vert   \leq R$$, and suppose that the sequence $$\{f_k(z)\}$$ converges uniformly to $$f(z)$$ for $$\vert  z-z_0\vert   \leq R$$. Then for each $$r < R$$ and for each $$m \geq 1$$, the sequence of mth derivatives $$\{f_k^{(m)}(z)\}$$ converges uniformly to $$f^{(m)}(z)$$ for $$\vert  z-z_0\vert   \leq r$$.

### Normal Convergence

A sequence of $$\{ f_k(z)\}$$ of analytic functions on a domain $$D$$ converges normally to the analytic function $$f(z)$$ on $$D$$ if it converges uniformly to $$f(z)$$ on each closed disk contained in $$D$$.

This occurs if and only if $$\{f_k(z)\}$$ converges to $$f(z)$$ uniformly on each bounded subset $$E$$ of $$D$$ at a strictly positive distance from the boundary of $$D$$.


### Normal Convergence of Derivatives
Suppose that $$\{f_k(z)\}$$ is a sequence of analytic functions on a domain $$D$$ that converges normally on $$D$$ to the analytic function $$f(z)$$. Then for each $$m \geq 1$$, the sequence of mth derivatives $$\{f_k^{(m)}(z) \}$$ converges normally to $$f^{(m)}(z)$$ on $$D$$.

## 5.3 - Power Series

A power series (centered at $$z_0$$) is a series of the form $$\sum_{k=0}^\infty a_k (z - z_0)^k$$.

### Convergence of Power Series
Let $$\sum_k a_k z^k$$ be a power series. Then there is $$R$$, $$0 \leq R \leq +\infty$$, such that $$\sum a_k z^k$$ converges absolutely if $$\vert  z\vert  < R$$, and $$\sum a_k z^k$$ does not converge if $$\vert  z\vert  > R$$. For each fixed $$r$$ satisfying $$r < R$$, the series $$\sum a_k z^k$$ converges uniformly for $$\vert  z\vert  \leq r$$.

R - Radius of Convergence. Depends only on tail of series. I.E. remains the same after altering a finite number of terms

### Coefficients of a Power Series

Suppose $$\sum a_k z^k$$ is a power series with radius of convergence $$R > 0$$. Then the function  

$$ f(z) = \sum_{k=0}^\infty a_k z^k, \qquad \vert  z\vert  < R$$

is analytic. The derivatives of $$f(z)$$ are obtained by differentiating the series term by term,

$$ f'(z) = \sum_{k=1}^\infty k a_k z^{k-1}, $$

$$ f''(z) = \sum_{k=2}^\infty k(k-1)a_k z^{k-2}, $$

$$ \vert  z\vert  <R $$

and similarly for the higher-order derivatives. The coefficients of the series are given by

$$ a_k = \frac{1}{k!}f^{(k)}(0), \qquad k \geq 0$$

The radius of convergence can be determined from the coefficients.

### Ratio Test
If $$\vert  a_k/a_{k+1}\vert  $$ has a limit as $$k \to \infty$$, either finite or $$+ \infty$$, then the limit is the radius of convergence $$R$$ of $$\sum a_k z^k$$,

$$R = \lim\limits_{k \to \infty} \vert  \frac{a_k}{a_{k+1}}\vert  $$

### Root Test
If $$\sqrt[k]{\vert  a_k\vert  }$$ has a limit as $$k \to \infty$$, either finite or $$+ \infty$$, then the radius of convergence of $$\sum a_k z^k$$ is given by

$$ R = \frac{1}{\lim \sqrt[k]{\vert  a_k\vert  }}$$


### Cauchy-Hadamard formula
gives radius of convergence of any power series

$$ R = \frac{1}{\text{lim sup} \sqrt[k]{\vert  a_k\vert  }}$$


## 5.4 - Power Series Expansion of an Analytic Function

### Expansion of Analytic Functions

Suppose that $$f(z)$$ is analytic for $$\vert  z-z_0\vert   < p$$. Then $$f(z)$$ is represented by the the power series

$$f(z) = \sum_{k=0}^\infty a_k(z-z_0)^k, \qquad \vert  z-z_0\vert   < p$$

where

$$a_k = \frac{f^{(k)}(z_0)}{k!}, \qquad k \geq 0$$

and where the power series has radius of convergence $$R \geq p$$. For any fixed $$r$$, $$0 < r < p$$, we have

$$a_k = \frac{1}{2 \pi i} \oint_{\vert  \varsigma - z_0\vert  =r} \frac{f(\varsigma)}{(\varsigma - z_0)^{k+1}}d\varsigma$$

Further, if $$\vert  f(z)\vert   \leq M$$ for $$\vert  z-z_0\vert  =r$$, then
$$\vert  a_k\vert   \leq \frac{M}{r^k}, \qquad k \geq 0 $$


### Analytic Function Determined by Values at Center of Disk
Suppose that $$f(z)$$ and $$g(z)$$ are analytic for $$\vert  z-z_0\vert   < r$$. If $$f^{(k)(z_0)}= g^{(k)}(z_0)$$ for $$k \geq 0$$, then $$f(z) = g(z)$$ for $$\vert  z-z_0\vert  <r$$

### Radius of Convergence is Distance to Nearest Singularity

Suppose that $$f(z)$$ is analytic at $$z_0$$, with power series expansion $$f(z) = \sum a_k (z-z_0)^k$$ centered at $$z_0$$. Then the radius of convergence of the power series is the largest number $$R$$ such that $$f(z)$$ extends to be analytic on the disk $$\{\vert  z-z_0\vert   < R\}$$.


## 5.5 - Power Series Expansion at Infinity

$$f(z)$$ is analytic at $$z= \infty$$ if the fnction $$g(w) = f(1/w)$$ is analytic at $$w=0$$.

If $$g(w)$$ has power series expansion:

$$ g(w) = \sum_{k=0}^\infty b_k w^k \qquad \vert  w\vert  < p\vert  $$

Then $$f(z)$$ is defined:

$$f(z) = \sum_{k=0}^\infty \frac{b_k}{z^k} \qquad \vert  z\vert  > \frac{1}{p}$$

Coefficients given by formula:

$$b_k = \frac{1}{2 \pi i} \int_{\vert  z\vert  =r} f(z)z^{k-1}dz $$

## 5.6 - Manipulation of Power Series

### Properties of Operations on Power Series

$$f(z) = \sum_{k=0}^\infty a_k z^k, \qquad g(z) = \sum_{k=0}^\infty b_k z^k $$

$$f(z) + g(z) = \sum_{k=0}^\infty (a_k + b_k )z^k $$

$$ c f(z) = \sum_{k=0}^\infty c a_k z^k$$

$$ f(z)g(z) = \sum_{k=0}^\infty c_k z^k$$
Where $$c_k$$ are given by:

$$c_k = a_k b_0 + a_{k-1}b_1 + \cdots + a_1 b_{k=1} + a_0 b_k $$

Power series of a quotient: $$f(z)/g(z)$$ can be computed by first computing the series of $$\frac{1}{g(z)}$$.

## 5.7 - The Zeros of an Analytic Function

### Zeros of Order N
$$f(z)$$ has a zero of order $$N$$ at $$z_0$$ if $$f(z_0) = f'(z_0) = \cdots = f^{(N-1)}(z_0) = 0$$, but $$f^N(z_0 ) \neq 0$$.

This only occurs if $$f$$ has the formula
$$ f(z) = a_N(z-z_0)^N + a_{N+1}(z-z_0)^{N+1} + \cdots$$

We can express this as

$$f(z)  = (z-z_0)^N h(z) $$

Where $$h(z)$$ is analytic at $$z_0$$ and $$h(z_0) = a_N \neq 0$$

Simple Zero: Zero of order 1
Double Zero: Zero of order 2

### Uniqueness Principle

If $$D$$ is a domain, and $$f(z)$$ is an analytic function on $$D$$ that is not identically zero, then the zeros of $$f(z)$$ are isolated.

If $$f(z)$$ and $$g(z)$$ are analytic on a domain $$D$$, and if $$f(z) = g(z)$$ for $$z$$ belonging to a set that has a nonisolated point, then $$f(z) = g(z)$$ for all $$z \in D$$.

### Principle of Permanence of Functional equations
A natural extension of the uniqueness principle

Let $$D$$ be a domain, and let $$E$$ be a subset of $$D$$ that has a nonisolated point. Let $$F(z,w)$$ be a function defined for $$z,w \in D$$ such that $$F(z,w)$$ is analytic in $$z$$ for each fixed $$w \in D$$ and analytic in $$w$$ for each fixed $$z \in D$$. If $$F(z,w) = 0$$ whenever $$z$$ and $$w$$ both belong to $$E$$, then $$f(z,w)=0$$ for all $$z,w \in D$$.

# Chapter 6: Laurent Series and Isolated Singularities

## 6.1 - The Laurent Decomposition

Laurent Decomposition splits a function analytic in an annulus into the sum of a function analytic inside the annulus and a function analytic outside the annulus.

### Laurent Decomposition Theorem
Suppose $$0 \leq \rho < \sigma \leq +\infty$$, and suppose $$f(z)$$ is analytic for $$p < \vert  z-z_0\vert  < \sigma$$. Then $$f(z)$$ can be decomposed as a sum

$$ f(z) = f_0(z) + f_1(z)$$
where $$f_0(z)$$ is analytic for $$\vert  z-z_0\vert  < \sigma$$, and $$f_1(z)$$ is analytic for $$\vert  z-z_0\vert  >\rho$$ and at $$\infty$$. If we normalize the decomposition so that $$f_1(\infty)=0$$, then the decomposition is unique.

### Special Cases

If $$f(z)$$ is analytic for $$\vert  z-z_0\vert  < \sigma$$, or $$\vert  z-z_0\vert  >\rho$$, then the Laurent decomposition becomes the function itself, along with $$f_1(z)=0$$, or $$f_0(z)=0$$, respectively.

### Laurent Series Expansion
Suppose $$0 \leq \rho < \sigma \leq \infty$$, and suppose $$f(z)$$ is analytic for $$\rho < \vert  z-z_0\vert   < \sigma$$. Then $$f(z)$$ has a Laurent expansion

$$f(z) = \sum\limits_{k=-\infty}^\infty a_k(z-z_0)^k, \qquad \rho < \vert  z-z_0\vert   < \sigma$$

that converges absolutely at each point of the annulus, and that converges uniformly on each subannulus $$r \leq \vert  z-z_0\vert   \leq s$$, where $$p<r<s<\sigma$$. The coefficients are uniquely determined by $$f(z)$$, and they are given by

$$ a_n = \frac{1}{2\pi i} \oint_{\vert  z-z_0\vert  =r} \frac{f(z)}{(z-z_0)^{n+1}}dz, \qquad -\infty < n < \infty$$

for any fixed $$r$$, $$\rho< r < \sigma$$.



## 6.2 - Isolated Singularities of an Analytic Function

### Definition of Isolated Singularities
A point $$z_0$$ is an isolated singularity if $$f(z)$$ is analytic in some punctured disk centered at $$z_0$$.

Eg. $$1/z$$ has an isolated singularity at 0, while $$\text{log}(z)$$ does not.

### Laurent Series Representation
Suppose $$f(z)$$ has an isolated singularity at $$z_0$$. Then $$f(z)$$ has a Laurent series expansion

$$f(z)= \sum\limits_{k=-\infty}^\infty a_k(z-z_0)^k $$

### Removable Singularity
The isolated singularity of $$f(z)$$ is said to be a removable singularity if $$a_k =0$$ for all $$k<0$$. And the Laurent Series becomes a power series

$$ (z)= \sum\limits_{k=0}^\infty a_k(z-z_0)^k$$

And if we define $$f(z_0)=a_0$$, the function becomes analytic on the entire disk.

### Riemann's Theorem on Removable Singularities
Let $$z_0$$ be an isolated singularity of $$f(z)$$. If $$f(z)$$ is bounded near $$z_0$$, then $$f(z)$$ has a removable singularity at $$z_0$$.

### Poles
The isolated singularity is defined to be a pole if there is $$N > 0$$ such that $$a_{-N} \neq 0$$ but $$a_k = 0$$ for all $$k < -N$$. The integer $$N$$ is the order of the pole.

#### Laurent Series Representation
$$ f(z) = \sum\limits_{k=-N}^\infty a_k(z-z_0)^k$$

#### Principal Part

The sum of the negative powers,
$$P(z)= \sum_{k=-N}^{-1} a_k(z-z_0)^k $$

is called the principal part of $$f(z)$$ at $$z_0$$. This corresponds to the summand $$f_1(z)$$ of the Laurent decomposition.

All of the bad behavior of $$f$$ is contained in the principle part, i.e. $$f(z)- P(z)$$ is analytic at $$z_0$$

### Pole Theorem 1
Let $$z_0$$ be an isolated singularity of $$f(z)$$. Then $$z_0$$ is a pole of $$f(z)$$ of order $$N$$ if and only if $$f(z)= g(z)/(z-z_0)^N$$, where $$g(z)$$ is analytic at $$z_0$$ and $$g(z_0) \neq 0$$.

### Pole Theorem 2
Let $$z_0$$ be an isolated singularity of $$f(z)$$. Then $$z_0$$ is a pole of $$f(z)$$ of order $$N$$ if and only if $$1/f(z)$$ is analytic at $$z_0$$ and has a zero of order $$N$$.

### Meromorphic Functions
A function $$f$$ is meromorphic on a domain $$D$$ if $$f$$ is analytic on $$D$$ except at isolated singularities, each of which is a pole. Sums, products, and quotients of meromorphic.

### Pole Theorem 3
Let $$z_0$$ be an isolated singularity of $$f(z)$$. Then $$z_0$$ is a pole if and only if $$\vert  f(z)\vert   \to \infty$$ as $$z \to z_0$$ .

### Essential Singularities
An isolated singularity of $$f(z)$$ at $$z_0$$ is an essential singularity if $$a_k \neq 0$$ for infinitely many $$k<0$$.

E.x. $$e^{1/z} = 1 + \frac{1}{z^2} + \frac{1}{2!}\frac{1}{z^2} + \cdots$$ So there are infinitely many negative powers of z.

### Casorati-Weierstrass Theorem
Suppose $$z_0$$ is an essential isolated singularity of $$f(z)$$. Then for every complex number $$w_0$$, there is a sequence $$z_n \to z_0$$ such that $$f(z_n) \to w_0$$.

## 6.3 - Isolated Singularity at Infinity

$$f(z)$$ has an isolated singularity at $$\infty$$ if $$f(z)$$ is analytic outside some bounded set, that is, there is $$R > 0$$ such that $$f(z)$$ is analytic for $$\vert  z\vert  >R$$.

$$f(z)$$ has an isolated singularity at $$\infty$$ if and only if $$g(w) = f(1/w)$$ has an isolated singularity at 0.

We classify singularities by their appearance in $$g(w)$$. (think about how the inverse formula cross multiplies terms, then the below formulas make sense. )

### Laurent Series Classification

$$f(z) = \sum\limits_{k=-\infty}^\infty b_k z^k $$

Singularity at $$\infty$$ is removable if $$b_k = 0$$ for all $$k>0$$.

Singularity at $$\infty$$ is essential if $$b_k \neq 0$$ for infinitely many $$k>0$$.

Singularity at $$\infty$$ is a pole of order $$N$$ if $$b_N \neq 0$$, but $$b_k = 0$$ for all $$k>N$$.

### Principal part

$$P(z) = b_N z^N + B_{N-1}z^{N-1} + \cdots + b_1 z + b_0 $$

## 6.4 - Partial Fractions Decomposition

### Meromorphic functions on extended complex plane
A meromorphic function on the extended complex plane $$\mathbb{C}^* $$ is rational.

Note: meromorphic functions can only have a finite number of poles.

### Partial Fractions Decomposition
Every rational function has a partial fractions decomposition, expressing it as the sum of a polynomial in $$z$$ and its principal parts at each of its poles in the finite complex plane.

$$f(z) = P_\infty (z) + \sum\limits_{j=1}^m P_j(z) $$

### Division Algorithm

TODO: Figure this out.

# Chapter 7: The Residue Calculus

## 7.1 The Residue Theorem

## Residues

Suppose $$z_0$$ is an isolated singularity of $$f(z)$$ and that $$f(z)$$ has Laurent Series

$$f(z) = \sum_{n= -\infty}^\infty a_n(z-z_0)^n, \qquad 0< \vert  z-z_0\vert  < p $$

We define the residue of $$f(z)$$ at $$z_0$$ to be the coefficient $$a_{-1}$$ of $$1/(z-z_0)$$ in this laurent expansion,

$$\text{Res}[f(z), z_0] = a_{-1} = \frac{1}{2 \pi i} \oint_{\vert  z-z_0\vert  =r} f(z) dz, $$

where $$r$$ is any fixed radius satisfying $$0<r<p.$$

## Residue Theorem

Let $$D$$ be a bounded domain in the complex plane with piecewise smooth boundary. Suppose that $$f(z)$$ is analytic on $$D \cup \partial D$$, except for a finite number of isolated singularities $$z_1, \cdots, z_m$$ in $$D$$. then

$$\int_{\partial D} f(z)dz = 2\pi i \sum_{j=1}^m \text{Res}[f(z),z_j] $$

### Rule 1
If $$f(z)$$ has a simple pole at $$z_0$$, then

$$\text{Res}[f(z),z_0] = \lim\limits_{z \to z_0} (z-z_0) f(z)$$

In this case the Laurent series of $$f(z)$$ is

$$f(z) = \frac{a_{-1}}{z- z_0} + [\text{analytic at } z_0] $$,

from which the rule follows immediately.

### Rule 2
If $$f(z)$$ has a double pole at z_0, then

$$\text{Res}[f(z),z_0] = \lim\limits_{z \to z_0} \frac{d}{dz}[(z-z_0)^2 f(z)] $$.

In this case the Laurent expansion is

$$f(z) = \frac{a_{-2}}{(z-z_0)^2} + \frac{a_{-1}}{z-z_0} + a_0 + \cdots $$.

Thus

$$(z-z_0)^2 f(z) = a_{-2} + a_{-1}(z-z_0) + a_0 (z-z_0)^2  + \cdots $$

### Rule 3
If $$f(z)$$ and $$g(z)$$ are analytic at $$z_0$$, and if $$g(z)$$ has a simple zero at $$z_0$$, then

$$\text{Res}[\frac{f(z)}{g(z), z_0}] = \frac{f(z_0)}{g'(z_0)} $$.

### Rule 4

If $$g(z)$$ is analytic and has a simple zero at $$z_0$$, then

$$\text{Res}[\frac{1}{g(z)},z_0]= \frac{1}{g'(z_0)}$$

## 7.2 Integrals Featuring Rational Functions



NOTE: take some notes while doing hw problems

## 7.3 Integrals of Trigonometric Functions

NOTE: take some notes while doing hw problems

## 7.4 Integrands with Branch Points

NOTE: take some notes while doing hw problems
