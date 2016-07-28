---
layout: post
title: Fourier Analysis Assignment 1
author: Jay Reddy 
category: "Fourier Analysis"
---

# P 18:

Let $$f$$ and $$g$$ be $$2\pi$$-periodic, piecewise smooth functions having Fourier series $$f(x) = \sum_n \alpha_n e^{inx}$$ and $$g(x)= \sum_n \beta_n e^{inx}$$, and define the convolution of $$f$$ and $$g$$ to be $$f * g(x) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t)g(x-t)dt$$. Show that the complex form of the Fourier series for $$f * g$$ is

$$ f * g(x)= \sum_{n=-\infty}^\infty \alpha_n \beta_n e^{inx}$$

----

Substituting the series representation for $$g(x-t)$$ yields

$$ f * g(x) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t)(\sum_{n} \beta_n e^{in(x-t)})dt$$
$$ f * g(x) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t)(\sum_{n} \beta_n e^{inx} e^{-int})dt$$

Since $$e^{int}, \ n = \cdots, -1, 0,1,\cdots$$ are orthogonal, we can separate the series for $$g(x)$$ since all of the cross terms will be zero.

$$ f * g(x) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t)(\sum_{n} \beta_n  e^{-int}) (\sum_n e^{inx} )dt$$

If we define a new function $$h(x) = \sum_{n} \overline{\beta_n} e^{inx}$$, we can see that $$\overline{h(x)} = \sum_{n} \beta_n  e^{-inx}$$ which we can plug in to yield:

$$ f * g(x) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t)\overline{h(t)} (\sum_n e^{inx} )dt$$

Since the series does not contain $$t$$, we can separate it from the integral.

$$ f * g(x) = (\frac{1}{2\pi} \int_{-\pi}^\pi f(t)\overline{h(t)} dt) \ (\sum_n e^{inx} )$$

Then by Parseval's Equation,
$$\frac{1}{2 \pi} \int_{-\pi}^\pi f(t) \overline{g(t)}dt = \sum_{n = - \infty}^\infty \alpha_n \overline{\beta_n}$$

So we can plug in:
$$ f * g(x) = (\frac{1}{2\pi} \int_{-\pi}^\pi f(t)\overline{h(t)} dt) \ (\sum_n e^{inx} ) = \sum_{n = - \infty}^\infty \alpha_n \overline{\overline{\beta_n}} \ (\sum_n e^{inx})$$

$$ f * g(x) =  \sum_{n = - \infty}^\infty \alpha_n \beta_n \ \sum_n e^{inx}$$

Again because of the orthogonality of the exponentials, we can recombine the sums since all of the cross terms are $$0$$.

$$ f * g(x)= \sum_{n=-\infty}^\infty \alpha_n \beta_n e^{inx}$$

End Proof.


# P 30:

Prove the following version of the Riemann-Lebesgue Lemma for infinite intervals: Suppose $$f$$ is a continuous function on $$a \leq t < \infty$$ with $$\int_a^\infty \vert f(t)\vert dt < \infty$$; show that

$$ \lim\limits_{n\to \infty} \int_a^\infty f(t) \text{cos}(nt)dt = \lim\limits_{n \to  \infty} \int_a^\infty f(t)\text{sin}(nt) dt = 0$$

Hint: break up the interval $$a \leq \infty$$ into two intervals: $$a \leq t \leq M$$ and $$M \leq t < \infty$$ where $$M$$ is chosen so large that $$\int_M^\infty \vert f(t)\vert dt$$ is less than $$\epsilon/2$$; apply the usual Riemann-Lebesgue Lemma to the first interval.

----

Since $$\int_a^\infty \vert f(t)\vert  < \infty$$, $$\exists C \in \mathbb{R}$$ s.t. $$\int_a^\infty \vert f(t)\vert  dt = C$$

This implies $$\lim\limits_{M \to \infty} \int_{a}^M  = C$$ and therefore $$\forall \frac{\epsilon}{2}, \exists M_0$$ s.t. for $$M > M_0, \ \vert \int_a^M\vert f(t)\vert dt - C\vert < \frac{\epsilon}{2}$$

It follows that for $$M > M_0, \ \int_M^\infty \vert f(t)\vert < \frac{\epsilon}{2}$$ (Lemma 1)

We next split the integrals into two intervals:

$$\lim\limits_{n \to \infty} \int_a^\infty f(t) \text{cos}(nt)dt = \lim\limits_{n \to \infty} \int_a^M f(t) \text{cos}(nt)dt + \int_M^\infty f(t) \text{cos}(nt)dt $$

$$\lim\limits_{n \to \infty} \int_a^\infty f(t) \text{sin}(nt)dt = \lim\limits_{n \to \infty} \int_a^M f(t) \text{sin}(nt)dt + \int_M^\infty f(t) \text{sin}(nt)dt $$

In each case, we can bound the second term by $$\frac{\epsilon}{2}$$ for a fitting choice of $$M$$ since $$\vert \text{cos}(nt)\vert ,\vert \text{sin}(nt)\vert  \leq 1$$:

$$\int_M^\infty f(t) \text{sin}(nt)dt \leq \int_M^\infty \vert f(t)\vert  dt  < \frac{\epsilon}{2}$$

$$\int_M^\infty f(t) \text{cos}(nt)dt \leq \int_M^\infty \vert f(t)\vert  dt  < \frac{\epsilon}{2}$$

For $$M > M_0$$, which we can choose for arbitrary epsilon according to Lemma 1.

So for arbitrary $$\frac{\epsilon}{2}$$, we can choose $$M$$ s.t.

$$\lim\limits_{n \to \infty} \int_a^\infty f(t) \text{cos}(nt)dt < \lim\limits_{n \to \infty} \int_a^M f(t) \text{cos}(nt)dt + \frac{\epsilon}{2} $$

$$\lim\limits_{n \to \infty} \int_a^\infty f(t) \text{sin}(nt)dt < \lim\limits_{n \to \infty} \int_a^M f(t) \text{sin}(nt)dt + \frac{\epsilon}{2}$$

To bound the first portion of the interval, consider the Riemman-Lebesgue Lemma, which states:

$$ \lim\limits_{n \to \infty} \int_a^b f(x) \text{cos}(nx) dx = \lim\limits_{n \to \infty} \int_a^b f(x) \text{sin}(nx) dx = 0 $$

And so for arbitrary $$\frac{\epsilon}{2},$$ we can choose $$n$$ s.t.

$$\int_a^M f(x)\text{cos}(nx) < \frac{\epsilon}{2}$$

$$\int_a^M f(x)\text{sin}(nx) < \frac{\epsilon}{2}$$

This is independent of our choice of $$M$$.

Plugging in above, we know that for arbitrary $$\epsilon$$, we can chooose $$n > n_0$$ s.t.

$$\int_a^\infty f(t) \text{cos}(nt)dt < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon $$

$$\int_a^\infty f(t) \text{sin}(nt)dt < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon $$

Thus, by making $$n$$ large enough, the integral can be made arbitrarily small. It therefore follows by the definition of a limit that

$$ \lim\limits_{n\to \infty} \int_a^\infty f(t) \text{cos}(nt)dt = \lim\limits_{n \to  \infty} \int_a^\infty f(t)\text{sin}(nt) dt = 0$$

# P 40:

The goal of this problem is to prove Poisson's formula which states that if $$f(t)$$ is a piecewise smooth function on $$- \pi \leq t \leq \pi$$, then

$$u(r,\phi) - \frac{1}{2 \pi} \int_0^{2\pi} f(t) \frac{1 - r^2 dt}{1 - 2 \text{cost}(\phi - t) + r^2} $$

for $$0 \leq r \leq 1, 0 \leq \phi \leq 2\pi$$ solves Laplace's equation

$$\nabla u = u_{xx} + u_{yy} = 0 $$

in the unit disc $$x^2 + y^2 \leq 1$$ (in polar coordinates $$\{ r \leq 1\}$$) with boundary values $$u(r=1,\phi) = f(\phi), -\pi \leq \phi \leq \pi$$ Follow the outline given to establish this formula.

----

A. Show that the functions $$u(x,y) = (x+iy)^n$$ and $$u(x,y) = (x-iy)^n$$ both solve Laplace's equation for each value of $$n=0,1,2,\cdots$$ Using complex notation $$z= x+ iy$$, these solutions can be written as $$u(z)= z^n$$ and $$u(z)= \overline{z}^n$$

----

First, I will show that $$u(x,y) = (x+iy)^n$$ is a solution:

$$ u_{x} = n(x+iy)^{n-1}$$
$$ u_{xx} = n(n-1)(x+iy)^{n-2}$$
$$ u_y = in(x+iy)^{n-1}$$
$$ u_{yy} = i^2 n (n-1)(x+iy)^{n-2} = - u_{xx}$$

So $$u_{xx}+ u_{yy} = u_{xx} - u_{xx} = 0$$, satisfying Laplace's Equation.

Next, I will show that $$u(x,y) = (x-iy)^n$$ is a solution:
$$ u_{x} = n(x-iy)^{n-1}$$
$$ u_{xx} = n(n-1)(x-iy)^{n-2}$$
$$ u_y = -in(x-iy)^{n-1}$$
$$ u_{yy} = (-1)^2 i^2 n (n-1)(x-iy)^{n-2} = - u_{xx}$$

So $$u_{xx}+ u_{yy} = u_{xx} - u_{xx} = 0$$, satisfying Laplace's Equation.

B. Show that any finite sum of the form $$\sum_{n=0}^N A_nz^n + Z_{-n}\overline{z}^n$$, where $$A_n$$ and $$A_{-n}$$ are real (or complex) numbers solves Laplace's equation. It is a fact that if the infinite series (i.e., as $$\vert N\vert  \to \infty$$) converges uniformly and absolutely for $$\vert z\vert =1$$. Write this function in polar coordinates with $$z = re^{i \phi}$$ and show that we can express it as $$\sum_{n=-\infty}^\infty A_n r^{\vert n\vert }e^{in \phi}$$

----

First, I will show that any finite sum of the above form satisfies Laplace's equations by showing that each term of the sum forms its own additive inverse.

First, consider a term of the form $$u = A_n z^n = A_n(x+iy)^n$$

$$ u_{xx} = A_n n(n-1) (x+iy)^{n-2}$$
$$ u_{yy} = -A_n n(n-1)(x+iy)^{n-2} = -u_{xx}$$

So the second partial derivatives of any term of the form $$A_n(x+iy)^n$$ form their own additive inverses and thus sum to 0.

Second, consider a term of the form $$u = A_n \overline{z}^n = A_{-n} (x - iy)^n$$

$$u_{xx} = A_{-n} n (n-1) (x-iy)^{n-2} $$
$$u_{yy} = -A_{-n} n (n-1)(x-iy)^{n-2} = -u_{xx} $$

The same property holds.

If we now examine the sum: $$U = \sum_{n=0}^N A_n z^n + A_{-n} \overline{z}^n$$ and take the partial derivatives:

$$U_{xx}, U_{yy}$$

The derivatives of each of the elements of the sum are evaluated independently. So the expression: $$U_xx + U_yy$$ becomes:

$$\sum_{n=0}^N (A_n z^n)_{xx} + (A_n z^n)_{yy} + (A_{-n} \overline{z}^n)_{xx} + (A_{-n} \overline{z}^n)_{yy}$$

Since the two partial derivatives of each term are additive inverses as shown above, they all cancel each other and the sum becomes:

$$\sum_{n=0}^N 0 = 0$$

Thus, any sum of the above form satisfies Laplace's Equation.

Now examining the infinite series $$\sum_{n=0}^\infty A_n z^n + A_{-n} \overline{z}^n$$

$$z^n = r^n e^{in\phi}, \overline{z}= r^n e^{-in\phi}$$

So substituting into our previous expression of the summation, we see:

$$ \sum_{n=0}^\infty A_nz^n + A_{-n}\overline{z}^n = \sum_{n=0}^\infty A_n r^n e^{in \phi}+ A_{-n}r^n e^{-in \phi}  $$
Which can be split into two sums:
$$ \sum_{n=0}^\infty A_n r^n e^{in \phi} + \sum_{n=0}^\infty A_{-n}r^n e^{-in \phi} $$

Rewriting the indices on the second summation:
$$ \sum_{n=0}^\infty A_n r^n e^{in \phi} + \sum_{n=-
\infty}^0A_{n}r^{\vert n\vert } e^{in \phi} $$

And combining the summations:

$$ \sum_{n=-\infty}^\infty A_n r^{\vert n\vert } e^{in \phi}$$

And the desired result is shown.


C. In order to solve Laplace's equaiton, we therefore must hunt for a solution of the form $$u(r,\phi)= \sum_{n=-\infty}^\infty A_n r^{\vert n\vert }e^{i n \phi}$$ with boundary condition $$u(r=1,\phi) = f(\phi)$$. Show the boundary condition is satisfied if $$A_n$$ is set to the Fourier coefficients of $$f$$ in complex form.

----

With $$u(r, \phi) = \sum_{n=-\infty}^\infty A_n r^\vert n\vert  e^{in \phi}$$:

$$ u(r=1,\phi) = \sum_{n= -\infty}^\infty A_n e^{in \phi}$$

To fulfill the boundary condition, we must find valus for $$A_n$$ s.t. $$u(r=1,\phi) = f(\phi)$$\\

By examining the form of the complex fourier series of $$f$$:

$$f(\phi) = \sum_{n=-\infty}^\infty\alpha_n e^{in \phi} $$

Where $$a_n$$ are the Fourier coefficients given by:

$$\alpha_n= \frac{1}{2\pi}\int_{-\pi}^\pi f(t)e^{-in t} dt $$

We can see by pattern matching that $$u(r=1,\phi)= f(\phi)$$ if $$A_n = \alpha_n$$, and so the boundary conditions are satisfied if $$A_n$$ is set to the complex Fourier coefficients of $$f$$.

D. Using the formula for the complex Fourier coefficients, show that if $$f$$ is real-valued, then $$A_{-n} = \overline{A_n}$$. Use this fact to rewrite the solution in the previous step as

$$ u(r,\phi) = \frac{1}{2 \pi} \text{Re}\{ \int_{-\pi}^\pi f(t) [2(\sum_{n=0}^\infty r^n e^{in(\phi -t)})-1] \}$$

----

$$ A_n = \frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{-int}$$
$$ A_{-n} = \frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{int}$$

$$ \overline{A_n} = \overline{\frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{-int}}$$
$$ \overline{A_n} = \overline{\frac{1}{2\pi}} \int_{-\pi}^\pi \overline{f(t)} \overline{e^{-int}}$$

$$ \overline{A_n} = \frac{1}{2\pi} \int_{-\pi}^\pi \overline{f(t)} e^{int}$$

If $$f(t)$$ is real-valued, then $$\overline{f(t)} = f(t)$$ so

$$ \overline{A_n} = \frac{1}{2\pi} \int_{-\pi}^\pi f(t) e^{int} = A_{-n}$$

Next, to rewrite the solution in the desired form:

$$ u(r,\phi) = \sum_{n=-\infty}^\infty A_n \ e^{in \phi}$$

$$ u(r,\phi) = \sum_{n=-\infty}^{-1} A_n  e^{in \phi} + \sum_{n=0}^\infty A_n e^{in \phi}$$

$$ u(r,\phi) = \sum_{n=1}^{\infty} A_{-n}  e^{in \phi} + \sum_{n=0}^\infty A_n e^{in \phi}$$

Shifting the terms in the summation and subtracting to account for double-counting $$A_0$$

$$ u(r,\phi) = \sum_{n=0}^{\infty} \overline{A_{n}}  e^{in \phi} - \overline{A_0} + \sum_{n=0}^\infty A_n e^{in \phi}$$

Next I make the argument that since $$u(r=1, \phi) = f(\phi)$$ and $$f$$ is real and $$r$$ has no bearing on the complex component of $$u$$, $$u = \text{Re}(u)$$. Therefore:

$$ u(r,\phi) = \text{Re} \{\sum_{n=0}^{\infty} \overline{A_{n}}  e^{in \phi} - \overline{A_0} + \sum_{n=0}^\infty A_n e^{in \phi} \}$$

And since $$\text{Re}(\overline{A_n}) = A_n$$:

$$ u(r,\phi) = \text{Re}(\sum_{n=0}^{\infty} A_{n}  e^{in \phi} - A_0 + \sum_{n=0}^\infty A_n e^{in \phi})$$

$$ u(r,\phi) = \text{Re}(2 \sum_{n=0}^{\infty} A_{n}  e^{in \phi} - A_0)$$

Plugging in for the fourier coefficients:

$$ u(r,\phi) = \text{Re}(2\sum_{n=0}^\infty \frac{1}{2\pi}\int_{-\pi}^\pi f(t) r^{\vert n\vert }e^{-int} dt \ e^{in \phi} - \frac{1}{2\pi}\int_{-\pi}^\pi f(t) dt)$$

Since the integration is over a separate variable from the summation, we can switch their order. Similarly, since $$e^{in\phi}$$ is constant w.r.t. the integration we can move it inside the integral:

$$ u(r,\phi) = \text{Re}( \frac{1}{2\pi}\int_{-\pi}^\pi 2 \sum_{n=-0}^\infty f(t) r^{\vert n\vert }e^{-int} e^{in \phi} dt  - \frac{1}{2\pi}\int_{-\pi}^\pi f(t) dt)$$

$$ u(r,\phi) =  \text{Re}(\frac{1}{2\pi}\int_{-\pi}^\pi 2 \sum_{n=0}^\infty f(t) r^{\vert n\vert }e^{in(\phi -t)} dt -  \frac{1}{2\pi}\int_{-\pi}^\pi f(t) dt)$$

And because the terms of $$f(t)$$ are independent of teh summation, we can rearrange the equation:

$$ u(r,\phi) =  \text{Re}(\frac{1}{2\pi}\int_{-\pi}^\pi f(t) 2 \sum_{n=0}^\infty  r^{\vert n\vert }e^{in(\phi -t)} dt -  \frac{1}{2\pi}\int_{-\pi}^\pi f(t) dt)$$

Finally, grouping like terms:

$$ u(r,\phi) =  \frac{1}{2\pi} \text{Re}(\int_{-\pi}^\pi f(t) [2 \sum_{n=0}^\infty  r^{\vert n\vert }e^{in(\phi -t)} -1] dt)$$

Yielding the desired form.

E. Now use the geometric series formula to rewrite the solution in the previous step as

$$ u(r,\phi) = \frac{1}{2\pi} \int_{-\pi}^\pi f(t) P(r, \phi -t)dt$$

where

$$ P(r,u) = \text{Re} \{ \frac{2}{1- r e^{iu}} -1\}$$

----

By pattern matching the first equation, we can see that $$P$$ takes the place of the summation in the first equation:

$$  P(r,\phi) = \text{Re} \{[2 \sum_{n=0}^\infty  r^{n}e^{in(\phi -t)} -1] \}$$

Using the geometric formula:

$$\sum_{n=0}^\infty a^n = \frac{1}{1-a}$$

Where $$a$$ in this case is $$r^{n}e^{in(\phi -t)}$$

So we can make a substitution for $$\frac{1}{1- re^{i(\phi -t)}}$$:

$$  P(r,\phi) = \text{Re} \{2 (\frac{1}{1- re^{i(\phi -t)}}) -1] \}$$

$$ P(r,\phi) = \text{Re} \{ \frac{2}{1- r e^{i\phi}} -1\}$$

Thus the equivalence is shown.

F. Rewrite $$P$$ as

$$ P(r,u) = \frac{1- r^2}{1 -2r \text{cos}(u) + r^2}$$

Use this formula together with the previous integral formula for $$u$$ to establish Poisson's formula.

$$ P(r,\phi) = \text{Re} \{ \frac{2}{1- r e^{i\phi}} -1\}$$

----

$$ P(r,\phi) = \text{Re} \{ \frac{2 - 1 +re^{i\phi}}{1- r e^{i\phi}} \}$$

Multiplying by the complex conjugate of the numerator:

$$ P(r,\phi) = \text{Re} \{ \frac{1 +re^{i\phi}}{1- r e^{i\phi}} \frac{1 - re^{i\phi}}{1 - re^{i \phi}}\}$$

$$ P(r,\phi) = \text{Re} \{ \frac{1 - r^2}{1+ 2re^{i\phi} + r^2} \}$$

$$ P(r,\phi) = \frac{1 - r^2}{1+ \text{Re} \{ 2re^{i\phi}\} + r^2} $$

$$ P(r,\phi) = \frac{1 - r^2}{1+ 2r\text{cos}(\phi) + r^2} $$

And finally plugging back in to the original integration formula we get:


$$u(r,\phi) = \frac{1}{2 \pi} \int_0^{2\pi} f(t) \frac{1 - r^2 dt}{1 - 2 \text{cos}(\phi - t) + r^2} $$

The desired result.
