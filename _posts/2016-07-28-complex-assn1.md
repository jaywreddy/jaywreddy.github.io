---
layout: post
title: Complex Analysis Assignment 1
category: "Complex Analysis"
---
#P 143.2
Determine for which $$z$$ the following series converge.

a. $$\sum_{k=1}^\infty (z-1)^k$$

Using the change of variables: $$w = (z-1),$$ the series can be rewritten:

$$\sum_{k=1}^\infty 1*w^k$$

And thus it is apparent that $$a_n = 1$$

Via the Ratio Test, the radius of convergence is:

$$ R = \lim\limits_{k \to \infty} \vert \frac{a_k}{a_{k+1}}\vert  = \frac{1}{1} = 1$$

So the function has a radius of convergence of 1 around $$z_0 = 1$$. So the series converges for the set $$z: \vert z-1\vert  < 1$$.

b. $$\sum_{k=10}^\infty \frac{(z-i)^k}{k!}$$

Using the change of variables: $$w = (z-i)$$, the series can be rewritten:

$$ \sum_{k=10}^\infty \frac{1}{k!}w^k$$

So $$a_n = \frac{1}{k!}$$

Using the ratio test:

$$R = \lim\limits_{k \to \infty } \vert \frac{a_k}{a_{k+1}}\vert   =\lim\limits_{k \to \infty } \frac{1}{k!}\frac{(k+1)!}{1} = \lim\limits_{k \to \infty } k+1 = + \infty$$

So the radius of convergence is infinite and the series converges for all values of z.

c. $$\sum_{m=0}^\infty 2^m(z-2)^m$$

Using the change of variables: $$w = z-2$$, the series can be rewritten

$$\sum_{m=1}^\infty 2^m w^m $$

and $$a_m = 2^m$$

Thus using the root test:

$$R = \frac{1}{\lim\limits_{m \to \infty} \sqrt[m]{2^m}} = \frac{1}{2} $$

Thus the series converges for $$z: \vert z-2\vert < \frac{1}{2}$$

d. $$\sum_{m=0}^\infty \frac{(z+i)^m}{m^2}$$

Using change of variables: $$w = z +i$$, we can see $$a_m = m^-2$$

Then using the root test we have:

$$R = \frac{1}{\lim\limits_{m \to \infty} \sqrt[m]2^{-m}} = 2$$

Therefore, the series converges for $$z: \vert z+i\vert <2$$

e. $$\sum_{n=1}^\infty n^n (z-3)^n$$

Using the change of variables: $$w= z-3$$, we can see that $$a_n = n^n$$.

Therefore, by the root test:

$$R = \frac{1}{\lim\limits_{n \to \infty} \sqrt[n]{n^n}} = \lim\limits_{n \to \infty} \frac{1}{n} = 0$$

Therefore, this series only converges for $$z=3$$, when each term becomes the trivial 0.

f. $$\sum_{n=3}^\infty \frac{2^n}{n^2} (z-2-i)^n$$

Using the change of variables: $$w = (z-2-i),$$ we can see that $$a_n = \frac{2^n}{n^2}$$

Using the root test:

$$R = \frac{1}{\lim \sqrt[n]{a_n}} = \frac{1}{2} $$

Thus the series converges for $$\vert z -2 -i\vert < \frac{1}{2}$$.

#P 143.5
What functions are represented by the following power series:

a. $$\sum_{k=1}^\infty k z^k$$

We know that

$$\sum_{k=0}^\infty z^k = \frac{1}{1-z}$$

And by differentiating we get:

$$\sum_{k=1}^\infty k z^{k-1} = \frac{1}{(1-z)^2} $$

And finally, by adding a factor of z we get the desired expression:

$$\sum_{k=1}^\infty kz^k = \frac{z}{(1-z)^2} $$


b. $$\sum_{k=1}^\infty k^2 z^k$$

Again, we start with the geometric series sum:

$$\sum_{k=0}^\infty z^k = \frac{1}{1-z}$$

This time, differentiating twice to get the desired $$k^2$$ term:

$$\sum_{k=2}^\infty (k^2 -k)z^{k-2} = \frac{2}{(1-z)^3}$$

And multiplying by $$z^2$$:

$$\sum_{k=2}^\infty (k^2 -k)z^k = \frac{2z^2}{(1-z)^3}$$

We add back the 1st term, which is conveniently 0:

$$\sum_{k=1}^\infty (k^2 -k)z^k = \frac{2z^2}{(1-z)^3}$$

And finally, we can add the expression for $$\sum_{k=1}^\infty k z^k$$ to yield the desired result:

$$\sum_{k=1}^\infty k^2 z^k = \frac{2 z^2}{(1-z)^3} + \frac{z}{(1-z)^2}$$



#P 147.3
Find the power series expansion of Log $$z$$ about the point $$z= i-2$$. Show that the radius of convergence of the series is $$R = \sqrt{5}$$. Explain why this does not contradict the discontinuity of Log $$z$$ at $$z=-2$$.

????

#P 148.7
Find the power series expansion of the principal branch $$\text{Tan}^{-1}(z)$$ of the inverse tangent function about $$z=0$$. What is the radius of convergence of the seres? Hint. Find it by integrating its derivative (a geometric series) term by term.

First I find the derivative:

$$ \frac{d}{dz} \text{Tan}^{-1} z = \frac{1}{1+z^2}$$

Which corresponds to an easy power series:

$$\frac{1}{1+z^2} = \sum_{k=0}^\infty (-z^2)^k $$

Integrating the first few terms:

$$\int 1 - z^2 + z^4 - \cdots = z - z^3/3 + z^5/5 - \cdots $$

We see that it follows a predictable pattern and thus:

$$\text{Tan}^{-1} z = \sum_{k=0}^\infty (-1)^k \frac{z^{2k+1}}{2k +1}$$

We can now apply the root test:

$$ R = \frac{1}{\lim\limits_{k \to \infty} \vert \frac{(-1)^k}{2k+1}\vert } = 1$$

And the Radius of convergence is 1.

#P 149.13
Prove the following version of L'Hospital's rule. If $$f(z)$$ and $$g(z)$$ are analytic, $$f(z_0) = g(z_0) = 0$$, and $$g(z)$$ is not identically zero, then

$$ \lim\limits_{z \to z_0} \frac{f(z)}{g(z)} = \lim\limits_{z \to z_0} \frac{f'(z)}{g'(z)}$$

in the sense that either both limits are finite and equal, or both limits are infinite.

soln given

#P 153.3

Show that

$$\frac{e^z}{1+z} = 1 + \frac{1}{2}z^2 - \frac{1}{3}z^3 + \frac{3}{8}z^4 - \frac{11}{30}z^5 + \cdots $$

Show that the general term of the power series is given by

$$a_n = (-1)^n [\frac{1}{2!} - \frac{1}{3!} + \cdots + \frac{(-1)^n}{n!}] $$

What is the radius of convergence of the series.

soln given

# P 157.1acegi
Find the zeros and orders of zeros of the following functions.

a. $$\frac{z^2 + 1}{z^2 -1}$$  

The function has simple zeros at $$\pm i$$, since these values make the numerator 0.

c. $$z^2 \text{sin} (z)$$

Due to the periodicity of the sin function, the function has simple zeros at $$n\pi, \ n \in \mathbb{Z}$$

Since sin has a simple zero at $$0$$, we can express it as the product:

$$\text{sin}(z) = z h(z) $$

For some analytic function $$h$$, where $$h(0) \neq 0$$.

Thus our function becomes $$z^3 h(z)$$, and we can see that there is in fact a zero of order 3 at $$z=0$$.

Thus, the function has a root of order 3 at $$z=0$$, and a simple root at $$z = n \pi, \ n\in Z,\ n \neq 0$$


e. $$\frac{\text{cos}(z)- 1}{z}$$

The function is only 0 when $$\text{cos}(z)=1$$, which happens when $$z = 2n\pi, \ n\in \mathbb{Z}$$.

Thus, the function has simple zeros at $$z = 2n\pi, \ n\in \mathbb{Z}$$.

g. $$e^z -1$$

By the power series for $$e^z-1$$:

$$z + \frac{z^2}{2!} + \frac{z^3}{3!} + \cdots $$

We can see that there is a simple zero at 0.

Additionally, since $$e^z$$ is periodic in the complex plane, we also have simple zeros whenever $$e^z=1$$, which happens for $$z = i 2\pi n \qquad n\in \mathbb{Z}$$.

i. $$\frac{\text{Log}(z)}{z}$$ (principal value)

We know the power series expansion of Log($$1-z$$):

$$-z -\frac{1}{2}z^2 - \frac{1}{3}z^3 - \cdots $$

By plugging in log$$(z) = \text{log}(1-(1-z))$$

$$-(1-z) - \frac{1}{2}(1-z)^2 - \frac{1}{3}(1-z)^3 - \cdots $$

We see that log$$(z)$$ has a simple zero at 1.

Thus the function only has a simple zero at 1, since the denominator does not contribute to the zeros.


# P 158.8
With the convention that the function that is identically zero has a zero of infinite order at each point, show that if $$f(z)$$ and $$g(z)$$have zeros of order $$n$$ and $$m$$ respectively at $$z_0$$, then $$f(z) + g(z)$$ has a zero of order $$k \geq \text{min}(n,m)$$. Show that strict inequality can occur here, but that equality holds whenever $$m \neq n$$

no soln
