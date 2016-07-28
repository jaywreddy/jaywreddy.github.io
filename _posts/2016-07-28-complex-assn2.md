---
layout: post
title: "Complex Analysis Assignment 2"
category: "Complex Analysis"
---

# P 170.1
Find all possible Laurent expansions centered at 0 of the following functions:

(a) $$\frac{1}{z^2-z}$$

Since there is a pole at $$z=1$$, we have the regions $$0 < \vert z\vert  < 1$$ and $$\vert z\vert  > 1$$

On the region $$0 < \vert z\vert <1$$:

$$\frac{1}{z^2 -z} = -\frac{1}{z}\frac{1}{1-z} $$
Using the geometric expansion of $$\frac{1}{1-z}$$:

$$-\frac{1}{z} \sum\limits_{k=0}^\infty z^k = -\sum\limits_{k=-1}^\infty z^k $$

On the region $$\vert z\vert >1$$:

$$\frac{1}{z^2-z}= \frac{1}{z^2}\frac{1}{1- \frac{1}{z}} $$

Using the geometric expansion of $$\frac{1}{1- \frac{1}{z}}$$:

$$\frac{1}{z^2}\sum\limits_{k=0}^\infty z^{-k} = \sum\limits_{k = -\infty}^{-2}z^k $$

(b) $$\frac{z-1}{z+1}$$
Since there is a pole at $$-1$$, we have the regions $$\vert z\vert <1$$ and $$\vert z\vert >1$$.

On $$\vert z\vert <1$$, we have:

$$\frac{z-1}{z+1} = 1 - \frac{2}{z+1} = 1 - \frac{2}{1- (-z)} = 1 - 2\sum\limits_{k=0}^\infty (-1)^n z^n$$

On $$\vert z\vert <1$$, we have:
$$\frac{z-1}{z+1} = 1 - \frac{2}{z+1} = 1 - \frac{2}{z}\frac{1}{1 - (-\frac{1}{z})} $$

Using the geometric expansion of the second term, we have:

$$1 - \frac{2}{z} \sum\limits_{k=0}^\infty (-1)^k z^{-k} = 1 - \frac{2}{z} \sum\limits_{k=-\infty}^0 (-1)^k z^k $$

And finally, decrementing the sum by factoring out the last z, we have

$$  1 - 2 \sum\limits_{k=-\infty}^{-1} (-1)^k z^k $$

(c) $$\frac{1}{(z^2 - 1)(z^2 - 4)}$$

The function has poles at 1 and 2, so we have three regions: $$\vert z\vert <1$$, $$1<\vert z\vert <2$$, and $$\vert z\vert >2$$

First, on the region $$\vert z\vert <1$$

Using the partial fraction expansion:

$$\frac{1}{(z^2-1)(z^2-4)} =  -\frac{1}{3}\frac{1}{z^2-1} + \frac{1}{3}\frac{1}{z^2-4}$$

We have

$$ -\frac{1}{3}\frac{1}{z^2-1} + \frac{1}{3}\frac{1}{z^2-4} = \frac{1}{3}\frac{1}{1-z^2} - \frac{1}{12}\frac{1}{1- z^2/4}$$

Then using the geometric series for the first and second terms we obtain

$$\frac{1}{3}\sum\limits_{k=0}^\infty z^{2k} - \frac{1}{12} \sum\limits_{k=0}^\infty \frac{z^{2n}}{4^n}$$

Finally, simplifying the summations, we get the expression
$$\frac{1}{12}\sum\limits_{k=0}^\infty (4 - 4^{-k})z^{2k} $$

For the region $$1<\vert z\vert <2$$

We begin with the same partial fractions expansion

$$ -\frac{1}{3}\frac{1}{z^2-1} + \frac{1}{3}\frac{1}{z^2-4} = -\frac{1}{3}\frac{1}{z^2(1-1/z^2)} - \frac{1}{12}\frac{1}{1-z^2/4}$$

And substituting the geometric expansions, we get:

$$-\frac{1}{3 z^2} \sum\limits_{k=0}^\infty z^{-2k} - \frac{1}{12}\sum\limits_{k=0}^\infty \frac{z^{2k}}{4^k} $$

Lastly, shifting the first summation to account for the extra $$z^{-2}$$ term

$$- \frac{1}{3} \sum\limits_{k=-\infty}^{-1} z^{2k} - \frac{1}{12} \sum\limits_{k=0}^\infty \frac{z^{2n}}{4^n} $$

For the final region $$\vert z\vert >2$$, we begin with the now familiar partial fraction expansion.

$$-\frac{1}{3}\frac{1}{z^2-1} + \frac{1}{3}\frac{1}{z^2-4} = - \frac{1}{3} \frac{1}{z^2(1-1/z^2)} + \frac{1}{3}\frac{1}{z^2(1-4/z^2)} $$

Using geometric series expansions and factoring the remaining $$z^{-2}$$ term into the series indices:

$$-\frac{1}{z^2}\sum\limits_{k=0}^\infty z^{-2k} + \frac{1}{3z^2}\sum\limits_{k=0}^\infty 4^k z^{-2k} = -\frac{1}{3}\sum\limits_{k=-\infty}^{-1}z^{2k} + \frac{1}{3}\sum\limits_{k=-\infty}^{-1}4^{-1-k} 2^k$$

And finally, combining the summations, we get:

$$\frac{1}{3}\sum\limits_{k=-\infty}^{-1}(4^{-1-k}-1)z^{2k} $$

# P 170.2
For each of the function sin Exercise 1, find the Laurent expansion centered at $$z=-1$$ that converges at $$z= \frac{1}{2}$$. Determine the largest open set on which each series converges.

(a) $$\frac{1}{z^2-z}$$

Using partial fractions expansion, we get:

$$- \frac{1}{z} + \frac{1}{z-1} $$

For the first term, we can find the laurent expansion at $$z= -1$$:

$$- \frac{1}{z} = - \frac{1}{z+1-1} = -\frac{1}{z+1}\frac{1}{1 - \frac{1}{z+1}}= -\frac{1}{z+1}\sum\limits_{k=0}^\infty \frac{1}{(z+1)^k} $$

Reordering the indices of the sum, we finally get
$$- \frac{1}{z} = - \sum\limits_{-\infty}^{-1} (z+1)^k $$

Next, we look at the second term: $$\frac{1}{z-1}$$

$$\frac{1}{z-1}= \frac{1}{z+1-2} = - \frac{1}{2}\frac{1}{1-(z^2+1)/2} = -\frac{1}{2}\sum\limits_{k=0}^\infty \frac{(z+1)^n}{2^n} $$

Offsetting the indices to account for the extra $$2^{-1}$$ term, we have

$$ -\sum\limits_{k=0}^\infty \frac{(z+1)^n}{2^{n+1}} $$

Thus we have obtained expressions in the form of

$$\frac{1}{z^2-z} = \sum\limits_{k=-\infty}^\infty a_n (z+1)^n $$

Where $$a_n$$ is:

$$-1 \qquad \text{if } n \leq -1, \qquad -\frac{1}{2^{n+1}} \qquad \text{else}  $$

Since the original function has poles at 0 and 1, we can see that the function converges for $$1 < \vert z+1\vert < 2$$

(b) $$\frac{z-1}{z+1}$$

$$\frac{z-1}{z+1} = 1 - \frac{2}{z+1} $$

so we can see we have an expression where:

$$\frac{z-1}{z+1} = \sum\limits_{k=-\infty}^\infty a_n (k+1)^n $$

Where $$a_n$$ is:

$$-2 \qquad \text{if }n=0, \qquad 1 \qquad \text{if }n=0, \qquad 0 \text{ else} $$

Since the function has a pole at $$-1$$, the series centered at $$-1$$ converges for $$0 < \vert z+1\vert  < \infty$$

(c) $$\frac{1}{(z^2-1)(z^2-4)}$$

Obtaining a full partial fractions expansion, we get:

$$-\frac{1}{6}\frac{1}{z-1} + \frac{1}{6}\frac{1}{z+1} + \frac{1}{12}\frac{1}{z-2} - \frac{1}{12}\frac{1}{z+2} $$

Expressing each of the terms in terms of $$z+1$$

$$ -\frac{1}{6}\frac{1}{z+1 -2} + \frac{1}{6}\frac{1}{z+1} + \frac{1}{12}\frac{1}{z+1 -3} - \frac{1}{12}\frac{1}{z+1 +1}$$

And dividing by the trailing terms in the denominator of each expression we get:

$$ \frac{1}{12}\frac{1}{1 - (z+1)/2} + \frac{1}{6}\frac{1}{z+1} - \frac{1}{36}\frac{1}{1 -(z+1)/3} - \frac{1}{12}\frac{1}{z+1}\frac{1}{1 +1/(z+1)}$$

Substituting geometric series...

$$ \frac{1}{12}\sum\limits_{k=0}^\infty \frac{(z+1)^k}{2^k}
 + \frac{1}{6}\frac{1}{z+1}
 - \frac{1}{36}\sum\limits_{k=0}^\infty \frac{(z+1)^k}{3^k}
 + \frac{1}{12}\sum\limits_{-\infty}^{k=-1} (-1)^n (z+1)^n$$

 Finally, combining like terms we get:

 $$\sum\limits_{k=0}^\infty (\frac{1}{12 * 2^k} - \frac{1}{36 * 3^k})(z+1)^k + \frac{1}{12}\sum\limits_{-\infty}^{k=0}(-1)^k(z+1)^k + (\frac{1}{6}- \frac{1}{12})(z+1^{-1}) $$

 And since the initial function has poles at -1 and 1, our expansion around $$-1$$ converges on $$0< \vert z+1\vert <2$$

# P 176.1 acegi
Find the isolated singularities of the following functions, and determine whether they are removable, essential, or poles. Determine the order of any pole, and find the principle part at each pole.

(a) $$\frac{z}{(z^2-1)^2}$$

The function has two poles of order 2 at $$-1, 1$$,

since for $$z=1$$,

We can express the function as $$\frac{b(z)}{(z-1)^2}$$ and $$b(z) = \frac{z}{(z+1)^2}$$.

Thus we can see the second order pole in the first expression.

For $$z=-1$$,

We can express the function as $$\frac{c(z)}{(z+1)^2}$$ and $$c(z) = \frac{z}{(z-1)^2}$$

To find the principle part at each pole, I use the partial fraction expansion:

$$\frac{z}{(z^2-1)^2} = \frac{1}{4(z-1)^2} - \frac{1}{4(z+1)^2} $$

So we can see that the principle part at $$z=1$$ is $$\frac{1}{4(z-1)^2}$$

And at $$z=-1$$ it is $$\frac{1}{4(z+1)^2}$$


(c) $$\frac{e^{2z} - 1}{z}$$

Observe that there exists an easy power series expansion, signifying that there is a removable discontinuity at 0.

$$\frac{e^{2z} - 1}{z} = \frac{1}{z}(-1 + \sum\limits_{k=0}^\infty)\frac{2^k z^k}{k!} = \frac{1}{z}\sum\limits_{k=1}^\infty \frac{2^k z^k}{k!} = \sum\limits_{k=1}^\infty \frac{2^{k+1} z^k}{(k+1)!} $$

(e) $$z^2 \text{sin}(\frac{1}{z})$$

Using the series expansion of $$\text{sin}(x)$$

$$z^2 \sum\limits_{k=0}^\infty \frac{(-1)^k}{(2n+1)!}z^{-(2k+1)} =\sum\limits_{k=0}^\infty \frac{(-1)^k}{(2n+1)!}z^{-(2k-1)}  $$

Thus there are infinitely many nonzero terms with negative exponents and we can classify the singularity as an essential singularity.

(g) $$\text{Log}(1-\frac{1}{z})$$

There are no isolated singularities since $$\text{Log}(z)$$ is analytic on $$\mathbb{C}\setminus(-\infty,0]$$, we can see that $$\text{Log}(1-\frac{1}{z})$$ is analytic on $$\mathbb{C}\setminus(0,1]$$. So there are no isolated singularities in the domain.

(i) $$e^{1/(z^2+1)}$$

I observe that there are singularities at $$z=-i$$ and $$z=i$$

Using the partial fractions expansion:
$$\frac{1}{z^2+1} = \frac{i}{2}[\frac{1}{z+i} - \frac{1}{z-i}] $$

We can split the function as so:

$$e^{\frac{1}{z^2+1}} = e^{\frac{i}{2}\frac{1}{z+i}}e^{-\frac{i}{2} \frac{1}{z-i}} $$

And using the series expansion of $$e^x$$:

$$e^x = \sum\limits_{n=0}^\infty \frac{x^n}{n!} $$

We can expand either term of the equation:

$$ e^{\frac{i}{2}\frac{1}{z+i}} \sum\limits_{k=0}^\infty \frac{(-i)^n}{2^n n!} \frac{1}{(z-i)^n}$$

or
$$ e^{\frac{i}{2}\frac{1}{z-i}} \sum\limits_{k=0}^\infty \frac{(-i)^n}{2^n n!} \frac{1}{(z+i)^n}$$

We can observe that in the expression, the exponential in front is analytic and non-zero at $$z=i$$, but the series contains infinitely many negative exponents, so there is an essential singularity at $$z=i$$.

Similarly in the second expression, the exponential in front is analytic and non-zero at $$z=-i$$, but the series contains infinitely many negative exponents, so there is an essential singularity at $$z=-i$$.

# P 176.3
Consider the function $$f(z)=\text{tan}(z)$$ in the annulus $$\{3 < \vert z\vert  < 4\}$$. Let $$f(z)= f_0(z)+f_1(z)$$ be the Laurent decomposition of $$f(z)$$, so that $$f_0(z)$$ is analytic for $$\vert z\vert <4$$, and $$f_1(z)$$ is analytic for $$\vert z\vert >3$$ and vanishes at $$\infty$$.

(a) Obtain an explicit expression for $$f_1(z)$$.

Since $$\text{tan}(z) = \frac{\text{sin}(z)}{\text{cos}(z)}$$, and cosine has simple zeros at $$\pm \frac{\pi}{2}$$
, $$\text{tan}(z)$$ has simple poles at $$\pm \frac{\pi}{2}$$. These are the only poles on the disk $$\vert z\vert <4$$.

To find the principal part, I use the sin and cos series expansions:

$$\text{sin}(z) =  z - z^3/3! + O(z^5)$$

$$\text{cos}(z) =  1 - z^2/2! + O(z^4)$$

To find the principle part of the pole at $$z = - \pi/2$$,
I see that
$$\text{tan}(z) = \frac{\text{sin}(z)}{\text{cos}(z)} = - \frac{\text{cos}(z+ \pi/2)}{\text{sin}(z + \pi/2)} = - \frac{1 - (z+\pi/2)^2/2! + O((z+\pi/2)^4)}{(z+\pi/2) - (z+\pi/2)^3/3! + O((z+\pi/2)^5)}$$

$$\text{tan}(z) = -(z+\pi/2)^{-1} + \frac{1}{3}(z+\pi/2)+ O((z+\pi/2)^3) $$

So we can see the principal part at $$z=-\pi/2$$ is $$-(z+\pi/2)^{-1}$$.

We can perform a similar procedure to find the principal part at $$z=\pi/2$$:

$$\text{tan}(z) = \frac{\text{sin}(z)}{\text{cos}(z)} = - \frac{\text{cos}(z- \pi/2)}{\text{sin}(z - \pi/2)} = - \frac{1 - (z-\pi/2)^2/2! + O((z-\pi/2)^4)}{(z-\pi/2) - (z-\pi/2)^3/3! + O((z-\pi/2)^5)}$$

$$\text{tan}(z) = -(z-\pi/2)^{-1} + \frac{1}{3}(z-\pi/2)+ O((z-\pi/2)^3) $$

So we can see the principal part at $$z=\pi/2$$ is $$-(z-\pi/2)^{-1}$$.

So if we define $$f_1(z) = -(z+\pi/2)^{-1} + -(z-\pi/2)^{-1}$$,

Then $$f_0(z) = f(z)-f_1(z)$$ is analytic on the disk $$\vert z\vert <4$$, by the properties of the principal part.

Thus, since we can see that $$f_1(z)$$ is analytic on $$\vert z\vert >3$$, by uniqueness these are the $$f_1,f_2$$ of the Laurent decomposition.

(b) Write down the series expansion for $$f_1(z)$$, and determine the largest domain on which it converges.

From the previous part, we have $$f_1(z) = -(z+\pi/2)^{-1} + -(z-\pi/2)^{-1}$$

Combining the fractions, we get
$$ f_1(z) = - \frac{z+ \pi/2 + z - \pi/2}{z^2 - \pi/4}
= - \frac{2z}{z^2 - \pi^2/4} = -\frac{2z}{z^2}\frac{1}{1- \frac{\pi^2}{4 z^2}}
$$

Using geometric series expansion, we get

$$f_1(z) = \frac{-2}{z}\sum\limits_{k=0}^\infty (\frac{\pi^2}{4})^k \frac{1}{z^{2k}} = -2\sum\limits_{k=0}^\infty (\frac{\pi^2}{4})^k \frac{1}{z^{2k+1}}$$

Since the series can only converge if the terms vanish at infinity, we can see that $$\vert z\vert >\pi/2$$, or the first term in the summation will dominate.

Thus the series converges for $$\vert z\vert >\pi/2$$.

(c) Obtain the coefficients $$a_0, a_1, \text{and } a_2$$ of the power series expansion of $$f_0(z)$$.

We know from the power series expansion of $$\text{tan}(z)$$:

$$\text{tan}(z)= z + O(z^3) $$

that the first three coefficients of $$\text{tan}(z)$$ are:

$$a_0 = 0, a_1 = 1, a_2 = 0 $$

Additionally, we know that $$a_n = 0$$ for $$n\geq 0$$ in $$f_1$$ by examining its powerseries.

Therefore, since $$f_0 = f - f_1$$, we can see that the coefficients of $$f_0$$ are:

$$a_0= 0, a_1 = 1, a_2 =0 $$

(d) What is the radius of convergence of the power series expansion for $$f_0(z)?$$

Since $$f_0(z) = f(z)-f_1(z)$$ has poles at $$\pm 3\pi/2$$, and is analytic for $$\vert z\vert <3$$ (as stated in part a), the series converges for $$\vert z\vert < 3\pi/2$$ and therefore has a radius of convergence of $$2\pi/2$$.


# P 181.1 acd
Find the partial fractions decompositions of the following functions.
(a) $$\frac{1}{z^2-z}$$

We want to achieve the following form:
$$\frac{1}{z^2-z} = \frac{A}{z} + \frac{B}{z-1}$$

$$\frac{1}{z^2-z} = \frac{Az-A + Bz}{z-1} $$

Thus we can see $$A = -1, B=1$$,

and the partial fractions expansion is

$$\frac{-1}{z} + \frac{1}{z-1} $$


(c) $$\frac{1}{(z+1)(z^2+2z+1)}$$

Looking at the poles of the function, we want to achieve the following form:

$$\frac{1}{(z+1)(z^2+2z+1)} = \frac{A}{z+1} + \frac{B}{z+1+i} + \frac{C}{z+1-i} $$

Cross-multiplying, we obtain

$$\frac{1}{(z+1)(z^2+2z+1)} = \frac{Az^2 + 2Az + 2A + +Bz^2 + 2Bz + B - Biz -Bi + Cz^2 + 2Cz +C +Ciz +Ci}{z^2 + 2z+1} $$

Observing that the the only terms containing $$i$$ on the right-hand side must cancel, we see that $$B=C$$, and so the expression simplifies:

$$1 = (A+2B)z^2 +2(A+2B)z + 2A +2B$$

This is fulfilled by $$A=1$$, $$B=C= -\frac{1}{2}$$

So we get the partial fractions expansion:

$$\frac{1}{z+1} - \frac{1/2}{z+1+i} - \frac{1/2}{z+1-i} $$



(d) $$\frac{1}{(z^2+1)^2}$$

Looking at the poles of the function, we know our solution will take the form

$$\frac{A}{(z+i)^2} + \frac{B}{z+i} + \frac{C}{(z-i)^2} + \frac{D}{z-i} $$

Thus we must solve:

$$1 = A(z-i)^2 + B(z-i)(z+i)^2 + C(z+i)^2 + D (z+i)(z+i)^2 $$

Expanding the expressions for each unknown:

$$A(z-i)^2 = Az^2 - 2Aiz -A $$

$$B(z-i)^3 =Bz^3 - 3Biz^2   -3Bz + Bi $$

$$C(z+i)^2 = Cz^2 + 2Ciz -C $$

$$D(z+i)(z-i)^2 = Dz^3 - Diz^2  +Dz - Di $$

Grouping by like terms, we get the following simultaneous equations:

$$z^3 :\qquad B+D=0 $$
$$z^2: \qquad A - 3Bi + C - Di =0 $$
$$z: \qquad -2Ai -3B +2Ci +D = 0 $$
$$1: \qquad -A + Bi - C -Di = 1 $$

And solving them yields the following partial fraction expansion:

$$-\frac{1}{4(z-i)^2} - \frac{i}{4(z+i)} - \frac{1}{4(z+i)^2} + \frac{i}{4(z-i)} $$


# P 182.3
Let $$V$$ be the complex vector space of functions that are analytic on the extended complex plane except possibly at the points $$0$$ and $$i$$, where they have poles of order at most two. What is the dimension of $$V$$? Write down explicitly a vector space basis for $$V$$.  

A function that is analytic except for a finite number of points is meromorphic by definition. We also that any meromorphic function on the extended complex plane is rational, so we only need consider rational functions, meaning that they can be decomposed via partial fractional decomposition into a combination of the principal parts of the poles (including the pole at infinity).

Thus the function can be decomposed as:

$$f(z) = P_\infty(z) +\sum P_j(z) $$

Notice that since the function is analytic on the extended complex plane, it does not have an isolated singularity at $$\infty$$. However, the principal part of $$f(z)$$ at $$\infty$$ could still contain a constant term. This constant term provides one dimension in the vector space.

Additionally, we know the forms of the principal components of the two other singularities, $$0$$ and $$i$$

Since they have poles of degree at most two, they must be of the forms:
$$P_0(z) = \frac{\alpha_1}{z} + \frac{\alpha_2}{z^2}$$

and

$$P_i(z) = \frac{\alpha_1}{z-i} + \frac{\alpha_2}{(z-i)^2} $$

Respectively.

Each of the other principal components adds 2 degrees to our vector space.

Thus, the vector space has degree 5.

And we can show an explicit basis:

$$[1, \frac{1}{z}, \frac{1}{z^2}, \frac{1}{z-i}, \frac{1}{(z-i)^2}] $$
