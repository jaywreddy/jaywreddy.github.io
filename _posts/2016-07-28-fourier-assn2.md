---
layout: post
title: Fourier Analysis Assignment 2
category: "Fourier Analysis"
---
Jay Reddy HW 7 - Math 118
==========================

# Ex H2.1
Show that $$K = D^2 - x^2$$ is a symmetric operator on $$L^2(\mathbb{R})$$: for nice smooth functions $$f,g \in L^2 (\mathbb{R})$$ we have

$$\int_{-\infty}^\infty f(x)Kg(x)^* dx = \langle f, Kg \rangle = \langle Kf,g\rangle $$

First, I assume that for the class of "nice, smooth" functions,

$$\int_{-\infty}^\infty \vert f(x)\vert ^2 dx < \infty \implies \lim\limits_{\vert x\vert \to \infty} f(x) = 0$$

Therefore,

$$\langle f, Kg \rangle =\int_{-\infty}^\infty f(x) K g(x) dx = \int_{-\infty}^\infty f(x)g''(x) - f(x)x^2g(x) dx $$

Using integration by parts:

$$f(x)g'(x) - f'(x)g(x) \Big\vert _ {-\infty}^\infty + \int_{-\infty}^\infty f''(x)g(x) - x^2f(x)g(x) $$

Since the functions are "nice", I argue that most non-pathological functions that are square-integrable go to 0 at $$\pm\infty$$. (A search of the internet shows that there is no rigorous backing to this claim, in fact it is false even for infinitely differentiable functions - interesting.) However I think that this reasoning applies to the types of functions that we have been dealing with in this class.

This convenient hand-waving gets rid of the residuals and thus we can see:

$$\int_{-\infty}^\infty f''(x)g(x) - x^2f(x)g(x) $$

$$\int_{-\infty}^\infty Kf(x)g(x) dx = \langle Kf,g\rangle $$

# Ex H2.2

TODO

# Ex H2.3
Calculate the first three Hermite polynomials and use them to compute

$$\int_{-\infty}^\infty x^2 e^{-x^2} dx$$

We can calculate the hermite functions from the formula:

$$h_n(x) = \frac{(-1)^n}{n!}e^{x^2/2}D^n e^{-x^2}$$

The first three hermite functions are as follows:

$$h_0(x) = e^{-x^2/2} $$

$$h_1(x) = 2xe^{-x^2/2} $$

$$h_2(x) = (2x^2 -1)e^{-x^2/2} $$

And since the Hermite polynomials $$H_n$$ correspond to the hermite functions via:

$$h_n(x) = \frac{1}{n!}H_n(x)e^{-x^2/2}$$

We can see that the corresponding Hermite Polynomials are:

$$H_0(x) = 1 $$

$$H_1(x) = 2x$$

$$H_2(x) = 4x^2 -2 $$

Now to evaluate the integral, I notice that

$$x^2 e^{-x^2} = (h_2 +h_0)h_0/2 $$

So we can evaluate:

$$ \int_{-\infty}^\infty(h_2(x) +h_0(x))h_0(x)/2dx = \frac{1}{2} \int_{-\infty}^\infty h_2(x) h_0(x)dx + \frac{1}{2}\int_{-\infty}^\infty h_0(x) \cdot h_0(x)dx$$

$$= \langle h_2, h_0 \rangle /2 + \langle h_0, h_0\rangle/2 $$

The first term is zero by the orthogonality of the hermite functions:

$$= \langle h_0, h_0\rangle/2 = \vert \vert h_0\vert \vert ^2/2$$

We can evaluate the norm using the formula from the previous problem:

$$= \frac{\sqrt{\pi}}{0!}2^0/2 = \frac{\sqrt{\pi}}{2} $$

Thus

$$\int_{-\infty}^\infty x^2 e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$

# Ex H2.4

a. Show that

$$ - \langle Kf,f\rangle = \int_{-\infty}^\infty f'(x)^2 + x^2 f(x)^2 dx = \sum_{n=0}^\infty (2n+1)\frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2}$$

for real-valued $$f \in L^2(\mathbb{R})$$

I again use my assumption of "nice" functions from pt 1. namely:

$$\int_{-\infty}^\infty \vert f(x)\vert ^2 dx < \infty \implies \lim\limits_{\vert x\vert \to \infty} f(x)= 0$$

$$-\langle Kf,f \rangle = -\int_{-\infty}^\infty Kf(x) \cdot f(x) dx = -\int_{-\infty}^\infty f''(x)f(x) + x^2 f(x)^2$$

First, I get rid of the $$f''$$ term using integration by parts. However, this leaves the problem of the residuals. Based on my assumption of "nice" functions, it is easy to show that the derivative also has a limit of 0 at $$\pm \infty$$, as a consequence of the mean-value theorem (if the function approaches a finite limit at infinity, it will have a horizontal asymptote, and the limit of the differences will become 0).

$$= - \Big(f'(x)^2 \Big\vert _ {-\infty}^\infty - \int_{-\infty}^\infty f'(x)^2 + x^2 f(x)^2 dx \Big) = \int_{-\infty}^\infty f'(x)^2 + x^2 f(x)^2 dx$$

Thus showing the desired result.

To get the second result, consider the expansion of f:

$$f(x) = \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f,h_n \rangle h_n(x)$$

$$ -\langle Df,f \rangle = -\int_{-\infty}^\infty Kf(x)\cdot f(x) dx  = - \int_{-\infty}^\infty K \Big(\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle h_n \Big) f(x) dx$$

And by linearity of K,
$$=  - \int_{-\infty}^\infty \Big(\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle K h_n \Big) f(x) dx $$

By thm 3.
$$=  \int_{-\infty}^\infty \Big(\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle (2n+1)h_n  \Big) f(x) dx$$

$$=  \int_{-\infty}^\infty \Big(\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle (2n+1)h_n  \Big) \Big(\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle h_n  \Big) dx$$

Since the $$h_n$$ are orthogonal, we can combine the summations since the cross terms will be 0:

$$=  \int_{-\infty}^\infty \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^4} \langle f, h_n \rangle^2 (2n+1)h_n \cdot h_n  dx$$

Since the integration is over a different variable than the summation, we can switch their positions:

$$=   \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^4} \langle f, h_n \rangle^2 (2n+1) \Big(\int_{-\infty}^\infty h_n \cdot h_n  dx \Big)$$

Notice we are just taking the $$L^2$$ norm of $$h_n$$, which cancels with the first norm term.

$$=   \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^4} \langle f, h_n \rangle^2 (2n+1) \langle h_n, h_n\rangle$$

$$=   \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f, h_n \rangle^2 (2n+1) $$

$$ = \sum_{n=0}^\infty (2n+1)\frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2}$$

The desired result.

b. Prove the weak Heisenberg inequality

$$ \int_{-\infty}^\infty f'(x)^2 + x^2 f(x)^2dx \geq \int_{-\infty}^\infty f(x)^2 dx $$

Substituting our expression from the previous part:

$$ \sum_{n=0}^\infty (2n+1)\frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2} \geq \int_{-\infty}^\infty f(x)^2 dx $$

Plugging in the expansion for f:

$$f(x) = \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f,h_n \rangle h_n(x)$$

We get:

$$ \sum_{n=0}^\infty (2n+1)\frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2} \geq \int_{-\infty}^\infty \sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f,h_n \rangle h_n(x)\sum_{n=0}^\infty \frac{1}{\vert \vert h_n\vert \vert ^2} \langle f,h_n \rangle h_n(x) dx$$

Following the same simplification steps I used in the last section (combining the series and rearranging the sum and the integral), this expression becomes:

$$ \sum_{n=0}^\infty (2n+1)\frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2} \geq\sum_{n=0}^\infty \frac{\langle f,h_n \rangle^2}{\vert \vert h_n\vert \vert ^2} $$

With the expression in this form, we can easily see that this relation is true, since each of the terms on the left-hand side is greater than or equal to the corresponding term on the right-hand side.

Indeed, it appears that the only time there would be strict equality is if the only nonzero terms involved $$h_0$$.

# Ex H2.5

TODO

# Ex H3.1

TODO

# Ex H3.2
Use the Poisson Summation Formula to prove the Euler-Maclaurin summation formulas:

$$\sum_{n=0}^\infty = \frac{1}{2}f(0) + \int_{0}^\infty f(x) dx - \frac{1}{12}f'(0) + \frac{1}{720}f'''(0) - \cdots$$

for an even function $$f$$.

First, if $$f$$ is an even function, then we know that

$$\sum_{n = -\infty}^\infty f(n) = -f(0) + 2 \sum_{n=0}^\infty f(n) $$

rearranging, we get:

$$\sum_{n=0}^\infty f(n) = \frac{1}{2} f(0)+ \frac{1}{2} \sum_{n=-\infty}^\infty f(n)$$

Now, using Poisson's summation formula on the second term, we get:

$$\frac{1}{2} \sum_{n=-\infty}^\infty f(n) = \frac{\sqrt{2\pi}}{2} \sum_{n=\infty}^\infty \hat{f}(2\pi n)$$

$$\frac{1}{2} \sum_{n=-\infty}^\infty f(n) = \frac{\sqrt{2\pi}}{2} \hat{f}(0) +\frac{\sqrt{2\pi}}{2} \sum_{n \neq 0} \hat{f}(2\pi n ) $$

The first term, $$\frac{\sqrt{2\pi}}{2} \hat{f}(0)$$, can be rewritten as follows:

$$\frac{\sqrt{2\pi}}{2} \hat{f}(0) = \frac{\sqrt{2\pi}}{2} \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty f(x) e^{-i0x} dx =  \frac{1}{2} \int_{-\infty}^\infty f(x) dx$$

However, because f is even, this expression can be simplified even further:

$$ \frac{1}{2} \int_{-\infty}^\infty f(x) dx = \int_0^\infty f(x)  dx$$

Plugging back into our original expression, we now have:

$$ \sum_{n=0}^\infty f(n) = \frac{1}{2} f(0) + \int_0^\infty f(x)dx + \frac{\sqrt{2\pi}}{2} \sum_{k \neq 0} \hat{f}(2\pi k)$$

To simplify the next step in the equation, I am going to show an alternate expression for $$\hat{f}(n)$$:

We know

$$\hat{f}(k) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty f(x) e^{-ikx}dx$$

Since $$f$$ is even, this can equivalently be expressed:

$$ \hat{f}(k) = \frac{2}{\sqrt{2\pi}} \int_{0}^\infty f(x)e^{-ikx}dx$$  

Using integration by parts, we can expand $$\int_{0}^\infty f(x) e^{-ikx} dx$$ into terms containing derivatives of $$f$$, which we hope will lead to the form desired for the Eurler-Maclaurin Summation Formula.

$$\int_{0}^\infty f(x)e^{-ikx} dx  = \frac{if(x)e^{-ikx}}{k}\Big\vert _ 0^\infty  - \int_{0}^\infty \frac{if'(x)e^{-ikx}}{k} $$

Evaluating the first term, and applying a second iteration of integration by parts to the second, we get:

$$\int_{0}^\infty f(x)e^{-ikx} dx  = -\frac{if(0)}{k}+ \frac{f'(x)e^{-ikx}}{k^2}\Big\vert _ 0^\infty  - \int_{0}^\infty \frac{f''(x)e^{-ikx}}{k^2} $$

And by repeating this process, we obtain:

$$\int_{0}^\infty f(x)e^{-ikx} dx  = -\frac{if(0)}{k}-  \frac{f'(0)}{k^2} - \frac{if''(x)e^{-ikx}}{k^3}\Big\vert _ 0^\infty  - \int_{0}^\infty \frac{if'''(x)e^{-ikx}}{k^3} $$

And another one.

$$\int_{0}^\infty f(x)e^{-ikx} dx  = -\frac{if(0)}{k}-  \frac{f'(0)}{k^2} + \frac{if''(0)}{k^3} - \frac{f'''(x)e^{-ikx}}{k^4}\Big\vert _ 0^\infty  - \int_{0}^\infty \frac{if''''(x)e^{kx}}{k^4} $$

We can see that we can repeat this process forever to generate more terms in the series. Instead, I will truncate the expression:

$$\int_{0}^\infty f(x)e^{-ikx} dx  = -\frac{if(0)}{k}-  \frac{f'(0)}{k^2} + \frac{if''(0)}{k^3} + \frac{f'''(0)}{k^4}  - \cdots $$

And thus

$$\hat{f}(k) =  \frac{2}{\sqrt{2\pi}}\big[  -\frac{if(0)}{k}-  \frac{f'(0)}{k^2} + \frac{if''(0)}{k^3} + \frac{f'''(0)}{k^4}  - \cdots   \big]$$

If we plug the above into

$$ \sum_{n=0}^\infty f(n) = \frac{1}{2} f(0) + \int_0^\infty f(x)dx + \frac{\sqrt{2\pi}}{2} \sum_{k \neq 0} \hat{f}(2\pi k)$$

We get

$$ \sum_{n=0}^\infty f(n) = \frac{1}{2} f(0) + \int_0^\infty f(x)dx +  \sum_{k \neq 0} \big[  -\frac{if(0)}{2 \pi k}-  \frac{f'(0)}{(2\pi k)^2} + \frac{if''(0)}{(2\pi k)^3} + \frac{f'''(0)}{(2\pi k)^4}  - \cdots   \big]$$

I observe that we can remove all of the odd powers of $$k$$ from the summation since they will form their own additive inverses for the corresponding positive and negative values of $$k$$ in the summation.

$$ \sum_{n=0}^\infty f(n) = \frac{1}{2} f(0) + \int_0^\infty f(x)dx +  \sum_{k \neq 0} \big[ -  \frac{f'(0)}{(2\pi k)^2}  + \frac{f'''(0)}{(2\pi k)^4}  - \cdots   \big]$$

In the remaining terms, since $$k$$ is squared, the positive and negative values will be the same, so we can group like terms and rewrite the summation:

$$ \sum_{n=0}^\infty f(n) = \frac{1}{2} f(0) + \int_0^\infty f(x)dx -  \sum_{k=1}^\infty \frac{2}{(2\pi k)^2 } f'(0) +  \sum_{k=1}^\infty \frac{2}{(2\pi k)^4 } f'''(0) - \cdots$$

Finally, I note the following identities:

$$  \sum_{k=1}^\infty \frac{2}{(2\pi k)^2 } = \frac{1}{12} $$

$$ \sum_{k=1}^\infty \frac{2}{(2\pi k)^4 }  = \frac{1}{720}$$

To obtain the desired expression:

$$\sum_{n=0}^\infty = \frac{1}{2}f(0) + \int_{0}^\infty f(x) dx - \frac{1}{12}f'(0) + \frac{1}{720}f'''(0) - \cdots$$

This is the Euler-Maclaurin Summation formula. End proof.
