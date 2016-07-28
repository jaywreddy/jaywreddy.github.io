---
title: Complex Analysis Assignment 5
layout: post
category: "Complex Analysis"
---

# P 234.1
Suppose $$D$$ is a bounded domain with piecewise smooth boundary. Let $$f(z)$$ be meromorphic and $$g(z)$$ analytic on $$D$$. Suppose that both $$f(z)$$ and $$g(z)$$ extend analytically across the boundary of $$D$$, and that $$f(z)\neq 0$$ on $$\partial D$$. Show that

$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \sum\limits_{j=1}^n m_j g(z_j) $$

where $$z_1, \cdots, z_n$$ are the zeros and poles of $$f(z)$$, and $$m_j$$ is the order of $$f(z)$$ at $$z_j$$.

Since $$f(z)$$ is meromorphic, we know that it has a finite number of finite order, as well as a finite number of isolated zeros in $$D$$. Therefore, by the fundamental theorem of algebra, we can factorize $$f(z)$$ as

$$f(z) =\prod\limits_{j=1}^n (z-z_j)^{m_j}$$

Where $$z_1, \cdots, z_n$$ are the zeros and poles of $$f(z)$$, and $$m_j$$ are their order.

Then by the product rule,

$$f'(z) = \sum\limits_{j=1}^n m_j (z-z_j)^{m_j -1} * \prod_{k=1,k\neq j}^n (z-z_k)^{m_k}$$

And dividing by $$f(z)$$:

$$\frac{f'(z)}{f(z)} = \sum\limits_{j}^n \frac{m_j}{(z-z_j)}$$

So

$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \frac{1}{2 \pi i}\oint_{\partial D} \sum\limits_{j=1}^n \frac{m_j g(z)}{z-z_j} $$

We can evaluate the right hand side via the residue theorem:

$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \frac{1}{2 \pi i} 2 \pi i \text{Res} \Big [\sum\limits_{j=1}^n \frac{m_j g(z)}{z-z_j}] $$

$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \sum\limits_{j=1}^n \text{Res} \Big [ \frac{m_j g(z)}{z-z_j}] $$

Seeing as these are all simple poles, ($$g(z)$$ is analytic on the domain and has no poles), we can evaluate the residues by multiplying by $$(z-z_j)$$ and taking $$\lim\limits_{z \to z_j}$$


$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \sum\limits_{j=1}^n m_j g(z) \big\vert _ {z = z_j} $$

Thus, we can see

$$\frac{1}{2 \pi i} \oint_{\partial D} g(z) \frac{f'(z)}{f(z)}dz = \sum\limits_{j=1}^n m_j g(z_j)$$

# P 235.4
Let $$f(z)$$ be an analytic function on the open unit disk $$\mathbb{D}= \{\vert z\vert <1\}$$. Suppose there is an annulus $$U= \{r < \vert z\vert  < 1\}$$ such that the restriction of $$f(z)$$ to $$U$$ is one-to-one. Show that $$f(z)$$ is one-to-one  on $$\mathbb{D}$$.

First, consider a disk $$C = \{\vert z\vert <d\}, r<d<1$$. Since $$\partial C$$ is in the annulus, $$f(z)$$ is one-to-one on $$\partial C$$.

If we consider the path that $$f(z)$$ maps out on $$\partial C$$: $$f(z)\big\vert _ {\partial C}$$ we get some interesting results. First, the path can never cross itself, since this would violate the one-to-one nature of $$f$$ on $$\partial C$$.

So if we consider the increase in the argument of $$f(z)$$ on $$\partial C$$, we have three possibilities: $$0,\pm 2\pi$$. The path $$f(z)\big\vert _ {\partial C}$$ cannot encircle the origin more than once, as this would require a self-intersection in order to connect with its starting point.

Since $$f(z)$$ is analytic on $$C$$, there are actually only two possibilities: $$0,2\pi$$. Via the argument principle this value corresponds to the number of zeros minus the number of poles. Since $$f(z)$$ is analytic on $$C$$, it does not have any poles and thus the value must be positive. This means that $$f(z)$$ has zero or one roots on $$C$$.

Let's now extend the argument to look for roots of any function $$f(z)-w, w\in \mathbb{C}$$ within the disk $$C$$. We can do this via the argument principle by following the path that the function traces along $$\partial C$$: $$[f(z)-w]\big\vert _ {\partial C} = f(z)\vert _ {\partial C} - w$$. Since $$w$$ is a constant, it does not vary along the path. Thus, we can see that the path of any function $$f(z)-w$$ is simply a shifted version of $$f(z)$$.

Repeating our previous argument of the one-to-one and analyticity of $$f(z)$$ on $$\partial C$$, we can see that the increase in the argument along this path has only two possibilities, $$0,2\pi$$, corresponding to zero or one roots of $$f(z)-w$$ in $$C$$.  

This means that for all $$w\in \mathbb{C}$$, $$f(z)$$ attains the value of $$w$$ once or not at all in $$C$$. Thus, for all $$w\in \{f(z), z \in C\}$$, $$f(z)$$ attains the value exactly once. Thus, $$f(z)$$ is one-to-one on $$C$$.

Finally, we consider $$\lim\limits_{d \to 1} C = \mathbb{D}$$. We are free to take this limit without loss of generality of the previous claims, since the only two facts that we needed were that $$f(z)$$ is one-to-one on $$\partial C$$, and analytic on $$C$$, which remain true in the limit.

Thus, $$f(z)$$ is one-to-one on $$\mathbb{D}$$.

# P 245.1
Sketch the closed path $$\gamma(t)= e^{it}\sin(2t), 0\leq t \leq 2\pi$$, and determine the winding number $$W(\gamma, \varsigma)$$ for each point $$\varsigma$$ not on the path.

``` python
import numpy as np  
import matplotlib.pyplot as plt

# Define the Function
def gamma(t):
  return np.sin(2*t)*np.exp(t*1j)

# Take 10,000 samples form 0 to 2pi
dom = np.linspace(0,2*np.pi, 10000)

# Calculate gamma values
vals = np.array([gamma(t) for t in dom])

# Plot the result
plt.figure()
plt.plot(vals.real,vals.imag)
plt.savefig("imgcache/pathplot.png")
```

![Sketch of Path](/images/complex_analysis/pathplot.png)

Here we can see that there are 5 connected domains traced out by the path, one in each quadrant, and the unbounded region around the path.

For each of the domains in each quadrant, the winding number is $$1$$, and for the unbounded domain outside of the path, the winding number is $$0$$.

# P 245.3
Let $$f(z)$$ be analytic on an open set containing a closed path $$\gamma$$, and suppose $$f(z) \neq 0$$ on $$\gamma$$. Show that the increase in $$\text{arg } f(z)$$ around $$\gamma$$ is $$2\pi W(f \circ \gamma,0)$$

Increase in argument is defined :


$$W(f \circ \gamma ,0)= \frac{1}{2\pi i} \int_{f\circ\gamma} \frac{dz}{z}$$

Let's do a change of variables $$z = f(z)$$ and $$dz= f'(z)dz$$

$$ W(f \circ \gamma,0) = \frac{1}{2\pi i}\int_\gamma \frac{f'(z)}{f( z)}dz$$

This expression imposes the restriction that $$f(z)\neq 0$$ on $$\gamma$$ so that the integral is not indeterminate at any point.

This is the logarithmic integral, so we know.

$$\frac{1}{2\pi i}\int_{\gamma} \frac{f'(z)}{f(z)}dz = \frac{1}{2\pi i}\int_\gamma d \ \text{log}(f(z)) = \frac{1}{2\pi i} \int_\gamma d \ \text{log}\vert f(z)\vert + \frac{1}{2 \pi} \int_\gamma d\ \text{arg}(f(z))$$

And since the differential $$d\ \text{log}\vert f(z)\vert $$ is exact, its integral along a closed curve is 0. So we have
$$W(f \circ \gamma,0) = \frac{1}{2 \pi} \int_\gamma d\ \text{arg}(f(z))$$

And finally

$$2 \pi W(f \circ \gamma,0) = \int_\gamma d\ \text{arg}(f(z))$$

The right hand expression is the increase in the argument of $$f(z)$$ around $$\gamma$$. And so we have our desired result.


# P 245.7
Evaluate

$$ \frac{1}{2 \pi i}\int_\gamma \frac{dz}{z(z^2-1)}$$

where $$\gamma$$ is the closed path indicated in the figure. Hint. Either use Exercise 6, or proceed directly with partial fractions.

![Path](/images/complex_analysis/gammaimg.png)

Via partial fractions expansion, we get:

$$\frac{1}{z(z^2 -1)}  = \frac{a}{z} + \frac{b}{z-1} + \frac{c}{z+1}$$

Solving the equations:

$$a z^2 -a + bz^2 +bz + cz^2 -cz = 1 $$

Grouping like terms, we have:

$$a = -1 $$
$$a + b + c = 0$$
$$b = c $$

We get the partial fractions expansion

$$\frac{-1}{z} + \frac{1/2}{z-1} + \frac{1/2}{z+1} $$

We can then evaluate the integral of the closed path using Residue calculus:

$$\frac{1}{2\pi i}\int_\gamma \frac{dz}{z(z^2-1)} = \frac{1}{2\pi i} 2\pi i \text{Res}[\frac{-1}{z} + \frac{1/2}{z-1} + \frac{1/2}{z+1}]$$

$$\frac{1}{2\pi i}\int_\gamma \frac{dz}{z(z^2-1)} = \text{Res}[\frac{-1}{z}] + \text{Res}[\frac{1/2}{z-1}] + \text{Res}[\frac{1/2}{z+1}]$$

These are all simple poles, so we can multiply by the denominator in each case and evaluate at the poles:

$$ \frac{1}{2\pi i}\int_\gamma \frac{dz}{z(z^2-1)} = -1 \vert _ 0 + 1/2 \vert _ 1 + 1/2 \vert _ {-1} = 0$$

So the integral evaluates to $$0$$.

# P 257.1
Which of the following domains in $$\mathbb{C}$$ are simply connected? Justify your answers.

1. $$D= \{ \text{Im} z>0\} \backslash [0,i]$$, the upper half-plane with a vertical slit from $$0$$ to $$i$$.

I use the definition (iv) to prove this fact. Every curve in $$D$$ has winding number $$W(\gamma, z_0)=0$$ about $$z_0\in \mathbb{C}\backslash D$$. Any curve in $$D$$ begins and terminates in the upper half-plane. Thus, any $$z_0$$ in the lower half plane have winding number 0. The only troublesome points are in the slit $$[0,i]$$, but these will also have winding number 0 since the slit connects with the lower half-plane, requiring the curve to traverse the lower half-plane in order to encircle it. This cannot happen and so points in $$\mathbb{C}\backslash D$$ have winding number 0 and $$D$$ is simply connected.

2. $$D= \{\text{Im} z>0\} \backslash [i,2i]$$, the upper half-plane with a vertical slit from $$i$$ to $$2i$$.

$$D$$ is not simply connected. I use definition $$(iv)$$ to show this. Consider the rectangular path in $$D$$ that connects points [$$-1/2 + 1/2i$$, $$1/2+1/2i$$, $$1/2 + 3i$$, $$-1/2+ 3i$$]. Connecting these points sequentially yields a closed curve in $$D$$, and yet the winding number around the point $$i$$ in $$\mathbb{C} \backslash D$$ is 1. This is a contradiction to thm (iv), and so $$D$$ is not closed.

3. $$D= \mathbb{C}\backslash[0, +\infty]$$, the complex plane slit along the positive real axis.

$$D$$ is connected since its complement in the extended complex plane :$$\mathbb{C}^* \backslash D$$ is connected.

In the extended complex plane, $$[0, +\infty]$$ corresponds to half the diameter of the Riemann Sphere starting at the top $$\infty$$, proceeding along a longitudinal line representing the x-axis, through $$+1$$, and to $$0$$. Since this line is simply connected, $$D$$ is simply connected.  

4. $$D = \mathbb{C}\backslash[-1,1]$$, the complex plane with an interval deleted.

$$D$$ is simply connected. We know this by Thm (v), which states that $$D$$ is simply connected if the complement is simply connected. Since $$\mathbb{C}^{* } \backslash D = [-1,1]$$, a line on the real axis. In the Riemann Sphere representation of the extended complex plane, this represents a half-diameter longitudinal line passing through 0 Starting at $$-1$$ and ending at $$1$$. Since this line is simply connected, complement to $$D$$ is connected and therefore $$D$$ is simply connected.

# P 257.2
Show that a domain $$D$$ in the extended complex plane $$\mathbb{C}^* = \mathbb{C} \cup \{\infty\}$$ is simply connected if and only if its complement $$\mathbb{C}^* \backslash D$$ is connected.

Hint: If $$D \neq \mathbb{C}^{* }$$, move a point in the complement of $$D$$ to $$\infty$$. If $$D = \mathbb{C}^{* }$$, first deform a given closed path to one that does not cover the sphere, then deform it to a point by pulling along arcs of great circles.

There are two cases I will consider:

Case 1: $$D = \mathbb{C}^{* }$$.

Now for every closed curve $$\gamma$$ on $$D$$, we can chose a point $$z_0$$. By rotation, this point can be set to $$\infty$$. Now, for every other point $$z_1$$ on $$\gamma$$, there is a great circle $$C$$ that goes through $$z_0$$ and $$z_1$$. We can therefore retract $$z_1$$ to $$z_0$$ along $$C$$. Repeating this for every point $$z_1$$ on $$\gamma$$, we have shown that $$\gamma$$ is homotopic to a point on $$\mathbb{C}^{* }$$.

Therefore, every closed curve $$\gamma$$ on $$D$$ is homotopic to a point. This implies that the winding number is $$0$$ everywhere on $$D$$, since the integral evaluated on any closed path $$\gamma$$ will be homotopic to a point, and thus will evaluate to 0. Therefore, since the winding number is 0, $$D$$ must be closed.

Case 2: $$D \neq \mathbb{C}^{* }$$

If $$D$$ does not include the entire extended complex plane, then there is a point $$z_0 \in \mathbb{C}^{* } \backslash D$$. Through rotation, we can set this point $$z_0$$ to be the north point on the Riemann Sphere, $$\infty$$. We can then consider the stereographic projection with the north point at $$z_0$$ of $$D: D' \in \mathbb{C}$$.

Now since under this stereographic projection, $$D$$ does not contain $$\infty$$, it follows that $$D=D'$$. This implies, via substitution, that $$\mathbb{C}^{* } \backslash D = \mathbb{C}^{* } \backslash D'$$.

So via the equality, connectedness must be preserved in these relations, and we have: $$D$$ is connected $$\iff$$ $$D'$$ is connected.

For $$D'$$ in the complex plane, we have by thm (iv), that it is connected iff its complement in the extended complex plane is connected: $$\mathbb{C}^{* } \backslash D'$$.

Thus, we have the chain of relations:

$$D$$ is connected $$\iff$$ $$D'$$ is connected $$\iff$$ $$\mathbb{C}^{* }\backslash D'$$ is connected $$\iff$$ $$\mathbb{C}^{* }\backslash D$$ is connected

Finally:

$$D$$ is connected $$\iff$$ $$\mathbb{C}^{* } \backslash D$$ is connected.

Thus, under both cases (all possibilities), I have shown necessity and sufficiency, so the if and only if relation holds.



# P 257.3
Which of the following domains in $$\mathbb{C}^{* }$$ are simply connected? Justify your answers.

1. $$D =\mathbb{C}^{* }\backslash[-1,1]$$, the extended complex plane with an interval deleted

By the previous problem, we have that $$D \in \mathbb{C}^{* }$$ is connected if and only if $$\mathbb{C}^{* } \backslash D$$ is connected.

Since the complement $$\mathbb{C}^* \backslash D = [-1,1]$$, a line, the complement is simply connected and therefore $$D$$ is simply connected.

2. $$D = \mathbb{C}^{* } \backslash \{-1,0,1\}$$, the thrice-punctured sphere

By the previous problem, we have that $$D \in \mathbb{C}^{* }$$ is connected if and only if $$\mathbb{C}^{* } \backslash D$$ is connected.

The complement of $$D$$, $$\mathbb{C}^* \backslash D = \{-1,0,1\}$$ is not connected since if we take $$E = 0$$, then $$E$$ has positive distance $$1$$ to the rest of $$\mathbb{C}^* \backslash D$$. Therefore, $$\mathbb{C}^* \backslash D$$ is not connected and hence $$D$$ is not connected.


# P 258.4
Show that a domain $$D$$ in the complex plane is simply connected if and only if any analytic function $$f(z)$$ on $$D$$ that does not vanish at any point of $$D$$ has an analytic logarithm on $$D$$.

Hint: If $$f(z) \neq 0$$ on $$D$$, consider the function

$$G(z) = \int_{z_0}^z \frac{f'(w)}{f(w)}dw$$

First, I would like to show sufficiency:

If $$D$$ is simply connected, and there is an analytic function $$f(z)$$ on $$D$$ that does not vanish at any point on $$D$$, then the function $$G(z)$$ as stated above is well-defined on $$D$$.

I will next prove that this constitutes an analytic logarithm. We can differentiate $$G(z)$$ and get:

$$G'(z) = \frac{f'(z)}{f(z)} $$

$$G'(z)f(z) = f'(z) $$

now consider the expression:

$$e^{-G(z)}f(z)$$

Taking derivatives and substituting for $$f'(z)$$ from our relations above, we have:

$$ \frac{d}{dz}e^{-G(z)}f(z) = f'(z)e^{-G(z)} - G'(z)e^{-G(z)}f(z)$$

$$= G'(z)e^{-G(z)}f(z) - G'(z)e^{-G(z)}f(z) = 0 $$

And since the derivative is 0, the function must be constant for some constant $$C$$:

$$e^{-G(z)}f(z) = C \implies f(z) = C e^{G(z)} $$

Thus, $$G(z)$$ is the analytic logarithm of $$f(z)$$ (modulo a constant). And it is analytic since quotients of analytic functions on a simply connected domain are analytic (provided $$f(z)\neq 0$$, which is given), and analytic functions have analytic antiderivatives on a simply connected domain.

Thus, we have shown the existence of an analytic logarithm of $$f(z)$$ on $$D$$ based on the fact that $$D$$ is simply connected and the analyticity and definiteness of $$f(z)$$. Thus we have proved sufficiency.

Next, I must show necessity: That the existence of an analytic logarithm on a domain $$D$$ implies that $$D$$ is simply connected.

First, consider the winding number expressed as a function of the logarithm:

$$W(f, z_0) = \frac{1}{2\pi i} \int_{\gamma} d \text{log}(f(z- z_0)) $$

Now, we know that $$\text{log}(f(z-z_0))$$ has an analytic antiderivative: The logarithm on $$D$$, which I will call $$h(z)$$.

Thus, since $$h$$ is analytic, we can define the integral by its antiderivative at the starting and ending points.


$$W(f, z_0) = \frac{1}{2\pi i}h(a) - h(b) $$

And since $$\gamma$$ is a closed curve, $$a=b \implies h(a) = h(b) = 0$$. And so:

$$W(f, z_0) = 0 $$

The winding number of an analytic function $$f$$ around any point $$z_0$$ is 0, and thus the domain is simply connected. Thus, I have shown sufficiency and necessity and proved the if and only if relationship.  
