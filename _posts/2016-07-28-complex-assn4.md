---
title: Complex Analysis Assignment 4
category: "Complex Analysis"
layout: post
---


# 211.2
Show using residue theory that

$$\int_{-\infty}^\infty \frac{\sin(ax)}{x(x^2+1)} dx = \pi (1-e^{-a}), \qquad a > 0 $$

_Hint_. Replace $$\sin(ax)$$ by $$e^{iax}$$, and integrate around the boundary of a half-disk indented at $$z=0$$

If we have $$f(z) = \frac{e^{iaz}}{z(z^2+1)}, I =  \text{Im}[\frac{e^{iax}}{x(x^2+1)}]$$. Then using residue theorem we can integrate along the upper half disk indented at 0. We have four paths: $$\gamma_1$$ goes from $$\epsilon \to R$$ along the x-axis. $$\gamma_2$$ is a $$180$$-degree arc of length $$R$$. $$\gamma_3$$ goes from $$-R \to \epsilon$$ along the x-axis. $$\gamma_4$$ is an $$180$$-degree arc of length $$\epsilon$$.

Defining the domain $$D$$ to be the indented half-plane, we can see that:

$$2\pi i \text{Res}[f(z)] = \oint_\gamma f(z) dz$$

First, I observe that the function $$f(z)$$ has poles at $$f(z) = \pm i,0$$. We will use the residue theorem at $$i$$ and the fractional residue theorem at $$0$$.

Since $$f(z)$$ has a simple pole at $$i$$, we can evaluate the residue as:

$$ \text{Res}[f(z),i] = \lim\limits_{z \to i} (z-i) \frac{e^{iaz}}{z(z^2+1)} = \lim\limits_{z\to i}\frac{e^{iaz}}{z(z+i)}$$

Next, to calculate the residue at $$0$$, we observe that it is a simple pole at $$0$$.

$$\text{Res}[f(z),0] = \lim\limits_{z \to 0} z \frac{e^{iaz}}{z(z^2+1} = \frac{e^0}{1} = 1$$

Next, I use the Residue Theorem:

$$\lim\limits_{R \to \infty, \epsilon \to 0}\int_{\epsilon}^R f(x) dx = \int_{\theta = 0}^\pi f(R e^{i\theta}) d\theta + \int_{-R}^\epsilon f(x)dx + \int_{\theta = \pi}^0 f(\epsilon e^{i \theta}) d\theta = 2\pi i \text{Res}[f(z),i] $$

Combining the two integrals along the x-axis after we take the limits:

$$\int_{-\infty}^\infty f(x) dx + \lim\limits_{R \to \infty} \int_{\theta = 0}^\pi f(Re^{i\theta})d\theta + \lim\limits_{\epsilon \to 0} \int_{\theta = \pi}^0 f(\epsilon e^{i \theta} d\theta)  = -\pi ie^{-a}$$

Applying the ML-Estimate to the integral, with path length $$\pi R$$:

$$\vert \int_{\theta = 0}^\pi f(Re^{i\theta}) d\theta\vert  \leq \pi R/R(R^2-1)  $$

Since $$\vert e^{iaz}\vert \leq 1$$ in the upper half plane if $$a>0$$.

Thus, continuing from our previous expression,

$$\vert \int_{\theta = 0}^\pi f(Re^{i\theta}) d\theta\vert  \sim \frac{\pi}{R^2} \to 0 $$

As $$R \to \infty$$.

Next, we use the fractional residue formula to evaluate the puncture at 0:

$$\int_{\theta = \pi}^0 f(\epsilon e^{i \theta} d\theta) \to -i \pi \text{Res}[f(z),0] = - \pi i$$

Plugging these results into our initial integral equation, we have:
$$\int_{-\infty}^\infty f(x) dx +0 + -i \pi= -\pi ie^{-a}$$

$$ \int_{-\infty}^\infty f(x) dx = i\pi(1- e^{-a})$$

Finally, we must remember that $$\sin(\alpha x) = \text{Im}[e^{iax}]$$, so our original function is given by the imaginary part of $$f(z)$$, giving us:

$$\text{Im}\Big[\int_{-\infty}^\infty \frac{e^{iax}}{x(x^2+1)}\Big] = \text{Im}\big[ \pi i (1-e^{-a})\big]  $$

$$\text{Im}\Big[\int_{-\infty}^\infty \frac{e^{iax}}{x(x^2+1)}\Big] = \text{Im}\big[ \pi i (1-e^{-a})\big]  $$

$$\int_{-\infty}^\infty \frac{\sin(ax)}{x(x^2+1)} = \pi(1-e^{-a}) , \qquad a>0$$

# 211.4
Show using residue theory that

$$\int_0^\infty \frac{1-\cos(x)}{x^2} dx = \frac{\pi}{2} $$

In order to evaluate this integral, I will integrate over the punctured 1st upper half disk. If we define $$f(z) = \frac{1- e^{iz}}{z^2}$$, then we can see that our original function is equal to the real part of our new function since $$e^{i \theta} = \cos(\theta) + i \sin(\theta)$$
, and using the residue theorem we have:

$$\oint_{\partial D} = \lim\limits_{R \to \infty, \epsilon \to 0} \int_{x=\epsilon}^R f(x)dx + \int_{\theta = 0}^{\pi} f(Re^{i\theta}) + \int_{-R}^\epsilon f(x) dx + \int_{\theta = \pi}^0 f(\epsilon e^{i\theta})  = 2 \pi i\text{Res}[f(z)] $$

Since the function only has a double pole at $$z=0$$, which is inside the puncture of the unit disk, there are no poles on the domain, so the left hand side of the above equation is 0.

We can find the residue at $$z=0$$ by observing that it is a second order pole.

$$\text{Res}[f(z),0] = \lim\limits_{z\to 0} \frac{d}{dz} 1-e^{iz} \Big\vert _ 0 = -i e^{0} = -i $$

To evaluate the outer arc of the circle, we use the ML-estimate and the path length $$\pi R$$:

$$\vert \int_{\theta = 0}^\pi f(Re^{i\theta})d\theta \vert  \leq \frac{2}{R^2} \cdot \pi/R \ d\theta \sim \frac{2\pi}{R} \to 0$$

As $$R \to \infty$$, by observing that $$\vert e^{ix}\vert  \leq 1$$

We can also evaluate the integral around the puncture in the half-disk by applying the Fractional Residue Theorem:

$$ \lim\limits_{\epsilon \to 0} \int_{\theta = \pi}^0 f(\epsilon e^{i\theta}) = -i\pi \text{Res}[f(z),0] = -\pi$$

Plugging back into our original expression and taking the limits, we have:

$$ \int_{x=0}^\infty f(x)dx + 0 + \int_{-\infty}^0 f(x) dx - \pi = 0 $$

Combining the integrals:


$$ \int_{x=-\infty}^\infty f(x)dx  = \pi $$

Taking the Real Part of both sides, we are back to our original respresentation of the function:
$$ \int_{x=-\infty}^\infty \frac{1-\cos(x)}{x^2} dx  = \pi $$

And finally, I observe that our function is even, (since it is the sum of two even functions, $$\frac{1}{x^2}$$, and $$\frac{-\cos(x)}{x^2}$$. Thus, the integral across a symmetric interval is equal to twice the integral across the half interval:

$$2 \int_{x=0}^\infty \frac{1-\cos(x)}{x^2} = \pi$$

$$ \int_{x=0}^\infty \frac{1-\cos(x)}{x^2} = \frac{\pi}{2} $$

The desired result.

# 228.3
Find the number of zeros of the polynomial $$p(z)=z^6 + 4z^4 + z^3 + 2z^2 + z + 5$$ in the first quadrant $$\{\text{Re }z >0 \}$$

In order to find the number of zeros, we will apply the argument principle and look at the change in the argument around the border of the 1st quadrant quarter disk.

This path consists of three integrals: The integral along the x-axis to infinity, the integral along an arc from the x-axis to the y-axis, and the integral from infinity to 0 along the y-axis.

Since the polynomial has all positive coefficients and a positive constant term, the value of the polynomial will be positive along the entire x-axis. Therefore, the value of the the change in the argument along the first part of the path is 0.

On the arc from the x-axis to the y-axis, we can make the substitution $$z= Re^{i\theta}$$, and our function becomes:

$$p(R,\theta) = R^6 e^{6i\theta} + 4 R^4 e^{4 i \theta} + 3R^3 e^{4i\theta} + 2 R^2 e^{2 i \theta} + Re^{i \theta} + 5 $$

For large $$R$$, the leading term dominates and so we have:

$$p(R,\theta) \approx R^6 e^{6i\theta} $$

and

$$\text{arg}(p) \approx 6  \theta  \implies \Delta \text{arg}(p) = 6 \theta \vert _ 0^{\pi/2} = 3\pi$$

Finally, we have to evaluate along the y-axis. If we express our variable $$z = iy$$,

$$p(iy) = - y^6 + 4 y^4 -iy^3 - 2y^2 +iy +5$$

Separating it into real and imaginary parts:

$$p(iy) = -y^6 + 4y^4 -2y^2 + 5 + i(y - y^3) $$

We can see that at $$y \to \infty$$, the leading term dominates and so the Real component is negative. Similarly, the $$y^3$$ term dominates and the imaginary component is also negative.

The imaginary component has roots at $$0,1$$, and the real component has 4 imaginary roots and real roots at $$\sim \pm 1.95$$.

Thus, we can inspect the behavior of the function as we travel along the y-axis:

The real part of the starts in the lower hald-plane, and crosses the x-axis once at $$y=1.95$$.

The function starts in the third quadrant and has argument approximately $$-\pi$$. The function remains negative in the real axis as it crosses the imaginary axis into the second quadrant. It then crosses the real-axis at $$y=1$$. Thus, the function has thus far made a loop in the lower-half plane and has change in argument $$\pi$$.

After $$y=1$$, the function crosses from the real axis into the 1st quadrant before terminating on the real axis in the first quadrant at $$y=0$$. We know that it remains in the first quadrant since we do not observe any real zeros in this interval, indicating that it does not cross the imaginary axis again.  

After the function crosses the real-axis, the function segment begins and terminates on the real-axis in the first quadrant, indicating no change in argument.

Thus, collecting the $$\pi$$ from the traversal of the y-axis and $$3\pi$$ from traversing the ark of the quarter disk, we have

$$ \Delta \text{arg}(p) = 4\pi$$

So there must be 2 zeros in the first quadrant.



# 228.4
Find the number of zeros of the polynomial $$p(z) = z^9 + 2z^5 -2z^4 + z + 3$$ in the right half-plane.

First, I observe that the polynomial does not have any real roots, and so the zeros must occur in complex conjugate pairs. Thus, we can evaluate the number of zeros in the right half-plane by doubling the number of zeros in the right upper quadrant.

I divide the path around this quadrant into three sections: path 1 from 0 to $$\infty$$ along the x-axis, path 2 along the arc from the x-axis to the y-axis, and path 3 from $$\infty$$ to 0 along the y-axis.

For path 1, $$z=x$$, and so $$p(z) = p(x) = x^9 + 2x^5 -2x^4 + x + 3$$. This function is positive for the whole x-axis, since all of the coefficients are positive except for $$-2x^4$$, which is compensated for by $$2x^5$$ for $$x\geq1$$, and by the term $$+3$$ for $$x \in [0,1]$$. Thus, the function is positive along the whole right-hand x-axis, and the argument does not change.

For path 2, we write $$z = Re^{i\theta}$$, and $$p(z)= p(R,\theta) = R^9 e^{9it} + 2 R^5 e^{5it} + \cdots$$, which is dominated by $$R^9 e^{9it}$$ for large $$R$$. Thus,

$$\text{arg}( p) \approx 9 \theta $$

and

$$ \Delta \text{arg}(p) \approx 9 \theta \Big\vert _ 0^{\pi/2} = \frac{9}{2} \pi$$

So we say that the change in the argument for path 2 is $$\frac{9}{2} \pi$$

Finally, along path 3, we parameterize the path $$z = iy$$, and

$$p(z)= p(iy) = iy^9 +2iy^5 -2y^4 +iy + 3 $$

Separating the real and imaginary components

$$ p(iy) =   -2y^4 + 3 + i(y^9 +2 y^5 + y)$$

Factoring the imaginary part, we have:

$$ p(iy) = -2y^4 +3 + iy(y^4 +1)^2$$

Thus, we can see that the real part has a zero at $$\sqrt[4]{3/2}$$, and the imaginary part has a zero at $$0$$.

I can additionally observe that at $$\infty$$, the largest terms of the series dominate and so the real parts starts negative, and the imaginary part starts positive.

Using these facts, I can characterize the behavior of the function.

Since the imaginary part is larger than the real part, at $$\infty$$, the argument with be $$\frac{\pi}{2}$$. When the function crosses the imaginary axis, the imaginary part is positive and so the argument is still $$\frac{\pi}{2}$$, and so the change of argument along this segment of the path is 0.

Once the function is in the first quadrant (real and imaginary parts are positive), the next zero in the imaginary component at $$y=0$$ means that the function reaches the real axis, where the argument is 0, without a change in the sign of the real component. Thus, the change in argument along this segment, and the total along path 3, is $$\frac{-\pi}{2}$$.

Summing the changes of argument along all the paths, we have that

$$\Delta \text{arg}(p) \approx 4\pi $$

Thus, there are two zeros in the first quadrant, and since we know that these are complex conjugate pairs, there are 4 zeros in the right half-plane.  

# 230.2
How many roots does $$z^9 + z^5 - 8z^3 +2z + 1$$ have between the circles $$\{\vert z\vert =1\}$$ and $$\{\vert z\vert =2\}$$?

We can say $$p(z) = f(z)+ h(z)$$, $$f(z) = z^9$$ and $$h(z) = z^5 -8 z^3 +2z + 1$$.
Indeed, we can see for the circle $$\vert z\vert =2$$, $$\vert f(z)\vert  = 512, \vert h(z)\vert  \leq 101$$. Thus, we can apply Rouche's theorem and $$p(z)$$ has the same number of zeros on the circle as $$z^9$$, which has 9.

On the circle of $$\vert z\vert  = 1$$, I split the function based on the highest coefficient. So $$f(z) = -8z^3$$, $$h(z) = z^9 + z^5 + 2z + 1$$, so $$\vert f(z)\vert = 8, \vert h(z)\vert = 5$$, and we can apply Rouche's theorem to say that $$p(z)$$ has three roots on the circle, the same as $$f(z)$$.

Thus, with 9 roots within the outer circle and 3 in the inner circle, there are 6 roots between the circles.  

# 230.4
Fix a complex number $$\lambda$$ such that $$\vert \lambda\vert <1$$. For $$n\geq 1$$, show that $$(z-1)^n e^z - \lambda$$ has $$n$$ zeros satisfying $$\vert z-1\vert <1$$ and no other zeros in the right half-plane. Determine the multiplicity of the zeros.

If we say $$p(z) = (z-1)^n e^z -\lambda$$, then we can define $$p(z)=f(z)+h(z)$$, with $$f(z) = (z-1)^n e^z$$, and $$h(z)=-\lambda$$.

Thus, we can apply Rouche's Theorem on the circle $$\vert z-1\vert =1$$, since on this circle, $$z = 1 + e^{i\theta}$$, and

$$\vert f(z)\vert  = \vert 1+e^{i\theta}-1\vert ^n \vert e^{1+e^i\theta}\vert  = 1*e*e^{\cos(\theta)}e^{i \sin(\theta)}  \geq 1$$

Since $$\vert e^{i\sin(\theta)}\vert  = 1$$, and $$1 \leq e*e^{\cos(\theta)}\leq e^2$$, so the product is greater than or equal to one.

On the other hand, we know by definition that $$\vert \lambda\vert <1$$, so we can apply Rouche's theorem to say that $$p(z)$$ has the same number of zeros as $$f(z)$$ in the circle $$\vert z+1\vert =1$$. Thus, $$p(z)$$ has $$n$$ zeros in the circle since we can see $$f(z)$$ has $$n$$ roots at $$z=1$$.

To prove that there are no other zeros in the right-half plane, I consider the number of zeros within a disk centered at $$R$$ with radius $$R$$, which in the limit includes the entire right-hand plane.

$$z = R + Re^{i\theta} $$

Along this disk,

$$\vert f(z)\vert  = \vert R + Re^{i\theta}\vert ^n e^{R+Re^{i\theta} - 1} = \vert 2R-1\vert ^n \vert e^R e^{R\cos(\theta)}\vert \vert e^{iR\sin \theta}\vert  \geq \vert 2R-1\vert ^n $$

Again, I make the observations that $$\vert e^{iR\sin{\theta}}\vert =1,$$ and $$\vert 1<e^R e^{R\cos(\theta)}\vert <e^{2R}$$, so $$\vert f(z)\vert $$ is larger than $$\vert h(z)\vert =\vert \lambda\vert <1$$.

Thus, for $$R$$ large, $$p(z)$$ has the same number of zeros on the entire right-hand plane as $$f(z),$$ which has $$n$$ zeros.

Thus, $$p(z)$$, has $$n$$ zeros on the entire right-half plane, which is the same as in the circle $$\vert z+1\vert =1$$, so the only zeros in the right-half plane occur in the circle.

To find the multiplicities of the zeros, I inspect the derivative of the function, since via the power series expansion of a function we know that any zero of multiplicity greater than 1 must also be a zero of the derivative.

$$p'(z) = n(z-1)^{n-1}e^z + (z-1)^n e^z = (n+z-1)(z-1)^n e^z $$

So the derivative has zeros at either $$z=1,1-n$$. Of these, only $$z=1$$ is in the right-half plane, meaning that it is our only candidate for a zero of multiplicity greater than 1.

However, $$f(1)=-\lambda$$. Means that there are no zeros at $$z=1$$ unless $$\lambda=0$$. Thus, we can say that all of the zeros in the right-half plane are simple zeros, unless $$\lambda=0$$, in which case there is a zero of multiplicity $$n$$ at $$z=1$$.

# 230.7
Let $$f(z)$$ and $$g(z)$$ be analytic functions on the bounded domain $$D$$ that extend continuously to $$\partial D$$ and satisfy $$\vert f(z) + g(z)\vert  < \vert f(z)\vert  + \vert g(z)\vert $$ on $$\partial D$$. Show that $$f(z)$$ and $$g(z)$$ have the same number of zeros in $$D$$, counting multiplicity.

First, I would like to observe that neither of the functions can be zero at any point on their boundary is implied by the inequality. If one of them were to be zero at a point $$z_0$$, ($$g$$, without loss of generality), then:

$$\vert f(z_0)+ g(z_0)\vert  = \vert f(z_0)\vert  = \vert f(z_0)\vert  = \vert f(z_0)\vert +\vert g(z_0)\vert $$

 violates the strict inequality, a contradiction of our assumption.

Additionally, if $$f(z)$$ and $$g(z)$$ satisfy $$\vert f(z)+g(z)\vert < \vert f(z)\vert  + \vert g(z)\vert $$ on $$\partial D$$, their arguments can never be equal. Since if for some point $$z_0$$ on $$\partial D$$, if $$f(z_0) = R_1e^{i\theta}$$ and $$g(z_0) = R_2 e^{i \theta}$$, for the same argument $$\theta$$, then

$$\vert f(z_0) + g(z_0)\vert  = \vert (R_1+ R_2) e^{i \theta}\vert  = R_1 + R_2 = \vert R_1 e^{i \theta}\vert  + \vert R_2 e^{i \theta}\vert  = \vert f(z_0)\vert  + \vert g(z_0)\vert  $$

violating the strict inequality, a contradiction of our assumption.

Next, I would like to point out an implication of the argument principle for analytic functions:

$$\int_{\partial D}d \ \text{arg}(f(z)) = 2\pi N_0 $$

Because zeros must appear in integer amounts, the change in the argument must be an integer multiple of $$2\pi$$. This signifies that the function $$f(z)$$ must do exactly $$N_0$$ loops in the complex plane while traversing the boundary $$\partial D$$.

I now combine the observed facts: Both $$f(z)$$ and $$g(z)$$ must encircle the complex plane an integer number of times. Since they cannot be $$0$$ as a result of the inequality, they will form loops about the origin in the complex plane while they traverse the boundary. Finally, because their arguments can never be equal, they must encircle the complex plane the same number of times.

More precisely, if one function were to encircle the complex plane additional times (remember they must be integer), then at some point it's argument would have to overtake the other's. At this point their arguments would be equal, a contradiction to the initial assumption of the inequality.

So we have shown that $$f(z)$$ and $$g(z)$$ must encircle the complex plane an identical integer number of times, $$N_0$$, while traversing $$\partial D$$. This means that their change in argument along the boundary is $$2\pi N_0$$. Finally, by applying the argument principle symmetrically to $$f(z)$$ and $$g(z)$$ with this change in argument, we see that they must have the same number of zeros counting multiplicities, $$N_0$$.  



# 230.8
Let $$D$$ be a bounded domain, and let $$f(z)$$ and $$h(z)$$ be meromorphic functions on $$D$$ that extend to be analytic on $$\partial D$$. Suppose that $$\vert h(z)\vert <\vert f(z)\vert $$ on $$\partial D$$. Show by example that $$f(z)$$ and $$f(z)+h(z)$$ can have different numbers of zeros on $$D$$. What can be said about $$f(z)$$ and $$f(z)+h(z)$$? Prove your assertion.

A meromorphic function is analytic except for a finite number of poles. Thus, the problem become that when we apply the argument principle of a meromorphic function instead of an analytic function, the change in the argument is equal to the number of zeros minus the number of poles rather than just the number of zeros.

I'll illustrate with an example:

$$h(z) = \frac{1}{z}$$
$$f(z) = 4 $$
$$p(z) = f+h =  \frac{1}{z} +4$$
On the domain of the unit disk: $$\vert z\vert =1$$, it appears that we could naively apply Rouche's theorem since indeed, $$\vert h\vert  = 1 < 4 = \vert f\vert $$,

But we can see that the conclusion, that $$f$$ and $$p$$ have the same number of zeros does not hold, since $$f$$ has no zeros on the unit disk (or anywhere), but $$p(z)$$ has a zero at $$z = -1/4$$.

Instead, I will say that the number of zeros minus the number of poles will be the same for both functions, which I will prove as follows:

Consider $$p(z) = f(z) + h(z) = f(z)[1 + \frac{h(z)}{f(z)}]$$

Then we have:

$$\text{arg}[f(z)+h(z)] = \text{arg}[f(z)(1+\frac{h(z)}{f(z)})] = \text{arg}[f(z)] + \text{arg}[1 + \frac{h(z)}{f(z)}] $$

Then if we wanted to know the numbers of zeros and poles of the functions, we could apply the argument principle on a domain $$D$$:

$$\int_{\partial D} d \ \text{arg}[f(z)+ h(z)] = 2 \pi (N_{0,f+h} - N_{\infty, f+h}) $$

Where $$N_0$$ is the number of zeros and $$N_\infty$$ is the number of poles on the domain.

Substituting our expression for $$\text{arg}[f(z)+h(z)]$$, we have

$$\int_{\partial D} d \Big[ \text{arg}[f(z)] + \text{arg}[1 + \frac{h(z)}{f(z)}]\Big] = 2 \pi (N_{0,f+h} - N_{\infty, f+h}) $$

$$\int_{\partial D} d\  \text{arg}[f(z)] +\int_{\partial D} d\ \text{arg}[1 + \frac{h(z)}{f(z)}]= 2 \pi (N_{0,f+h} - N_{\infty, f+h}) $$

However, we can observe that if $$\vert h(z)\vert <\vert f(z)\vert $$ on $$D$$, then $$\Big\vert \frac{h(z)}{f(z)}\Big\vert  <1$$, and so $$1 + \frac{h(z)}{f(z)}$$ lies in the right-half plane, and so the increase in $$\text{arg}[1 + \frac{h(z)}{f(z)}]$$ around a closed boundary will be 0. This leaves us with the equation:

$$\int_{\partial D} d\  \text{arg}[f(z)] = 2 \pi (N_{0,f+h} - N_{\infty, f+h}) $$

And applying the argument principle again to the left-hand side,

$$2 \pi (N_{0,f} - N_{\infty, f}) = 2 \pi (N_{0,f+h} - N_{\infty, f+h}) $$

$$ (N_{0,f} - N_{\infty, f}) =  (N_{0,f+h} - N_{\infty, f+h}) $$

So the number of zeros minus the number of poles (counting multiplicities) of $$f$$ is the same as the number of zeros minus the number of poles (counting multiplicities) of $$f+h$$.

Returning to our toy example, we can see that this is indeed the case, since $$f(z) = 4$$ has no zeros or poles $$(N_0 - N_\infty) = (0-0)=0$$, and $$p(z) = 1/z + 4$$ has a zero at $$z=-1/4$$, and a pole at $$z=0$$, so $$(N_0 - N_\infty) = (1-1)= 0$$.
