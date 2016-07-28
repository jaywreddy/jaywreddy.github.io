---
layout: post
title: Complex Analysis Assignment 3
category: "Complex Analysis"
---
Jay Reddy Math 185 HW 8
=========
SID: 23471486

Problems: 198.3ace, 202.4, 202.8, 205.2, 205.3, 207.1, 207.2.

# P 198.3
Evaluate the following integrals using the residue theorem

## a
$$\oint_{\vert z\vert =1} \frac{\text{sin} (z)}{z^2}dz $$

We can see that there is a single pole at $$z=0$$.

So using the residue theorem:
$$\oint_{\vert z\vert =1} \frac{\text{sin} (z)}{z^2}dz = 2 \pi i \text{Res}[\frac{\text{sin} (z)}{z^2},0]$$

Via Rule 1:
$$ \text{Res}[\frac{\text{sin} (z)}{z^2},0] = \lim\limits_{z \to z_0} (z)\frac{\text{sin} (z)}{z^2} = \lim\limits_{z \to z_0} \frac{\text{sin} (z)}{z} = 1$$

So,

$$\oint_{\vert z\vert =1} \frac{\text{sin} (z)}{z^2}dz = 2 \pi i $$


## c
$$\oint_{\vert z\vert =2} \frac{z}{\text{cos}(z)}dz $$

We can see that there are poles at $$\pm\frac{\pi}{2}$$

Thus using residue theorem:
$$\oint_{\vert z\vert =2} \frac{z}{\text{cos}(z)}dz =2 \pi i (\text{Res}[\frac{z}{\text{cos}(z)},\frac{\pi}{2}] + \text{Res}[\frac{z}{\text{cos}(z)}, -\frac{\pi}{2}]) $$

Since both the numerator and denominator are analytic at $$\pm\frac{\pi}{2}$$, we can apply rule 3:

 $$\oint_{\vert z\vert =2} \frac{z}{\text{cos}(z)}dz =2 \pi i (\frac{z}{-\text{sin}(z)}\Big\vert _ {\pi/2} + \frac{z}{-\text{sin}(z)} \Big\vert _ {\pi/2}) $$

  $$\oint_{\vert z\vert =2} \frac{z}{\text{cos}(z)}dz =2 \pi i (-\frac{\pi}{2} - \frac{\pi}{2} ) = -2\pi^2 i $$


## e
$$\oint_{\vert z-1\vert =1} \frac{1}{z^8 - 1}dz $$

Of the $$8$$ roots of unity, only 3 are within distance $$1$$ of $$x=1$$: [$$1, e^{i\pi/4}, e^{-i\pi/4}$$]

So by the residue theorem:
$$\oint_{\vert z-1\vert =1} \frac{1}{z^8 - 1}dz = 2\pi i(\text{Res}[\frac{1}{z^8 -1}, 1]+\text{Res}[\frac{1}{z^8 -1}, e^{i\pi/4}] + \text{Res}[\frac{1}{z^8 -1},e^{-i\pi/4}] )$$

Using rule 4:

$$\oint_{\vert z-1\vert =1} \frac{1}{z^8 - 1}dz = 2\pi i(\frac{1}{8z^7} \Big \vert _ {z=1} + \frac{1}{8z^7} \Big \vert _ {z=e^{i\pi/4}} + \frac{1}{8z^7} \Big \vert _ {z=e^{-i\pi/4}} )$$

$$\oint_{\vert z-1\vert =1} \frac{1}{z^8 - 1}dz = \frac{\pi i}{4}(1 + e^{-i\pi/4} + e^{i\pi/4} ) = \frac{\pi i}{4}(1 + \frac{1-i}{\sqrt{2}} + \frac{1+i}{\sqrt{2}}) $$

$$\oint_{\vert z-1\vert =1} \frac{1}{z^8 - 1}dz = \frac{\pi i}{4}(1 + \sqrt{2}) $$


# P 202.4
Using residue theory, show that $$\int_{-\infty}^\infty \frac{dx}{x^4 +1} = \frac{\pi}{\sqrt{2}}$$

We can evaluate this on the half disk. Looking at the four roots of unity: [$$\frac{1+i}{\sqrt{2}},\frac{-1+i}{\sqrt{2}},\frac{-1-i}{\sqrt{2}},\frac{1-i}{\sqrt{2}}$$]

Two of these fall within the half disk:

$$\frac{\pm1 + i}{\sqrt{2}} $$

Thus by the residue theorem:
$$\int_{\partial D_R} \frac{dx}{x^4 +1} = i2\pi(\text{Res}[\frac{1}{z^4+1},\frac{1+i}{\sqrt{2}}] + \text{Res}[\frac{1}{z^4 + 1}, \frac{-1 +i}{\sqrt{2}}])$$

By rule 2, we can evaluate the residuals:

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = i2\pi(\frac{1}{4z^3}\Big\vert _ {\frac{1+i}{\sqrt{2}}} + \frac{1}{4z^3}\Big\vert _ {\frac{-1 +i}{\sqrt{2}}})$$

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = i2\pi(\frac{\sqrt{2}}{8}(-1-i) + \frac{\sqrt{2}}{8}(1-i))$$

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = i2\pi(-i\frac{\sqrt{2}}{4})$$

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = \frac{\pi}{\sqrt{2}}$$

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = \int_{-R}^R \frac{dx}{x^4 +1} + \int_{\Gamma_R} \frac{dx}{x^4 +1}$$

Now, let's take the limits of $$R$$ in order to equate the integral on the disk to the desired integral.

The length of $$\Gamma_R = \pi R$$, so via the $$ML$$-estimate, we have

$$\Bigg\vert  \int_{\Gamma_R} \frac{dz}{z^4+1}\Bigg\vert \leq \frac{1}{R^4-1} * \pi R \sim \frac{\pi}{R^3} \to 0$$

as $$R\to \infty$$.

So taking the limit $$R \to \infty$$

$$\int_{\partial D_R} \frac{dx}{x^4 +1} = \int_{-\infty}^\infty \frac{dx}{x^4 +1} + 0 $$

$$ \int_{-\infty}^\infty \frac{dx}{x^4 +1} = \frac{\pi}{\sqrt{2}}$$


# P 202.8
Show that

$$\int_{-\infty}^\infty \frac{\text{cos}(x)}{(x^2+1)^2}dx = \frac{\pi}{e}$$

To evaluate this integral, I will use residue theory the evaluate the integral on the half disk, which contains a double pole at $$i$$.

To do this, we integrate the function:

$$\frac{e^{iz}}{(z^2+1)^2}$$

Noting that our original function is the real component of this function.

On the half disk, and using the residuals theorem

$$\oint_{\partial D}\frac{e^{iz}}{(z^2+1)^2} dz = i2\pi \text{Res}[\frac{e^{iz}}{(z^2+1)^2},i] $$

By rule 2,

$$ \text{Res}[\frac{e^{iz}}{(z^2+1)^2},i] = \lim\limits_{z\to i } \frac{d}{dz} (z-i)^2 \frac{e^{iz}}{(z+1)^2} = \lim\limits_{z\to i } \frac{d}{dz}  \frac{e^{iz}}{(z+i)^2} $$

With a little scratch calculation on paper... (the derivative is kind of big to LaTeX)

$$ \text{Res}[\frac{e^{iz}}{(z^2+1)^2},i] = -\frac{1}{2e}i $$

So, plugging into the equation above the value of the residual.

$$\oint_{\partial D_R}\frac{e^{iz}}{(z^2+1)^2} dz =\frac{\pi}{e}$$

Now we know that integrating the half-disk is the same as integrating the arc and the center:

$$\oint_{\partial D_R}\frac{e^{iz}}{(z^2+1)^2} dz = \int_{-R}^R \frac{e^{iz}}{(z^2+1)^2} dz + \int_{\Gamma_R} \frac{e^{iz}}{(z^2+1)^2} dz$$

To evaluate the arc with length $$\pi R$$, via the $$ML$$-estimate:

$$\Big\vert \int_{\Gamma_R} \Big\vert  \leq \frac{1}{(R^2-1)^2} \pi R \sim \frac{\pi}{R^3} \to 0$$

as $$R \to \infty$$.

Taking the limit of $$R \to \infty$$:
$$\oint_{\partial D_R}\frac{e^{iz}}{(z^2+1)^2} dz = \int_{-\infty}^\infty \frac{e^{iz}}{(z^2+1)^2} dz + \int_{\Gamma_R} \frac{e^{iz}}{(z^2+1)^2} dz$$

$$\frac{\pi}{e} = \int_{-\infty}^\infty \frac{e^{iz}}{(z^2+1)^2} dz + 0 $$

And expanding the complex exponential:

$$ i\int_{-\infty}^\infty \frac{\text{sin}(x)}{(x^2+1)^2} dx + \int_{-\infty}^\infty \frac{\text{cos}(x)}{(x^2+1)^2} dx = \frac{\pi}{e}$$

We can see by matching terms that the real value of the right hand side (all of it) belongs to the cosine term.

Thus,


$$\int_{-\infty}^\infty \frac{\text{cos}(x)}{(x^2+1)^2}dx = \frac{\pi}{e}$$

# P 205.2
Show using residue theory that

$$\int_0^{2\pi} \frac{d \theta}{a+b \text{sin}(\theta)} = \frac{2\pi}{\sqrt{a^2 -b^2}}, \qquad a > b> 0 $$

I will do this by integrating the unit disk, and to do so I will make the usual parameterization:

$$ d\theta = \frac{dz}{iz}$$

Thus, we have:

$$\int_0^{2\pi} \frac{d \theta}{a+b \text{sin}(\theta)} = \oint_{\vert z\vert =1} \frac{dz}{(a+ b/2i (z- 1/z))(iz)} = \oint_{\vert z=1\vert } \frac{2}{bz^2 + 2azi - b} dz $$

Via the quadratic formula, we can see that there are simple poles at:

$$z = \frac{-a\pm \sqrt{a^2 - b^2}}{b}i $$

Additionally, the relation $$a>b>0$$ ensures that the discriminant is positive.

Of which only the positive root falls in the unit circle.

By the residue theorem,

$$ \oint_{\vert z=1\vert } \frac{2}{bz^2 + 2azi - b} dz = 2\pi i \text{Res}[\frac{2}{bz^2 + 2azi - b}, \frac{-a + \sqrt{a^2-b^2}}{b}i] $$

By rule 3, we can find this residual by evaluating the derivative of the denominator at the pole:

$$ \oint_{\vert z=1\vert } \frac{2}{bz^2 + 2azi - b} dz =(2\pi i) \frac{2}{2bz + 2ai} \Bigg\vert _ {z = \frac{-a + \sqrt{a^2-b^2}}{b}i} =(2\pi i) \frac{1}{\sqrt{a^2 - b^2}i}=  \frac{2 \pi}{\sqrt{a^2 - b^2}}$$

Thus, plugging into our earlier equality provided by the change of variables:

$$\int_0^{2\pi} \frac{d \theta}{a+b \text{sin}(\theta)} = \frac{2\pi}{\sqrt{a^2 -b^2}}$$


# P 207.1
By integrating around the keyhole contour, show that

$$\int_0^\infty \frac{x^{-a}}{1 + x}dx = \frac{\pi}{\text{sin}(\pi a)}, \qquad 0<a<1 $$

To find this I will integrate the principal branch of the function $$z^{-a}/(1+z)$$:

$$ \frac{r^{-a} e^{-ia \theta}}{1+z} \qquad 0<\theta< 2\pi$$

This function has a simple pole at $$-1$$, so by rule 3, evaluating at the derivative of the denominator,

$$\text{Res}[\frac{r^{-a} e^{-ia \theta}}{1+z}] = r^{-a} e^{-ia\theta} $$

$$ z=-1 \implies r = 1, \theta = \pi$$

so
$$\text{Res}[\frac{r^{-a} e^{-ia \theta}}{1+z}] = e^{-i\pi a}$$

First, we must integrate out from the center of the keyhole,

$$ \int_{\gamma_1} \frac{r^{-a} e^{-ia \theta}}{1+z} dz$$.

Since we are integrating along the positive x axis, (the argument of the complex number is 0),

$$ z = x, dz = dx$$

$$ \int_{\gamma_1} \frac{r^{-a} e^{-ia \theta}}{1+z} dz = \int_{\epsilon}^R \frac{x^{-a} }{1+x} dx $$.

Taking the limits $$\epsilon \to 0, R \to \infty$$, this becomes:

$$\int_{-\infty}^\infty \frac{x^{-a} }{1+x} dx =I$$

This is the expression we are trying to evaluate.

Second, we must integrate around the circumference of the keyhole.

Using the $$ML$$-estimate, and the circumference of the circle, $$2\pi R$$

$$\Big\vert \int_{\gamma_2}\frac{r^{-a} e^{-ia \theta}}{1+z} dz  \Big\vert  \leq \frac{R^{-a}}{R-1} 2\pi R \sim \frac{2 \pi}{R^a} \to 0 $$

as $$R \to \infty$$.

Third, we must integrate back to the center of the keyhole

Now, we are integrating along the -x axis, so

$$dz = dx $$
$$z = xe^{2\pi i} $$

$$\int_{\gamma_3} \frac{r^{-a} e^{-ia \theta}}{1+z} = \int_{R}^\epsilon \frac{x^{-a}e^{-2\pi i a}}{1+x} dx$$

Taking the limits as $$\epsilon \to 0, R\to \infty$$:
$$\int_{\gamma_3} \frac{r^{-a} e^{-ia \theta}}{1+z} = \int_{0}^\infty \frac{x^{-a}e^{-2\pi i a}}{1+x} dx = (- e^{-2 \pi i a} ) \int_0^\infty \frac{x^{-a}}{1+x}dx = (-e^{-2\pi i a}) I$$

Where $$I$$ is our desired expression.

Finally, we must integrate around the keyhole.

Using the $$ML$$-estimate, and the inner radius $$\epsilon \to 0$$, and keyhole circumference $$2\pi \epsilon$$:
here we know $$r = \epsilon$$
$$\Big\vert \int_{\gamma_4} \frac{r^{-a} e^{-ia\theta}}{1+z} \Big\vert  \leq \frac{\epsilon^{-a}}{1-\epsilon}2\pi \epsilon \sim 2\pi \epsilon^{1-a} \to 0 $$.

Notice here we are able to tell that the limit $$\epsilon \to 0$$ of the function indeed goes to 0, because the problem states $$0<a<1$$.

Now using the Residue Theorem, we have:

$$\int_{\gamma_1} f(z) dz + \int_{\gamma_2} f(z) dz + \int_{\gamma_3} f(z) dz + \int_{\gamma_4} f(z) dz = \text{Res}  $$

$$I + 0 - e^{-2\pi i a}I  +0 = w\pi i e^{-\pi i a} $$

$$I (e^{\pi i a} - e^{-\pi i a}) = 2\pi i  $$

$$I 2i sin(\pi a) = 2\pi i  $$

$$I = \frac{\pi}{\text{sin}(\pi a)}  $$

Thus,

$$\int_0^\infty \frac{x^{-a}}{1 + x}dx = \frac{\pi}{\text{sin}(\pi a)}, \qquad 0<a<1 $$

# P 207.2
By integrating around the boundary of a pie-slice domain or aperture $$2\pi/b$$, show that

$$\int_0^\infty \frac{dx}{1+x^b} = \frac{\pi}{b \text{sin}(\pi/b)}, \qquad b>1 $$

I will integrate this function in four parts, first, travelling along the positive x-axis to R, then following an arc of length $$2\pi /b$$, then integrating back to the center, $$\epsilon$$, of the pie along the radius, and finally integrating in an arc about $$\epsilon$$ back to the x-axis.

with $$f(z) = \frac{1}{1+z^b}$$, and the residue theorem, this means:

$$\int_{\gamma_1} f(z) dz  + \int_{\gamma_2} f(z) dz + \int_{\gamma_3} f(z) dz+ \int_{\gamma_4} f(z) dz = \text{Res}$$

First, I observe that this function will have b roots of unity occurring at $$e^{\pi/b}$$.

However, since our outside arc only passes along $$1/b$$ the circumference of the circle, it can only contain one root. Thus, there will be a simple pole at $$z = e^{\pi i /b}$$

So to evaluate the residuals, we can use Rule 3, taking the derivative of the denominator and evaluating at the root:

$$\text{Res}[\frac{1}{1+z^b},e^{\pi i /b}] = \frac{1}{bz^{b-1}}\Big\vert _ {z = e^{\pi i /b}} = \frac{- e^{\pi i /b}}{b}  $$  

Next, we must evaluate each of our integrals.

First, traveling in the positive direction along the x-axis and taking the limits as $$R \to \infty,$$ $$\epsilon \to 0$$:

We must make the variable substitutions: $$dz = dz, z = x$$

$$ \int_{\gamma_1} f(z) dz = \int_\epsilon^R \frac{dx}{1+x^b} \to \int_{0}^\infty \frac{dx}{1+x^b} = I$$,

where $$I$$ is our desired function.

Next, Integrating along the arc of length $$2\pi R/B$$ and using the $$ML$$-estimate:

$$\Big \vert  \int_{\gamma_2} f(z)dz \Big\vert  \leq \frac{1}{R^b -1} \frac{2\pi R}{b} \sim \frac{2\pi}{R^{b-1}} \to 0 $$

as $$R \to \infty$$, since $$b>1$$ in the problem statement.
Next, we integrate back towards the center along the axis, making the variable substitutions: $$z = xe^{2\pi i/b}, dz = e^{2\pi i/b}dz$$. Since we are staying at the same angle and integrating inward along the length of $$\vert z\vert $$.

$$ \int_{\gamma_3} f(z) dz = \int_R^\epsilon \frac{1}{1+x^b} e^{2\pi i /b}$$

We then take the limits as $$R\to \infty, \epsilon \to 0$$, and move out everything that is constant w.r.t the variable of integration.

$$\int_R^\epsilon \frac{1}{1+x^b} e^{2\pi i /b} \to e^{2 \pi i /b}\int_{\infty}^0 \frac{1}{1+x^b}dx  $$

Flipping the limits of integration, we see that this a product involving our desired function:

$$\int_{\gamma_3} f(z) dz = e^{2 \pi i /b}\int_{\infty}^0 \frac{1}{1+x^b}dx = - e^{2 \pi i /b}\int_{0}^\infty \frac{1}{1+x^b}dx = -e^{2 \pi i /b} I   $$

Finally, we must integrate back to the origin axis around $$\epsilon$$, and take the limit as $$\epsilon \to 0$$.

Using the $$ML$$-estimate:

$$ \Big \vert  \int_{\gamma_4} f(z) dz \Big \vert   \leq \frac{1}{1- \epsilon^b}\frac{2\pi \epsilon}{b} \sim \frac{2\pi \epsilon}{b} \to 0 $$

Now returning to our residue theorem equation:

$$\int_{\gamma_1} f(z) dz  + \int_{\gamma_2} f(z) dz + \int_{\gamma_3} f(z) dz+ \int_{\gamma_4} f(z) dz = \text{Res}$$

And plugging in everything we just calculated:

$$ I + 0 -e^{2\pi i /b}I + 0 = 2\pi i (\frac{-e^{\pi i /b}}{b})$$

$$(e^{-\pi i /b} - e^{\pi i /b} ) I - \frac{-2 \pi i }{b}$$

$$-2i\text{sin}(\pi/b)I = \frac{-2 \pi i}{b} $$

$$I = \frac{\pi}{b \text{sin}(\pi/b)} $$

Thus,

$$\int_0^\infty \frac{dx}{1+x^b} = \frac{\pi}{b \text{sin}(\pi/b)}, \qquad b>1 $$
