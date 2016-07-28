---
layout: post
title: Fourier Analysis Assignment 3
category: "Fourier Analysis"
---

# Question 1

## a.
Use Fourier transform to find a bounded solution $$u$$ of
$$ u_{xx} + u_{tt}=0$$
in the upper half plane $$x\in \mathbb{R},\ t>0$$, with boundary conditions
$$ u(x,0) = g(x)$$
where $$g\in\mathcal{L}^2(\mathbb{R})$$ is bounded and continuous.

Since the problem does not appear to be solvable on the time domain, and the function is extends infinitely on x, I take the Fourier Transform on the variable x:

$$\mathcal{F}_x(u_{xx}+ u_{tt} = 0) $$

$$\mathcal{F}_x(u_{xx}) +\mathcal{F}(u_{tt})= 0 $$

Using the transform variable $$\lambda$$, and using the Fourier Transform property of derivatives, I get

$$\hat{u}(\lambda,t)_ {tt}  -\lambda^2 \hat{u}(\lambda,t) = 0$$

This forms a common second order homogeneous differential equation in $$t$$ with characteristic equation:

$$r^2 - \lambda^2 = 0 $$

Suggesting general solutions of the form:

$$\hat{u}(\lambda,t) = A(\lambda)e^{\lambda t} + B(\lambda)e^{-\lambda t} $$

Since we are looking for a bounded function for $$u$$, the function must decrease to $$0$$ as $$t \to \infty$$. Via Parseval's Equation, this means that the same must be true of $$\hat{u}$$.

Thus, $$A(w)=0$$ for $$w >0,$$ and $$B(w)=0$$ for $$w<0$$. Additionally, since $$u$$ vanishes for $$y<0$$, we can combine the two sides of the equation and say:

$$\hat{u}(\lambda, t) = C(\lambda)e^{-\vert \lambda\vert t} $$

All we have done is taken the general solution to the ODE and constrained it based on the boundaries of the problem.

Putting aside this equation for a bit, let's return to the boundary conditions. If we take the Fourier Transform in x, we get:

$$\mathcal{F}(u(x,0) = g(x)) = \hat{u}(\lambda,0) = \hat{g}(\lambda) $$

This look suspiciously like the equation we just put aside. In fact, plugging in 0, we find:

$$\hat{u}(\lambda,0) = C(\lambda) \implies C(\lambda) = \hat{g}(\lambda) $$

So our general form of the equation is actually

$$\hat{u}(\lambda,t) = \hat{g}(\lambda)e^{-\vert \lambda\vert t} $$

 Now all we have to do is take the inverse Fourier transform in order to find a solution.

 $$u(x,t)= \mathcal{F}^{-1}_x (\hat{g}(\lambda)e^{-\vert \lambda\vert  t})$$

 $$u(x,t) = \mathcal{F}^{-1}_\lambda (\mathcal{F}_x(g(x))e^{-\vert \lambda\vert  t})$$

 Expanding the Fourier Transform, and noting that $$e^{-\vert \lambda\vert t}$$ can be moved inside since it is constant w.r.t. the variable of integration:

 $$ u(x,t) = \mathcal{F}^{-1}_\lambda(\frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty g(x)e^{-\vert \lambda\vert t}e^{-i\lambda x}dx) $$

 When we expand the inverse Fourier transform, we will have a problem because we have an $$e^{i\lambda x}$$ term which is not part of the integrand of the integral over x. To avoid confusion, I will do a variable substitution in the current equation: $$dx = d\tau$$

 $$ u(x,t) = \frac{1}{\sqrt{2\pi}} \int_{\lambda=-\infty}^\infty \Big[ \frac{1}{\sqrt{2\pi}} \int_{\tau=-\infty}^\infty g(\tau)e^{-\vert \lambda\vert t}e^{-i\lambda \tau}d\tau \Big] e^{i\lambda x}d\lambda$$

Switching the order of integration and factoring out everything constant w.r.t $$d\lambda$$
  $$ u(x,t) = \frac{1}{2\pi} \int_{\tau=-\infty}^\infty f(\tau) \int_{\lambda=-\infty}^\infty \Big[   e^{\lambda i(\tau - x) - \vert \lambda\vert t} \Big]d\lambda d\tau$$

Next, to evaluate the inner integral:

$$ \int_{\lambda=-\infty}^\infty \Big[   e^{\lambda i(\tau - x) - \vert \lambda\vert t} \Big]d\lambda $$

$$ \int_{\lambda=-\infty}^0   e^{\lambda i(\tau - x) +\lambda t} d\lambda + \int_{\lambda=0}^\infty    e^{\lambda i(\tau - x) - \lambda t} d\lambda $$

$$ \int_{\lambda=-\infty}^0   e^{\lambda [t+i(\tau - x)]} d\lambda + \int_{\lambda=0}^\infty   e^{-\lambda[t- i(\tau -x)]}d\lambda $$

$$\frac{1}{t+i(\tau-x)} e^{\lambda [t+i(\tau - x)]}\Bigg\vert _ {\lambda =-\infty}^0 +  \frac{-1}{t-i(\tau-x)} e^{-\lambda[t- i(\tau -x)]} \Bigg\vert _ {\lambda=0}^\infty$$

$$\frac{1}{t + i(\tau-x)} + \frac{1}{t - i(\tau-x)} $$

$$\frac{2t}{t^2 + (\tau-x)^2} $$

And substituting, we finally have:

$$u(x,t)=\frac{t}{\pi} \int_{\tau = -\infty}^\infty \frac{g(\tau)}{t^2 + (\tau-x)^2} d\tau $$

Which is a solution to Laplace's Equations with boundary condition $$g(x)$$

## b.
Show that $$u$$ attains its boundary values in the sense that
$$ u(x,t) \to g(x)$$
as $$t\to 0$$

$$\lim\limits_{t \to 0}\frac{t}{\pi} \int_{\tau = -\infty}^\infty \frac{g(\tau)}{t^2 + (\tau-x)^2} d\tau $$

This comes out fairly easily if we make the smart substitution: $$\tau = x + st$$. Since $$t$$ is guaranteed to be positive, the limits of integration remain the same and $$d\tau = t ds$$. Thus the expression becomes:

$$\lim\limits_{t \to 0}\frac{t}{\pi} \int_{s = -\infty}^\infty \frac{g(x+ ts)}{t^2 + (x+ts-x)^2} tds $$

$$\lim\limits_{t \to 0}\frac{t}{\pi} \int_{s = -\infty}^\infty \frac{g(x+ ts)}{t^2 + t^2s^s} tds $$

$$\lim\limits_{t \to 0}\frac{t}{\pi} \int_{s = -\infty}^\infty \frac{1}{t}\frac{g(x+ ts)}{1+s^2} ds $$

Thus, since $$t$$ is not part of the integral, we can move it outside to cancel with the leading t.

$$\lim\limits_{t \to 0}\frac{1}{\pi} \int_{s = -\infty}^\infty \frac{g(x+ ts)}{1+s^2} ds $$

Via the dominated convergence theorem, we can apply the limit inside the integral, which makes $$g$$ independent of the integral.

$$\lim\limits_{t \to 0}\frac{1}{\pi} \int_{s = -\infty}^\infty \frac{g(x)}{1+s^2} ds $$

$$\frac{g(x)}{\pi} \int_{s = -\infty}^\infty \frac{1}{1+s^2} ds $$

The integral is conveniently equal to $$\pi$$, allowing us to cancel with the last term and get:

$$\lim\limits_{t \to 0} u(x,t) = g(x) $$




## c.
Assume that $$g' \in \mathcal{L}^2(\mathbb{R})$$ is also bounded and continuous. Argue directly from the Laplace equation that if

$$u_t(x,t) \to \Lambda g(x) $$
the the Dirichlet-Neumann operator $$\Lambda$$ must satisfy

$$\Lambda^2g(x)=-g''(x) $$


Assuming that the limit referred to above is $$\lim\limits_{t \to 0}$$ (the only one that really makes sense), we have:

$$\lim\limits_{t \to 0} u_t(x,t) = \Lambda g(x) $$

First, we know from part (a) that the solution to the Laplace equation looks like:

$$\hat{u}(\lambda,t) = \hat{g}(\lambda)e^{-\vert \lambda\vert t} $$

Taking the derivative with respect to $$t$$ of both sides, and noting that the Fourier transform we have performed is with respect to $$x$$, we have

$$\hat{u}(\lambda,t)_ t =\frac{d}{dt} \hat{g}(\lambda)e^{-\vert \lambda\vert t} $$

Noting that $$\hat{g}(x)$$ is constant w.r.t $$t$$,

$$\hat{u}(\lambda,t)_ t =- \vert \lambda\vert  \hat{g}(\lambda)e^{-\vert \lambda\vert t} $$

And taking the limit

$$\lim\limits_{t \to 0} \hat{u}(\lambda,t)_ t =- \vert \lambda\vert  \hat{g}(\lambda)$$

Taking the inverse fourier transform, we have:

$$\lim\limits_{t \to 0} u(\lambda,t)_ t = \mathcal{F}^{-1}(- \vert \lambda\vert  \hat{g}(\lambda))$$

$$ \Lambda g(x)= \mathcal{F}^{-1}(- \vert \lambda\vert  \hat{g}(\lambda))$$

Thus, we can see that the Dirichlet-Neumann operator in the time domain corresponds to a multiplication by $$-\vert \lambda\vert $$ in the frequency domain. Thus,

$$ \Lambda^2 g(x)= \mathcal{F}^{-1}(\lambda^2 \hat{g}(\lambda))$$

And since we know that

$$\mathcal{F}(g'(x)) = i\lambda\hat{g}(x)$$

$$\mathcal{F}(g''(x)) = -\lambda^2\hat{g}(x)$$

$$\mathcal{F}(-g''(x)) = \lambda^2\hat{g}(x)$$

We can evaluate the inverse Fourier Transform above and get our final result:

$$ \Lambda^2 g(x)= -g''(\lambda)$$

## d.
Find the kernel of the Hilbert transform operator $$H$$ such that

$$\Lambda g = H(g') $$

We know from the previous parts that the general solution to $$u(x,t)$$ is:

$$u(x,t)=\frac{t}{\pi} \int_{\tau = -\infty}^\infty \frac{g(\tau)}{t^2 + (\tau-x)^2} d\tau $$

Making the substitution: $$\tau = x + ts$$, we get:

$$u(x,t)=\frac{1}{\pi} \int_{s = -\infty}^\infty \frac{t^2g(x+ts)}{t^2 + (ts)^2} ds $$

$$u(x,t)=\frac{1}{\pi} \int_{s = -\infty}^\infty \frac{g(x+ts)}{1 + s^2} ds $$

Then taking the time derivative, we have:

$$u(x,t)_ t=\frac{1}{\pi} \int_{s = -\infty}^\infty \frac{sg'(x+ts)}{1 + s^2} ds $$

And remembering from part (b) that: $$\lim\limits_{t \to 0} u(x,t)_ t = \Lambda g(x)$$, we take this limit and get:

$$\Lambda g(x) = \lim\limits_{t \to 0} \frac{1}{\pi} \int_{s = -\infty}^\infty \frac{sg'(x+ts)}{1 + s^2} ds $$

But before evaluating this limit, we reverse our variable substitution: $$\tau = x + ts \implies s = (\tau - x)/t$$

$$\Lambda g(x) = \lim\limits_{t \to 0} \frac{1}{\pi} \int_{t = -\infty}^\infty \frac{t^2}{t}\frac{(\tau - x)g'(\tau)}{t^2 + (\tau - x)^2} \frac{1}{t} d\tau  $$

$$\Lambda g(x) = \lim\limits_{t \to 0} \frac{1}{\pi} \int_{t = -\infty}^\infty \frac{(\tau - x)g'(\tau)}{t^2 + (\tau - x)^2} d\tau  $$

Evaluating the limit:

$$\Lambda g(x) = \lim\limits_{t \to 0} \frac{1}{\pi} \int_{t = -\infty}^\infty \frac{(\tau - x)g'(\tau)}{ (\tau - x)^2} d\tau  $$

$$\Lambda g(x) = \frac{1}{\pi} \int_{t = -\infty}^\infty \frac{g'(\tau)}{ \tau - x} d\tau  $$

Thus, from the initial information of the problem we can see:

$$H(g')(x) = \frac{1}{\pi} \int_{t = -\infty}^\infty \frac{g'(\tau)}{ \tau - x} d\tau$$

We can see that $$H$$ is a straightforward integral transform kernel with convolution varialbe $$\tau$$:

$$ \frac{1}{\pi} \frac{1}{\tau - x}$$



# Question 2

## a
Use Fourier transform to show that the bounded solution $$u$$ of the free-space heat equation

$$ u_t = u_{xx}$$

for $$x \in \mathbb{R}$$ and $$t>0$$, with bounded continuous initial conditions $$u(x,0)= u_0(x)$$, is given by

$$u(x,t) = K_t * u_0(x) = \frac{1}{\sqrt{4 \pi t}} \int_{-\infty}^\infty e^{-(x-y)^2/4t} u_0(y) dy$$
for $$t > 0$$.



First, we begin as in the laplace equation by taking the fourier transform w.r.t. x of the heat equation.

$$ \hat{u}(\lambda)_ t = -\lambda^2\hat{u}(\lambda)$$

This is a first order ode in $$t$$ with characteristic equation: $$r+\lambda^2 = 0$$, implying a general solution

$$\hat{u}(\lambda,t) = A(\lambda)e^{-\lambda^2 t}$$

In conjunction with the fourier transform of the boundary condition:

$$\hat{u}(\lambda,0) = \hat{u}_0(\lambda) $$

We see that $$A=\hat{u}_0$$, and so we have

$$ \hat{u}(\lambda,t) = \hat{u}_0(\lambda)e^{-\lambda^2 t}$$

This function looks like a simple multiplication in the frequency domain, which I know corresponds to a convolution in the time domain.

$$\mathcal{F}[(f*u_0)(x)] = \sqrt{2\pi}\hat{f}(\lambda)\hat{u_0}(\lambda) $$

So I only need take the inverse fourier transform of the function:

$$ K_t = \mathcal{F}^{-1}(\frac{1}{\sqrt{2\pi}}e^{-\lambda^2t})$$

This corresponds to a common fourier transform pair:

$$\mathcal{F}(e^{-\alpha x^2})= \frac{1}{\sqrt{2\alpha}}e^{-\frac{\lambda^2}{4\alpha}} $$
Which can be reversed:

$$\mathcal{F}( \frac{1}{\sqrt{2\alpha}}e^{-\frac{x^2}{4\alpha}}) = e^{-\alpha \lambda^2} $$

Which appears to fit the bill with $$\alpha=t$$

$$\mathcal{F}( \frac{1}{\sqrt{2 t}}e^{-\frac{x^2}{4t}}) = e^{- t \lambda^2} $$

$$\mathcal{F}( \frac{1}{\sqrt{4 \pi t}}e^{-\frac{x^2}{4t}}) =\frac{1}{\sqrt{2\pi}} e^{- t \lambda^2} $$

Which implies

$$K_t = \frac{1}{\sqrt{4 \pi t}}e^{-\frac{x^2}{4t}} $$

By pattern matching.

Since we know

$$u(x,t) = K_t * u_0(x) $$

We can evaluate the convolution in the time domain to get an explicit equation:

$$u(x,t) = \frac{1}{\sqrt{4 \pi t}} \int_{-\infty}^\infty e^{-(x-y)^2/4t} u_0(y) dy$$


## b
Show that $$u$$ attains its initial conditions in the sense that

$$u(x,t) \to u_0(x) $$
as $$t \to 0$$


We want to find:

$$\lim\limits_{t \to 0} \frac{1}{\sqrt{4 \pi t}} \int_{-\infty}^\infty e^{-(x-y)^2/4t} u_0(y) dy$$

To do this, I make the substitution suggested on Piazza:
$$y =x- \sqrt{4t}z$$. This substitution requires us to flip the limits of integration.

$$\lim\limits_{t \to 0} \frac{1}{\sqrt{4 \pi t}} \int_{\infty}^{-\infty} e^{-z^2} u_0(x - \sqrt{4t}z) (-\sqrt{4t}) dz$$

Since it is constant w.r.t the integration, I can move $$-\sqrt{4t}$$ out of the integral to cancel with the leading term.

$$\lim\limits_{t \to 0} \frac{-1}{\sqrt{\pi}} \int_{\infty}^{-\infty} e^{-z^2} u_0(x - \sqrt{4t}z) dz$$

Now I take the limit within the intergral, justified by the dominated convergence theorem.

$$ \frac{-1}{\sqrt{\pi}} \int_{\infty}^{-\infty} e^{-z^2} u_0(x) dz$$

Since $$u_0(x)$$ is now constant w.r.t. the integral:

$$ \frac{-u_0(x)}{\sqrt{\pi}} \int_{\infty}^{-\infty} e^{-z^2}dz$$

flipping the limits of integration again to remove the negative sign:

$$  \frac{u_0(x)}{\sqrt{\pi}} \int_{-\infty}^{\infty} e^{-z^2}dz$$

And since we know:

$$\int_{-\infty}^\infty e^{-x^2} dx = \sqrt{\pi}$$

We can evaluate the inner integral to cancel the remaining $$\frac{1}{\sqrt{\pi}}$$

Giving us the desired result:

$$ \lim\limits_{t \to 0}u(x,t) = u_0(x) $$

# Question 3

Solve the integral equation

$$D^{-1/2} h(t) = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}h(s)ds = g(t) $$
where $$g$$ is a nice function with $$g(0) = 0$$. (Hint: square $$D^{-1/2}$$)


To start, I apply another $$D^{-1/2}$$ to each side,

$$D^{-1/2}D^{-1/2} h(t)= \int_{s=0}^t \frac{1}{\sqrt{4\pi(t-s)}}\int_{\sigma =0}^s \frac{1}{\sqrt{4\pi(s-\sigma)}}h(\sigma)d\sigma ds = D^{-1/2}g(t)$$

Expanding the right hand side:

$$ \int_{s= 0}^t \frac{1}{\sqrt{4\pi(t-s)}}\int_{\sigma=0}^s \frac{1}{\sqrt{4\pi(s-\sigma)}}h(\sigma)d\sigma ds = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}g(s)ds $$

Since the first term is constant w.r.t. $$\sigma$$, I can move it inside the inner integral.

$$ \int_{s= 0}^t \int_{\sigma=0}^s \frac{1}{\sqrt{4\pi(t-s)}} \frac{1}{\sqrt{4\pi(s-\sigma)}}h(\sigma)d\sigma ds = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}g(s)ds $$

Now, I can switch the order of the integrals, which requires changing the limits of integration($$0<\sigma<s<t$$):

$$ \int_{\sigma= 0}^t \int_{s=\sigma}^t \frac{1}{\sqrt{4\pi(t-s)}} \frac{1}{\sqrt{4\pi(s-\sigma)}}h(\sigma)ds d\sigma = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}g(s)ds $$

To evaluate this integral, we make the smart substitution $$s = \sigma + (t - \sigma)\theta$$. Notice that the limits of integration of $$s: \ \sigma,t$$ correspond to $$\theta = 0,1$$ respectively.

Notice as well:

$$ s = \sigma + (t - \sigma)\theta \implies  t-s = (t-\sigma)(1 - \theta)$$

$$ s = \sigma + (t - \sigma)\theta \implies  s-\sigma = (t-\sigma)\theta$$

Making the required substitutions:

$$ \frac{1}{4 \pi}\int_{\sigma= 0}^t \int_{\theta=0}^1 \frac{1}{\sqrt{(t-\sigma)(1-\theta)}} \frac{1}{\sqrt{(s - \sigma) \theta}}h(\sigma)(t-\sigma)d\theta d\sigma = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}g(s)ds $$

$$ \frac{1}{4 \pi}\int_{\sigma= 0}^t h(\sigma)\int_{\theta=0}^1 \frac{1}{\sqrt{(1-\theta)}} \frac{1}{\sqrt{ \theta}}d\theta d\sigma = \int_0^t \frac{1}{\sqrt{4\pi(t-s)}}g(s)ds $$

Now making the substitution $$\theta = \text{cos}^2(\phi)$$, $$d\theta =  -2\text{cos}(\phi)\text{sin}(\phi) d\phi$$ we get:

$$\int_{\theta = 0}^1 \frac{d\theta}{\sqrt{\theta}\sqrt{1-\theta}} d\theta= \int_{\phi = \pi/2}^0 \frac{-2\text{cos}(\phi)\text{sin}(\phi) d\phi}{\sqrt{\text{cos}^2(\phi)}\sqrt{\text{sin}^2(\phi)}}$$

$$\int_{\theta = 0}^1 \frac{d\theta}{\sqrt{\theta}\sqrt{1-\theta}} d\theta= -2 \int_{\phi = \pi/2}^0 \frac{\text{cos}(\phi)\text{sin}(\phi) d\phi}{\text{cos}(\phi)\text{sin}(\phi)}$$

$$\int_{\theta = 0}^1 \frac{d\theta}{\sqrt{\theta}\sqrt{1-\theta}} d\theta= -2 \int_{\phi = \pi/2}^0 d\phi$$

$$\int_{\theta = 0}^1 \frac{d\theta}{\sqrt{\theta}\sqrt{1-\theta}} d\theta= \pi $$

Plugging back in, we have:

$$ \frac{1}{4}\int_{\sigma= 0}^t h(\sigma) d\sigma = \frac{1}{\sqrt{4 \pi}}\int_0^t \frac{g(s)}{\sqrt{(t-s)}}ds $$

And taking the derivative w.r.t $$t$$ on both sides, we have:

$$ \frac{1}{4} h(t) = \frac{1}{\sqrt{4 \pi}} \frac{d}{dt} \int_0^t \frac{g(s)}{\sqrt{(t-s)}}ds$$

First, I use integration by parts $$\int u dv  = uv - \int v du$$ ti get our desired $$g'$$ expression:

$$\int_{s=0}^t \frac{g(s)}{\sqrt{t-s}}ds = -2g(s)\sqrt{t-s}\Big\vert _ {0}^t + \int_{s=0}^t 2\sqrt{t-s} g'(s)ds $$

Notice that the first term vanishes at the limits since at $$t$$, $$\sqrt{t-t}=0$$, and at $$0$$, $$g(0)=0$$.

so we are left with:

$$\int_{s=0}^t \frac{g(s)}{\sqrt{t-s}}ds =  \int_{s=0}^t 2\sqrt{t-s} g'(s)ds $$

Plugging back in, we now have:

$$ \frac{1}{4} h(t) = \frac{1}{\sqrt{4 \pi}} \frac{d}{dt} \int_{s=0}^t 2\sqrt{t-s} g'(s)ds$$

We can switch the order of integration and differentiation since they are over different variables:

$$ \frac{1}{4} h(t) = \frac{1}{\sqrt{4 \pi}}  \int_{s=0}^t \frac{d}{dt} 2\sqrt{t-s} g'(s)ds$$

$$ \frac{1}{4} h(t) = \frac{1}{\sqrt{4 \pi}}  \int_{s=0}^t \frac{ g'(s)}{\sqrt{t-s}}ds$$

$$ h(t) = \frac{2}{\sqrt{ \pi}}  \int_{s=0}^t \frac{ g'(s)}{\sqrt{t-s}}ds$$

Solving the integral equation.

# Question 4

## a
Solve the intial-boundary value problem for the heat equation

$$ u_t = u_{xx}$$
for $$x>0,t>0$$, with homogeneous initial conditions

$$u(x,0)=0 $$

and boundary conditions

$$u(0,t)= g(t)$$
where $$g$$ is a nice function with $$g(0)=0$$ (Hint: Try $$u(x,t)= \int_0^t K_{t-s}(x)h(s)ds$$ and solve an integral equation for $$h$$).

We know from part 2 that:

$$K_t = \frac{1}{\sqrt{4 \pi t}}e^{-\frac{x^2}{4t}} $$
so

$$ K_{t-s} = \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} $$

So let's try an expression of the form:

$$ u(x,t) = \int_{s=0}^t  \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} h(s) ds$$

In order to show that this is indeed a solution, we have to show that this expression satisfies the boundary conditions and solves the heat equation.

For the first initial condition: $$u(x,0)=0$$,

$$u(x,0)= \int_{s=0}^0 \cdots ds = 0$$

Since the limits of integration are the same.

Next, to show that this expression solves the heat equation, I must show that

$$u_t  = u_{xx} $$

Starting with

$$ u(x,t) = \int_{s=0}^t  \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} h(s) ds$$

$$u_t = \frac{d}{dt} \int_{s=0}^t  \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} h(s) ds$$

And via the DCT

$$u_t = \int_{s=0}^t \frac{d}{dt}  \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} h(s) ds$$

Using the product rule with:

$$ f = \frac{1}{\sqrt{4 \pi (t-s)}}, \qquad g = e^{-\frac{x^2}{4(t-s)}}$$

and


$$ f_t = - \frac{1}{4 \sqrt{\pi} (t-s)^{3/2}} \qquad g_t = \frac{x^2}{4(t-s)^2} e^{-\frac{x^2}{4(t-s)}}$$

We get:

$$u_t = \int_{s=0}^t h(s) \Big[\frac{1}{\sqrt{4 \pi (t-s)}}\frac{x^2}{4(t-s)^2} e^{-\frac{x^2}{4(t-s)}} - \frac{1}{4 \sqrt{\pi} (t-s)^{3/2}} e^{-\frac{x^2}{4(t-s)}}  \Big] ds $$

$$ u_t = \int_{s=0}^t \Big[ \frac{x^2}{8 \sqrt{\pi} (t-s)^{5/2}} - \frac{1}{4 \sqrt{\pi} (t-s)^{3/2}}  \Big]e^{-\frac{x^2}{4(t-s)}} h(s) ds $$

Next, to find $$u_{xx}$$, by taking the derivative and using the DCT, we have

$$ u_x = \int_{s=0}^t \frac{d}{dx}  \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} h(s) ds = u_x \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}}h(s) \frac{d}{dx}e^{-\frac{x^2}{4(t-s)}} ds $$

$$u_x = \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}} \frac{-2x}{4(t-s)} e^{\frac{-x^2}{4(t-s)}} ds $$

Taking a second $$x$$ derivative and using the DCT, we have:

$$u_{xx} = \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}} \frac{d}{dx} \frac{-2x}{4(t-s)} e^{\frac{-x^2}{4(t-s)}} ds$$

Evaluating with the product rule,

$$u_{xx} = \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}} \frac{-2}{4(t-s)}e^{\frac{-x^2}{4(t-s)}} + \frac{1}{\sqrt{4 \pi (t-s)}} \frac{4x^2}{16(t-s)^2}e^{\frac{-x^2}{4(t-s)}}$$

$$u_{xx} = \int_{s=0}^t \Big [\frac{x^2}{8 \sqrt{\pi} (t-s)^{5/2}}-\frac{1}{4\sqrt{\pi} (t-s)^{3/2}} \Big] e^{-\frac{x^2}{4(t-s)}} h(s) ds = u_t$$

So we have verified that this expression solves the heat equation.

Lastly, we need our expression to satisfy $$u(0,t) = g(t)$$, so plugging in we have:


$$ K_t(0) = \frac{1}{\sqrt{4 \pi t}}e^{-\frac{0^2}{4t}} =  \frac{1}{\sqrt{4 \pi t}}$$

$$ u(0,t) = \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}}h(s)ds = g(t)$$

This is just question 3,

$$ D^{-1/2} h(t) = g(t)$$

So without repeating my previous work, we know:

$$ h(t) = \frac{2}{\sqrt{ \pi}}  \int_{s=0}^t \frac{ g'(s)}{\sqrt{t-s}}ds$$

Satisfies the boundary conditions, yielding the following formula for $$u(x,t)$$ with a little renaming of variables for sanity:

$$u(x,t) = \int_{s=0}^t \frac{1}{\sqrt{4 \pi (t-s)}}e^{-\frac{x^2}{4(t-s)}} \int_{\sigma=0}^s \frac{2}{\sqrt{\pi}} \frac{g'(\sigma)}{\sqrt{s-\sigma}} d\sigma ds$$

So to solve the heat equation with given initial conditions and boundary equations, we have:

$$u(x,t) = \frac{1}{\pi}\int_{s=0}^t \frac{1}{\sqrt{ (t-s)}}e^{-\frac{x^2}{4(t-s)}} \int_{\sigma=0}^s \frac{g'(\sigma)}{\sqrt{s-\sigma}} d\sigma ds$$

I will attempt to simplify the expression as well as show that it actually solves the heat equation.

$$u(x,t) = \frac{1}{\pi}\int_{s=0}^t \frac{1}{\sqrt{ (t-s)}}e^{-\frac{x^2}{4(t-s)}} \int_{\sigma=0}^s \frac{g'(\sigma)}{\sqrt{s-\sigma}} d\sigma ds$$

$$\frac{1}{\pi}\int_{s=0}^t \int_{\sigma=0}^s \frac{1}{\sqrt{ (t-s)}}e^{-\frac{x^2}{4(t-s)}}  \frac{g'(\sigma)}{\sqrt{s-\sigma}} d\sigma ds$$

Switching the order of integration:

$$\frac{1}{\pi} \int_{\sigma=0}^t \int_{s=\sigma}^t  \frac{1}{\sqrt{ (t-s)}}e^{-\frac{x^2}{4(t-s)}}  \frac{g'(\sigma)}{\sqrt{s-\sigma}} ds d\sigma $$

$$\frac{1}{\pi} \int_{\sigma=0}^t g'(\sigma)\int_{s=\sigma}^t  \frac{1}{\sqrt{ (t-s)}}  \frac{1}{\sqrt{s-\sigma}} e^{-\frac{x^2}{4(t-s)}}ds d\sigma$$

With a change of variables: $$s=\sigma + (t-\sigma)\theta$$

$$\frac{1}{\pi} \int_{\sigma=0}^t g'(\sigma)\int_{\theta =0}^1  \frac{1}{\sqrt{ 1+ \theta}}  \frac{1}{\sqrt{\theta}} e^{-\frac{x^2}{4(t-\sigma)(1+\theta)}}d\theta  d\sigma$$

And changing variables again: $$\theta = \cos^2(\phi)$$

$$\frac{1}{\pi} \int_{\sigma=0}^t g'(\sigma)\int_{\phi =\frac{\pi}{2}}^0  \frac{-2 \sin \phi \cos \phi}{\sin \phi \cos \phi}  e^{-\frac{x^2}{4(t-\sigma)(\sin^2 \phi)}} d\phi  d\sigma$$

$$\frac{-2}{\pi} \int_{\sigma=0}^t g'(\sigma)\int_{\phi =\frac{\pi}{2}}^0  e^{-\frac{x^2}{4(t-\sigma)(\sin^2 \phi)}}d\phi  d\sigma$$

Now that we have gotten rid of those terms, we can revert back to our original variables.

$$u(x,t) = \frac{-2}{\pi} \int_{\sigma=0}^t g'(\sigma)\int_{s=\sigma}^t  e^{-\frac{x^2}{4(t-s)}}ds d\sigma$$

Thus, I have shown that the above solution solves the heat equation and satisfies the given boundary conditions.

## b
Assume that $$g'\in L^2(\mathbb{R})$$ is also bounded and continuous. Argue directly from the heat equation that if

$$u_x(x,t) \to \Lambda g(t)$$
as $$x \to 0$$, then the Dirihlet-Neumann operator $$\Lambda$$ must satisfy
$$\Lambda^2 g(t) = g'(t)$$

First, we know that $$u(0,t) = g(t)$$, and since $$u$$ is a nice function, this means

$$\lim\limits_{x \to 0} u(x,t) = u(0,t)= g(t) $$

Next, observe via the Fourier Transform:

$$\mathcal{F}_x[ \lim\limits_{x \to 0} u_x(x,t)] = \mathcal{F}_x [\Lambda g(t)]$$

Exchanging limits via the dominated convergence theorem:
$$\lim\limits_{x \to 0} \mathcal{F}_x[  u_x(x,t)] = \mathcal{F}_x [\Lambda g(t)]$$

$$\lim\limits_{x \to 0} i \lambda \mathcal{F}_x[  u(x,t)] = \mathcal{F}_x [\Lambda g(t)]$$

And again:

$$ i \lambda \mathcal{F}_x[ \lim\limits_{x \to 0} u(x,t)] = \mathcal{F}_x [\Lambda g(t)]$$

And since we know $$\lim\limits_{x \to 0} u(x,t) = u(0,t)= g(t)$$,

$$ i \lambda \mathcal{F}_x[ g(t)] = \mathcal{F}_x [\Lambda g(t)]$$

This implies that $$\Lambda$$ in the time domain is a multiplication by $$i\lambda$$ in the fourier domain.

So to evaluate $$\Lambda^2$$ next:

$$\mathcal{F}_x [\Lambda^2 g(t) ]= -\lambda^2 F_x[g(t)] = -\lambda^2 F_x[\lim\limits_{x\to 0} u(x,t)] $$

Exchanging limits via the dominated convergence theorem.

$$\mathcal{F}_x [\Lambda^2 g(t) ]=\lim\limits_{x\to 0} -\lambda^2 F_x[ u(x,t)] = \lim\limits_{x\to 0}  F_x[ u_{xx}(x,t)]$$

And again:

$$\mathcal{F}_x [\Lambda^2 g(t) ] =  F_x[ \lim\limits_{x\to 0}  u_{xx}(x,t)]$$

Taking the inverse Fourier Transform:


$$ \Lambda^2 g(t)= \lim\limits_{x\to 0}  u_{xx}(x,t) $$

Since $$u_{xx} = u_t$$, and switching limits via the DCT.

$$\Lambda^2 g(t) = \lim\limits_{x \to 0} u_t(x,t) = \frac{d}{dt} \lim\limits_{x \to 0} u(x,t) = \frac{d}{dt} g(t)$$

So:

$$\Lambda^2 g(t) = g'(t) $$

## c
Find the Dirichlet-Neumann operator $$\Lambda$$.

We have two pieces of information about $$\Lambda$$:

From Q2,

$$ \Lambda^2 g(x) = -g''(x)$$

And from Q4:

$$ \Lambda^2 g(t) = g'(t)$$

Combining them, (and assuming it is a linear operator) we see:

$$\Lambda^2(g(x) + g(t)) = g'(t) - g''(x) \implies \Lambda^2 = \frac{d}{dt} - \frac{d^2}{dx^2}$$

Indeed, this makes sense because to each partial derivative, the function of the other variable is constant and so becomes 0 and disappears.

Thus, $$\Lambda$$ can be defined in terms of the Riemann-Liouville Fractional Derivative:

$$ \Lambda = D^{1/2}_t + iD_x$$

Notice that there is across term in $$\Lambda^2, \ 2i D^{1/2}_t D_x$$. However this vanishes when applies to the function $$g(x)$$ or $$g(t)$$, as one of the derivatives will be over a constant and therefore will be 0.

# Question 5

Use Fourier transform in the variable $$x$$ to solve the problem of Question 4. (Hint: Extend $$u$$ discontinuously to be zero for negative $$x$$. Integrate by parts. Use parity.)

We start by taking the fourier transform in the variable x of $$u_t(x,t)$$. Notice we integrate only on $$[0.\infty)$$ since $$u(x,t)=0 \text{ if } x<0$$.

$$\hat{u}_t(k,t) = \int_{x=0}^\infty e^{-ikx}u_t(x,t) dx$$

Substituting based on the heat equation:

$$\hat{u}_t(k,t) = \int_{x=0}^\infty e^{-ikx}u_{xx}(x,t) dx$$

Integrating by parts, we have:

$$\hat{u}_t(k,t) = e^{-ikx} u_x(x,t)\Big\vert _ {x=0}^\infty + \int_{x=0}^\infty ike^{-ikx} u_x(x,t) dx$$

Integrating by parts again,

$$\hat{u}_t(k,t) = u_x(0,t) + ike^{-ikx}u(x,t) \Big\vert _ {x=0}^\infty  + \int_{x=0}^\infty k^2 e^{-ikx} u(x,t)$$

Recognizing that the last term is a fourier transform, we have:

$$\hat{u}_t(k,t) = u_x(0,t) + iku(0,t)   + k^2  \hat{u}(x,t)$$

This is actually a nonhomogenous differential equation in $$t$$:

$$\hat{u}_t(k,t) - k^2 \hat{u}(x,t) = u_x(0,t) + iku(0,t)  $$

Using integrating factor $$e^{-k^2 t}$$:

$$e^{-k^2 t}\hat{u}_t(k,t) + e^{-k^2 t}(- k^2) \hat{u}(x,t) = e^{-k^2 t}[u_x(0,t) + iku(0,t)]  $$

We can see via the product rule that the left hand side is actually the derivative of another function:

$$\frac{d}{dt}[e^{-k^2 t} \hat{u}(k,t)] = e^{-k^2 t}[u_x(0,t) + iku(0,t)]  $$

Taking the integral and derivative of the right hand side:

$$\frac{d}{dt}[e^{-k^2 t} \hat{u}(k,t)] = \frac{d}{dt} \int_{s=0}^t e^{-k^2 s}[u_x(0,s) + iku(0,s)]ds   $$

Thus the inner expressions of the derivatives must be the same. So equating them and multiplying by $$e^{k^2 t}$$, we have

$$ \hat{u}(k,t) = \int_{s=0}^t e^{-k^2 (s-t)}[u_x(0,s) + iku(0,s)]ds   $$

I take the inverse fourier transform and separate the terms. I also observe that $$u(0,t)= g(t)$$. I am also switching the order of integration.

$$u(x,t) = \frac{1}{2 \pi} \int_{s=0}^t \int_{k = - \infty}^\infty e^{ikx} e^{-k^2 (s-t)} - ik g(t) dk ds - \frac{1}{2 \pi}\int_{s=0}^t \int_{k = - \infty}^\infty e^{ikx} e^{-k^2 (s-t)}u_x(0,s) dk ds  $$

We can see that the first term is odd, so the integral from $$-\infty$$ to $$\infty$$ will be 0. Accordingly, I remove the term and am left with:

$$u(x,t) = - \frac{1}{2 \pi}\int_{s=0}^t \int_{k = - \infty}^\infty e^{ikx} e^{-k^2 (s-t)}u_x(0,s) dk ds  $$

$$u(x,t) = - \frac{1}{2 \pi}\int_{s=0}^t u_x(0,s) \int_{k = - \infty}^\infty  e^{-k^2 (t-s) + ixk} dk ds$$

We must integrate the inner term:

$$\int_{-\infty}^\infty e^{-(t-s)k^2 + ixk} dk$$

If we substitute $$a=(t-s)$$, and $$b=ix$$, the denominator becomes $$-ak^2 + bk$$.

We can complete the square for the exponent to become $$-a(k - \frac{b}{2a})^2 + \frac{b^2}{2a}$$

So our integral becomes:

$$e^{b^2/4a} \int_{k=-\infty}^\infty e^{-a (k - b/2a)^2} dk$$

With a little manipulation of the exponent inside the integral, we can see that it is actually a gaussian and so the integral is 1.

$$e^{b^2/4a} \int_{k=-\infty}^\infty e^{-\frac{1}{2} \Big(\frac{k - b/2a}{1/\sqrt{2a}} \Big)^2} dk$$

Leaving us with

$$ e^{b^2/4a} $$

So plugging back in we have:

$$u_(x,t) = \frac{1}{2\pi} \int_{0}^t u_x(0,s) e^{-\frac{x^2}{4(t-s)}}$$

This is the same expression we had in problem (4a), if

$$g_x(0,s) = \frac{\sqrt{\pi}}{\sqrt{(t-s)}} h(s)$$
