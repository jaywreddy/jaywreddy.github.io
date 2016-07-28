---
title: Time Series Analysis Assignment 2
category: "Time Series Analysis"
layout: post
---

# 1.
Let $$x_t = e^{2\pi i \omega t}$$ for some $$\omega \in \Big( -\frac{1}{2}, \frac{1}{2} \Big]$$. Show that

  $$ d(\omega_j) = n^{-1/2} e^{\pi i (\omega - \omega_j)t} D_n(\omega - \omega_j)$$
Where $$D_n(\omega - \omega_j) = \frac{\sin(n \pi (\omega - \omega_j))}{\sin(\pi(\omega - \omega_j))}$$ is the Dirichlet kernel. Plot the Dirichlet kernel, and explain what happens to the DFT coefficients if the Fourier frequencies do not exactly match the frequency of the signal.

## Derivation

To derive the desired formula, we start with the definition of the Discreet Fourier Transform:

$$d(w_j) = n^{-1/2}\sum\limits_{t=1}^{n} x_t e^{-2\pi i \omega_j t}$$

Applying this to our $$x_t$$, we have:

$$ d(w_j) = n^{-1/2}\sum\limits_{t=1}^{n}e^{2\pi i \omega t} e^{-2\pi i \omega_j t}= n^{-1/2}\sum\limits_{t=1}^{n} e^{2\pi i (\omega - \omega_j) t}$$

Next, we can use the geometric series formula:

$$\sum\limits_{k=1}^n r^k =  \frac{1 - r^n}{1-r}$$

Substituting in our case for $$r=e^{2\pi i (\omega- \omega_j)}$$, we have:

$$d(\omega_j) = n^{-1/2} \frac{1- e^{2\pi i (\omega- \omega_j)n} }{1 - e^{2\pi i (\omega- \omega_j)}} $$

In order to coerce this expression into a sinusoid, I observe that

$$\sin(z) = \frac{e^z - e^{-z}}{2} $$

And so I multiply the numerator and denominator by $$r^{-1/2}=e^{-\pi i (\omega - \omega_j)}, r^{-n/2} =e^{-n\pi i (\omega - \omega_j)}$$ as so:

$$1= \frac{r^{-n/2}}{r^{-n/2}} \frac{r^{-1/2}}{r^{-1/2}} =r^{n/2 - 1/2} \frac{r^{-n/2}}{r^{-1/2}}  = e^{\pi i (\omega - \omega_j ) (n-1)} \frac{e^{-n\pi i (\omega - \omega_j ) }}{e^{-\pi i (\omega - \omega_j ) }}$$

So multiplying our expression for $$d(\omega_j)$$, we have:

$$d(\omega_j) = n^{-1/2} e^{\pi i (\omega - \omega_j ) (n-1)}  \frac{e^{-n\pi i (\omega - \omega_j ) } - e^{-n\pi i (\omega - \omega_j ) }}{e^{-\pi i (\omega - \omega_j ) } - e^{\pi i (\omega - \omega_j ) } }  $$

We can see that the fraction numerator and denominator are actually sinusoids:

$$ -2i\sin(n\pi(\omega - \omega_j)), \quad -2i\sin(\pi(\omega- \omega_j))$$

So plugging in and cancelling the leading $$-2i$$, we have

$$d(\omega_j) = n^{-1/2} e^{\pi i (\omega - \omega_j ) (n-1)}  \frac{\sin(n\pi(\omega - \omega_j))}{\sin(\pi(\omega- \omega_j))}$$

With a final observation that $$D_n(\omega - \omega_j ) =  \frac{\sin(n\pi(\omega - \omega_j))}{\sin(\pi(\omega- \omega_j))}$$, we arrive at the desired result.

$$d(\omega_j) = n^{-1/2} e^{\pi i (\omega - \omega_j ) (n-1)} D_n(\omega - \omega_j)$$

## Plotting the Dirichlet Kernel
Next, a code snippet to generate the kernel.

```python
from matplotlib import pyplot as plt
import numpy as np
from math import pi, sin, sqrt

# take care of errors at 0.
np.seterr(divide = 'ignore')

# Such partial, so function
# Somewhere, John DeNero just shed a tear of joy
def dirichlet(n):
  def D(w):
    num = sin(n*pi*w)
    den = sin(pi * w)
    return num/den
  return D

# Plotting Initialization
plt.figure()

# Plot some Dirchlet Kernels!
for n in [2,6,12,32]:
  kernel = dirichlet(n)
  domain = np.linspace(-1/2, 1/2, 200)
  data = [kernel(x) for x in domain]
  plt.plot(domain, data, label = "$$D_{ %d}$$"%n)

# Plotting Boilerplate
plt.title("Dirichlet Kernel Plot")
plt.xlabel("$$\omega- \omega_j$$")
plt.ylabel("$$D_n (\omega- \omega_j)$$")
plt.legend()
plt.savefig("figs/dirichlet.png")
```
![Dirichlet Kernel](/images/timeseries_analysis/dirichlet.png)

## Dirichlet Kernel Analysis

We can see from the Dirichlet Kernel Plot that there is a significant positive spike when $$\omega - \omega_j \approx 0$$, and then the function quickly tapers off in both direction. This spike becomes more pronounced as $$n$$ increases, and tapers off more quickly. For large $$n$$, this would become like the Dirac Delta function. Therefore, if the frequency of the signal is close to the Fourier frequencies, they will be greatly amplified, while others will be aggressively filtered out. This effect becomes more pronounced as $$n$$ increases. Thus, we can say that the Dirichlet Kernel "selects" the closest harmonic frequency to the underlying frequency of the function.

# 2.
Consider the following $$ARMA(p,q)$$ processes, all of the form $$\phi (B) x_t = \theta (B) w_t$$. Calculate the spectral density. Sketch the location of the poles and zeros in the complex plane, and describe how they affect the spectral density.

First, we know that for an $$ARMA(p,q)$$ process, the spectral density is given by:

$$f_x(\omega) = \sigma_w^2 \frac{\vert \theta(e^{-2\pi i \omega})\vert ^2}{\vert \phi(e^{- 2 \pi i \omega})\vert ^2} $$

Thus, zeros will occur at the roots of $$\theta(e^{-2\pi i \omega})$$, and poles will occur at the roots of $$\phi(e^{-2\pi i \omega})$$.


## a
$$\phi(z)=1+\Big(\frac{9}{10}z\Big)^2, \theta(z) = 1 + \frac{1}{3}z,$$

First, to find the poles:

The frequency response will have a pole when the denominator is 0.

$$\phi(z) = 1 + \Big(\frac{9}{10}z\Big)^2$$
has roots at $$z= \pm \frac{10}{9} i$$.

Thus, poles of the spectrogram appear at

$$z = \pm \frac{10}{9}i$$

Next, for the zeros:

$$\theta(z)$$ has only one root at $$z=-3$$.

Thus, the zero of the spectrogram appear at

$$z = -3 $$

Our actual spectrogram is

$$f_x(\omega) = \sigma_w^2 \frac{\vert 1 + \frac{1}{3}e^{- 2 \pi i \omega}\vert ^2}{\vert 1 + (\frac{9}{10}e^{-2\pi i \omega})^2\vert ^2} $$

$$f_x(\omega) = \sigma_w^2 \frac{(1 + \frac{1}{3}e^{-2\pi i \omega})(1 + \frac{1}{3}e^{2\pi i \omega})}{(1 + (\frac{9}{10}e^{-2\pi i \omega})^2)(1 + (\frac{9}{10}e^{2\pi i \omega})^2)} $$

$$f_x(\omega) = \sigma_w^2 \frac{\frac{10}{9} + \frac{1}{3}(e^{2 \pi i \omega}+ e^{-2 \pi i \omega})}{1 + (\frac{9}{10})^4 + (\frac{9}{10})^2 (e^{-4 i \pi \omega}  + e^{4 i \pi \omega})} $$

$$f_x(\omega) = \sigma_w^2 \frac{\frac{10}{9} + \frac{2}{3}\cos(2\pi \omega)}{1 + (\frac{9}{10})^4 + 2(\frac{9}{10})^2 \cos(4\pi \omega)} $$

I will construct a plot for when $$\sigma_w^2 =1$$

``` python
def analyze(ar_params, ma_params, name):
  # Matplotlib setup
  plt.figure()
  plt.subplot(2,1,1)

  ar_poly = np.poly1d(ar_params[::-1])
  ma_poly = np.poly1d(ma_params[::-1])

  def transfer(x):
    z= np.exp(-2j*pi*x)
    num = np.abs(ma_poly(z))**2
    den = np.abs(ar_poly(z))**2
    return num/den

  # Calculate Spectral Density
  w = np.linspace(0, 1/2, 1000)
  sd = [transfer(t) for t in w]

  # Plotting niceness
  plt.plot(w,sd)
  plt.title("Spectral Density")
  plt.xlabel("$$\omega$$")

  plt.subplot(2,1,2)

  #poles and zeros
  zeros = ma_poly.r
  plt.scatter(zeros.real,zeros.imag,
    color = "b",
    label= "Zeros",
    marker= "o")

  poles = ar_poly.r
  plt.scatter(poles.real, poles.imag,
    color = "r",
    label = "Poles",
    marker = "x")

  plt.title("Poles and Zeros of Spectral Density")
  plt.axhline(y=0, color='k')
  plt.text(.1,0,"$$\mathbb{R}$$")
  plt.axvline(x=0, color='k')
  plt.text(0,.12, "$$\mathbb{C}$$")
  plt.legend(loc='lower center',
    bbox_to_anchor=(0.5, -0.35),
    ncol=2)
    
  plt.savefig("figs/%s.png"%name)

ma_params = np.array([1,1/3])
ar_params = np.array([1, 0, 81/100])
name = "specdens2_a"
analyze(ar_params,ma_params,name)
```

![Pole/Zero Graph and Spectral Density](/images/timeseries_analysis/specdens2_a.png)

## b
$$\phi(z)=1-2z + 2z^2, \theta(z) = 1 - \frac{1}{2}z,$$

By the quadratic equation, the roots of $$\phi(z)$$ are:

$$z= \frac{2\pm \sqrt{4 - 8}}{4} $$

$$z = \frac{1}{2} \pm \frac{1}{2}i $$

And so the poles of the spectrogram appear at

$$z = \frac{1}{2} \pm \frac{1}{2}i $$

$$\theta(z)$$ has only one root at $$z = 2$$, so the zeros of the spectrogram appear at

$$z = 2 $$

Next, to calculate the spectral density:

$$f(\omega) = \sigma_w^2\frac{\vert 1 - \frac{1}{2}e^{-2\pi i \omega}\vert ^2}{\vert 1 -2(e^{-2\pi i \omega}) + 2 (e^{- 2 \pi i \omega})^2\vert ^2} $$

$$ f(\omega) = \sigma_w^2\frac{(1 - \frac{1}{2}e^{-2\pi i \omega})(1 - \frac{1}{2}e^{2\pi i \omega})}
{(1 -2(e^{-2\pi i \omega}) + 2 (e^{- 2 \pi i \omega})^2)(1 -2(e^{2\pi i \omega}) + 2 (e^{2 \pi i \omega})^2)} $$

$$ f(\omega) = \sigma_w^2 \frac{\frac{5}{4}  - \frac{1}{2}(e^{-2\pi i \omega} + e^{2\pi i \omega})}
{9 - 6 (e^{-2\pi i \omega} + e^{2\pi i \omega}) +2(e^{-4\pi i \omega} + e^{4\pi i \omega})} $$

$$f(\omega) = \sigma_w^2\frac{\frac{5}{4} - \cos(2\pi \omega)}{9 - 12\cos(2\pi \omega) + 4 \cos(4\pi \omega)} $$

``` python
ma_params = np.array([1,-1/2])
ar_params = np.array([1, -2, 2])
name = "specdens_b"
analyze(ar_params,ma_params,name)
```

![Pole/Zero Graph and Spectral Density](/images/timeseries_analysis/specdens_b.png)

## c
$$\phi(z)=1-4 z^2, \theta(z) = 1 - z + z^2,$$

$$\phi(z)$$ has roots at $$z= \pm \frac{1}{2}$$, and so the spectral density has poles at

$$z= \pm \frac{1}{2} $$

By the quadratic formula, $$\theta(z)$$ has roots at

$$z = \frac{1 \pm \sqrt{1 - 4}}{2} $$

$$z = \frac{1}{2} \pm \frac{\sqrt{3}}{2}$$

And so the spectral density has zeros at

$$z = \frac{1}{2} \pm \frac{\sqrt{3}}{2}$$

Next, to caculate the spectral density:


$$f(\omega) = \sigma_w^2 \frac{\vert 1 - e^{-2\pi i \omega} + (e^{-2\pi i \omega})^2\vert ^2}{\vert 1 - 4(e^{-2\pi i \omega})^2\vert ^2} $$

$$f(\omega) = \sigma_w^2 \frac{(1 - e^{-2\pi i \omega} + (e^{-2\pi i \omega})^2)(1 - e^{2\pi i \omega} + (e^{2\pi i \omega})^2)}
{(1 - 4(e^{-2\pi i \omega})^2)(1 - 4(e^{2\pi i \omega})^2)} $$

$$f(\omega) = \sigma_w^2 \frac{3 -2(e^{-2\pi i \omega} + e^{2\pi i \omega})+(e^{-4\pi i \omega} + e^{4\pi i \omega})}
{17 - 4(e^{-4\pi i \omega} + e^{4\pi i \omega})} $$

$$f(\omega) = \sigma_w^2 \frac{3 -4\cos(2\pi \omega)+2\cos(4\pi \omega)}
{17 - 16\cos(4\pi \omega)} $$


``` python
ma_params = np.array([1,-1,1])
ar_params = np.array([1, 0, -4])
name = "specdens_c"
analyze(ar_params,ma_params,name)
```

![Pole/Zero Graph and Spectral Density](/images/timeseries_analysis/specdens_c.png)

## d
$$\phi(z)=1+\frac{3}{4}z, \theta(z) = 1 + \frac{1}{9}z^2,$$

$$\phi(z)$$ has one root at $$z = -\frac{4}{3}$$, so the spectral density has a pole at

$$z = - \frac{4}{3} $$

$$\theta(z)$$ has two roots at $$z = \pm 3i$$, so the spectral density has two zeros at

$$z = \pm 3i $$

Next, to calculate the spectral density

$$f(\omega) = \sigma_w^2\frac{\vert 1 + \frac{1}{9}(e^{-2\pi i \omega})^2\vert ^2}{\vert 1 + \frac{3}{4}e^{-2\pi i \omega}\vert ^2} $$

$$f(\omega) = \sigma_w^2\frac{(1 + \frac{1}{9}(e^{-2\pi i \omega})^2)(1 + \frac{1}{9}(e^{2\pi i \omega})^2)}{(1 + \frac{3}{4}e^{-2\pi i \omega})(1 + \frac{3}{4}e^{2\pi i \omega})} $$

$$f(\omega) = \sigma_w^2\frac{\frac{82}{81} + \frac{1}{9}(e^{-4\pi i \omega}+e^{4 \pi i \omega})}{\frac{25}{16} + \frac{3}{4}(e^{-2\pi i \omega} + e^{2\pi i \omega})} $$

$$f(\omega) = \sigma_w^2\frac{\frac{82}{81} + \frac{2}{9}\cos(2\pi \omega)}{\frac{25}{16} + \frac{6}{4}\cos(4 \pi \omega)} $$

``` python
ma_params = np.array([1, 0, 1/9])
ar_params = np.array([1, 3/4])
name = "specdens_d"
analyze(ar_params,ma_params,name)
```

![Pole/Zero Graph and Spectral Density](/images/timeseries_analysis/specdens_d.png)

## Analysis

It appears that poles lead to a high frequency response and zeros lead to a small frequency response.

If you think of $$e^{2\pi i \omega}$$ traversing the unit circle from $$+1 \to -1$$ as $$\omega: 0 \to \frac{1}{2}$$. The frequency response at $$\omega$$ appears to be characterized by the poles/zeros near $$e^{2\pi i \omega}$$.

For example, in (a), we see a zero at $$-1$$, and a pole at approximately $$\pi/2$$ in the complex plane. Accordingly, there appears to be a spike at $$\omega = 1/4 \implies e^{2\pi i \omega} = \pi/2$$. Additionally, the frequency response damps at the limits, corresponding to the zero near $$-1$$ of the unit circle.

In (b), we see that there is a pole at approximately $$\pi/4$$ on the complex plane, corresponding to a peak at $$\omega = 1/8 \implies e^{2\pi i \omega} = \pi /4$$. There is also a zero near $$+1$$ on the unit circle, but the damping effects of this zero seem to be much less pronounced than in (a). Perhaps this is because there is less of a phase difference between the poles and zeros, so the effects are less distinct.

In (c), we see poles at $$\pm1$$ on the unit circle, and the frequency response has peaks at both of those locations. Additionally, the zero at $$\sim \pi/4$$ appears to contribute to the damping effect in the frequency response near $$\omega \sim 1/8$$.

In (d), we see a single pole near $$-1$$ on the unit circle, corresponding to a spike at $$\omega = 1/2$$. The pole at $$\pi/2$$ appears to be the cause of the frequency response remaining almost flat up to $$\omega = 1/8$$.

Based on these observations, I restate my theory that the poles and zeros in the complex plane correspond to peaks and lows in the frequency domain for $$\omega$$ corresponding to the location on the unit circle as it is traced out by $$e^{2\pi i \omega}$$.

# 3.
Generate four sample paths of length $$n=128,256,512,1024$$ of the $$AR(1)$$ process $$\phi(B)x_t = w_t$$, where $$\phi(z)=1-\frac{1}{2}z$$. Plot the periodogram for each sample path. Calculate an approximate $$95$$% confidence interval for $$f(0.1)$$. Does the confidence interval shrink as $$n$$ grows?

Here is my script to generate the relevant plots.

``` python
np.random.seed(8675309) # Jenny
#define the AR process
arparams = np.array([1, -.5])
ar = np.r_[1,-arparams]
ma = np.ones(1)
ar_process =  sm.tsa.ArmaProcess(ar,ma)

plt.figure()
sizes = np.array([128,256,512,1024])
pgrams = []
for i,n in enumerate(sizes):
  # Set up figure
  plt.subplot(2,2, i+1)

  # Create sample and Calculate Periodogram
  sample = ar_process.generate_sample(n)
  pgram = sm.tsa.stattools.periodogram(sample)
  pgrams.append(pgram)
  domain = np.linspace(-1/2,1/2,n)

  # Plot and make it pretty
  plt.plot(domain, pgram)
  plt.title("AR(1) Periodogram, $$n=%d$$"%n)
  plt.xlabel("$$\omega$$")
  plt.ylabel("$$\hat{f}(\omega)$$")

plt.savefig("figs/pgram.png")
```

![Periodograms](/images/timeseries_analysis/pgram.png)

Here, I have plotted sample periodograms of each of the paths generated by the AR(1) model.

Next, I establish confidence intervals for $$f(.1)$$.

We know that a $$100(1-\alpha)$$% confidence interval is of the form:

$$\frac{2 I(\omega_{j:n})}{\chi_2^2 (1- \alpha/2)} \leq f(\omega) \leq \frac{2 I (\omega_{j:n})}{\chi_2^2(\alpha/2)} $$

```python
import scipy as sp
import pandas as pd

chi_int = np.array(sp.stats.chi2.interval(.95,2))
ws =  np.rint((1/2 + .1)*sizes).astype(int)
cis = []
for s,w in zip(pgrams,ws):
  ci =  2*s[w] * chi_int
  cis.append(ci)

data = [[n,l,h, h-l] for n,(l,h) in zip(sizes, cis)]
labels = ["N", "CI low", "CI high", "CI length"]
df = pd.DataFrame(data, columns = labels)
#Manually Put data into a table.. **yawn**
```

$$ \begin{array}{ c \vert  c \vert  c \vert  c }
\text{n} & \text{CI low} & \text{CI high} & \text{CI length} \\
\hline
128 &0.005447 & 0.793625 & 0.788179 \\
256 &0.047258 & 6.885683 & 6.838425\\
512 & 0.015097 &2.199619 & 2.184523\\
1024 & 0.008431 & 1.228380 & 1.219950\\
\end{array} $$

Indeed, we can see that the confidence interval shrinks as $$n$$ grows. However, there is an oddity at $$n=128$$, where it actually has the smallest CI. I tried several other random instantiations of the data and this effect disappeared. Moreover, every once in a while one of the simulations would have an abnormally small or large CI, though the general decreasing trend remained consistent. It felt wrong to rerun the simulation until I got the "correct" outcome, so I made note of this and present my original data.

# 4.
Consider the smoothed periodogram
$$ \hat{f}(\omega) = \frac{1}{2[\sqrt{n}]+1} \sum\limits_{\vert j\vert < \sqrt{n}} I(\omega_{(j)} + \frac{j}{n})$$
Where $$I$$ is the periodogram and $$\omega_{(j)}$$ is the Fourier frequency closest to $$\omega$$. Plot the smoothed periodogram for each of the sample paths in Q3.

Here is my code to generate the smoothed periodograms:

``` python

# OOOOH more partial functions
# Bet you can't do this in R
def smooth_pgram(pgram):
  n = len(pgram)
  def fhat(w):
    wj =  n/2 + np.rint(w * n).astype(int)
    low = int(max(0,np.ceil(wj - sqrt(n))))
    high = int(min(n,np.floor(wj + sqrt(n))))
    summand = sum(pgram[low:high])
    term1 = 1/((2*sqrt(n))+1)
    return term1*summand
  return fhat

# Look at how lazy I am! It's practically Haskell
smoothed = [smooth_pgram(pgram) for pgram in pgrams]

plt.figure()
for i,(spgram,n) in enumerate(zip(smoothed,sizes)):
  # Set up figure
  plt.subplot(2,2, i+1)

  # Calculate the Smoothed Periodogram
  domain = np.linspace(-1/2,1/2,n)
  data = [spgram(d) for d in domain]

  # Plot and make it pretty
  plt.plot(domain, data)
  plt.title("$$n=%d$$"%n)
  plt.xlabel("$$\omega$$")
  plt.ylabel("$$\hat{f}(\omega)$$")
plt.savefig("figs/spgram.png")

```

![AR(1) Smoothed Periodograms](/images/timeseries_analysis/spgram.png)
