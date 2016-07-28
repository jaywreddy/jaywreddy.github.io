---
title: Fourier Analysis Assignment 5
layout: post
category: "Fourier Analysis"
---

# 1
Let

$$(Fx)_ k = \frac{1}{\sqrt{N}} \sum\limits_{j=0}^{N-1} e^{-2\pi i k j /N}x_j $$

for $$x\in \mathbb{C}^N = [x_0, \cdots, x_{N-1}]^\top$$

## a
Show that

$$(F^2 x)_ k = x_{N-k} $$

First, we have by definition of the DFT:

$$\hat{x}_k = (Fx)_ k $$

And the definition of the inverse Fourier Transform:

$$x_n = \frac{1}{\sqrt{N}}\sum\limits_{j=0}^{N-1} e^{2\pi i n j/N} \hat{x}_j $$

With a change of indices:

$$x_{N-n} =\frac{1}{\sqrt{N}}\sum\limits_{j=0}^{N-1} e^{2\pi i (N-n) j/N} \hat{x}_j   $$

Separating the exponential into two terms:

$$x_{N-n} =\frac{1}{\sqrt{N}}\sum\limits_{j=0}^{N-1} e^{2\pi i N j/N} e^{- 2\pi i n j/N}\hat{x}_j $$

Observing that $$e^{2 \pi i j N/N} = e^{2\pi i j} = 1$$ for integer $$j$$ (which is true because $$j$$ is our index variable)

$$x_{N-n} =\frac{1}{\sqrt{N}}\sum\limits_{j=0}^{N-1} e^{- 2\pi i n j/N}\hat{x}_j $$

With a simple change of variables: $$k=n$$, this takes the familiar form of the Fourier Transform, as defined above:

$$x_{N-k} = \frac{1}{\sqrt{N}}\sum\limits_{j=0}^{N-1} e^{- 2\pi i k j/N}\hat{x}_j  = (F \hat{x})_ k $$

Next, since we know that $$\hat{x} = Fx$$ by definition,

$$x_{N-k} = (F(Fx))_ k = (F^2 x)_ k $$

And I have shown the desired result.

## b
Show that $$F^4 = I$$

Since we have that

$$(F^2 x)_ k = x_{N-k}$$

Taking $$(F^2x)= u$$ and taking the Fourier Transform twice, we have:

$$ (F^2 u)_ l = u_{N-l} $$

And of course, to evaluate $$u_l$$:

$$u_{N-l} = (F^2 x)_ {N-l} = x_{N-(N-l)} = x_l $$

So substituting into the previous expression for $$u$$, we have:

$$(F^2 (F^2 x))_ l = x_l $$

Simplifying and with the change of variables $$l=k$$, we have:

$$(F^4 x)_ k = x_k $$

So under the transformation $$F^4,$$ all of the components of $$x$$ remain unchanged. Therefore:

$$(F^4 x)_ k = x_k = (Ix)_ k \implies F^4 =I $$

## c
Show that any eigenvalue $$\lambda$$ of $$F$$ must satisfy $$\lambda^4 = 1$$

Consider the diagonalization of $$F$$:

$$F = P^{-1} D P$$

Where $$D$$ is a diagonal matrix of the eigenvalues of $$F$$, and $$P$$ is a matrix of the corresponding eigenvalues.

Then,

$$F^4 = P^{-1} D^4 P = I $$

Where $$D^4$$ is a diagonal matrix of each of the eigenvalues of $$F$$ raised to the 4th power. Then

$$P^{-1} D^4 P = I  $$

Left-Multiplying both sides by $$P$$:

$$P P^{-1}D^4 P = D^4 P = P $$

Right-Multiplying both sides by $$P^{-1}$$:

$$D^4 P P^{-1} = D^4 = P P^{-1} = I$$

Thus, $$D^4 = I$$. $$D^4$$ is a diagonal matrix composed of the fourth power of each of the eigenvalues of $$F$$, and $$I$$ is a diagonal matrix of $$1$$'s. Since they are equal, the diagonal entries must be equal, and so for any eigenvalue of $$F$$.

$$\lambda^4 = 1 $$

Thus I have shown the desired result.

## d
Show that the operator

$$(Lu)_ j = u_{j+1} - 2u_j + u_{j-1} -(4 \sin^2(\frac{\pi j}{N}))u_j $$

satisfies

$$LF = FL $$

To show that these are equivalent, I am going to show that $$LFu =FLu$$.

First,


$$(FLu)_ j = F[u_{j+1} - 2u_j + u_{j-1} -(4 \sin^2(\frac{\pi j}{N}))u_j] $$

By linearity of the Fourier Transform:

$$(FLu)_ j = F[u_{j+1}] - 2F[u_j] + F[u_{j-1}] -F[4 \sin^2(\frac{\pi j}{N})u_j]$$

Expressing $$\sin^2$$ as a complex exponential,  

$$F[4\sin^2(\frac{\pi j }{N})u_j] =  F[-e^{-2i \pi j/N}u_j - e^{2 \pi i j/N}u_j + 2u_j]$$

Notice That the complex exponentials are time shifts of the Fourier Transform:

$$F[e^{-2 \pi i j m/N} u_j] = \hat{u}_{j-m} $$

So we have

$$F[4\sin^2(\frac{\pi j }{N})u_j] =  -\hat{u}_{j-1} - \hat{u}_{j+1} + 2\hat{u}_j $$

Next, consider the Fourier Transforms of each of the shifted components:

$$ Fu_{j-m} = e^{-2 \pi i j m/N} \hat{u}_j$$

So plugging these facts back into our expression for $$FLu$$, we have

$$(FLu)_ j = e^{-2 \pi i j/N} \hat{u}_j - 2 \hat{u}_j + e^{2\pi i j/N } + \hat{u}_{j-1} + \hat{u}_{j+1} + 2\hat{u}_j  $$

Notice by our trig identity:

$$-4\sin^2(\frac{\pi j }{N})u_j =  e^{-2i \pi j/N}u_j + e^{2 \pi i j/N}u_j - 2u_j$$

And so we can see that the leading terms are actually $$-4\sin^2(\frac{\pi j }{N})\hat{u}_j$$. Substituting, we have:

$$(FLu)_ j =  \hat{u}_{j-1} + \hat{u}_{j+1} + 2\hat{u}_j-4\sin^2(\frac{\pi j }{N})\hat{u}_j  $$

Next, I will compute $$LFu$$, but it is interesting to note that the first and second parts of the equation are Fourier Transform Pairs, so the operator itself is the same under the Fourier Transform. Thus, without showing the next part, it makes sense that it would commute with the Fourier Transform. For completeness, I show:

$$(LFu)_ j = L \hat{u}_j$$

$$(LFu)_ j = \hat{u}_{j+1} - 2\hat{u}_j + \hat{u}_{j-1} -(4 \sin^2(\frac{\pi j}{N}))\hat{u}_j $$

Thus, we have

$$(LFu)_ j  = (FLu)_ j \implies LFu = FLu \implies LF = FL$$

Showing the desired result.

## e
Show that

$$x^\top L x < 0 $$

whenever $$x\neq 0$$

We have that

$$x^\top L x = \sum\limits_{j = 0}^{N-1} x_j (Lx)_ j $$

$$x^\top L x = \sum\limits_{j = 0}^{N-1} x_j (x_{j+1} - 2x_j + x_{j-1} -(4 \sin^2(\frac{\pi j}{N}))x_j) $$

$$x^\top L x = \sum\limits_{j = 0}^{N-1} x_j (x_{j+1} + x_{j-1} -(2 + 4 \sin^2(\frac{\pi j}{N}))x_j) $$

$$x^\top L x = \sum\limits_{j = 0}^{N-1} x_j x_{j+1} + x_j x_{j-1} -2 \sum\limits_{j=0}^{N-1} x_j^2 - 4\sum\limits_{j=0}^{N-1}  \sin^2(\frac{\pi j}{N})x_j^2 $$

Considering the first term:

$$\sum_{j=0}^{N-1} x_j x_{j+1} + x_j x_{j-1} = \sum\limits_{j=0}^{N-1} x_j x_{j+1} + \sum\limits_{j=0}^{N-1} x_j x_{j-1}$$

By reordering the indices on the second summation, we can show that these expressions are in fact the same, with shifted indices.

$$\sum_{j=0}^{N-1} x_j x_{j+1} + x_j x_{j-1} = \sum\limits_{j=0}^{N-1} x_j x_{j+1} + \sum\limits_{j=1}^{N} x_j x_{j+1}$$

However, since all indices are interpreted mod $$N$$, $$j=N \text{ mod } N  = 0$$. So we can interpret the last index of the second summation as the $$0$$'th index, effectively wrapping the summation around. Thus, the second summation is actually the same as the first:

$$\sum_{j=0}^{N-1} x_j x_{j+1} + x_j x_{j-1} = \sum\limits_{j=0}^{N-1} x_j x_{j+1} + \sum\limits_{j=0}^{N-1} x_j x_{j+1} $$

$$\sum_{j=0}^{N-1} x_j x_{j+1} + x_j x_{j-1} = 2\sum\limits_{j=0}^{N-1} x_j x_{j-1} $$


To clarify what I just did, I noticed that the two terms are actually identical, but shifted by one index:
$$ x_k x_{k-1} \vert _ {j+1} = x_{j+1} x_{j} = x_k x_{k+1} \vert _ {j} $$

Therefore, if we think about the total summation, we could associatively rearrange the sum by grouping like terms to express the summation as above. (The 0th term matches with the 1st term, 1st with 2nd, etc... N-1th with 0th)

Therefore, I can express $$x^\top L x$$ as so:

$$x^\top L x = 2\sum\limits_{j = 0}^{N-1} x_j x_{j-1} -2 \sum\limits_{j=0}^{N-1} x_j^2 - 4\sum\limits_{j=0}^{N-1}  \sin^2(\frac{\pi j}{N})x_j^2$$

I can bound the result by showing that the third term is guaranteed to be negative.

First, $$\sin^2(x)$$ is guaranteed to be positive (or zero) since $$\sin^2(x): \mathbb{R}\to [0,1]$$.

Since $$x_j^2$$ is guaranteed to be positive as well, each of the terms of the summation is guaranteed to be positive (or 0), so the entire summation must be positive or 0. And since we are subtracting this quantity:

$$x^\top L x \leq 2\sum\limits_{j = 0}^{N-1} x_j x_{j-1} -2 \sum\limits_{j=0}^{N-1} x_j^2 $$

Next, consider that we can express the summations as the $$l^2$$ inner product of vectors:

$$x^\top L x \leq 2 [\langle x,Bx \rangle_{l^2} - \langle x,x \rangle_{l^2} ] $$

Where $$B$$ is the backshift operator on the indices of $$x$$.

Consider the Cauchy Schwarz Inequality, which states:

$$\langle x, Bx \rangle \leq \vert \vert x\vert \vert \ \vert \vert Bx\vert \vert   $$

Additionally,

$$\vert \vert Bx\vert \vert _ {l^2} = \sqrt{\sum\limits_{j=0}^{N-1} x_{j-1}}$$

With a change of variables: $$k = j+1$$, and letting the indices wrap around since they are interpreted as mod $$N$$, we have:

$$\sqrt{\sum\limits_{j=0}^{N-1} x_{j-1}} =\sqrt{\sum\limits_{k=1}^{N} x_{k}} = \sqrt{\sum\limits_{k=0}^{N-1} x_{k}} $$

The last expression is $$\vert \vert x\vert \vert _ {l^2}$$, so we can say:

$$\vert \vert Bx\vert \vert _ {l^2} = \vert \vert x\vert \vert _ {l^2} $$

And substituting this into our Cauchy-Schwarz equation:

$$ \langle x, Bx \rangle \leq \vert \vert x\vert \vert \ \vert \vert x\vert \vert  = \langle x, x\rangle_{l^2} $$

So by Cauchy-Schwarz, we have:

$$ \langle x, Bx \rangle_{l^2} \leq \langle x, x\rangle_{l^2} \implies \langle x, Bx \rangle_{l^2} - \langle x, x\rangle_{l^2} \leq 0 $$

If we plug this back into our expression for $$x^\top L x$$, we have:

$$x^\top L x \leq 0 $$

To establish a strict inequality, let's consider the only cases where the value can be $$0$$:

In order for

$$x^\top L x = 2[\langle x, Bx\rangle - \langle x, x \rangle] - 4\sum\limits_{j=0}^{N-1}  \sin^2(\frac{\pi j}{N})x_j^2$$

To be 0, both terms must be exactly 0, since they are individually bound to be $$\leq 0$$.

First, consider the case where

$$- 4\sum\limits_{j=0}^{N-1}  \sin^2(\frac{\pi j}{N})x_j^2 = 0$$

Since $$\sin^2(\frac{\pi j }{N}) \neq 0 \text{ if } j\neq 0$$, the only time the sum can be $$0$$ is if all $$x_j = 0 \text{ if } j\neq 0$$. Thus, for the second term to be exactly 0, the vector $$x$$ must have the form:

$$x = [\alpha, 0, \cdots, 0]^\top $$

With only the $$0$$'th term being nonzero. To recap, I have argued:

$$ - 4\sum\limits_{j=0}^{N-1}  \sin^2(\frac{\pi j}{N})x_j^2 = 0 \implies x = [\alpha, 0, \cdots, 0]^\top $$

And in order for $$x^\top L x = 0$$, most terms must be equal to $$0$$. So let's consider what happens to the first term when $$x$$ takes the form above.

$$\langle x, Bx \rangle - \langle x,x\rangle = -\vert \alpha\vert  $$

The inner product of $$x,Bx$$ will be zero, since $$x$$ is all zeros except for the $$0$$th term. Thus, if we multiply the vector by a shifted version of itself, the nonzero terms will be multiplied by $$0$$'s, and so the norm will be $$0$$.

The value of $$\langle x,x \rangle$$ follows from the form of $$x$$, $$x = [\alpha, 0, \cdots, 0]^\top$$

Thus, the first term is only $$0$$ if $$\alpha = 0 \implies x = 0$$.

So the only way the two conditions for exact equality are satisfied simultaneously is if $$x=0$$. Thus, we can restate

$$x^\top L x \leq 0 $$

as

$$x^\top L x < 0 $$

as long as $$x \neq 0$$.

Thus I have shown the desired result.


## f
Study the eigenvalues and eigenvectors of L for

$$N= 4^k +1, k = 1,2, \cdots $$

Are the eigenvalues simple (i.e. each eigenvalue has only one eigenvector up to scaling)? Do the eigenvectors have to be orthogonal? Plot eigenvectors and discover special properties such as sparsity, regularity, and so forth.

``` python
import numpy as np
import matplotlib.pyplot as plt

def L_mat(dim):
  L = np.zeros([dim,dim])
  for j in range(dim):

    #  (L u)_j = u_j+1 - 2 u_j + u_j-1 -(4sin^2(pi j/N))u_j
    L[j,j] = -2 -4*(np.sin(np.pi*j/dim)**2)

  for j in range(1,dim):
    L[j,j-1] = 1
    L[j-1,j] = 1

  L[0,dim-1] = 1
  L[dim-1,0] = 1
  return L

def analyze(k):
  d = 4**k -1
  L = L_mat(d)

  vals, vecs = np.linalg.eig(L)

  # eigenvectors orthogonal?
  mat = np.zeros([d,d])
  for i in range(d):
    for j in range(d):
      v1 = vecs[i]
      v2 = vecs[j]
      prod = np.dot(v1,v2)
      mat[i,j] = prod

  # Plot Eigenvalues orthogonality
  plt.matshow(mat)
  plt.colorbar().set_label("Dot Product Value")
  plt.title("Matrix of Eigenvector Dot Products")
  plt.savefig('figs/hw10ortho.png')
  plt.clf()

  #Plot Eigenvalues in Matrix
  plt.matshow(vecs.T)
  plt.colorbar().set_label("Component Value")
  plt.xlabel("Eigenvector #")
  plt.ylabel("Component #")
  plt.savefig('figs/hw10eigenvecs.png')
  plt.clf()


  #Singular Eigenvalues
  svals = sorted(vals)
  diffs = np.diff(svals)
  plt.subplot(2,1,1)
  plt.plot(svals)
  plt.title("Eigenvalues, sorted")
  plt.ylabel("Value")

  plt.subplot(2,1,2)
  plt.plot(diffs)
  plt.xlabel("Eigenvalue")
  plt.ylabel("Difference")
  plt.savefig('figs/eigenvals.png')
  plt.clf()

analyze(4)


```

I produced the following plots and analysis for $$k = [1,5]$$, however I found that $$k=4$$ best illustrated the desired properties of the eigenvectors.

![Eigenvalues](/images/fourier_analysis/eigenvals.png)
\ 

Here, we see the a plot of the sorted eigenvalues corresponding to each eigenvector. If two eigenvectors corresponded to the same eigenvalue, we would see two identical eigenvalues appear. However, as the plot shows, the eigenvalues all appear to be unique. This is confirmed by looking at the first-order differencing of the eigenvalues, as a a repeated eigenvalue would result in a difference of $$0$$, which does not occur. Thus, each of the eigenvalues corresponds to only a single eigenvector and all of the eigenvalues are simple.

![Orthogonality](/images/fourier_analysis/hw10ortho.png)
\ 

For general matrices, there is no requirement that eigenvalues be orthogonal. However, in this case we can see that the dot product of each of the eigenvectors with any other is $$0$$ (actually $$O(10^{-12})$$, which I attribute to numerical error). Similarly, the dot product of any eigenvector with itself is $$1$$, (actually $$1 \pm 10^{-12}$$, which I attribute to numerical error).

Thus, the eigenvectors of $$L$$ appear to be orthonormal.

![Eigenvectors](/images/fourier_analysis/hw10eigenvecs.png)
\ 

Analyzing the components of the eigenvectors, we see clear patterns. Notice that eigenvectors close together seem to fluctuate in "bands" of components. We can see hints of the orthogonality between different sets of eigenvectors. For example, in components $$[0,100]$$, eigenvectors $$[1,55]\cup[200,256]$$ and $$[100,150]$$ appear orthogonal as each set is zero where the other is fluctuating.

# 2
Let

$$h_n (x) = \frac{(-1)^n}{n!} e^{x^2/2} D^n e^{-x^2} $$
where $$D = \frac{d}{dx}$$

# a
Show that
$$(D^2 - x^2)h_n(x) = -(2n+1)h_n(x) $$

By theorems 1 and 2 from handout 2, we have:

$$h_n'(x) = xh_n(x) - (n+1)h_{n+1}(x) \implies (D-x)h_n(x) = -(n+1)h_{n+1}(x)$$
$$h_n'(x) = -x h_n(x) + 2h_{n-1}(x) \implies (D+x)h_n(x) = 2 h_{n-1}(x)$$

Next, observing the factorization of the operator:

$$D^2 - x^2 + I = (D-x)(D+x) $$

We can proceed to say:

$$(D^2 - x^2)h_n (x) = [(D^2 - x^2 + I) - I]h_n(x) = [(D-x)(D+x) - I] h_n(x)$$

By linearity of the operators:

$$(D^2 - x^2)h_n (x) = (D-x)(D+x)h_n(x) - h_n(x) $$

And we can evaluate the first term as follows:

$$(D-x)(D+x)h_n(x) $$

Applying Thm 2:

$$2(D-x)h_{n-1} $$

Applying Thm 1:

$$2[-(n-1+1)h_n] $$

$$-(2n)h_n $$

Which gives us

$$(D^2 - x^2)h_n (x) = -(2n)h_n - h_n $$

$$(D^2 - x^2)h_n (x) = -(2n+1)h_n $$

The desired result.

# b
Show that $$h_n$$ is an eigenfunction of the Fourier Transform

$$\hat{f}(k) = \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^\infty f(x)e^{-ikx}dx $$

I will prove the following result by induction:

$$\hat{h}_n(k) = (-i)^n h_n(k) $$


Since the hermite polynomials are related by the recurrence relations:

$$ (n+1)h_{n+1} = 2xh_n - 2h_{n-1}$$

Higher hermite functions can be successively derived from the lower ones. I can exploit this property for an inductive proof, since if I show that a result holds for $$h_0,h_1$$, and fulfills the same recurrence relation, then the result will hold for all $$h_n$$.

First, $$h_0 = e^{-x^2/2}$$, so to calculate $$\mathcal{F}[e^{-x^2/2}]$$, I first observe the common Fourier Transform Pair:

$$\mathcal{F}[e^{-\alpha x^2}] = \frac{1}{\sqrt{2\alpha}}e^{-\frac{k^2}{4\alpha}} $$

Which can be used to compute $$\mathcal{F}[h_0]$$ with $$\alpha = 1/2$$:

$$\mathcal{F}[e^{- x^2/2}] = e^{-k^2/2} $$

Thus, we have $$\mathcal{F}[h_0(x)]= h_0(k)$$.

Next, for $$h_1$$, we have by the recurrence relations that

$$ h_1(x)= -2h_0 '(x)$$

Taking the Fourier Transform,

$$ \mathcal{F}[h_1(x)]=\mathcal{F}[ -2h_0 '(x)]$$

By the derivative property of the Fourier Transform

$$ \hat{h}_1(k)= -2ik\hat{h}_0 (k)$$

Which, by the $$h_0$$ relation  we just proved:

$$ \hat{h}_1(k)= -2ik h_0 (x)$$

And by the Recurrence Relations,

$$ \hat{h}_1(k)= -ih_1 (x)$$

$$ i\hat{h}_1(k)= h_1 (x)$$


Now that we have shown that $$h_0 = \hat{h}_0$$ and $$i\hat{h}_1 = h_1$$, this suggests that there is a pattern relating $$\hat{h}$$ to $$h$$. To show the pattern holds for all $$n$$, I will derive the recurrence relations for $$\hat{h}$$.

First, by the product property of the fourier transform,

$$\mathcal{F}[x f(x)] = i \hat{f}'(k)$$

Which implies:

$$\mathcal{F}[-ix f(x)] = \hat{f}'(k)$$

Generally, and so:

$$\hat{h}'_ n(k)= \mathcal{F}[-ix h_n(x)] $$

Next, by the derivative property of the Fourier Transform,

$$\mathcal{F}[f'(x)] = ik \hat{f}(k) $$

Which implies:

$$k \hat{f}(k) = \mathcal{F}[-if'(x)] $$

Generally, and so:

$$k \hat{h}_n(k) = \mathcal{F}[-ih'_ n(x)] $$

By adding these functions, we can form a recurrence relation for $$\hat{h}$$:

$$\hat{h}'_ n(k) + k \hat{h}_n(k) =  \mathcal{F}[-ix h_n(x)] +  \mathcal{F}[-ih'_ n(x)] $$

And by linearity of the Fourier Transform:

$$\hat{h}'_ n(k) + k \hat{h}_n(k) =  -i\mathcal{F}[x h_n(x) +h'_ n(x)] $$

And by the recurrence relations,

$$h_n' (x) + xh_n(x) = 2 h_{n-1}(x)$$

So we have:

$$\hat{h}'_ n(k) + k \hat{h}_n(k) =  -i2\mathcal{F}[h_{n-1}(x)] $$

Which is simply

$$\hat{h}'_ n(k) + k \hat{h}_n(k) =  -2i\hat{h}_{n-1}(x) $$

We can show another recurrence relation by subtracting the values:

$$\hat{h}'_ n(k) - k \hat{h}_n(k) =  \mathcal{F}[-ix h_n(x)] -  \mathcal{F}[-ih'_ n(x)] $$

$$\hat{h}'_ n(k) - k \hat{h}_n(k) =  i\mathcal{F}[h'_ n(x)-x h_n(x) ] $$

But we know from the recurrence relations that:

$$\hat{h}_n ' (k) -k \hat{h}_n(k) = -i(n+1) \mathcal{F}[h_{n+1}(x)] $$

$$ \hat{h}_n ' (k) -k \hat{h}_n(k) =-i(n+1) \hat{h}_{n+1}(k) $$

Combining the two relations:

$$ \hat{h}_n ' (k) -k \hat{h}_n(k) = -i(n+1) \hat{h}_{n+1}(k) $$

$$ \hat{h}'_ n(k) + k \hat{h}_n(k) =  -2i\hat{h}_{n-1}(x) $$

Subtracting the first from the secibd to get ride of the derivatives,

$$2k\hat{h}_n(k) = i(n+1)\hat{h}_{n+1}(k) -2i \hat{h}_{n-1}(k)$$

$$i(n+1)\hat{h}_{n+1} (k)= 2k\hat{h}_n(k) +2i \hat{h}_{n-1}(k)$$

Multiplying both sides by $$(i)^n$$:

$$i(i)^n(n+1)\hat{h}_{n+1} (k)= (i)^n2k\hat{h}_n(k) +i(i)^n2 \hat{h}_{n-1}(k)$$

$$(n+1)(i)^{n+1}\hat{h}_{n+1} (k)= 2k(i)^n\hat{h}_n(k) -2 (i)^{n-1}\hat{h}_{n-1}(k)$$

Thus, $$(i)^n\hat{h}_n$$ satisfies the same recurrence relations as $$h_n$$

So by induction, if $$\hat{h}_0 = h_0$$ and $$i\hat{h}_1 = h_1$$, then

 $$(i)^n\hat{h}_n(k) = h_n(k) \implies\hat{h}_n(k) = (-i)^n h_n(k) $$

 So we can say:

 $$\mathcal{F}[h_n(x)] = (-i)^n h_n(k)$$

I have shown the fourier transform of an hermite function $$h_n$$ is a linear scaling by $$(-i)^n$$, and thus $$h_n$$ are eigenfunctions of the Fourier Transform, with eigenvalues $$(-i)^n$$.


# c
Show that eigenvalues $$\lambda$$ of the Fourier Transform satisfy $$\lambda^4 =1$$

Since I have shown in part (b) that the eigenvalues $$\lambda_n$$ corresponding to eigenfunctions $$h_n$$ are given by

$$\lambda_n = (-i)^n $$

We have

$$\lambda_n^4 = (-i)^{4n} = (-1)^{4n}(i)^{4n}= 1^n1^n = 1 $$

Thus
$$\lambda_n^4 = 1, \forall n $$

And so all eigenvalues $$\lambda \in \{\lambda_n\}$$ must satisfy $$\lambda^4=1$$

# d
Use the Poisson sum formula

$$ \sqrt{2 \pi} \sum\limits_{- \infty}^\infty f(2\pi n) = \sum\limits_{-\infty}^\infty \hat{f}(n)$$

to show that the vector $$f_n \in \mathbb{C}^N$$ with

$$f_n(p) = \sum\limits_{q = -\infty}^\infty h_n(\sqrt{\frac{2\pi}{N}} (p+qN)) $$

for $$0 \leq p \leq N-1$$ is an eigenvector of the discrete Fourier Transform $$F$$ from problem 1.

# e
Use the Euler-Maclaurin sum formula to show that $$\langle f_n, f_m \rangle$$ is small for $$0 \leq n, m\leq N-1$$.

We have

$$\langle f_n, f_m \rangle =\sum\limits_{p = 0}^{N-1} \frac{1}{2\pi} \sum\limits_{q \in \mathbb{Z}} \sum\limits_{r \in \mathbb{Z}} h_n\big(\sqrt{\frac{2\pi}{N}}(p+qN)\big)h_m\big(\sqrt{\frac{2\pi}{N}}(p+rN)\big)$$

The inner product will remain unchanged if we identically shift both functions, so I shift them both by $$-\sqrt{2\pi N} q$$:

$$\langle f_n, f_m \rangle =\sum\limits_{p = 0}^{N-1} \frac{1}{2\pi} \sum\limits_{q \in \mathbb{Z}} \sum\limits_{r \in \mathbb{Z}} h_n\big(\sqrt{\frac{2\pi}{N}}(p)\big)h_m\big(\sqrt{\frac{2\pi}{N}}(p+(r-q)N)\big)$$

According to the Eurler-Maclaurin Sum formula:

$$\sum\limits_{k=1}^{N-1}f_k \approx \int_0^N f(k)dk$$

so I replace the sum over $$p$$ with an integral:



$$\langle f_n , f_m \rangle \approx \frac{1}{2\pi} \sqrt{\frac{N}{2\pi}}\int_{\sqrt{\frac{2\pi}{N} }\cdot N}^{\sqrt{\frac{2\pi}{N} }\cdot(N+1)} \sum\limits_{q \in \mathbb{Z}}\sum\limits_{r \in \mathbb{Z}} h_n\big(x\big)h_m\big(x +(r-q)N\sqrt{\frac{2\pi}{N}}\big) dx $$

observing that $$\sum_{q}\sum_r (r-q)$$ takes on values: $$\sum_k k$$, I can rewrite the expression:

$$\langle f_n , f_m \rangle \approx \frac{1}{2\pi} \sqrt{\frac{N}{2\pi}}\int_{\sqrt{\frac{2\pi}{N} }\cdot N}^{\sqrt{\frac{2\pi}{N} }\cdot(N+1)} \sum\limits_{k \in \mathbb{Z}} h_n\big(x\big)h_m\big(x +(k)N\sqrt{\frac{2\pi}{N}}\big) dx$$

Next, I use part (f) to claim that the summation over $$k$$ will be dominated by a single term ($$k=0$$), so I can say that this is approximately equal to

$$\langle f_n , f_m \rangle \approx \frac{1}{2\pi} \sqrt{\frac{N}{2\pi}}\int_{\sqrt{\frac{2\pi}{N} }\cdot N}^{\sqrt{\frac{2\pi}{N} }\cdot(N+1)}  h_n\big(x\big)h_m\big(x \big) dx $$

And since $$h_n$$ is small outside of $$\sqrt{2n+1},$$ the functions are very small beyond the limits of the integral. Thus, I can expand the limits of the integral without changing its value:

$$\langle f_n , f_m \rangle \approx \frac{1}{2\pi} \sqrt{\frac{N}{2\pi}}\int_{-\infty}^{\infty}  h_n\big(x\big)h_m\big(x \big) dx $$

But this the $$L^2$$ norm. We can thus use our knowledge of the orthogonality of the hermite functions to complete the proof:

$$\langle f_n , f_m \rangle \approx \frac{1}{2\pi} \sqrt{\frac{N}{2\pi}}\int_{-\infty}^{\infty}  h_n\big(x\big)h_m\big(x \big) dx = \delta_{n,m} $$

So  

$$\langle f_n , f_m \rangle$$

Is very small for $$n \neq m$$

# f
Show that the sum over $$q$$ in (d) is dominated by a single term to high accuracy for $$0 \leq n \leq N-1$$.

Recall that $$h_n(x)$$ is small for $$\vert x\vert <\sqrt{2n+1}$$

Then

$$h_n(\sqrt{\frac{2\pi}{N}}(p+qN)) $$

is small for

$$\vert \sqrt{\frac{2\pi}{N}}p + q \sqrt{2\pi N}\vert <\sqrt{2n+1} $$

Since $$0 \leq n \leq N-1$$, we can bound the expression by:

$$n = N-1 $$

$$\vert \sqrt{\frac{2\pi}{N}}p + q \sqrt{2 \pi N}\vert  < \sqrt{2N-1} $$

Thus the terms are only significant for:

$$\vert \frac{p}{N}+q\vert < \sqrt{\frac{1}{\pi} - \frac{1}{2N}} $$

$$p \in [0,N-1], q \in \mathbb{Z}$$

Since $$\sqrt{1/\pi} \approx .56$$. We can immediately see that this condition will be violated for $$q\geq 1$$, or $$q<-1$$.

For a given value of $$p$$, if $$\frac{p}{N} \geq \sqrt{\frac{1}{\pi} - \frac{1}{2N}}$$, then the condition will only be satisfied by $$q=-1$$, so the summation will only have $$1$$ significant term.

If $$\frac{p}{N}\leq 1- \sqrt{\frac{1}{\pi} - \frac{1}{2N}}$$, then the condition will only be satisifed for $$q=0$$, and thus the summation will only have 1 significant term.

Finally, however, if

$$ \sqrt{\frac{1}{\pi} - \frac{1}{2N}}\geq \frac{p}{N} \geq 1- \sqrt{\frac{1}{\pi} - \frac{1}{2N}} $$

Then the condition may be satisfied by $$q=0,-1$$. And so the summation will actually be dominated by two terms.

# g
For $$N = 4^k +1, k = 1,2, \cdots$$ compute the Gram matrix

$$G_{ij} = \langle f_i, f_j \rangle $$

``` python
import numpy as np
from scipy import special
from matplotlib import pyplot as plt
from functools import reduce
import math

def factorial(n):
  if n == 0:
    return 1
  return  reduce(lambda x,y: x*y, range(1,n+1))

def make_h(n):
  H_n = special.hermite(n)
  def h(x):
    weighting = np.exp(-1*(x**2) /2)
    return  H_n(x)*weighting / factorial(n)
  return h

def make_f(n,dim):
    h_n = make_h(n)
    def f(p):
      # sum is dominated by a small terms:
      total = 0
      for q in range(-5,5):
        x = math.sqrt(2*np.pi / dim)*(p+q*dim)
        total = total + h_n(x)
      return total
    return f

def Gram(dim):
  G = np.zeros([dim,dim])
  for i in range(dim):
    for j in range(dim):
      fi = make_f(i,dim)
      fj = make_f(j,dim)
      v_fi = [fi(x) for x in range(dim)]
      v_fj = [fj(x) for x in range(dim)]
      G[i,j] = np.inner(v_fi, v_fj)
  return G

  G = Gram(25)
  plt.figure()
  plt.matshow(G)
  plt.colorbar()
  plt.savefig("figs/gram.png")

  plt.clf()
```

![Gram Matrix](/images/fourier_analysis/gram.png)
\ 

This was my best shot at the Gram matrix. After significant debugging I wasn't able to determine why it didn't work, but I know that my matrix is incorrect since I expect upper left quadrant of the matrix to be the identity.

# h
Study the projections $$P_\alpha$$ onto the four eigenspaces of $$F$$ given by $$(0 \leq \alpha \leq 3)$$

$$P_\alpha = \sum\limits_{n \equiv \alpha \ (\text{mod }N)} f_n G_\alpha^{-1} f_n^\top $$

where $$G_\alpha$$ contains rows and columns of $$G$$ with $$i \equiv \alpha$$ and $$j \equiv \alpha \ (\text{mod }N)$$

Though I couldn't calculate the gram matrix in order to find the actual projections, I would expect to have four projections:

$$P_0, P_1, P_2, P_3 $$

such that

$$P_0 + P_2 = \frac{F^2+I}{2}$$

$$P_1 +P_3 = \frac{F^2 - I}{2}$$
