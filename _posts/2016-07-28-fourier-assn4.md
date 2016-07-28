---
title: Fourier Analysis Assignment 4
layout: post
category: Fourier Analysis
---

# P 7
Filtering. Let
$$f(t) = e^{-t^2/10} (\sin(2t) + 2\cos(4t) + .4\sin(t)\sin(50t)) $$
Discretize $$f$$ by setting $$y_k = f(2k\pi/256), \quad k=1 \cdots 256$$. Use the fast fourier transform to computer $$\hat{y}_k$$ for $$0\leq k \leq 256$$. Recall from Theorem 3.4 that $$\hat{y}_{n-k}= \bar{\hat{y}}
_ k$$. Thus, the low-frequency coefficients are $$\hat{y}_0,\cdots,\hat{y}_m$$ and $$\hat{y}_{256-m} \cdots \hat{y}_{256}$$ for some low value of $$m$$. Filter out the high-frequency terms by setting $$\hat{y}_k=0$$ for $$m\leq k \leq 255-m$$ with $$m=6$$; then apply the inverse fast Fourier transform to this new set of $$\hat{y}_k$$ to compute the $$y_k$$ (now filtered); plot the new values of $$y_k$$ and compare with the original function. Experiment with other values of m.

``` python
import numpy as np
import matplotlib.pyplot as plt

# Create function f
def f(t):
  return np.exp(-t*t/10)*(np.sin(2*t) +
    2*np.cos(4*t) + .4*np.sin(t)*np.sin(50*t))

# Generate Data Points and Evaluate Function
x_k = [np.pi * k / 128 for k in range(256)]
y_k = np.array([f(t) for t in x_k])

# Plotting
def plot(data,name):
  plt.figure()
  plt.plot(data)
  plt.savefig("figs/hw9/%s.png"%(name))


# Filtering
def fft_filter(data,m):
  fft = np.fft.fft(data)
  fft_fil = [0 if m <= k <= (255-m) else y
    for (k,y) in enumerate(fft)]
  return np.fft.ifft(fft_fil)

# Filter for m=6
filtered = fft_filter(y_k,6)
plot(y_k, "yk")
plot(filtered,"fil6_yk")
```
![Original Function](/images/fourier_analysis/hw9/yk.png)

![Filtered Function](/images/fourier_analysis/hw9/fil6_yk.png)

We can see between the two images (Figures 1,2) that the filtered version has "smoothed" the data, in the sense that the high frequency components have been removed.

Now, to experiment with multiple values of m:

``` python
# Create a figure with multiple values of m
plt.figure()
for m in [5,20,100]:
  filtered = fft_filter(y_k,m)
  plt.plot(filtered, label = "m=%d"%m)
plt.legend()
plt.savefig("figs/hw9/multi_yk.png")
```

![Multiple Filterings](/images/fourier_analysis/hw9/multi_yk.png)

We can see as the value of m increases, two features of the data become more pronounced:

First, higher frequency oscillations from the $$\sin(50t)$$ term become visible.

Second, the tapering due to the $$e^{-t^2/10}$$ term becomes better. This is because $$e^x$$ requires an infinite fourier transform, so as m increases, our DCT better approximates the tapering.

# P 8
Compression. Let $$tol=1.0$$. In exercise 7, if $$\vert \hat{y}_k\vert  < tol$$, then set $$\hat{y}_k$$ equal to zero. Apply the inverse fast Fourier transform to this new set of $$\hat{y}_k$$ to compute the $$y_k$$; plot the new values of $$y_k$$ and compare with the original function. Experiment with other values of $$tol$$. Keep track of the percentage of Fourier coefficients which have been filtered out. Compute the relative $$l^2$$ error of the compressed signal as compared with the original signal.

Building on the code I developed for the previous section:

``` python
def compress(data,tol):
  fft = np.fft.fft(data)
  compressed = [0 if abs(y) < tol else y for y in fft]
  return np.fft.ifft(compressed)

# compress for tol = 1.0
compressed = compress(y_k,1.0)
plot(compressed, "tol1_yk")
```

![Original Function](/images/fourier_analysis/hw9/yk.png)

![Compressed Function](/images/fourier_analysis/hw9/tol1_yk.png)

Performing compression, we can see that unlike basic filtering the $$\sin(50t)$$ component is not lost even at a low tolerance. Presumably this is because the $$\sin(50t)$$ component corresponds to a large fourier coefficient at that frequency. The only feature of the data that seems to be affected is the tapering due to $$e^{-t^2/10}$$, which results in many small fourier coefficients to cancel out near the end.

Now to plot for multiple tolerances, including $$l^2$$ error:

``` python
# Create a figure with multiple values of tol
plt.figure()
for i,tol in enumerate([.3,100,5,1]):
  compressed = compress(y_k,tol)

  # Calculate Relative Error
  # Numpy defaults to l2 norm
  norm = np.linalg.norm
  error = norm(y_k - compressed)/norm(y_k)

  plt.subplot("22%d"%i)
  plt.plot(compressed)
  plt.title( "tol=%f, l2 error = %f"%(tol,error))
plt.tight_layout()
plt.savefig("figs/hw9/multi_compressed_yk.png")
```
![Compression with Multiple Tolerances](/images/fourier_analysis/hw9/multi_compressed_yk.png)

Thus, we can see that lower cutoff tolerances admit more fourier coefficients to our DCT, reducing the error in our approximations.

# P 10
Derive Eq. (3.11) by inserting Eq. (3.9) and Eq. (3.10) into Eq. (3.7) at $$t=t_k = 2\pi k/n$$.

Eq. (3.11)
$$ au_{k+1} + \beta u_k + \gamma u_{k-1} = h^2 f_k, \quad 1 \leq k \leq n-1$$
note: $$h= 2\pi/n$$, so it is a constant

Eq. (3.9)
$$u'(t_k) \approx \frac{u_k - u_{k-1}}{h} $$

Eq. (3.10)
$$ u''(t_k) \approx \frac{u_{k+1}+u_{k-1}-2u_k}{h^2}$$

Eq. (3.7)
$$au'' + bu' + cu = f(t) $$

This can be shown directly by evaluating Eq. (3.7) at $$t_k$$ and substituting values from the formulas above:

$$\frac{a}{h^2}(u_{k+1} + u_{k=1}-2u_k) + \frac{b}{h}(u_k - u_{k-1}) + c u_k = f_k $$

Multiplying each side by $$h^2$$ and grouping like terms, we have:

$$a u_{k+1} + [bh - 2a + ch^2]u_k + [a-bh]u_{k-1} = h^2 f(t) $$

Now substituting $$\beta = [bh - 2a + ch^2],$$ and $$\gamma = [a-bh]$$

We arrive at the desired expression:

$$ au_{k+1} + \beta u_k + \gamma u_{k-1} = h^2 f_k, \quad 1 \leq k \leq n-1$$


# P 14
Recall that the complex exponentials $$e^{int}$$ are $$2\pi$$-periodic eigenfunctions of the operator $$D^2[u]=u''$$. A discretized version of this operator acts on the periodic sequences in $$S_n$$. If $$u_k$$ is $$n$$-periodic, then define $$L[u]$$ via

$$L[u]_ k =u_{k+1} + u_{k-1} - 2u_k $$
(a) Show that $$L$$ maps $$S_n$$ into itself.

First, observing the definition of an $$n$$-periodic function, if $$u_k \in S_n$$, then $$u_{n+k} = u_{k}$$ for all $$k$$.

Next, let's say:

$$g_k = L[u_k] = u_{k+1} + u_{k-1} - 2u_k $$

To show $$L$$ maps $$S_n$$ into itself, we must show $$g_k \in S_n$$, which is to say $$L[u_k] \in S_n \quad \forall u_k \in S_n$$.

By the definition of $$n$$-periodic, $$g_k\in S_n$$ if and only if

$$g_{k}  = g_{k+n} \forall k $$

$$ g_{n+k} = u_{n+k+1} + u_{n+k-1}-2u_{n+k}$$

But remember that $$u_k$$ is $$n$$-periodic, which means $$u_k = u_{k-n}$$. We can therefore shift each of the terms by $$-n$$ and get the same result.

$$ g_{n+k} = u_{k+1} + u_{k-1} -2u_k = g_(k)$$

We have thus proved the result without loss of generality, so $$L$$ maps $$S_n$$ onto itself.

(b) For $$n=4$$, show the matrix $$M_4$$ that represents $$L$$ is

$$M_4 = \begin{pmatrix}
-2 & 1 & 0 & 1 \\
1 & -2 & 1 & 0 \\
0 & 1 & -2 & 1 \\
1 & 0 & 1 & -2 \end{pmatrix} $$

To do this, I observe that the unit function at each time shift represents a basis of $$S_4$$, which is to say that any $$4$$-periodic function can be characterized by its values at $$t=0,1,2,3$$. Or equivalently, that any function in $$S_4$$ is a linear combination of the time-shifted unit functions: $$\{e_0,e_1,e_2,e_3\}$$, where $$e_i(x) = \begin{cases} 1 & \text{if } i = x \\ 0 & \text{else} \end{cases}$$

A function on a vector space is a described by a matrix on a basis of the vector space which describes how the basis vectors are mapped into the basis vectors.

We can now easily compute:

$$L[e_0] = e^1 + e_3 -2 e_0 $$
$$L[e_1] = e^2 + e_0 -2 e_1 $$
$$L[e_2] = e^3 + e_1 -2 e_2 $$
$$L[e_3] = e^0 + e_2 -2 e_3 $$

And if we define the vector: $$e = (e_0, e_1, e_2, e_3)^\top$$, then we can write the Matrix form of the operation: $$L[e] = Me$$ as follows by matching the rows to the respective equations above:

$$M_4 = \begin{pmatrix}
-2 & 1 & 0 & 1 \\
1 & -2 & 1 & 0 \\
0 & 1 & -2 & 1 \\
1 & 0 & 1 & -2 \end{pmatrix} $$

Thus, I have shown the desired result.

(c) Observe that $$M_4$$ is self-adjoint and thus diagonalizable. Find its eigenvalues and corresponding eigenvectors. How are these related to the matrix columns for the matrix $$F_4$$ in Eq. (3.2)? Could you have diagonalized this matrix via an FFT? Explain.

$$F_4 = \begin{pmatrix}
1 & 1 & 1& 1\\
1 & i & -1 & -i \\
1 & -1 & 1 & -1 \\
1 & -i & -1 & i \\
\end{pmatrix} $$

I diagonalize the matrix to yield: $$M = PDP^{-1}$$

Where

$$ D = \begin{pmatrix}
-4 & 0 &0 & 0 \\
0 & -2 & 0 & 0 \\
0 & 0 & -2 & 0 \\
0 & 0 &0 & 0\end{pmatrix}$$

and

$$ P = \begin{pmatrix}
-1 & 0 & -1 & 1 \\
1 & -1 & 0 & 1 \\
-1 & 0 & 1 & 1 \\
1 & 1& 0 & 1 \\
\end{pmatrix} $$

The eigenvalue,eigenvector pairs are the diagonal entries and corresponding columns of the two matrices respectively.

It appears that the columns of $$P$$ have some relation to the columns of $$F_4$$, I reorder the eigenvalues in order to get a closer comparison:

$$ F_4 = \begin{pmatrix}
1 & 1 & 1& 1\\
1 & i & -1 & -i \\
1 & -1 & 1 & -1 \\
1 & -i & -1 & i \\
\end{pmatrix}, \qquad P = \begin{pmatrix}
1 & -1 & -1 & 0 \\
1 & 0 & 1 & -1 \\
1 & 1 & -1 & 0 \\
1 & 0& 1 & 1 \\
\end{pmatrix}$$

I can't tell an exact straightforward relation, but the columns of $$P$$ seem to be the real components of $$F$$, with the columns each shifted by some amount.

TODO

(d) Generalize this result for all $$n$$. (Hint: Use the DFT on $$L[u]_ k$$; recall that the FFT is really a fast DFT.)

$$L[u]_ k$$ can be interpreted as a convolution with the function: $$l = [-2,1, \cdots, 1]$$.

Indeed, writing out the convolution formula:

$$(l*u)_ k = \sum\limits_{j=0}^{n-1} u_j l_{k-j} = u_{k+1}+u_{k-1} -2 u_k = L[u]_ k$$

Which yields: $$(l*u)_ k = L[u]_ {k}$$

Since $$\mathcal{F}_n[(l*u)]_ k = \mathcal{F}_n[l]_ k \cdot \mathcal{F}_n[u]_ k$$, this implies that $$L[u]$$ corresponds to a simple scaling in the Fourier Domain, meaning that the matrix corresponding to $$\mathcal{F}_n[l]$$ must be diagonal. This is a little easier to see if we use another notation:

$$\widehat{(l*u)}_k = \hat{l}_k \cdot \hat{u}_k $$

The fourier coefficients of the convolution are all simple scalings of the fourier coefficients of the function $$u$$. Thus, $$\mathcal{M}(\hat{l})$$ must be diagonal, since no cross terms of $$\hat{u}$$ appear.  

 Now we have two valid representations of $$L[u]$$:

 $$L[u] = (l * u) = Mu $$

 We additionally know that the matrix of the Fourier transform of $$l$$ is diagonal. The above equation tells us that the matrix computation $$Mu$$ is equivalent to changing bases to the fourier domain, multiplying be a diagonal matrix, and changing back. Thus, writing the Fourier Transform in matrix notation, (parenthesis are used to separate the Fourier Transform of $$l$$).

 $$Mu = \frac{1}{n}F_n (\bar{F}_n l) \bar{F}_n u$$

 It is immediately apparent then, that

  $$M = \frac{1}{n}F_n (\bar{F}_n l) \bar{F}_n $$
  $$M = \frac{1}{\sqrt{n}}F_n (\bar{F}_n l) \frac{1}{\sqrt{n}}\bar{F}_n $$

With the notation, $$P = \frac{1}{\sqrt{n}}F_n$$

$$M = P(\bar{F}_n l)P^{-1} $$

Where we know that $$\bar{F}_n l$$ is diagonal. Thus, the fourier transform diagonalizes $$M$$. Furthermore, our diagonal matrix $$D = \bar{F}_n l$$.

We can thus diagonalize $$M$$ using the FFT, since:

$$D = \bar{F}_n l_n = \text{FFT}(l_n) $$

Remembering where $$l_n = [-2,1,\cdots,1]^\top$$


# P 15
(Circulants and the DFT.) An $$n \times n$$ matrix $$A$$ is called a circulant if all of its diagonals (main, super, and sub) are constant and the indices are interpreted "modulo" $$n$$. For example, this $$4 \times 4$$ matrix is a circulant:

$$ \begin{pmatrix} 9 & 2 & 1 & 7 \\
7 & 9 & 2 & 1 \\
1 & 7 & 9 & 2 \\
2 & 1 & 7 & 9 \end{pmatrix} $$

(a) Look at the $$n$$-periodic sequence $$a$$, where $$a_l = A_{l+1,1}, l=0,\cdots,n-1$$. Write the entries of $$A$$ in terms of the sequence $$a$$.

The sequence $$a$$ represents the first column of $$A$$. Since we can see that each of the columns of the matrix is "shifted" vertically by one, we can explicitly write the entries of $$A$$ as:

$$A_{i,j} = a_k \quad k=((i-1) - (j-1)) \% n$$

And since $$a_k = a_{n+k}$$, $$n$$-periodic sequences are always interpreted as modulo-n. Therefore,

$$A_{i,j} = a_k \quad k=(i-1) - (j-1)$$

(b) Let $$X$$ be an $$n \times 1$$ column vector. Show that $$Y=AX$$ is equivalent to $$y = a * x$$, if $$x,y$$ are the $$n$$-periodic sequences for which $$x_l = X_{l+1,1}$$ and similarly for $$y_l = Y_{l+1,1},l=0,\cdots,n-1$$.

The discrete convolution for $$n$$-periodic functions is defined:

$$(f*g)_ k = \sum_{j=0}^{n-1} f_{k-j}g_k$$

I can show that the matrix multiplication of a circulant matrix is actually a convolution by writing out the formula for matrix multiplication $$X = AY$$

$$X_{i,j} = \sum\limits_{k=1}^n A_{i,k} Y_{k,j}$$

Since $$X,Y$$ are column vectors, the secondary index can only be 1, thus:

$$X_i = \sum\limits_{k=1}^n A_{i,k} Y_k$$

Substituting the formula for $$A_{i,j}$$ from the previous part:

$$X_i = \sum\limits_{k=1}^n \alpha_{(i-1)-(k-1)} Y_k$$

Shifting the indices so that $$i = i+1, k = k+1$$, this expression becomes

$$X_{i+1} = \sum\limits_{k=0}^{n-1} \alpha_{i-k} Y_{k+1}$$

This looks very similar to the convolution formula, except for the pesky $$+1$$'s. However, this $$+1$$ is simply to account for the difference between $$0$$-indexed series and $$1$$-indexed matrices. So if we define series $$x,y$$ representing the column of $$X,Y$$ (as defined in the problem statement), we get:

$$x_i = \sum\limits_{k=0}^{n-1} \alpha_{i-k} y_{k} = (\alpha * y)_ k $$



(c) Prove that the DFT diagonalizes all circulant matrices; that is, $$n^{-1}F_n^\top A\bar{F}_n = D$$ where $$D$$ is diagonal. What are the diagonal entries of $$D$$ (i.e., what are the eigenvalues of $$A$$)?

From the previous part, we know there are two representations of a circular matrix, a convolution by $$\alpha$$, and a matrix multiplication.

$$\mathcal{M}[(\alpha * y)] =X = AY$$

Additionally, I observe the convolution theorem:

$$\mathcal{F}_n[\alpha * y] = \mathcal{F}_n[\alpha] \cdot \mathcal{F}_n [y]$$

Looking at what this means for the fourier coefficients,

$$\widehat{a*y}_k = \hat{\alpha}_k \cdot \hat{y}_k $$

Thus, convolution in the time domain is simple multiplication in the Fourier Domain. In matrix form, $$\hat{\alpha}$$ must be diagonal since there are no cross terms of $$\hat{y}$$, so there is only a simple scaling happening to each coefficient.

Thus, we know that if we transform $$Y$$ to the Fourier Domain, multiplication by $$A$$ is a multiplication by a diagonal matrix, and then a transformation back to the time domain:

$$X =AY = \frac{1}{n}F_n\mathcal{M}[\hat{\alpha}]\bar{F}_nY$$

Thus it easily follows that $$A$$ is diagonalized by the fourier transform for any circulant matrix $$A$$:

$$ A = \frac{1}{n}F_n\mathcal{M}[\hat{\alpha}]\bar{F}_n$$

If we define $$D=\mathcal{M}[\hat{\alpha}]$$, which we already know is a diagonal matrix, we get:

$$A = n^{-1}F_n D \bar{F}_n $$

And since $$F_n$$ is symmetric, this can be equivalently written:

$$A = n^{-1}F_n^\top D \bar{F}_n $$

The desired result.

Next, to find the diagonal entries of $$D$$ (and therefore the eigenvalues of $$A$$), we inspect the diagonal matrix:

$$\mathcal{M}[\hat{\alpha}] = \mathcal{M}[\bar{F}_n \alpha]$$

Thus, the eigenvalues of $$A$$ are provided by the fourier transform of $$\alpha$$, the first column of $$A$$:

$$\text{Eig}[A] = \bar{F}_n \alpha  $$


# P 16
Find the Z-transform for the sequence

$$x = \big(  \cdots, 0 \ 0 \ 1 \ \frac{1}{2} \ \frac{1}{4} \  \cdots \frac{1}{2^n} \ \cdots \big) $$

The Z-transform is defined:

$$\hat{x}(z) = \sum_{j=-\infty}^\infty x_j z^{-j} $$

where $$z = e^{i \phi}$$

Since the coefficients are 0 before a certain point, and choosing the indexing so that $$x_j =0$$ for $$j<0$$.  
$$\hat{x}(z) = \sum\limits_{k=0}^\infty 2^{-k}z^{-k}$$


By observing that $$\vert z/2\vert  < 1/2$$ and using the geometric series formula, we can rewrite this:

$$\hat{x}(z)= \frac{1}{1-1/2z}$$
