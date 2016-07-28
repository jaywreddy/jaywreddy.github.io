---
title: Time Series Analysis Exam
category: "Time Series Analysis"
layout: post
---

# Problem 1 (8 Points)
Let $$w_t$$ be a stochastic process that follows

$$\mathbb{E}[w_{t+1}\vert w_{t:1}]=0, \forall t > 0 $$

1. Assume $$w_t$$ is stationary, and its variance is finite. Show that $$w_t$$ is white noise.

    White Noise is a mean $$0$$, finite variance series of uncorrelated random variables.

    First, the conditional expectation: $$\mathbb{E}[w_{t+1}\vert w_{t:1}]$$ says that the expected value of the next value of the series given knowledge of the previous values of the series is $$0$$. However, this expectation does not depend on a specific instantiation of the previous values of the series, so we could equivalently say: $$\mathbb{E}[w_t] = 0$$.

    A more rigorous way to think of this is that the expectation is the the value of the R.V. weighted by its probability:

    $$\mathbb{E}[x] = \int_{x\in\Omega}x \mathbb{P}[x]$$

    Or equivalently

    $$\mathbb{E}[w_{t+1}] = \int_{w_{t+1} \in \Omega}w_{t+1} \mathbb{P}[w_{t+1}] $$

    Which we can express as a conditional probability:

    $$P[w_{t+1}] = \int_{w_{t:1}\in \Omega} \mathbb{P}[w_{t+1}\vert w_{t:1}]\mathbb{P}[w_{t:1}] $$

    $$\mathbb{E}[w_{t+1}] = \int_{w_{t+1} \in \Omega}w_{t+1} \int_{w_{t:1}\in \Omega} \mathbb{P}[w_{t+1}\vert w_{t:1}]\mathbb{P}[w_{t:1}]$$

    Since $$w_{t+1}$$ is a constant w.r.t. the inner integral

    $$\mathbb{E}[w_{t+1}] = \int_{w_{t+1}  \in \Omega} \int_{w_{t:1}\in \Omega}w_{t+1} \mathbb{P}[w_{t+1}\vert w_{t:1}]\mathbb{P}[w_{t:1}]$$

    Then, since we are integrating over separate regions, we may switch the order of integration

    $$\mathbb{E}[w_{t+1}] =  \int_{w_{t:1}\in \Omega}\int_{w_{t+1} \in \Omega} w_{t+1}  \mathbb{P}[w_{t+1}\vert w_{t:1}]\mathbb{P}[w_{t:1}]$$

    Then noting that the last term $$\mathbb{P}[w_{t:1}]$$ is a constant w.r.t. the inner integral

    $$\mathbb{E}[w_{t+1}] =  \int_{w_{t:1}\in \Omega} \mathbb{P}[w_{t:1}] \int_{w_{t+1} \in \Omega} w_{t+1}  \mathbb{P}[w_{t+1}\vert w_{t:1}]$$

    But we can see that the inner integral is the expectation: $$\mathbb{E}[w_{t+1}\vert w_{t:1}]=0$$ from the problem definition:

    $$\mathbb{E}[w_{t+1}\vert w_{t:1}] = \int_{w_{t+1} \in \Omega}  w_{t+1}  \mathbb{P}[w_{t+1}\vert w_{t:1}] = 0 $$

    Plugging in, we have

    $$\mathbb{E}[w_{t+1}] =  \int_{w_{t:1}\in \Omega} \mathbb{P}[w_{t:1}] \times 0$$

    $$\mathbb{E}[w_{t+1}] = 0, \ \forall t \implies \mathbb{E}[w_t] = 0, \ \forall t$$

    Thus, I have shown that $$w_t$$ is mean zero, fulfilling the first white noise criterion.

    We were given in the problem statement that the process is finite variance, fulfilling the second white noise criterion.

    The last criterion of white noise is that the variables be uncorrelated, which means that the covariance of two different terms in the series must be 0.

    Consider, without loss of generality, the covariance of two terms in the series: $$w_t, w_{t+h}, h>0$$.

    $$\text{Cov}(w_{t+h},w_t) = E[w_{t+h},w_t] - E[w_{t+h}]E[w_t] $$

    Since $$E[w_t]=0, \forall t$$:

    $$\text{Cov}(w_{t+h},w_t) = E[w_{t+h},w_t] $$

    $$\text{Cov}(w_{t+h},w_t) = \int_{w_{t+h}\in \Omega} \int_{w_t \in \Omega} w_t w_{t+h} \mathbb{P}[w_{t+h}, w_t] $$

    $$\text{Cov}(w_{t+h},w_t) = \int_{w_{t+h}\in \Omega} \int_{w_t \in \Omega} w_t w_{t+h} \mathbb{P}[w_{t+h}\vert  w_t]\mathbb{P}[w_t] $$

    Changing the order of integration and moving constants out of the inner integral, we have:

    $$\text{Cov}(w_{t+h},w_t) = \int_{w_t \in \Omega} \mathbb{P}[w_t]w_t\int_{w_{t+h}\in \Omega}   w_{t+h} \mathbb{P}[w_{t+h}\vert  w_t] $$

    But the inner integral is just:

    $$\mathbb{E}[w_{t+h}\vert w_t] $$

    Since $$\mathbb{E}[w_{t+h}] = 0$$ and $$\mathbb{E}[w_{t+h}\vert w_{t+h-1:1}]=0$$, we have that no information about the priors to random variable yields an expected value of $$0$$, and complete information yields an expected value of $$0$$.

    $$\mathbb{E}[w_{t+h}\vert w_t]$$ has access to the information of $$w_t$$, which is also contained in $$\mathbb{E}[w_{t+h}\vert w_{t+h-1:1}]$$, since $$t \in [t+h-1:1]$$.

    Concretely, complete knowledge of $$w_{t+h-1:1}$$ gives us no new information about $$\mathbb{E}[w_{t+h}]$$, so knowledge of $$w_t$$ will not help either.

    So,
    
    $$\mathbb{E}[w_{t+h} \vert  w_t] = 0$$

    Thus,

    $$\text{Cov}(w_{t+h},w_t) = \int_{w_t \in \Omega} \mathbb{P}[w_t]w_t \times 0 $$

    $$\text{Cov}(w_{t+h},w_t) =  0 $$

    And so $$w_t$$ is uncorrelated and therefore white noise.

2. Is $$w_t$$ necessarily independent white noise?

    No, since take for example the distribution:

    $$w_t \sim \begin{cases}
    5 & \text{w.p.} \frac{1}{4} \\
    2 & \text{w.p.} \frac{1}{4} \\
    -3& \text{w.p.} \frac{1}{4} \\
    -4& \text{w.p.} \frac{1}{4} \\
    \end{cases} \text{ if } w_{t-1}<0$$

    $$w_t \sim \begin{cases}
    4 & \text{w.p.} \frac{1}{4} \\
    3 & \text{w.p.} \frac{1}{4} \\
    -2& \text{w.p.} \frac{1}{4} \\
    -5& \text{w.p.} \frac{1}{4} \\
    \end{cases} \text{ if } w_{t-1}>0$$

    Then $$\mathbb{E}[w_t] = 0, \forall t$$ in both cases, and $$E[w_t^2] = 13.5, \forall t$$ in both cases, satisfying the descriptions of the process $$w_t$$ in part (1). However this series is not independent. The specific instantiation of each $$w_t$$ depends on instantiations of prior random variables.

# Problem 2 (12 points)
Let $$x_t$$ be the ARMA(2,1) process given by

$$x_t - \frac{1}{9}x_{t-2} = w_t - \frac{1}{3.1}w_{t-1}$$

1. Suggest a modification to the MA polynomial so that $$x_t$$ is a causal AR(1) process.

    We can rewrite the process as a one-sided AR model:

    $$\pi(B)x_t = w_t $$

    Where $$\pi(z)= \frac{\phi(z)}{\theta(z)}$$.

    Currently, the ARMA(2,1) process is given by:

    AR(2):

    $$\theta(z) = 1 - \frac{1}{9}z^2 = (1- \frac{1}{3}z)(1+ \frac{1}{3}z) $$

    MA(1):

    $$\phi(z) = 1 - \frac{1}{3.1}z$$

    However, if we modified the MA polynomial as such:

    $$\phi(z) = 1 - \frac{1}{3}z$$

    Then we would have:

    $$\pi(z) = \frac{(1- \frac{1}{3}z)(1+ \frac{1}{3}z)}{1- \frac{1}{3}z} = (1 + \frac{1}{3}z) $$

    So we could write the model as a one-sided AR process:

    $$\pi(B)x_t = w_t$$

    Which could equivalently be expressed as an AR(1) model:

    $$\phi(B)x_t = w_t $$

    $$\phi(z) = 1 + \frac{1}{3}z $$
    And indeed, $$\phi(z)$$ is degree 1, so this is an AR(1) process.

    To check causality of the model, we see that $$\phi(z)$$ has a single root at $$z=-3$$. Since $$\phi(z)$$ has no roots inside the unit circle, the modified process is causal.

2. We observe $$x_1, \cdots, x_n$$, but mistake the data generating process as an AR(1) process. What is the Yule-Walker estimator of the AR(1) coefficient?

    The Yule-Walker equations for an AR(1) process are:

    $$\gamma(h) = \phi_1\gamma(h-1), h=1  $$

    $$\sigma_w^2 = \gamma(0) - \phi_1 \gamma(1) $$

    Thus,

    $$ \phi_1= \frac{\gamma(1)}{\gamma(0)} = p(1)$$

    For the Yule-Walker estimator, we substitute the sample autocovariance from the data for the autocovariance and thus have:

    $$\hat{\gamma}(h) = n^{-1} \sum\limits_{t=1}^{n-h}(x_{t+h}- \bar{x})(x_t - \bar{x})$$

    $$\hat{\phi_1} = \frac{\hat{\gamma}(1)}{\hat{\gamma}(0)} = \hat{p}(1) $$

3. Do you expect the forecasts of the fitted AR(1) model to be accurate? Why or why not?

    I expect the forecasts to be accurate since our ARMA(2,1) model is almost an AR(1) process, in that the coefficients need only be modified slightly for the model to be completely accurate. So in effect, we will be forecasting the model:

    $$x_t - \frac{1}{9}x_{t-2} = w_t - \frac{1}{3}w_{t-1} $$

    instead of

    $$x_t - \frac{1}{9}x_{t-2} = w_t - \frac{1}{3.1}w_{t-1}$$

    In fact, since these are the idealized models and the actual predictions will be made off of the parameter estimates, the actual parameter estimates used in forecasting will vary from the ones above.

    Thus, evolutions of the forecasts made from the AR(1) model will accumulate small errors at each step based on the small difference in parameters. However, errors in any forecast, even those based on the correct model will accumulate errors based on inaccuracy of the parameter estimates.

    Thus, I expect that the AR(1) model will create accurate forecasts since it is so close to the AR(2) model of the actual process that the increased errors due to the differences in models are likely to be insignificant.

# Problem 3 (12 points)
We observe a sample path of a MA(q) process.

1. What is the $$\alpha$$-level test of

    $$H_0 : x_t \sim \text{MA}(q_0) \text{ versus } H_1: x_t \sim \text{MA}(q_1), \ q_0 < q_1? $$

    Show that the Type I error rate of your test is at most $$\alpha$$.

    First, we know that for an MA(q) polynomial, the ACF, $$p(h)$$ cuts off after $$q$$ terms:

    $$p(h) = 0, h>p $$

    Thus, if we calculate the sample autocorrelation function $$\hat{p}(h)$$, we would expect it to be near $$0$$ for $$h>q_0$$ if $$H_0$$ is true.

    So consider the vector:

    $$\hat{p}_{q_0:q_1} = [\hat{p}(q_0 + 1), \cdots, \hat{p}(q_1)]$$

    Under the null hypothesis, each of the $$p(h), h \in [q_0+1, q_1]$$ should be $$0$$. However, the sample autocorrelation function is approximately normally distributed with standard deviation:

    $$\sigma_{\hat{p}} = \frac{1}{\sqrt{n}}$$

    So we can form a confidence interval for $$\hat{p}_{q_0:q_1}$$ as a mean-$$0$$ Multivariate Gaussian with $$\Sigma = \text{diag}[\frac{1}{\sqrt{n}}, \cdots, \frac{1}{\sqrt{n}}]$$ is an $$(q_1 -q_0)\times(q_1 - q_0)$$ covariance matrix.

    Since each of the autocorrelations are normally distributed with mean $$0$$ and SD $$\sigma_{\hat{p}}$$, we have:

    $$\frac{1}{\sigma_{\hat{p}}}\hat{p}_{q_0:q_1} \to \mathcal{N}(0, I_{q_1- q_0}) $$

     So to create an $$\alpha$$ level test, we can use the $$1-\alpha$$ quantile of the $$\chi^2$$ distribution:


     $$B = \{z \in \mathbb{R}^{q_1 - q_0} : \vert \vert z\vert \vert _ 2^2 \leq \chi^2_{q_1- q_0, 1-\alpha} \}$$

     Then, our test is:

     $$P[\frac{1}{\sigma_{\hat{p}}} \hat{p}_{q_0 : q_1} \in B]   = P[\hat{p}_{q_0 : q_1} \in \sigma_{\hat{p}}B] \to 1 - \alpha $$

     This is a valid $$\alpha$$-level test with Type I error at most $$\alpha$$, since under the null hypothesis, each of the autocorrelation estimates should be independently normally distributed with mean $$0$$ and standard deviation $$\sigma_{\hat{p}}$$. Thus, the $$1-\alpha$$-level confidence set given by the $$\chi^2$$ distribution scaled by $$\sigma_{\hat{p}}$$ contains $$100(1-\alpha)\%$$ of the possible instantiations of $$\hat{p}_{q_0:q_1}$$ under the null hypothesis. Thus, only $$100\alpha \%$$ of the time will the null-hypothesis yield a false-positive (Type 1) error, and so the Type-1 error rate is $$\alpha$$.

2. Let $$\hat{p}:=[\hat{p}(1), \cdots, \hat{p}(q)]^\top$$, where $$\hat{p}(h)$$ is the sample ACF. What is an (asymptotic) $$1-\alpha$$-level confidence set for $$\hat{p}$$?

     Building upon part (1), we know that $$\hat{p}$$ is a $$q$$-dimensional vector with mean $$\hat{p}$$, and each of the components are normally distributed R.V.s with standard deviation $$\sigma_{\hat{p}} = \frac{1}{\sqrt{n}}$$.

     Thus,

     $$\frac{1}{\sigma_{\hat{p}}} (\hat{p}-p) \to \mathcal{N}(0, I_q) $$

     We can create a $$1-\alpha$$ confidence set for a $$q$$-dimensional multivariate gaussian using the $$\chi^2$$ distribution:

     $$B = \{z \in \mathbb{R}^q : \vert \vert z\vert \vert ^2_2 \leq \chi^2_{d,1-\alpha}\} $$

     meaning,

     $$P[\frac{1}{\sigma_{\hat{p}}}(\hat{p}-p) \in B] \to 1- \alpha $$

     This confidence set can then be scaled from variance $$I_q$$ to variance $$\sigma_{\hat{p}}^2 I_d$$ to match $$\hat{p}$$:

     $$ \sigma_{\hat{p}}B$$

     Meaning,

     $$P[\hat{p}- p \in \sigma_{\hat{p}}B] \to 1-\alpha$$

     Finally, we should shift the confidence set to be centered on $$\hat{p}$$:

     $$\hat{p} - \sigma_{\hat{p}}B $$

     $$P[p\in \hat{p}-\sigma_{\hat{p}}B ] \to 1-\alpha$$

     Is a $$1- \alpha$$-level confidence interval for $$\hat{p}$$, meaning that given an instantiation of $$\hat{p}$$, there is a $$1-\alpha$$ chance that the true value of the estimator is in the confidence set:

     $$\hat{p} - \sigma_{\hat{p}}B $$

3. What is a (not necessarily exact) $$1-\alpha$$-level simultaneous confidence band for the ACF? By a $$1-\alpha$$-level simultaneous confidence band, I mean two functions $$l(h)$$ and $$u(h)$$ such that

    $$\mathbb{P}\big[l(h) \leq \hat{p}(h)\leq u(h)), \forall h \in [n-1]\big] \geq 1-\alpha $$

    Hint: If $$B(x_{1:n})\subset I(x_{1:n})$$ is the $$1-\alpha$$-level confidence set you derived in (2), then you automatically have

    $$\mathbf{P}\big[\hat{p} \in I(x_{1:n})\big] \geq \mathbf{P}\big[\hat{p} \in B(x_{1:n}) \big] \geq 1- \alpha$$

    We know that the estimator of the autocorrelation has a standard deviation: $$\sigma = \frac{1}{\sqrt{n}}$$.

    And will be normally distributed. Therefore, we can create an $$1-\alpha$$ level simultaneous confidence set:

    $$ l(h) = \hat{p}(h) - z(1-\alpha)\sigma$$
    $$u(h) = \hat{p}(h) + z(1-\alpha)\sigma$$

    Where $$z(1-\alpha)$$ is the z-score associated with a $$100(1-\alpha)\%$$ confidence interval.

# Problem 4 (12 points)
Consider the stationary time series $$x_t$$ given by

$$x_t - (\frac{1}{1.01})^2 x_{t-2} = w_t + \frac{1}{4}w_{t-1} $$

1. What is the MA representation of $$x_t$$? You may leave your answer in the form $$x_t = \psi(B)w_t$$, where $$\psi(z)$$ is a rational function. Where are the zeros and poles of $$\psi(z)$$.

    We currently have the ARMA(2,1) model with AR,MA polynomials:

    $$\phi(z) = 1 - (\frac{1}{1.01})^2 z^2 $$

    $$\theta(z) =  1 + \frac{1}{4}z$$

    This can be expressed as a one-sided MA model of the form:

    $$x_t = \psi(B)w_t $$

    where
    $$\psi(z) = \frac{\theta(z)}{\phi(z)} = \frac{1 + \frac{1}{4}z}{1 - (\frac{1}{1.01})^2 z^2}$$

    From this representation, we can see that the zeros of $$\psi(z)$$ occur at the roots of $$\theta(z)$$: $$z=-4$$. Additionally, the poles of $$\psi(z)$$ occur at the roots of $$\phi(z)$$: $$z= \pm \frac{1.01}{1}$$

    Thus we have a single zero at $$z=-4$$ and a two poles at $$z= \pm \frac{1.01}{1}$$

2. Sketch the spectral density of $$x_t$$.


    First, we know the Spectral Density of an ARMA model is defined:
    
    $$f_x(\omega)= \sigma_w^2 \frac{\vert \theta(e^{-2 \pi i \omega})\vert ^2}{\vert \phi(e^{-2 \pi i \omega})\vert } $$
    
    equivalently, 
    
    $$f_x(\omega) = \sigma_w^2 \frac{\theta(e^{-2 \pi i \omega}) \bar{\theta(e^{-2 \pi i \omega})}}{\phi(e^{-2 \pi i \omega}) \bar{\phi(e^{-2 \pi i \omega})}} $$
    
    So plugging in for our AR, MA polynomials, we have:
    
    $$f_x(\omega) = \sigma_w^2 \frac{(1 + \frac{1}{4}e^{-2\pi i \omega})(1 + \frac{1}{4}e^{2\pi i \omega})}{(1 - \big(\frac{1}{1.01}\big)^2 e^{- 4 \pi i \omega})(1 - \big(\frac{1}{1.01}\big)^2 e^{ 4 \pi i \omega})} $$
    
    And simplifying:
    
    $$f_x(\omega) = \sigma_w^2 \frac{\frac{17}{16} + \frac{1}{4}(e^{2\pi i \omega} + e^{-2\pi i \omega})}{1 + \big(\frac{1}{1.01}\big)^4 - \big(\frac{1}{1.01} \big)^2 (e^{4\pi i \omega} + e^{-4\pi i \omega})} $$
    
    The latter terms are actually sinusoids as we can see by the complex exponential: 
    
    $$\cos(x) = \frac{1}{2}[e^{ix} + e^{-ix}] $$
    
    So expressing the spectral density as such:
    
    $$f_x(\omega) = \sigma_w^2 \frac{\frac{17}{16} + \frac{1}{2}\cos(2\pi \omega)}{1 + \big(\frac{1}{1.01}\big)^4 - 2\big(\frac{1}{1.01} \big)^2 \cos(4\pi \omega)} $$
    
    In order to characterize the function, I will look at the values of the numerator and denominator for $$\omega \in [0,1/2]$$.
    
    We see that the denominator reaches its minimum when $$\cos(4\pi \omega) =1$$, $$\omega = 0,1/2$$. When this happens, the denominator $$D$$:
    
    $$D = 1 + (\frac{1}{1.01})^4 - 2(\frac{1}{1.01})^2 \approx .0004$$
    
    The denominator is at its maximum when $$\omega = 1/4$$ and 
    
    $$D = 1 + (\frac{1}{1.01})^4 + 2(\frac{1}{1.01})^2 \approx 4 $$
    
    The numerator, N, on the other hand, changes much less drastically. It has a max when $$\cos(2\pi \omega )  = 1$$, $$\omega = 0$$ and 
    
    $$N  = \frac{25}{16} $$
    
    And is at a minimum when $$\omega = 1/2$$ and 
    
    $$N = \frac{9}{16} $$
    
    Thus, the behavior of the spectral density is affected mainly by the poles when $$\omega$$ is at the limit of its range. On each edge, the denominator of the transfer function will cause the magnitude to be $$\approx 2500$$ times greater. Thus, there are two massive spikes in the spectral density, which I illustrate in my sketch:

    ![Spectrogram](/images/timeseries_analysis/finalspectrogram.jpg)

    We can see two large spikes near $$\omega= 0,1/2$$ are caused by the points $$e^{-2\pi i \omega}$$ being very close to the poles of the transfer function: $$z = \pm \frac{1.01}{1}$$

3. Suggest a filter to remove the spikes in the spectral density. That is, write down an expression for a sequence of coefficients $$(\psi_j)^\infty_{j=0}$$ such that the spectral density of $$y_t = \sum\limits_{j=0}^\infty \psi_j x_{t-j}$$ does not exhibit the spikes that are present in the spectral density of $$x_t$$.

    In order to remove the spikes near the poles of the transfer function, I would like to create a linear filter with zeros at the same points.

    If we have a linear filter:

    $$A(\omega) = \phi(e^{2 \pi i \omega})$$

    Then this will introduce zeros into the same regions that the original spectrogram has poles.

    And since a linear filter affects the spectrogram as:

    $$f_y(\omega) = \vert A(\omega)\vert ^2 f_x(\omega) $$

    I can compute the filtered spectrogram:
    
    $$f_y(\omega) = \theta(e^{-2\pi i \omega}) $$

    We would now like to find a series of coefficients:

    $$(\psi_j)^\infty_{j=0}$$

    So that the filtered spectra will correspond to the convolution with $$x_t$$:

    $$y_t = \sum\limits_{j=0}^\infty \psi_j x_{t-j}$$

    We have already found the Frequency domain representation: $$A(\omega)$$, where $$f_y(\omega) = \vert A(w)\vert ^2 f_x(\omega)$$. Since multiplication in the frequency domain corresponds to convolution in the time domain, I am looking for the Fourier transform pair of $$A(\omega)$$, which I can calculate using the inverse DFT. Notice the problem definition requires a causal filter (no $$\psi_j, \ j < 0$$). So I only take the positively indexed terms to filter based off of only the previous elements of the time series. 

    $$\psi_j = \sum\limits_{j=0}^\infty A(\omega_j) e^{2\pi \omega_j t} $$


# Problem 5 (16 points)

Recall the periodogram of a linear process is distributed like a vector of independent $$\chi_2^2$$ random variables:

$$\frac{2 I (\omega_0)}{f(\omega_0)}, \cdots, \frac{2 I(\omega_{n-1})}{f(\omega_{n-1})} \stackrel{i.i.d.}{\sim} \chi_2^2 $$

where $$w_j = \frac{j}{n}$$ are the Fourier frequencies. To consistently estimate the spectral density, we consider a smoothed periodogram:

$$\hat{f_k}(\omega_j) = \frac{1}{2k+1}\sum\limits_{h=-k}^k I(\omega_{j+h}) $$

One possible approach to choosing the bandwidth $$k$$ is choosing it to minimize the mean squared error (MSE): $$\mathbb{E}\bigg[\big(\hat{f}_ k(\omega_j)- f(\omega_j) \big)^2\bigg]$$.

We assume the periodogram is L-Lipschitz continuous:

$$\vert f(\omega_1)-f(\omega_2)\vert  \leq L\vert \omega_1-\omega_2\vert  \ \text{ for any }\omega_1,\omega_2 $$  

1. Show that
    $$
    \begin{split}
    \mathbb{E}\bigg[ \big(\hat{f_k} (\omega_j) -f(\omega_j)\big)^2\bigg] & \\ =\big(\mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j) \big)^2 &+ \mathbb{E}\bigg[ \big( \hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)]\big)^2 \bigg]
     \end{split}
     $$

     Hint: The preceding expression is true for any estimator and target, so you do not need to use the fact that $$\hat{f_k}(\omega_j)$$ is a smoothed periodogram.

     First, I observe that the two terms in the solution appear to be the result of a square of the form:
     $$\bigg(\big(\mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j) \big) +\big( \hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)]\big)  \bigg)^2$$

     With cross-terms removed. I therefore add and subtract $$\mathbb{E}[\hat{f_k}(\omega_j)]$$ to our initial expression for the MSE in hopes that the cross terms will be $$0$$:

     $$\mathbb{E}\bigg[ \big(\hat{f_k} (\omega_j) -f(\omega_j)\big)^2\bigg]$$

     $$\mathbb{E}\bigg[ \big(\hat{f_k} (\omega_j)+ \mathbb{E}[\hat{f_k}(\omega_j)] - \mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\big)^2\bigg]$$

     Grouping the $$\mathbb{E}[\hat{f_k}(\omega_j)]$$ terms with the terms they are found with in the desired expression:

    $$\mathbb{E}\bigg[ \bigg(\Big(\hat{f_k} (\omega_j)-  \mathbb{E}[\hat{f_k}(\omega_j)]\Big) + \Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)\bigg)^2\bigg]$$

    Squaring the terms:

    $$\begin{split}\mathbb{E}\bigg[ \Big(\hat{f_k} (\omega_j)-  \mathbb{E}[\hat{f_k}(\omega_j)]\Big)^2 &\\
    -2\Big(\hat{f_k} (\omega_j) -\mathbb{E}[\hat{f_k}(\omega_j)]\Big)&\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big) \\
    +&\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)^2\bigg]\end{split}$$

    And by Linearity of Expectation:

    $$\begin{split}\mathbb{E}\bigg[ \Big(\hat{f_k} (\omega_j)-  \mathbb{E}[\hat{f_k}(\omega_j)]\Big)^2 \bigg]&\\
    -2\mathbb{E}\bigg[\Big(\hat{f_k} (\omega_j) -\mathbb{E}[\hat{f_k}(\omega_j)]\Big)&\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big) \bigg]\\
    +&\mathbb{E}\bigg[\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)^2\bigg]\end{split}$$

    The first term,

    $$\mathbb{E}\bigg[ \Big(\hat{f_k} (\omega_j)-  \mathbb{E}[\hat{f_k}(\omega_j)]\Big)^2 \bigg]$$

    Appears in our final expression so I will leave it as is.

    The second term,

    $$ -2\mathbb{E}\bigg[\Big(\hat{f_k} (\omega_j) -\mathbb{E}[\hat{f_k}(\omega_j)]\Big)\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big) \bigg] $$

    $$\begin{split} -2\mathbb{E}\bigg[ \hat{f_k} (\omega_j) \mathbb{E}[\hat{f_k}(\omega_j)] -& f(\omega_j)\hat{f_k}(\omega_j) \\ - \mathbb{E}[\hat{f_k}(\omega_j)]^2& +\mathbb{E}[\hat{f_k}(\omega_j)]f(\omega_j)\bigg]\end{split} $$

    Then by linearity of expectation and observing that $$\mathbb{E}[\hat{f_k}(\omega_j)]$$ and $$f(\omega_j)$$ are constant and so can be factored out of the expectation,

    $$\begin{split} -2\bigg( \mathbb{E}[\hat{f_k} (\omega_j)] \mathbb{E}[\hat{f_k}(\omega_j)] -& f(\omega_j)\mathbb{E}[\hat{f_k}(\omega_j)] \\- \mathbb{E}[\hat{f_k}(\omega_j)]^2 +&\mathbb{E}[\hat{f_k}(\omega_j)]f(\omega_j)\bigg) \end{split}$$

    Grouping like terms to better illistrate the terms cancelling:

    $$\begin{split} -2\bigg( \mathbb{E}[\hat{f_k} (\omega_j)]^2 - \mathbb{E}[\hat{f_k}(\omega_j)]^2 &\\ + f(\omega_j)\mathbb{E}[\hat{f_k}(\omega_j)]& - f(\omega_j)\mathbb{E}[\hat{f_k}(\omega_j)]\bigg) = 0 \end{split}$$

    We can see the cross term is actually $$0$$.

    And the third term,

    $$\mathbb{E}\bigg[\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)^2\bigg] $$

    Can be written simply:

    $$\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)^2$$

    Since $$\mathbb{E}[\hat{f_k}(\omega_j)]$$ and $$f(\omega_j)$$ are constant, so their sum and square are constant, so the outer expectation is superfluous.

    Now, recombining the first and third terms, we have:

    $$\Big(\mathbb{E}[\hat{f_k}(\omega_j)] -f(\omega_j)\Big)^2 + \mathbb{E}\bigg[ \Big(\hat{f_k} (\omega_j)-  \mathbb{E}[\hat{f_k}(\omega_j)]\Big)^2 \bigg]$$

    The desired result. So I have shown that the two expressions are in fact equivalent.

2. Show that
    $$\big\vert \mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j)\big\vert  \leq \frac{Lk}{2n} $$

    Hint: $$\sum_{h=1}^k h = \frac{1}{2}k(k-1)$$

    Substituting for the definition of $$\hat{f_k}(\omega_j)$$:

    $$ \big\vert \mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j)\big\vert  = \big\vert \mathbb{E}[\frac{1}{2k+1}\sum\limits_{h=-k}^k I(\omega_{j+h})]- f(\omega_j)\big\vert $$

    Which is, by linearity of expectation,

    $$ =\big\vert \frac{1}{2k+1}\Big(\sum\limits_{h=-k}^k \mathbb{E}[I(\omega_{j+h})] \Big)- f(\omega_j)\big\vert  $$

    The expected value of the periodogram of a fourier frequency is the spectral density:

    $$= \big\vert  \frac{1}{2k+1}\Big(\sum\limits_{h=-k}^k f(\omega_{j+h}) \Big)- f(\omega_j)\big\vert  $$

    We can move the $$f(\omega_j)$$ term into the summation by multiplying by $$\frac{2k+1}{2k+1}$$, factoring out the denominator to move it inside the parentheses, and then dividing by $$2k+1$$ to distribute the value among each of the terms of the summation:

    $$= \frac{1}{2k+1}\big\vert  \sum\limits_{h=-k}^k \Big(f(\omega_{j+h})- f(\omega_j) \Big)\big\vert  $$

    Next, noting that the sum of an absolute value is less than or equal to the absolute value of the sum:

    $$\leq \frac{1}{2k+1} \sum\limits_{h=-k}^k \big\vert f(\omega_{j+h})- f(\omega_j) \big\vert  $$

    And applying the L-Lipschitz continuous criterion:

    $$\leq \frac{1}{2k+1} \sum\limits_{h=-k}^k L \big\vert \omega_{j+h}- \omega_j \big\vert  $$

    Factoring $$L$$ out of the summation and expressing $$\omega_j$$ in terms of $$n$$:

    $$= \frac{L}{2k+1} \sum\limits_{h=-k}^k  \big\vert \frac{j+h}{n}- \frac{j}{n} \big\vert  $$

    $$= \frac{L}{n(2k+1)} \sum\limits_{h=-k}^k  \big\vert h\big\vert  $$

    Since we are summing the absolute value of $$h$$, we can rewrite the sum as 2-times the one-sided summation. We can also ignore the $$0$$'th index since $$h=0$$ is the additive identity:

    $$= \frac{2L}{n(2k+1)} \sum\limits_{h=1}^k  h $$

    Applying the formula supplied in the hint:

    $$= \frac{2L}{n(2k+1)} \frac{1}{2}k(k-1) $$

    $$= \frac{Lk}{n} \frac{k-1}{2k+1} $$

    Which is less than if the denominator were decreased by 1, so that we could factor out the $$2$$:

    $$\leq \frac{Lk}{2n} \frac{k-1}{k} $$

    Since $$\frac{k-1}{k} < 1$$,

    $$\leq \frac{Lk}{2n} $$

    Thus, I have shown:

    $$\big\vert \mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j)\big\vert  \leq \frac{Lk}{2n} $$

    The desired result.

3. Show that

    $$\mathbb{E}\bigg[\big(\hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)] \big)^2 \bigg] \leq \frac{\text{max}_j \ f(\omega_j)^2}{2k} $$

    Hint: The variance of a $$\chi_2^2$$distribution is 4.

    First,

    $$\mathbb{E}[(X- \mathbb{E}[X])^2] = \text{Var}[X]$$

    So we can see that

    $$\mathbb{E}\bigg[\big(\hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)] \big)^2 \bigg] = \text{Var}[\hat{f_k}(\omega_j)]$$

    Substituting for the definition of $$\hat{f_k}(\omega)$$:

    $$= \text{Var}[\frac{1}{2k+1}\sum\limits_{h=-k}^{k} I(\omega_{j+h})]$$

    Rearranging using the addition and scaling properties of variance:

    $$= \frac{1}{(2k+1)^2}\text{Var}[\sum\limits_{h=-k}^{k} I(\omega_{j+h})]$$

    $$= \frac{1}{(2k+1)^2}\sum\limits_{h=-k}^{k}\text{Var}[ I(\omega_{j+h})]$$

    And since $$\frac{2 I (\omega_j)}{f(\omega_j)} \sim \chi^2_2$$,

    $$\text{Var}[I(\omega_j)] = \text{Var}[\frac{f(\omega_j)}{2} \chi^2_2] = \frac{f(\omega_j)^2}{4}\text{Var}[\chi_2^2] $$

    Using the hint that $$\text{Var}[\chi_2^2]=4$$:

    $$ \text{Var}[I(\omega_j)] = f(\omega_j)^2$$

    And plugging in, we have:

    $$\mathbb{E}\bigg[\big(\hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)] \big)^2 \bigg] = frac{1}{(2k+1)^2}\sum\limits_{h=-k}^{k} f(\omega_j)^2$$

    We can bound the sum of a sequence of numbers to be less than or equal to the maximum element of the sequence times the length of the sequence:

    $$\leq \frac{1}{(2k+1)^2} (2k+1) \text{max}_j f(\omega_j)^2 $$

    $$= \frac{\text{max}_j f(\omega_j)^2}{2k+1} $$

    $$\leq \frac{\text{max}_j f(\omega_j)^2}{2k} $$

    So I have shown

    $$\mathbb{E}\bigg[\big(\hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)] \big)^2 \bigg] \leq \frac{\text{max}_j \ f(\omega_j)^2}{2k} $$

    The desired result.

4. By combining the results of the preceding three parts, deduce that an upper bound of the MSE is

    $$E \bigg[ \big( \hat{f_k}(\omega_j) - f(\omega_j)\big)^2\bigg] \leq \frac{L^2 k^2}{4n^2} + \frac{\text{max}_j \ f(\omega_j)^2}{2k} $$

    Minimize the right hand side in $$k$$ to show that the optimal bandwidth is (up to a multiplicative constant that does not depend on $$n$$) $$n^{\frac{2}{3}}$$.

    Using the inequalities found in parts (2) and (3):

    $$\big\vert \mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j)\big\vert  \leq \frac{Lk}{2n} $$

    $$\mathbb{E}\bigg[\big(\hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)] \big)^2 \bigg] \leq \frac{\text{max}_j \ f(\omega_j)^2}{2k} $$

    We can substitute into our expression from part (1):

    $$\begin{split}\mathbb{E}\bigg[ \big(\hat{f_k} (\omega_j) -f(\omega_j)\big)^2\bigg]& \\
     =\big(\mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j) \big)^2 + &\mathbb{E}\bigg[ \big( \hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)]\big)^2 \bigg]\end{split}$$

     $$ =\Big\vert \mathbb{E}[\hat{f_k}(\omega_j)]- f(\omega_j) \Big\vert ^2 + \mathbb{E}\bigg[ \big( \hat{f_k}(\omega_j) - \mathbb{E}[\hat{f_k}(\omega_j)]\big)^2 \bigg] $$

     $$\leq \frac{L^2k^2}{4n^2} + \frac{\text{max}_j \ f(\omega_j)^2}{2k} $$

     This is an upper bound for the MSE, which can be minimized w.r.t. k by taking the derivative and setting it to $$0$$ in order to find minima:

     $$\frac{d}{dn} \Big[\frac{L^2k^2}{4n^2} + \frac{\text{max}_j \ f(\omega_j)^2}{2k}\Big] = 0$$

     $$ \frac{2L^2k}{4n^2} - \frac{\text{max}_j \ f(\omega_j)^2}{2k^2} = 0$$

     $$ \frac{2L^2k}{4n^2} = \frac{\text{max}_j \ f(\omega_j)^2}{2k^2} $$

     $$ k^3 = \frac{n^2 \big(\text{max}_j \ f(\omega_j)^2\big)}{L^2} $$

     $$k = \sqrt[3]{\frac{ \big(\text{max}_j \ f(\omega_j)^2\big)}{L^2}} n^{\frac{2}{3}} $$

     Thus, the optimal bandwidth, $$k$$, that minimizes the MSE is:

     $$ n^{\frac{2}{3}}$$

     Up to a multiplicative constant

     $$\sqrt[3]{\frac{ \big(\text{max}_j \ f(\omega_j)^2\big)}{L^2}} $$
