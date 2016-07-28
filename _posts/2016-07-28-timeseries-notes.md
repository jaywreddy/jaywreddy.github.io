---
layout: post
title: Time Series Analysis Notes 
category: "Time Series Analysis"
---
# Chapter 1: Characteristics of Time Series

## 1.3 Time Series Statistical Models
Time Series $$\{x_t\}$$

### White Noise
$$ w_t \sim wn(0,\sigma_w^2)$$
Zero mean, finite variance random process.
Can be iid or usually Gaussian.

### Some Different White Noise Models:

#### Moving Average Filtered White Noise
Smoothes a signal
i.e.
$$v_t = \frac{1}{3} (w_{t-1} + w_t + w_{t+1}) $$

#### Autoregressions
Current value is based on previous values:
i.e.
$$ x_t = t_{t-1} - .9x_{t-2} + w_t$$

#### Random Walk with Drift
Values fluctuate based on previous value and white noise, along with a slope called the drift: $$\delta$$

$$x_t = \delta + x_{t-1}  + w_t$$

#### Signal in Noise
Most time series are assumed to have an underlying signal plus a noise component:

$$x_t = 2 \cos(2\pi t) + w_t $$

## 1.4 Measures of Dependence: Autocorrelation and Cross-Correlation

### Probability Models
A probability model of a time series is usually described by the density function
$$F_t(x) = P\{x_t \leq x\}$$
or marginal density function

$$f_t(x) = \frac{\partial F_t(x)}{\partial x} $$

### Mean Function
$$\mu_{xt} = E(x_t) = \int_{-\infty}^\infty xf_t(x)dx $$

### Autocovariance Function
$$\gamma_x(s,t) = \text{cov}(x_s,x_t) = E[(x_s = \mu_s)(x_t-\mu_t)] $$

The autocovariance measures the linear dependence between two points on a time series at different times. Smooth series have large autocovariances even when $$s,t$$ are far apart.

$$\gamma_x(t,t) = \text{var}(x_t) $$

### Some example Autocovariances:

#### White Noise
$$\gamma_w(s,t) = \text{cov}(w_s,w_t) = \begin{cases} \sigma^2_w & s=t, \\ 0 & s \neq t.\end{cases} $$

#### Random Walk
Without drift, $$x_t = \sum_{k=1}^t w_j$$

$$ \gamma_x(s,t) = \text{cov}(x_s,x_t) = \text{cov} \Big(\sum\limits_{j=1}^s w_j , \sum\limits_{k=1}^t w_k \Big) = \text{min}\{s,t\} \sigma^2_w$$

Since the covariance is the linear combination of the covariances of each of the terms, which is of course $$\sigma_w^2$$ for shared terms and $$0$$ for different terms.

### Autocorrelation Function (ACF)

$$p(s,t) = \frac{\gamma(s,t)}{\sqrt{\gamma(s,s)\gamma(t,t)}} $$
$$ -1 \leq p(s,t) \leq 1$$

Measures the linear predictability of the series at time $$t$$ basesd on the value at time $$s$$. It is normalized based on the variances at each of the times.

### Cross-covariance Function
Defined between two series $$x_t, y_t$$

$$ \gamma_{x,y}(s,t) = \text{cov}(x_s,y_t) = E[(x_s - \mu_{xs})(y_t - \mu_{yt})] $$

### Cross-correlation Function (CCF)

$$p_{xy}(s,t) = \frac{\gamma{xy}(s,t)}{\sqrt{\gamma_x(s,s)\gamma_y(t,t)}} $$

This can be extended to multivariate series as well.

## 1.5 Stationary Time Series

### Strictly Stationary
Behavior of series $$\{x_t\}$$ is identical to shifted series $$\{x_{t+h}\}$$

Implies $$\mu_s = \mu_t$$ for all $$s,t$$
Also,

$$\gamma(s,t) = \gamma(s+h,t+h) $$
Depends only on the difference in the times, not the times themselves.

### Weakly Stationary
$$x_t$$ is a finite variance process with

1. mean value function $$\mu_t$$ is constant
2. autocovariacne function $$\gamma(s,t)$$ depends on $$s,t$$ only through their difference: $$h = \vert  s-t\vert  $$

#### Constant Average
Since $$\mu_t$$ is constant, we can simply call it $$\mu$$

#### Time-Invariant Autocovariance
Similarly, the autocovariance function for stationary time series can be defined:

#### Time-Invariant Autocorrelation (ACF)
$$\gamma(h) = \text{cov}(x_{t+h},x_t) $$
$$ p(h) = \frac{\gamma(h)}{\gamma(0)}$$

#### Examples of Stationarity

##### White Noise
$$ \gamma_w(h) = \begin{cases} \sigma_w^2 & h=0 \\ 0 & h\neq 0 \end{cases} $$

##### Trend Stationarity

For $$x_t = \alpha + \beta t+ y_t$$, where $$y_t$$ is zero-mean stationary, the average $$\mu_{x,t} = \alpha + \beta t$$ is not stationary.

However, the autocovariance function is time-independent: $$\gamma_x(h) = \gamma_y(h)$$

Thus we say this model has stationary behavior around a ternd, called trend stationarity.

##### Stationarity of a Moving Average
using $$x_t = \frac{1}{3}(w_{t+1} + w_t + w_{t-1})$$
we have: $$\mu_{xt}=0$$

$$ \gamma_x(h) = \begin{cases}
\frac{3}{9}\sigma_w^2 & h=0, \\
\frac{2}{9}\sigma_w^2 & h=\pm 1, \\
\frac{1}{9}\sigma_w^2 & h=\pm 2, \\
\frac{3}{9}0 & \vert  h\vert  >2. \\
\end{cases}
$$

#### Symmetric Autocovariance
For stationary time series, the Autocovariance is symmetric:
$$\gamma(h) = \gamma(-h) $$
since $$\gamma(t-(t+h))=\text{cov}(x_t, x_{t+h}) = \text{cov}(x_{t+h},x_t) = \gamma((t+h)-t$$

### Joint Stationarity
Two time series are said to be jointly stationary if they are each stationary, and the cross-covariance function is a function only of lag $$h$$:
$$\gamma_{xy}(h)= \text{cov}(x_{t+h},y_t) = E[(x_{t+h}-\mu_x)(y_t - \mu_y)] $$

#### Cross Correlation Function (CCF)
CCF of jointly stationary time series $$x_t,y_t$$ becomes:

$$p_{xy}(h) = \frac{\gamma_{xy}(h)}{\sqrt{\gamma_x(0),\gamma_y(0)}} $$

and $$p_{xy}(h) = p_{xy}(-h)$$

### Linear Processes
Linear Process $$x_t$$ is a linear combination of white noise variances:

$$x_t = \mu + \sum\limits_{j=-\infty}^\infty \psi_j w_{t-h} \qquad \sum\limits_{j=-\infty}^\infty \vert  \psi\vert   \leq \infty $$

Has autocovariance:

$$\gamma(h) = \sigma_w^2 \sum\limits_{-\infty}^\infty \psi_{j+h}\psi_j $$

#### Causal Processes
Notice that Linear Processes can be dependent on noise in the future $$j < 0$$. thus, we define a causal process as $$\psi_j = 0, j<1$$.

### Gaussian Process
n-dimensional time series (each time point is a vector) $$\textbf{x}_t$$ is said to be Gaussian if each component of the vector is the output of a Gaussian process. So collectively, each of the components of the time series has a normal distribution, and the time series itself follows a multivariate normal.

Multivariate Normal Density:

$$ f(\textbf{x}) = (2\pi)^{-n/2} \vert  \Gamma\vert  ^{-1/2} \exp\Big\{-\frac{1}{2}(\textbf{x} - \boldsymbol\mu)^\top \Gamma^{-1} (\textbf{x}- \boldsymbol\mu) \Big\}$$

where $$\boldsymbol\mu$$ is straightforward and
$$\Gamma = \text{var}(\textbf{x}) = \{\gamma(t_i,t_j); i,j = 1,\cdots,n\}$$

Note: $$\vert  \Gamma\vert  $$ is the determinant

#### Properties of Gaussian Processes
1. If Gaussian Process is weakly stationary, $$\mu_t = \mu$$, and $$\gamma(t_i,t_j) = \gamma(\vert  t_i - t_j\vert  )$$, so $$\mathbf{\mu}$$ and $$\Gamma$$ are independent of time.
2. A linear process need not be Gaussian, but if a time series is Gaussian, then it is a causal linear process (Wold Decomposition)
3. Even if the marginal distributions are Gaussian, the process may not be Gaussian. EG, $$X,Y$$ may be normal but $$(X,Y)$$ is not bivariate normal.

## 1.6 Estimation of Correlation
Our theoretical ACF and CCF require us to know the probability distributions in advance.

We construct the following functions via only the sample points $$\{x_1, x_2, \cdots x_n\}$$.

We proceed assuming that the time series is stationary, or nothing makes sense.

### Sample Mean
$$ \bar{x} = \frac{1}{n}\sum\limits_{t=1}^n x_t $$

Variance of Estimate:
$$\text{var}(\bar{x}) = \frac{1}{n} \sum\limits_{h=-n}^n(1 - \frac{\vert  h\vert  }{n})\gamma_x(h)$$

In the case of white noise, this reduces to the familiar $$\sigma_x^2/n$$

### Sample Autocovariance Function

$$ \bar{\gamma}(h) = n^{-1}\sum\limits_{t=1}^{n-h}(x_{t+h} - \bar{x})(x_t - \bar{x})$$

### Sample Autocorrelation Function

$$\hat{p}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}$$

From the sample ACF, we can determine if the data comes from a random series or if correlations are significant at specific lags.

### Large-Sample Distribution of the ACF
If $$x_t$$ is white noise, then for $$n$$ large, the sample ACF is approximately normaly distributed with zero mean and standard deviation:

$$\sigma_{\hat{p}_x(h)} = \frac{1}{\sqrt{n}} $$

Thus, we can tell if a spike is statistically significant by establishing confidence intervals: 95%: $$\pm2/\sqrt{n}$$.

After transformations, the ACF of the residuals should fall within these limits.

### Sample Cross-Covariance Function

$$\hat{\gamma}_{xy}(h) = n^{-1} \sum\limits_{t=1}^{n-h}(x_{t+h}- \bar{x})(y_t - \bar{y}) $$

### Sample Cross-Correlation Function (CCF)

$$\hat{p}_{xy}(h) = \frac{\hat{\gamma}_{xy}(h)}{\sqrt{\hat{\gamma}_x(0)\hat{\gamma}_y(0)}} $$

For negative lags, remember $$\hat{\gamma}_{xy}(-h) = \hat{\gamma}_{xy}(h)$$

### Large-Sample Distribution of CCF
For $$n$$ large, the distribution of $$\hat{p}_{xy}(h)$$ is approximately normal with mean zero and s.d.:

$$\sigma_{\hat{p}_{xy}} = \frac{1}{\sqrt{n}}$$

If at least one of the processes is **independent white noise**.
If neither process is white, then the confidence intervals do not apply.

#### Prewhitening and Cross Correlation Analysis
Somehow prewhitening allows us to use the Large-Sample dist. I don't really get it.

# Chapter 2: Exploratory Data Analysis
Scroll through, not explicitly required, a couple things catch my eye that I should return to if I think its important:
AIC, BIC information criterions
Linear Regression
Detrending: Backshift Operator, Differences Operator
Smoothing
Lagged Scatterplot


# Chapter 3: ARIMA Models

## 3.2 Autoregressive Moving Average Models

Autoregressive models say that the current value of a series can be influenced by past values of the series.

### Autoregressive Model, AR(p)
An autoregressive model of order p is defined:

$$ x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + w_t $$

Where $$x_t$$ is stationary, $$w_t \sim wn(0,\sigma_w^2)$$, and $$\{\phi_i\}$$ are constants.

Mean of $$x_t$$ must be 0, or else replace $$x_t$$ with $$x_t-\mu$$, leading to either:

$$  x_t - \mu = \phi_1 (x_{t-1}-\mu) + \phi_2 (x_{t-2}-\mu) + \cdots + \phi_p (x_{t-p}-\mu) + w_t$$
or
$$ x_t = \alpha + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + w_t  $$
where $$\alpha = \mu(1 - \phi_1 - \cdots - \phi_p)$$

This can be simplified using the

### Backshift Operator B
$$Bx_t = x_{t-1} $$

So the autoregressive model becomes:
$$(1- \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p)x_t = w_t$$

### Autoregressive Operator
The autoregressive operator is defined:

$$\phi(B) =1- \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p  $$

So we can write an AR(p) model as:

$$\phi(B)x_t = w_t $$

#### Causality in AR(1) processes
Consider an AR(1) process: $$x_t = \phi x_{t-1} + w_t$$
By recursive substitution we get:
$$ x_t = \sum\limits_{j=0}^\infty \phi^j w_{t-j}$$
$$ \gamma(h) = \frac{\sigma_2^2 \phi^h}{1- \phi^2}$$
$$ p(h) = \phi^h$$

If $$\vert  \phi\vert  >1$$, the process will increase in magnitude without bound, and is therefore not stationary. However, it can be modified to not explode: $$x_t = \phi^{-1} x_{t+1} - \phi^{-1} w_{t+1}$$

This will give us a stationary (but non-causal) model.

So if we have $$\vert  \phi\vert  <1$$, the model is causal.

This backwards-substitution is good for demonstration, but doesn't apply to higher orders of p.

### Moving Average Model, MA(q)
Moving Average Models assume that the outcome of the process is the result of combinations of noise.

$$x_t = w_t + \theta_1 w_{t-1} + \theta_2 w_{t-2} + \cdots + \theta_q w_{t-q} $$

With Moving Average Operator:
$$\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q $$

We can write an MA(q) model as:
$$x_t = \theta(B)w_t $$

#### Non-uniqueness and Invertability in MA(1) Processes
Consider MA(1) process: $$x_t = w_t + \theta w_{t-1}$$. Then, $$E(x_t)=0$$,

$$\gamma(h) = \begin{cases}(1+\theta)^2\sigma^2_w & h=0,\\ \theta \sigma^2_w & h=1 \\ 0 & h>1 \end{cases} $$

With ACF:
$$p(h) = \begin{cases} \frac{\theta}{1+\theta^2} & h=1 \\ 0 & h>1 \end{cases} $$

Notice for an MA(1) model, $$p(h)$$ is the same for $$\theta$$ and $$1/\theta$$. i.e., $$\sigma_w^2 = 1, \theta = 5$$ is equivalent to $$\sigma_w^2 = 25, \theta = 1/5$$.

To mimic the AR(1) causality condition, we choose the representation with $$\vert  \theta\vert  <1$$.


### Autoregressive Moving Average Models, ARMA(p,q)
A time series is ARMA(p,q) if it is stationary and:

$$ x_t = \phi_1 x_{t-1} +\cdots + \phi_p x_{t-p} + w_t + \theta_1 w_{t-1}  + \cdots + \theta_q w_{t-q} $$

If mean is not 0, we correct in same way as for AR models:

$$ x_t = \alpha + \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t + \theta_1 w_{t-1} + \cdots + \theta_q w_{t-q} $$

Where $$\alpha = \mu(1 - \phi_1 - \cdots - \phi_p)$$

This can be written concisely as:

$$\phi(B)x_t = \theta(B)w_t $$

### Parameter Redundancy

Notice that we could complicate the model by multiplying each side by an extra polynomials $$\zeta(B)$$

Therefore, there are 3 possible problems we should look out for in ARMA models:

1. Parameter Redundancy
2. AR non-causality
3. Non-unique MA models

These can all be alleviated by looking at:

### The AR and MA polynomials
AR(p):
$$\phi(z) =1- \phi_1 z - \cdots - \phi_p z^p  $$
MA(q):
$$\theta(z) = 1 + \theta_1 z  + \cdots + \theta_q z^q $$


#### Solving Parameter Redundancy
To address problem 1, parameter redundancy, when we refer to an ARMA(p,q) model, we require that $$\phi(z)$$ and $$\theta(z)$$ have no common factors.

#### Solving Causality
An ARMA(p,q) model is said to be causal if we can write the time series as a one-sided linear process:

$$ x_t = \sum_{j=0}^\infty \psi_j w_{t-j} = \psi(B)w_t$$
where $$\sum_{j=0}^\infty \vert  \psi_j\vert  < \infty$$, and we set $$\psi_0=1$$

The general requirement for this is that $$\phi(z)$$ contains no roots inside the unit circle: $$\phi(z) \neq 0\big\vert  _{ \vert  z\vert  \leq 1}$$. Then,

$$\psi(z) = \sum\limits_{j=0}^\infty \psi_j z^j =  \frac{\theta(z)}{\phi(z)}, \quad \vert  z\vert  \leq 1 $$

Gives the coefficients of the one-sided representation.
#### Solving Uniqueness
An ARMA(p,q) model is said to be invertible if we can write it as the one-sided model:

$$\pi(B)x_t = \sum\limits_{j=0}^\infty \pi_j x_{t-j} = w_t $$

Where $$\sum_{j=0}^\infty \vert  \pi_j\vert  < \infty,$$ and we set $$\pi_0=1$$.

The general requirement for this is that $$\theta(z)$$ have no roots inside the unit circle. Then,

$$\pi(z) = \sum\limits_{j=0}^\infty  \pi_j z^j = \frac{\phi(z)}{\theta(z)}, \quad \vert  z\vert   \leq 1$$

Gives the coefficients to the one-sided equation. So we say: A process is invertible only when the roots of $$\theta(z)$$ lie outside the unit circle.

#### AR, MA polynomial review
To recap, we should always attempt to simplify the polynomials.

Second, if it is causal, then we may write the process as a linear process. (MA) model

Third, if it is invertible, then we may write the process as an AR model.

#### Method of Matching Coefficients
TODO

## 3.3 Difference Equations

### Difference Equations of Order 1
For diffe: $$u_n - \alpha u_{n-1}=0$$, with zero: $$z_0$$ and initial conditions $$u_0=c$$

$$u_n = \alpha^n c = (z_0^{-1})^n c$$

### Difference Equations of Order 2
For diffe: $$u_n - \alpha_1 u_{n-1} - \alpha_2 u_{n-2} = 0$$, initial conditions $$u_0 = c_1, u_1 = c_2$$.

Char eq: $$\alpha(z) = 1 - \alpha_1z -\alpha_2 z^2$$

With roots: $$z_1 \neq z_2$$

has solution: $$u_n = c_1 z_1^{-n} + c_2 z_2^{-n}$$

If they are equal: $$u_n = z_0^{-n}(c_1 + c_2 n)$$

## 3.4 Autocorrelation and Partial Autocorrelation

### Autocorrelation and MA(q)
Consider an MA(q) process: $$x_t = \theta(B)w_t$$, where $$\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$$:

Recall the MA(q) for a linear function of white noise variables.

$$\gamma(h) = \begin{cases} \sigma_w^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h} & 0 \leq h \leq q \\ 0 & h>q \end{cases}$$

Thus the ACF form an MA(q) process drops to 0 for $$h>q$$. (obvious if we divide by $$\gamma(0)$$)

For an AR or ARMA process, the ACF damps exponentially.

### Partial Autocorrelation and AR(p)
The partial autocorrelation for three random variables $$X,Y,Z$$ is the correlation between $$X,Y$$ with $$Z$$ conditioned out:

$$p_{XY\vert  Z} = \text{corr}(X,Y\vert  Z) = \text{corr}\{X-\hat{X},Y-\hat{Y}\} $$
Where $$\hat{X}, \hat{Y}$$ are the linear regressions of the variables on $$Z$$.

For a stationary, 0-mean process $$x_t$$, (if not 0 mean, just replace $$x_t$$ with $$x_t -\mu$$), $$\hat{x}_{t+h}$$ is the regression of $$\hat{x}(t)$$ on $$\{x_{t+h-1}, \cdots, x_{t+1}\}$$

Which will come out to something of the form: $$\hat{x}_{t+h} = \beta_1 x_{t+h-1} + \beta_2 x_{t+h-2} + \cdots + \beta_{h-1}x_{t+1}$$

The PACF of a stationary process $$x_t$$, denoted $$\phi_{hh}$$ is defined:

$$\phi_{hh} = \text{corr}(x_{t+h} - \hat{x}_{t+h}, x_t - \hat{x}_t)$$
Note, this means $$\phi_{11} = \text{corr}(x_{t+1},x_t) = p(1)$$

For an AR(p) process, $$\phi_{hh} = 0$$ for $$h>p$$

For an MA(q) process, the PACF damps exponentially.

### Summary Table

$$
\begin{array}{ \vert  c \vert   c \vert   c \vert   c \vert  }
  & AR(p) & MA(q) & ARMA(p,q) \\ \hline
 ACF & \text{Tails off} & \text{Cuts off after lag }q & \text{Tails off} \\ \hline
 PACF & \text{Cuts off after lag } p & \text{Tails off} & \text{Tails off} \\ \hline
\end{array}
$$

## 3.5 Forecasting
Forecasting wants us to predict the next value of a time series given the past values. For now we will assume the model is known.

If $$\mathbf{x} = \{ x_n, x_{n-1}, \cdots x_1\}$$, the minimum mean squared error predictor is the conditional expectation:

Notation note: The superscript is the number of previous values we will be using for a prediction.

$$x^n_{n+m} = E(x_{n+m}\vert  \mathbf{x})$$

### Best Linear Predictors for Stationary Processes
First, we look at linear predictors, of the form:

$$x^n_{n+m} = \alpha_0 + \sum\limits_{k=1}^n \alpha_k x_k $$

Given data $$\mathbb{x}$$, the best linear predictor, $$x_{n+m}^n$$ for $$m\geq 1$$ is found by solving the *prediction equations*:
$$E[(x_{n+m}-x^n_{n+m})x_k]=0, \qquad k= 0,1,\cdots,n $$

where $$x_0 = 1$$, for $$\alpha_0, \alpha_1, \cdots, \alpha_n$$

#### Mean and the Prediction Equations
For $$k=0$$:
$$x_0 = 1 \implies E[x_{n+m}] = E[x^n_{n+m}]$$

And since it is stationary and $$E(x_t) = \mu$$, and replacing $$x_{n+m}^n$$ with the general equation and taking expectation, we get:

$$\mu = \alpha_0 + \sum\limits_{k=1}^n a_k \mu \implies x_{n+m}^n = \mu + \sum\limits_{k=1}^n a_k (x_k - \mu) $$
Thus, there is no loss in generality in considering $$\mu,\alpha_0 = 0$$

#### One Step Ahead Prediction
Given $$\{x_1, \cdots, x_n \}$$, we wish to forecast the value at $$x_{n+1}$$ The BLP will have the form:

$$x_{n+1}^n = \phi_{n1}x_n + \phi_{n2}x_{n-1} + \cdots+ \phi_{nn}x_1 $$

This notation shows the dependence of the coefficients on $$n$$. $$\alpha_k = \phi_{n,n+1-k}$$

These must satisfy:

$$E[(x_{n+1} - \sum\limits_{j=1}^n \phi_{nj}x_{n+1-j})x_{n+1-k}] =0 \qquad k=1,\cdots,n $$

Equivalently:
$$\sum\limits_{j=1}^n \phi_{nj} \gamma(k-j) = \gamma(k) \qquad k=1,\cdots,n $$

### Matrix Formulate on Prediction
Which in matrix notation is:
$$\Gamma_n \mathbf{\phi}_n = \mathbf{\gamma}_n$$

Where $$\Gamma_n = \{\gamma(k-j)\}_{j,k=1}^n$$ is an $$n\times n$$ matrix, $$\mathbf{\phi}_n = (\phi_{n1}, \cdots , \phi_{nn})^\top$$ is an $$n \times 1$$ vector, and $$\mathbf{\gamma}_n = (\gamma(1), \cdots, \gamma(n))^\top$$ is an $$n \times 1$$ vector.

Assuming $$\Gamma_n$$ isn't singular (in which case there is still a unique prediction, it's just harder to find).

$$\mathbf{\phi}_n = \Gamma_n^{-1}\mathbf{\gamma}_n $$

and

$$x_{n+1}^n = \mathbf{\phi}_n^\top \mathbf{x} $$

With Mean square prediction error:

$$P^n_{n+1} = E(x_{n+1} - x_{n+1}^n)^2 = \gamma(0) - \mathbf{\gamma}^\top_n\Gamma_n^{-1}\mathbf{\gamma}_n $$

#### AR(p) model predictions
For AR(p) models, prediction is easy, since we aready have the evolution of the function. Simply shift everything one over from the model: $$x^n_{n+1} = \phi_1 x_n + \cdots + \phi_p x_{n-p+1}$$

### Durbin-Levinson Algorithm
Sometimes a large matrix multiplication is intractible, but we can find the solution iteratively as follows:
Start with $$\phi_{00} = 0, P^0_1 = \gamma(0)$$ then,

For $$n \geq 1$$:

$$\phi_{nn} = \frac{p(n) - \sigma_{k=1}^{n-1} \phi_{n-1,k} p(n-k)}{1 - \sigma_{k=1}^{n=1} \phi_{n-1,k}p(k)}, \quad P_{n+1}^n = P_n^{n-1}(1 - \phi_{nn}^2) $$

Where for $$n \geq 2$$:

$$\phi_{nk} = \phi_{n-1,k}-\phi_{nn}\phi_{n-1,n-k} \quad k=1,\cdots,n-1$$

Generally, the Standard Error of the one-step ahead forecast is the square root of:

$$P^n_{n+1}=\gamma(0)\prod\limits_{j=1}^n [1 - \phi_{jj}^2]  $$

### Iterative Solution for the PACF:
The Durbin-Levinson also provides an iterative solution for the PACF.

### M-step ahead prediction
Same shit, we now want to find constants for the linear m-step ahead predictor, called $$\phi^{(m)}_{ni}$$:

$$x^n_{n+m} = \phi_{n1}^(m)x_n + \cdots + \phi^{(m)}_{nn}x_1 $$
That solve the prediction equations:

$$\sum\limits_{j=1}^n \phi^{(m)}_{nj}E(x_{n+1-j}x_{n+1-k})= E(x_{n+m} x_{n+1-k}), \qquad k=1,\cdots,n $$
or
$$\sum\limits_{j=1}^{n} \phi_{nj}^{(m)} \gamma(k-j)= \gamma(m+k-1), \qquad k=1,\cdots,n $$

In Matrix Form:

$$\Gamma_n \mathbf{\phi}_{n}^{(m)} = \mathbf{\gamma}_n^{(m)} $$

Where $$\mathbf{\gamma}_n^{(m)} = (\gamma(m),\cdots,\gamma(m+n-1))^\top$$, and $$\mathbf{\phi}_n^{(m)} = (\phi_{n1}^{(m)},\cdots, \phi_{nn}^{(m)})^\top$$

The mean square m-step-ahead prediction error is:

$$P_{n+m}^n  = E(x_{n+m}-x_{n+m}^n)^2 = \gamma(0)-\mathbf{\gamma}_n^{(m)^\top} \Gamma_n^{-1}\mathbf{\gamma}_n^{(m)}$$

### The Innovations Algorithm
Instead of doing the big matrix inversion again, we can iteratively calculate one-step-ahead predictors, $$x^t_{t+1}$$ and mean-squared-errors, $$P^t_{t+1}$$ with:

$$x_1^0 =0 \quad P_1^0 = \gamma(0)$$

$$x^t_{t+1} = \sum\limits_{j=1}^t \theta_{tj}(x_{t+1-j} - x_{t+1-j}^{t-j}), \quad t = 1,2,\cdots $$

$$P_{t+1}^t  = \gamma(0)- \sum\limits_{j=1}^{t-1} \theta^2_{t,t-j}P_{j+1}^j, \quad t= =1,2,\cdots $$

Where for $$j=0,1,\cdots,t-1$$
$$\theta_{t,t-j} = (\gamma(t-j)- \sum\limits_{k=0}^{j-1}\theta_{j,j-k}\theta_{t,t-k}P_{k+1}^k) /P_{j+1}^j$$

The final predictions for $$x^n_{n+1}$$ and $$P_{n+1}^n$$ are made at step $$t=n$$ and given by:

$$x_{n+m}^n = \sum_{j=m}^{n+m-1} \theta_{n+m-1,j}(x_{n+m-j}-x^{n+m-j-1}_{n+m-j})$$

$$P^n_{n+m} = \gamma(0) - \sum_{j=m}^{n+m-1} \theta^2_{n+m-1,j}P^{n+m-j-1}_{n+m-j} $$

Innovations Algorithm is good for predicting MA(p) processes

### Forecasting General ARMA Processes:
Assume $$x_t$$ is a causal and invertible ARMA(p,q) process, $$\phi(B)x_t = \theta(B)w_t$$, where $$w_t \sim iid N(0, \sigma_w^2)$$ In the non-zero mean case: $$E[x_t] =\mu_x$$, simply replace $$x_t$$ with $$(x_t - \mu_x)$$ in the model.

Best Linear Predictor:
$$x_{n+m}^n = E[x_{n+m}\vert  (x_n,\cdots,x_1)] $$

It is easier to calculate assuming we have a complete history of the process, so we say

$$\tilde{x}_{n+m} = E(x_{n+m}\vert  x_n,x_{n=1},\cdots) $$

For large samples, $$\tilde{x}_{n+m} \approx x_{n+m}^n$$

We write $$x_{n+m}$$ in causal and invertible forms:

$$x_{n+m} = \sum\limits_{j=0}^\infty \psi_j w_{n+m-j}, \quad \psi_0=1 $$
$$w_{n+m} = \sum\limits_{j=0}^\infty \pi_j x_{n+m-j}, \quad \pi_0=1 $$

Taking conditional expectations on the causal model, we get:

$$\tilde{x}_{n+m} = \sum\limits_{j=0}^\infty \psi_j \tilde{w}_{n+m-j} = \sum\limits_{j=m}^\infty \psi_j w_{n+m-j}$$,

Since,

$$\tilde{w}_t = E(w_t\vert   x_n,x_{n-1},\cdots) = \begin{cases}0 & t> n \\ w_t & t \leq n \end{cases} $$

Similarly, taking conditionals of the invertible expression, we have

$$0 = \tilde{x}_{n+m} + \sum_{j=1}^\infty \pi_j \tilde{x}_{n+m-j} $$
$$ \tilde{x}_{n+m} = - \sum_{j=1}^{m-1}\pi_j \tilde{x}_{n+m-j} - \sum_{j=m}^\infty \pi_j x_{n+m-j}$$

Since we already have all of the information about $$x_t$$ in the conditional for $$t \leq n$$. We can recursively predict like this for $$m=1,2,\cdots$$

Mean-square prediction error then becomes:

$$P_{n+m}^n = E(x_{n+m} - \tilde{x}_{n+m}) =\sigma_w^2 \sum\limits_{j=0}^{m-1} \psi_j^2 $$

Notice that for a fixed sample size $$n$$, the prediction errors are correlated, for $$k \geq 1$$,

$$E[(x_{n+m}-\tilde{x}_{n+m})(x_{n+m+k} - \tilde{x}_{n+m+k})] = \sigma_w^2\sum\limits_{j=1}^{m-1} \psi_j \psi_{j+k} $$

Notice from the Causal form, we can see that
$$\tilde{x}_{n+m} \to \mu_x$$

And from the limit of the Mean-square prediction error eq:

$$P_{n+m}^n \to \sigma_w^2 \sum\limits_{j=0}^\infty \psi_j^2 = \gamma_x(0) = \sigma^2_x $$

So long range forecasts tend to the mean with variance of white noise.

### Truncated Prediction for ARMA
For AR(p) models, we already can create an exact predictor as noted above.

For ARMA(p,q) models, the truncated predictors for $$m=1,2,\cdots$$ are

$$\tilde{x}_{n+m}^n = \phi_1 \tilde{x}_{n+m-1}^n + \cdots + \phi_p \tilde{x}^n_{n+m-p} + \theta_1 \tilde{w}_{n+m-1}^n + \cdots + \theta_q\tilde{w}_{n+m-q}^n$$

where $$\tilde{x}_t^n = x_t$$ for $$1\leq t\leq n$$ and $$\tilde{x}^n_t=0$$ for $$t\leq 0$$. The truncated prediction errors are given by: $$\tilde{w}_t^n=0$$ for $$t\leq 0$$ or $$t >n$$ and

$$ \tilde{w}_t^n = \phi(B)\tilde{x}_t^n - \theta_1 \tilde{w}_{t-1}^n - \cdots - \theta_q \tilde{w}_{t-q}^n$$

for $$1\leq t \leq n$$.

#### Backcasting
Backcasting can be performed via the same equations, but flipping the orders of all of the indices.

## 3.6 Estimation
Now, what if we have $$n$$ observations from an ARMA(p,q) model where we know $$p,q$$, but not $$\phi(B),\theta(B), \sigma_w^2$$. How could we find them?

### Method of Moments
Equate population moments to sample moments. This produces optimal estimators for AR(p) models.

### Yule-Walker Equations
For an AR(p) model: $$x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t$$

The Yule-Walker equations are:
$$ \gamma(h)= \phi_1 \gamma(h-1) + \cdots + \phi_p + \gamma(h-p), \quad h=1,2,\cdots,p $$
$$\sigma_w^2 = \gamma(0) - \phi_1 \gamma(1) - \cdots - \phi_p \gamma(p) $$

In Matrix notation:

$$\Gamma_p \mathbf{\phi} = \mathbf{\gamma}_p, \quad \sigma_w^2 = \gamma(0) - \mathbf{\phi}^\top \mathbf{\gamma}_p  $$

Where $$\Gamma_p = \{\gamma(k-j)\}_{j,k=1}^p$$, $$\mathbf{\phi} = (\phi_1, \cdots , \phi_p)^\top$$, and $$\mathbf{\gamma}_p = (\gamma(1), \cdots, \gamma(p))^\top$$.

Using the method of moments, we replace $$\gamma(h)$$ by the sample $$\hat{\gamma}(h)$$ and solve the above equations. These are usually called the Yule-Walker estimates.

Sometimes, it's easier to work with the sample ACF, so by factoring $$\hat{\mathbf{\gamma}}_p$$:

$$\hat{\mathbb{\phi}} = \hat{\mathbf{R}}_p^{-1}\hat{\mathbf{p}}_p, \quad \hat{\sigma}_w^2 = \hat{\gamma}(0)[1- \hat{\mathbf{p}}^\top_p \hat{\mathbf{R}}_p^{-1}]\hat{\mathbf{p}}_p $$

Where $$\hat{\mathbf{R}}_p = \{ \hat{p}(k-j)\}_{j,k=1}^p$$, and $$\hat{\mathbf{p}}_p = (\hat{p}(1),\cdots \hat{p}(p))^\top$$

#### Large Sample Results for Yule-Walker Estimators
For AR(p) models, the asymptotic behavior of the estimators is:

$$\sqrt{n}(\hat{\mathbf{\phi}}-\mathbf{\phi})\to N(0, \sigma_w^2 \Gamma_p^{-1}), \qquad \hat{\sigma}_w^2 \to \sigma_w^2 $$

If we don't want to invert $$\Gamma,R$$, we can run the Durbin-Levinson algorithm, replacing $$\gamma(h)$$ by the sample autocovariance. Running the Algorithm will also give us the sample PACF.

#### Large Sample Distribution of the PACF
For a causal AR(p) process, asymptotically

$$ \sqrt{n}\hat{\phi}_{hh} \to N(0,1), \quad \text{for } h>p$$

### Maximum Likelihood and Least Squares Estimation

Likelihood function is the chance of seeing data given parameters based on our model. We take the log likelihood to turn products into sums (log is convex). And then take the derivative and solve for 0 to get the max.

### AR(p)
For AR(p), we can explicitly write the likelihood and solve numerically.

### MA(q), ARMA(p,q)
It is difficult to write the likelihood as an explicit function of the parameters, so we write it in terms of the innovations, or one-step-ahead prediction erorrs, $$x_t - x_t^{t-1}$$

Let $$\mathbf{\beta} = (\mu,\phi_1,\cdots, \phi_p,\theta_1,\cdots,\phi_q)^\top$$ is the (p+q+1) dimensional vector of the model paramters.

$$L(\mathbf{\beta},\sigma_w^2) = \prod\limits_{t=1}^n f(x_t\vert   x_{t-1},\cdots, x_1)$$

The conditional distribution of $$x_t$$ given $$x_{t-1},\cdots,x_1$$ is Gaussian with mean $$x_t^{t-1}$$ and variance $$P_t^{t-1}$$.

From the innovations equations, we know: $$P_t^{t-1} = \gamma(0)\prod_{j=1}^{t-1}(1- \phi_{jj}^2)$$. For ARMA models, $$\gamma(0)=\sigma_w^2 \sum_{j=0}^\infty \psi_j^2$$. Now we can write

$$P_t^{t-1} = \sigma_w^2 \Big\{\Big[\sum\limits_{j=0}^\infty \psi_j^2 \Big]\Big[ \prod\limits_{j=1}^{t-1} (1- \phi_{jj}^2)\Big] \Big\} = \sigma_w^2 r_t$$

$$r_t$$ is the bit in braces. Notice $$r_{t+1} = (1-\phi_{tt}^2)r_t$$, $$r_1 = \sigma_{j=0}^\infty \psi_j^2$$. Thus these are functions only of the model parameters. We can now write everything as a multivariate gaussian:

$$L(\mathbf{\beta},\sigma_w^2) = (2\pi \sigma_w^2)^{-n/2}[r_1(\mathbf{\beta})\cdots r_n(\beta)]^{-1/2}\exp\Big[-\frac{S(\mathbf{\beta})}{2\sigma_w^2} \Big] $$

$$S(\mathbf{\beta}) = \sum\limits_{t=1}^n \Big[ \frac{(x_t - x_t^{t-1}(\mathbf{\beta}))^2)}{r_t(\mathbf{\beta})}\Big] $$

We have:

$$\hat{\sigma}_w^2 = n^{-1}S(\hat{\mathbf{\beta}}) $$

Where $$\hat{\mathbf{\beta}}$$ minimizes the log likelihood:

$$l(\mathbf{\beta}) = \log[n^{-1}S(\mathbf{\beta})] + n^{-1}\sum\limits_{t=1}^n \log r_t(\mathbf{\beta})$$

### Numerical Methods
Do I need to know this? Come back if I see anything, I don't think so

### Large Sample Distribution of the Estimators
Under appropriate conditions, for causal and invertible ARMA processes, the maximum likelihood, the unconditional least squares, and the conditional least squares estimators, each initialized by the method of moments estimator, all provide optimal estimators of $$\sigma_w^2$$ and $$\mathbf{\beta}$$, in the sense that $$\hat{\sigma}_w^2$$ is consistent, and the asymptotic distribution of $$\hat{\mathbf{\beta}}$$ is the best asymptotic normal distribution: In the limit
$$\sqrt{n}(\hat{\mathbf{\beta}}-\mathbf{\beta}) \to N(0,\sigma_w^2 \mathbf{\Gamma}^{-1}_{p,q}) $$

The asymptotic variance-covariance matri of the estimator $$\hat{\mathbf{\beta}}$$ is the inverse of the information matrix: $$(p+q)\times(p+q)$$ matrix $$\mathbf{\Gamma}_{p,q}$$, defined:

$$\mathbf{\Gamma}_{p,q} = \begin{pmatrix}
\Gamma_{\phi\phi} & \Gamma_{\phi \theta} \\
\Gamma_{\theta\phi} & \Gamma_{\theta\theta}
\end{pmatrix} $$

Where the $$p \times p$$ matrix $$\Gamma_{\phi \phi}$$ is $$\Gamma_{\phi \phi}(i,j) = \gamma_x(i-h)$$ for AR(p) process $$\phi(\beta)x_t = w_t$$. Similarly, $$q \times q$$ matrix $$\Gamma_{\theta \theta}(i,j) = \gamma_y(i-j)$$ from an AR(q) process $$\theta(\beta) = w_t$$.

Finally, $$p \times q$$ matrix $$\Gamma_{\phi\theta}(i,j) = \gamma_{xy}(i-j)$$, the covariance between the two different processes. and $$\Gamma_{\theta \phi} = \Gamma^\top_{\phi\theta}$$ is $$(q \times p)$$.

### Overfitting
We can see that by fitting higher parameter ARMA models to lower parameter ARMA processes, the variance of the estimator will increase, making the estimator less precise.

### Specific Asymptotic Variances

AR(1):
$$\hat{\phi} \sim AN[\phi, n^{-1}(1-\phi^2)] $$

AR(2):

$$\begin{pmatrix}\hat{\phi}_1\\ \hat{\phi}_2 \end{pmatrix} \sim AN\Big[ \begin{pmatrix}\phi_1\\ \phi_2 \end{pmatrix}, n^{-1} \begin{pmatrix}
1 - \phi_2^2 & -\phi_1 (1+ \phi_2) \\ \text{sym} & 1- \phi_2^2
\end{pmatrix} \Big]$$

There are more if I need them.

### Bootstrapping.
I don't think this will come up.

## 3.7 Integrated Models for Nonstationary Data

### Differencing Operator:
$$\nabla = (1- B)  $$

Makes Random walks and Random walks with drift stationary. If the drift is a k'th order polynomial, the series will be come stationary with $$\nabla^k$$

### ARIMA(p,d,q) models
A process is said to be ARIMA(p,d,q) if

$$\nabla^d x_t = (1-B)^d x_t $$
is ARMA(p,q). We can write the model as

$$\phi(B)(1-B)^d x_t = \theta(B) $$

If $$E(\nabla^dx_t)= \mu$$, we write the model as:

$$\phi(B)(1-B)^d x_t = \delta + \theta(B) $$

Where $$\delta = \mu(1-\phi_1 - \cdots \phi_p)$$.

### Working with ARIMA Models
Since $$y_t = \nabla^d x_t$$ is ARMA, we can use regular ARMA methods for forecasts of $$y_t$$, then transform them back into forecasts for $$x_t$$ by undoing the differencing.

It's harder to get the prediction errors $$P^n_{n+m},$$ but for large $$n$$, the same formula derived for ARMA models wors well:

$$P^n_{n+m} = \sigma_w^2 \sum\limits_{j=0}^{m-1}\psi_j^{* 2} $$

where $$\psi_j^{* }$$ is the coefficient of $$z^j$$ in $$\phi^* (z) = \theta(z)/\phi(z)(1-z)^d$$

Unlike the ARMA case, ARIMA models have prediction errors that increase without bound.


## 3.8 Building ARIMA Models

### Time-Domain Analysis
If the variance of the process appears to change with time, we might have to transform the data to stabilize the variance.

Differencing order $$d$$ can be inferred by whether or not the time series is increasing after different orders of $$d$$. Don't overdifference.

Look at sample ACF, PACF to get an idea of $$p,d$$. It can be difficult to tell if it is tailing off or cutting off.

### Model Diagnostics
Create a time plot of innovations (residuals): $$x_t - \hat{x}_t^{t-1}$$ or standardized innovations:

$$e_t = (x_t - \hat{x}_t^{t-1})/\sqrt{\hat{P}_t^{t-1}}$$

Where $$\hat{x}_t^{t-1}$$ is one-step-ahead prediction of $$x_t$$, and $$\hat{P}_t^{t-1}$$ is the estimated one-step-ahead error variance.

Well-fitted model should produce standardized residuals that are an iid sequence with mean zero and variance 1.

To test for the independence assumption of residuals, we could plot the sample autocorrelation: $$\hat{p}_e(h)$$ of the residuals.

$$\hat{p}_e(h)$$ should be distributed with zero mean and variance $$1/n$$.

#### Ljung-Box-Pierce Q-statistic

$$ Q = n(n+2) \sum\limits_{h=1}^H \frac{\hat{p}_e^2(h)}{n-h} $$

Typically, $$H=20$$. Under the null hypothesis of model adequacy, asymptotically, $$Q \sim \chi^2_{H-p-q}$$.

## 3.9 Regression with Autocorrelated errors

Usually, we assume that our model has uncorrelated errors $$w_t$$, however, the errors might actually be correlated. Consider the regression model

$$y_t - \sum\limits_{j=1}^r \beta_j z_{tj} + x_t $$

Where $$x_t$$ is a process with some covariance function $$\gamma_x(s,t)$$. In ordinary least squares, we assume $$x_t$$ is Gaussian noise.

Consider the vector model:

$$\textbf{y} = Z \textbf{B} + \textbf{x} $$

If $$\Gamma = {\gamma_x(s,t)}$$, then

$$\Gamma^{-1/2}\mathbf{y} = \Gamma^{-1/2}Z \mathbf{B} + \Gamma^{-1/2}\mathbf{x} = \mathbf{y}^* = Z^* + \mathbf{\delta}$$

Where the covariance matrix of $$\mathbf{\delta}$$ is the identity.

Thus, we can derive the weighted least squares estimate, and the variance-covariance matrix of the estimator:

$$\hat{\mathbf{\beta}}_w = (Z^\top \Gamma^{-1} Z)^{-1} Z^\top \Gamma^{-1}\mathbf{y}$$

$$\text{var}(\hat{\mathbf{\beta}}_w) = (Z^\top \Gamma^{-1} Z)^{-1} $$

If $$x_t$$ is white noise, then $$\Gamma = \sigma^2 I$$ and these reduce to the usual least squares equations.

To deal with autocorrelated errors:

1. First, run an ordinary regression of $$y_t$$ on $$z_{t,1}, \cdots ,z_{tr}$$ (acting as if the errors are uncorrelated). Retain the residuals, $$\hat{x}_t = y_t - \sum\_{j=1}^r \hat{\beta}_j x_{tj}$$

2. Identify ARMA models for the residuals $$\hat{x}_t$$

3. Run weighted least squares on the regression model with autocorrelated errors using the models we found

4. Inspect the residuals $$\hat{w}_t$$ for whiteness, and adjust model accordingly


# Chapter 4: Spectral Analysis and Filtering

## 4.2 Cyclic Behavior and Periodicity

### General Periodic Processes

A periodic component is defined:

$$x_t = A\cos (2\pi \omega t + \phi) $$

Where $$A$$ is the amplitude and $$\phi$$ is the phase.

or

$$x_t = U_1 \cos(2 \pi \omega t) + U_2 \sin(2\pi \omega t) $$

$$U_1,U_2$$ are usually taken to be normally distributed random variables, so $$A- \sqrt{U_1^2 + U_2^2}$$, and $$\phi = \text{tan}^{-1}(-U_2/U_1)$$.

$$U_1, U_2$$ are independent, standard normal random variables IFF $$A^2$$ is chi-squared with 2 dof, and $$\phi$$ is uniformly distributed on $$[-\pi,\pi]$$.

We can thus define a general periodic process as the sum of individual components:

$$ x_t = \sum\limits_{k=1}^q [U_{k1} \cos(2 \pi \omega_k t) + U_{k2}\sin(2 \pi \omega_k t)] $$

where $$U_{k1},U_{k2}, \quad k=1,\cdots,q$$ are independent zero-mean random variables with variance $$\sigma_k^2$$, and the $$w_k$$ are distinct frequencies.

We can then derive the autocovariance function

$$\gamma(h) = \sum\limits_{k=1}^q \sigma_k^2 \cos(2\pi \omega_k h) $$

Notice for $$h=0$$, the variance of the function is the sum of the component variances:

$$\gamma(0) = E(x_t^2) = \sum\limits_{k=1}^q \sigma_k^2 $$

### Scaled Periodogram
The scaled periodogram is given by:

$$ P(j/n) = \Big(\frac{2}{n} \sum\limits_{t=1}^n x_t \cos(2 \pi t j/n) \Big)^2 + \Big(\frac{2}{n} \sum\limits_{t=1}^n x_t \sin(2 \pi t j/n) \Big)^2 $$

Is the measure of the squared correlation of the data with sinusoids oscillating at $$w_j = j/n$$, $$j$$ cycles in the $$n$$ times points.

## 4.3 The spectral Density

### Spectral Representation of a Stationary Process
In nontechnical terms, any stationary time series may be thought of, approximately, as the random superposition of sines and cosines oscillating at various frequencies.

### The Spectral Density
If the autocovariance function $$\gamma(h)$$ of a stationary process satisfies:

$$\sum\limits_{h = -\infty}^\infty \vert  \gamma(h)\vert   < \infty$$

then it has the representation

$$\gamma(h) = \int_{-1/2}^{1/2} e^{2\pi i \omega h} f(\omega)d\omega $$

as the inverse transform of the spectral density, which has the representation

$$f(\omega) = \sum\limits_{h = - \infty}^\infty \gamma(h) e^{-2\pi i \omega h}, \quad -1/2 \leq \omega \leq 1/2 $$

The spectral Density, as defined above, is a probability density function. It is also symmetric: $$f(\omega) = f(-\omega)$$. Thus we can plot only half of it.

Conveniently for $$h=0$$,
$$ \gamma(0) = \text{var}(x_t) = \int_{-1/2}^{1/2} f(\omega)d \omega$$

For stationary processes, $$\gamma(h)$$ and $$f(\omega)$$ provide equivalent information.

We can also express all of these formulas in terms of $$\lambda = 2 \pi \omega$$.

### Linear Filters
Linear filters in the time domain are usually specified by a convolution with coefficients $$a_j$$. Thus:

$$y_t = \sum\limits_{j=-\infty}^\infty a_j x_{t-j}, \quad \sum\limits_{j=-\infty}^\infty \vert  a_j\vert   < \infty $$

Coefficients are the impulse response, and the frequency response is defined:

$$A(\omega)  = \sum\limits_{j=-\infty}^\infty a_j e^{-2 \pi i \omega j}$$

#### Output Spectrum of a Filtered Time Series
Under the established conditions, the spectrum of the filtered output $$y_t$$ is related to the spectrum of the input $$x_t$$ by:

$$f_y(\omega) = \vert  A(\omega)\vert  ^2 f_x(\omega) $$

### Spectral Density of ARMA
If $$x_t$$ is an ARMA(p,q), $$\phi(B)x_t = \theta(B)w_t$$, its spectral density is given by

$$f_x(\omega) = \sigma_w^2 \frac{\vert  \theta(e^{-2 \pi i \omega})\vert  ^2}{\vert  \phi(e^{-2\pi i \omega})\vert  ^2} $$

Where $$\phi(z) = 1- \sum_{k=1}^p \phi_l z^k$$ and $$\theta(z) = 1 + \sum_{k=1}^q \theta_k z^k$$.

Note: $$\vert  p(z)\vert  ^2 = p(z)* \bar{p(z)}$$. The magnitude squared is the product of the complex conjugates.

Spectral Density can also be obtained from first principles by calculating $$\gamma(h)$$, then performing the Fourier Transform.

### Causality in the Spectral Domain.
The causal counterparts of data are related to the complex conjugates:

Consider $$x_t = 2 x_{t-1} + w_t$$, and $$w_t \sim \text{iid} N(0,\sigma_w^2)$$ with spectral density

$$f_x(\omega) = \sigma_w^2\vert  1-2e^{-2\pi i \omega}\vert  ^{-2} $$

Can be transformed:

$$\vert  1- 2e^{-2\pi i \omega}\vert   = \vert  1-2e^{2\pi i \omega}\vert   = \vert  (2 e^{2\pi i \omega})(\frac{1}{2}e^{-2\pi i \omega} -1)\vert   = 2\vert  1-\frac{1}{2}e^{-2\pi i \omega}\vert   $$

so

$$f_x(\omega) = \frac{1}{4}\sigma_w^2 \vert  1-\frac{1}{2} e^{-2\pi i \omega}\vert  ^{-2} $$

Which implies $$x_t = \frac{1}{2}x_{t-1} + v_t, \quad v_t \sim N(0, \frac{1}{4} \sigma_w^2)$$ is an equivalent model.

## 4.4 Periodogram and the Discrete Fourier Transform

The Periodogram is a sample-based concept and the spectral density is a population-based concept. The DFT relates them rigorously.

### Discrete Fourier Transform
Given data $$x_1, \cdots, x_n$$, we define the discrete Fourier transform (DFT) to be

$$d(w_j) = n^{-1/2} \sum\limits_{t=1}^n x_t e^{-2 \pi i \omega_j t} $$

for $$j= 0, 1, \cdots, n-1$$, where the frequencies $$\omega_j = j/n$$ are called the Fourier or Fundamental frequencies.

The inverse is defined:

$$x_t = n^{-1/2} \sum\limits_{j=0}^{n-1} d(\omega_j)e^{2\pi i \omega_j t} $$

### Periodogram
The Periodogram is the squared modulus of the DFT:

$$ I(\omega_j) = \vert  d(\omega_j)\vert  ^2$$

With a little algebra, it can be analogously defined in terms of the sample autocovariance:

$$ I(\omega_j) = \sum\limits_{h = -(n-1)}^{n-1} \hat{\gamma}(h)e^{-2\pi i \omega_j h}$$

The scaled periodogram is then defined: $$P(\omega_j) = \frac{4}{n}I(\omega_j)$$

### Cosine and Sine Transforms
Sometimes it is useful to work with the real and imaginary parts of the DFT separately. Thus we have the cosine transform:

$$d_c(\omega_j) = n^{-1/2} \sum\limits_{t=1}^n x_t \cos(2 \pi \omega_j t) $$

and the sine transform

$$d_s(\omega_j) = n^{-1/2} \sum\limits_{t=1}^n x_t \sin(2 \pi \omega_j t) $$

where $$\omega_j = j/n$$ for $$j=0,1,\cdots, n-1$$.

These are related to the DFT and periodogram by:

$$d(\omega_j) = d_c(\omega_j) - i d_s(\omega_j) $$

and

$$I(w_j) = d_c^2(\omega_j)+ d_s^2(\omega_j) $$

### Large Sample Properties of the Periodogram

$$E[I(\omega_j)]=  \sum\limits_{h = -(n-1)}^{n-1} \Big( \frac{n - \vert  h\vert  }{n}\Big)\gamma(h) e^{-2\pi i \omega_j h} $$

For any given $$\omega \neq 0$$, we say $$\omega_{j:n}$$ is the closest $$\omega_j$$ in our DCT to the desired frequency $$\omega$$. Thus, as $$n \to \infty$$, $$\omega_{j:n} \to \omega$$, so

$$E[I(\omega_{j:n})] \to f(\omega) = \sum\limits_{-\infty}^\infty \gamma(h)e^{-2\pi i h \omega} $$

This means that the spectral density is the long-term average of the periodogram.

### Distribution of the Periodogram Ordinates
If
$$x_t = \sum\limits_{j=-\infty}^\infty \psi_j w_{t-j}, \quad \sum\limits_{j = -\infty}^\infty \vert  \psi_j\vert   < \infty $$

where $$w_t \sim iid(0,\sigma_w^2)$$, and

$$\theta = \sum\limits_{h = - \infty}^\infty \vert  h\vert   \vert  \gamma(h)\vert   < \infty $$

then for any collection of $$m$$ distinct frequencies $$w_j \in (0,1/2)$$ with $$\omega_{j:n} \to \omega_j$$

$$\frac{2I(\omega_{j:n})}{f(\omega_j)} \to \text{iid} \chi_2^2 $$

provided $$f(\omega_j) > 0$$, for $$j=1,\cdots,m$$.

#### Confidence Intervals
With $$\chi_v^2(\alpha)$$ be the lower $$\alpha$$ probability tail for the chi-squared distribution with $$v$$ degrees of freedom:

$$\text{Pr}\{\chi_v^2 \leq \chi_v^2(\alpha)\} = \alpha $$

Then a $$100(1-\alpha)$$% confidence interval is given by

$$\frac{2 I(\omega_{j:n})}{\chi_2^2(1-\alpha/2)} \leq f(\omega) \leq \frac{2 I(\omega_{j:n})}{\chi_2^2(\alpha/2)} $$

## 4.8 Linear Filters

### Convolution Definitions
If we have series $$y_t$$ and $$x_t$$ with spectral densities $$f_y(\omega)$$ and $$f_x(\omega)$$, and

$$ y_t = \sum\limits_{j=-\infty}^\infty a_j x_{t-j}, \quad \sum\limits_{j=-\infty}^\infty \vert  a_j\vert  < \infty$$

then

$$ f_y(\omega) = \vert  A(\omega)\vert  ^2 f_x(\omega) $$

where

$$ A(\omega) = \sum\limits_{j=-\infty}^\infty a_j e^{-2\pi i \omega j}$$

is the frequency response function of the filter.

### Matrix Formulation
The spectral matrix of the filtered output $$\mathbf{y}_t$$ is related to the spectrum of the input $$\mathbf{x}_t$$ by

$$f_y(\omega) = \mathcal{A}(\omega)f_x(\omega)\mathcal{A}^* (\omega) $$
where the matrix frequency response function $$\mathcal{A}(\omega)$$ is defined by

$$\mathcal{A}(\omega) = \sum\limits_{j=-\infty}^\infty A_j e^{-2\pi i \omega j}  $$


# Appendix A

## A.1 Convergence Modes

## A.2 Central Limit Theorems

### Model Parameter change
A sequence of random vectors is asymptotically normal

$$ \mathbf{x}_n \sim AN(\boldsymbol\mu_n , \Sigma_n)$$

iff

$$\mathbf{c}'\mathbf{x}_n \sim AN(\mathbf{c}'\boldsymbol\mu_n, \mathbf{c}'\Sigma_n \mathbf{c})$$

# Appendix C

## C.1 Spectral Representation Theorem




# Handouts

## Big O, little o

Big O - of the same order as
little o - ultimately smaller than
## Probability Spaces and Hilbert Spaces

## Fourier Series
