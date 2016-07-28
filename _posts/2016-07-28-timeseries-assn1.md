---
layout: post
title: "Time Series Analysis Assignment 1"
category: "Time Series Analysis"
---

To preface, here are all of the packages that I used to complete the assignment:

```python
import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
%matplotlib inline
```

# P1
Generate $$n=1000$$ observations from an MA(1) process. Fit an ARMA(1,2)
model to the data. What are the fitted AR and MA coefficients? Keeping
in mind the data generating process in an MA(1) process, are the fitted
coefficients reasonable? Explain your answer

Here is my code to generate the time series:

```python
maparams = np.r_[1,.74]
samples = 1000
y = arma_generate_sample(np.ones(1), maparams, samples)

#Plot the time series
plt = sns.tsplot(data=y)
plt.set_title("MA(1) Time Series")
plt.get_figure().savefig("MA1.png")
```
And the plot that it generated.

![MA(1)](/images/timeseries_analysis/MA1.png)
\

Fitting an ARMA(1,2) process, we get the coefficients

\vert coeff \vert  value \vert  std err\vert 
\vert ------\vert -------\vert --------\vert 
\vert const \vert  -0.0011\vert  0.064\vert 
\vert  AR1 \vert  0.6099 \vert  0.211\vert 
\vert MA1\vert  0.1728 \vert  0.217 \vert 
\vert MA2\vert  00.4068\vert  0.168 \vert 

```python
#Fit an ARMA(1,2) process and get coefficients
arma_mod = sm.tsa.ARMA(y, order=(1,2))
arma_res = arma_mod.fit()
arma_res.summary()
```

Compared to an MA(1) model:

```python
#Try fitting an MA(1) process to compare
ma_mod = sm.tsa.ARMA(y, order=(0,1))
ma_res = ma_mod.fit()
ma_res.summary()
```

\vert coeff \vert  value \vert  std err\vert 
\vert ------\vert -------\vert --------\vert 
\vert const \vert  -0.0012\vert  0.057\vert 
\vert MA1\vert  0.7501 \vert  0.022 \vert 

The ARMA(1,2) parameter estimates are obviously very far from the 'true' value of $$.74$$, while the MA(1) estimates are pretty accurate.

This discrepancy is reasonable, as the ARMA(1,2) coefficients are overfitting the data with a different model. The values in the ARMA(1,2) no doubt better describe the data, as given extra parameters we will always achieve more accuracy. The question however, is whether this is a desirable tradeoff. To find this, we can compare the BIC statistics, which weigh the tradeoff between accuracy and extra degrees of freedom.

The ARMA(1,2) model has BIC 2931.581, while the MA(1) model has BIC 2921.378.

Thus, the increase in parameters is not justified and the MA(1) model is a better choice.

To verify this assessment, we can compare the BIC's of several possible ARMA models:

```python
#Compare all models to validate conclusions
sm.tsa.stattools.arma_order_select_ic(y)
```


\vert  BIC \vert  q=0         \vert  1           \vert  2           \vert 
\vert -----\vert -------------\vert -------------\vert -------------\vert 
\vert  p=0 \vert  3379.809145 \vert  2921.378256 \vert  2927.549953 \vert 
\vert  1   \vert  3066.500779 \vert  2927.383301 \vert  2931.581370 \vert 
\vert  2   \vert  3010.627563 \vert  2928.472952 \vert  2935.230591 \vert 
\vert  3   \vert  2975.924328 \vert  2935.268508 \vert  2941.294055 \vert 
\vert  4   \vert  2959.321499 \vert  2941.796385 \vert  2948.696158 \vert 

Indeed, we see that the best BIC is achieved by an MA(1) process, indicating that other models, while they may achieve better errors, are probably victims of overfitting as their explanatory power does not justify the increased degrees of freedom.

Thus, fitted coefficients are 'reasonable' in that they correspond to the best fit given more parameters to play with. However, this would not be a good choice in model since the extra parameters do not sufficiently reduce the error, suggesting that overfitting is occurring.

> With four parameters I can fit an elephant, and with five I can make him wiggle his trunk.
>> - John von Neumann

# P2 - Johnson & Johnson quarterly earnings

## pt 1
Is the time series stationary? Support your answer with numerical evidence

```python
#load and plot data
jj_raw = pd.read_csv('data/jj.csv')
jjx = jj_raw['x']
pt = sns.tsplot(data=jjx)
pt.set_title("Johnson & Johnson Quarterly Earnings")
pt.get_figure().savefig('figs/jjraw.png')
```

Here is the plot of the data:

![Johnson & Johnson Quarterly Earnings](/images/timeseries_analysis/jjraw.png)
\

First, by visual inspection the data does not appear stationary.

For a more numerical approach, I will show that the dataset violates both conditions of weak stationarity: the mean value and variance functions are not constant across the dataset.

First, to show that the mean is time-dependent, I take a 40 point moving average and plot it:

```python
#To show that the mean is not constant, I will use a
#moving average filter:
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#40 point moving average
jj_mean = moving_average(np.array(jjx),40)
plt2 = sns.tsplot(data= jj_mean)
plt2.set_title("Johnson & Johnson Moving Average")
plt2.get_figure().savefig('figs/jj_mean.png')
```

![Johnson & Johnson Moving Average](/images/timeseries_analysis/jj_mean.png)
\

We can see clearly that the mean is increasing.

Next, to show that the variance is also time-dependent, I will plot the variance across 40 point moving windows.

```python
#Variance across 40 pt moving window
partials = [jjx[i:i+40] for i in range(60)]
tvar = [np.var(partial) for partial in partials]
pt = sns.tsplot(data = tvar)
pt.set_title("Johnson & Johnson Variance on Moving Window")
pt.get_figure().savefig('figs/jj_vars.png')
```

![J&J Variance on Moving Window](/images/timeseries_analysis/jj_vars.png)
\

Here as well, we can see that the variance of the function is time-dependent and changes based on the position in the dataset.

With these two pieces of evidence, we can conclude that the series is in fact nonstationary.  

## pt 2
Let $$y_t = \frac{x_{t+1}}{x_t}$$, where $$\{x_t\}$$ is the raw time series. Calculate the sample autocorrelation and partial autocorrelation of $$y_t$$.

```python
#calculate y_t
num_x = np.array(jjx)
jjy = num_x[1:]/num_x[:-1]
```

First, I plot the time series:

```python
pt = sns.tsplot(data= jjy)
pt.set_title("Johnson & Johnson Detrended Plot")
pt.get_figure().savefig('figs/jjy.png')
```
![J&J y plot](/images/timeseries_analysis/jjy.png)
\

Next, I find the sample ACF

```python
jjacf = sm.tsa.stattools.acf(jjy)
pt = sns.tsplot(data=jjacf)
pt.set_title("Johnson & Johnson Detrended ACF")
pt.get_figure().savefig('figs/jjy_acf.png')
```
![J&J detrended ACF](/images/timeseries_analysis/jjy_acf.png)
\

And finally, the PACF

```python
jjpacf = sm.tsa.stattools.pacf(jjy)
pt = sns.tsplot(data=jjpacf)
pt.set_title("Johnson & Johnson detrended PACF")
pt.get_figure().savefig('figs/jjy_pacf.png')
```
![J&J detrended PACF](/images/timeseries_analysis/jjy_pacf.png)
\


## pt 3

Is $$\{y_t\}$$ stationary? Support your answer with numerical evidence.

While the time series plot certainly looks stationary, I will verify by again plotting the sliding window mean and variance across the time series.

```python
# 40 point moving average
jjy_mean = moving_average(jjy,40)
pt = sns.tsplot(data= jjy_mean)
pt.set_title("Johnson & Johnson Detrended Moving Average")
pt.get_figure().savefig('figs/jjy_mean.png')
```
![J&J Detrended Moving Average](/images/timeseries_analysis/jjy_mean.png)
\

Though there are still fluctuations in the graph, they are relatively small ($$\pm 1.5\%$$). Since this is sampled data, we would not expect to see a perfectly constant mean, so this data validates the hypothesis that the data is stationary.

Next, I plot the variance

```python
#Variance across 40 pt moving window
partials = [jjy[i:i+40] for i in range(60)]
yvar = [np.var(partial) for partial in partials]
pt = sns.tsplot(data = yvar)
pt.set_title("Johnson & Johnson Detrended Moving Window Variance")
pt.get_figure().savefig('figs/jjy_vars.png')
```

![J&J Detrended Moving Window Variance](/images/timeseries_analysis/jjy_vars.png)
\

Here, there are much more significant variations in the variance ($$\pm50\%$$). I argue that this is too much variance to attribute to the data coming from a real source rather than a model.

Finally, I would like to note that from pt 2, the PACF does not decay to 0, signifying that the plot is not in-fact stationary.

Therefore, with significantly nonzero variance and a PACF that suggests against stationarity, I believe that this time series is not actually stationary, even though the simple time series and mean function seem to suggest so.

# P3 - Eurozone GDP

## pt 1

For $$d \in \{0,\cdots,3\}$$, plot $$y_t = \nabla^d x_t$$. What seems to be a good choice of differencing order to obtain a stationary time series?

First, I import the GDP data and plot it for sanity

```python
gdp_raw = pd.read_csv('data/eurozone.csv', skiprows=1,
  skipfooter= 5, header=None)
gdp_vals =gdp_raw.transpose()[4:] #prune weird starting stuff
np_gdp_data = vals.values.flatten()[:-1] #prune weird ending stuff

gdp_data=np_gdp_data.astype('float')
time = range(1960,2015)

europlot = sns.tsplot(time=time, data= gdp_data)
europlot.set(xlabel="Year", ylabel = "GDP")
europlot.set_title("Eurozone GDP")
europlot.get_figure().savefig("figs/europlot.png")
```
![Eurozone GDP](/images/timeseries_analysis/europlot.png)
\

Next, to evaluate which choice of differencing to make, I plot all of the differenced times series.

```python
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,
  squeeze=True, figsize=(6,6))

#generate 0 differences
europlot0 = sns.tsplot(time=time, data= gdp_data, ax= ax1)
europlot0.set(xlabel="Year", ylabel = "GDP")
ax1.set_title("d=0")

#generate 1 differences

gdp_data_d1 =np.diff(gdp_data,1)
time_d1 = time[1:]
europlot1 = sns.tsplot(time=time_d1, data = gdp_data_d1, ax= ax2)
europlot1.set(xlabel="Year", ylabel = "GDP")
ax2.set_title("d=1")

#generate 2 differences
gdp_data_d2 = np.diff(gdp_data,2)
time_d2 = time[2:]
europlot2 = sns.tsplot(time=time_d2, data = gdp_data_d2, ax= ax3)
europlot2.set(xlabel="Year", ylabel = "GDP")
ax3.set_title("d=2")

#generate 3 differences
gdp_data_d3 = np.diff(gdp_data,3)
time_d3 = time[3:]
europlot3 = sns.tsplot(time=time_d3, data = gdp_data_d3, ax= ax4)
europlot3.set(xlabel="Year", ylabel = "GDP", )
ax4.set_title("d=3")

f.tight_layout()
f.savefig("figs/eurodiff.png")
```

![Differencing of Eurozone Data](/images/timeseries_analysis/eurodiff.png)
\

In order to evaluate which differencing best, I also plot the autocorrelation functions of each time series

```python
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,
  squeeze=True, figsize=(6,6))
acf = sm.tsa.stattools.acf #renaming for convenience

#generate 0 differences
europlot0 = sns.tsplot(data= acf(gdp_data), ax= ax1)
europlot0.set(xlabel="Years", ylabel = "ACF")
ax1.set_title("d=0")

#generate 1 differences
europlot1 = sns.tsplot(data = acf(gdp_data_d1), ax= ax2)
europlot1.set(xlabel="Years", ylabel = "ACF")
ax2.set_title("d=1")

#generate 2 differences
europlot2 = sns.tsplot(data = acf(gdp_data_d2), ax= ax3)
europlot2.set(xlabel="Years", ylabel = "ACF")
ax3.set_title("d=2")

#generate 3 differences
europlot3 = sns.tsplot(data = acf(gdp_data_d3), ax= ax4)
europlot3.set(xlabel="Years", ylabel = "ACF", )
ax4.set_title("d=3")

f.tight_layout()
f.savefig("figs/euroacf.png")
```
![Eurozone ACF with Differencing](/images/timeseries_analysis/euroacf.png)
\

From the ACFs, we see that they correlations trail off to zero for $$d \geq 1$$. Therefore, to avoid overdifferencing, I think that $$d=1$$ is the best choice of differencing.

## pt 2
Fit an AR(2) model to $$\{y_t\}$$

```python
euro_y = gdp_data_d1

#Fit an AR(2) process and get coefficients
ar_mod = sm.tsa.ARMA(euro_y, order=(2,0))
ar_res = ar_mod.fit()
ar_res.summary()  
```
| coef |  value |  std err | 
| -----| -------|  ------- | 
|  const|  714.4711 |  237.310 | 
|  AR1  |  0.1941  |   0.134   | 
|  AR2  |   -.1522 |   0.132   | 

Here are the fitted parameters for the AR(2) model.

## pt 3

Assuming $$x_0=0$$, show that the best linear predictor of $$y_{t+1}$$ given $$y_{1:t}$$, where $$y_t = \nabla x_t$$, is $$P(x_{t+1}\vert x_{1:t})-x_t$$

$$ y_t = \nabla x_t \implies y_t = x_t - x_{t-1} \implies x_t = y_t + x_{t-1} $$

Substituting again for $$x_{t-1}$$, we get:

$$x_t = y_t + y_{t-1} + x_{t-2} $$

Iterating back to $$x_0=0$$, we get:

$$x_t = \sum\limits_{k=1}^t y_k $$

And substituting into the expression for the prediction of $$x_{t+1}$$,

$$P(x_{t+1}\vert x_{1:t}) - x_t  = P(y_{t+1}+ \sum\limits_{k=1}^t y_k\vert x_{1:t}) - x_t$$

Since $$P$$ is a linear operator, we can split this into two terms:
$$P(y_{t+1}\vert x_{1:t}) + P(\sum\limits_{k=1}^t y_k\vert  x_{1:t}) - x_t $$

However, since each $$y_k$$ up until $$y_t$$ can be uniquely determined from the $$x$$ values, we have perfect prediction in the second term. Thus we can write:

$$ P(y_{t+1}\vert x_{1:t}) + \sum\limits_{k=1}^t y_k - x_t$$

However, I previously showed through recursive substitution that $$x_t = \sum\limits_{k=1}^t y_k$$, so the latter terms of the expression are identical and cancel. So we are left with:

$$ P(y_{t+1}\vert x_{1:t}) $$

Next, notice that each $$y_{1:t}$$ can be determined uniquely by $$x_{1:t}$$, and vice versa. Therefore, without loss of generality we can replace $$x_{1:t}$$ with $$y_{1:t}$$, giving us our final expression and completing the proof:

$$ P(y_{t+1}\vert y_{1:t}) = P(x_{t+1}\vert x_{1:t})-x_t$$

# P4 - Global temperature data

## pt 1
Plot the ACF and APCF of global temperature till 2010 and the first differences. Does the ACF or PACF vanish after a finite lag?

First I will import the data and plot it for sanity

```python
temp_raw = pd.read_csv('data/temp.txt',
  delim_whitespace=True, skiprows=16)
#trim to 2010
temps = temp_raw["TEMP"].values[:-4]
times = temp_raw["YEAR"].values[:-4]
temp_plot = sns.tsplot(time=times,data=temps)
temp_plot.set_title("Temperature Plot")
temp_plot.get_figure().savefig("figs/tempplot.png")
```
![Temperature Plot](/images/timeseries_analysis/tempplot.png)
\
Next, I calculate the ACF and PACF of the temperature data and the first differences

```python
#first differences
temp_d = np.diff(temps,1)

#rename for convenience
pacf = sm.tsa.stattools.pacf

#setup plot
ftemp, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,
  squeeze=True, figsize=(6,6))

#vanilla acf
tempplot0 = sns.tsplot(data= acf(temps), ax= ax0)
tempplot0.set(xlabel="Years", ylabel = "ACF")
ax0.set_title("ACF of Temp Data")

#vanilla pacf
tempplot1 = sns.tsplot(data = pacf(temps), ax= ax1)
tempplot1.set(xlabel="Years", ylabel = "PACF")
ax1.set_title("PACF of Temp Data")

#differenced acf
tempplot2 = sns.tsplot(data = acf(temp_d), ax= ax2)
tempplot2.set(xlabel="Years", ylabel = "ACF")
ax2.set_title("ACF of Differenced Temp Data")

#differenced pacf
tempplot3 = sns.tsplot(data = pacf(temp_d), ax= ax3)
tempplot3.set(xlabel="Years", ylabel = "PACF", )
ax3.set_title("PACF of Differenced Temp Data")

ftemp.tight_layout()
ftemp.savefig("figs/tempacf.png")
```
![ACF and PACF of Temperature Data](/images/timeseries_analysis/tempacf.png)
\


## pt 2
Choose an ARMA model based on the ACF and PACF. Justify your choice using a model selection criteria.

Since the initial data does not appear to be stationary, I do not think that it offers much help in choosing a model. I will focus my investigation on the differenced data.

Using the model criterion we learned in class:

\vert    \vert  ACF \vert  PACF \vert 
\vert :---\vert :---:\vert :---:\vert 
\vert  AR($$p$$) \vert  decays \vert  0 for $$h>p$$\vert 
\vert  MA($$q$$) \vert  0 for $$h > q$$ \vert  decays \vert 
\vert ARMA($$p,q$$)\vert  decays \vert  decays \vert 

Looking at the differenced data, it is difficult to tell if the values are dropping to 0 or decaying. The fluctuations around 0, which are to be expected from a real dataset, complicate the judgement. Qualitatively however, it appears that the ACF drops steeply and begins to fluctuate, while the PACF has a more gradual curve. I will interpret this as the ACF dropping to 0 and the PACF decaying. With this in mind, I would expect an MA(q) model with $$q<5$$.

In order to numerically evaluate what model to use, I will use the BIC, which takes into account the residuals of the model as well as the number of parameters.

I evaluate the BIC on multiple models in attempt to find the best solution. Based on my hunch, I will evaluate the BIC for models with $$p \leq 2, \ q \leq 5$$.

```python
# calculate the BIC for a variety of ARMA(p,q) models
sm.tsa.stattools.arma_order_select_ic(temp_d, max_ar=2, max_ma=5)
```

\vert BIC\vert  q=0    \vert  1      \vert  2      \vert  3      \vert  4      \vert  5      \vert 
\vert ---\vert --------\vert --------\vert --------\vert --------\vert --------\vert --------\vert 
\vert p=0\vert -229.361\vert -248.653\vert -255.037\vert -250.180\vert -249.141\vert -244.158\vert 
\vert  1 \vert -232.865\vert -253.059\vert -250.056\vert -249.297\vert -244.214\vert -239.211\vert 
\vert  2 \vert -241.199\vert -251.481\vert -246.711\vert -243.750\vert -240.248\vert -235.582\vert 


Here, we see that the best model using the BIC selection criteria is an MA(2) model.

## pt 3
Fit the chosen ARMA model, and plot the residuals and their ACF.

```python
#fit MA(2) model
ma_mod = sm.tsa.ARMA(temp_d, order=(0,2))
ma_res = ma_mod.fit()

#get residuals
residuals = ma_res.resid

#plot residuals and their ACF
#setup plot
fres, (ax0,ax1) = plt.subplots(2, squeeze=True, figsize=(6,6))

#vanilla acf
resplot0 = sns.tsplot(data= residuals, ax= ax0)
resplot0.set(xlabel="Years", ylabel = "Residuals")
ax0.set_title("Residuals of MA(2) Model")

#vanilla pacf
resplot1 = sns.tsplot(data = acf(residuals), ax= ax1)
resplot1.set(xlabel="Years", ylabel = "ACF")
ax1.set_title("ACF of Residuals of MA(2) Model")

fres.tight_layout()
fres.savefig("figs/residuals.png")
```
![Residuals and ACF from MA(2) Model](/images/timeseries_analysis/residuals.png)
\

## pt 4
Test for serial correlation in the residuals (eg. using the Box-Ljung test). Report a p-value

Here I perform the Box-Ljung test to determine if the residuals are independent:

```python
resid_stats = sm.stats.acorr_ljungbox(residuals, lags=20)
p_values = resid_stats[1]
pt =sns.tsplot(data=p_values, interpolate=False)
pt.set_title("Box-Ljung p-values")
pt.get_figure().savefig("figs/pvals.png")
```
I plot the p-values at different lags below:

![Box-Ljung p-values](/images/timeseries_analysis/pvals.png)

Since none of the p-values are significant (which I define generously as $$p<.1$$), we fail to reject the null-hypothesis that the values are independent. Thus, I conclude that the residuals are uncorrelated.

I report the minimum p-value of $$.39$$ which occurs at a lag of 4. This still falls short of significance however.

## pt 5
Predict global temperatures from 2011 to 2015. Report $$95\%$$ confidence intervals for your predictions.

```python
(forecast, stderr, conf_int) = ma_res.forecast(5)

low_conf = [conf[0] for conf in conf_int]
high_conf = [conf[1] for conf in conf_int]

#create complete time series for each of the datasets
#we must add back in the first value which was removed
# through differencing.
init = np.array([temps[0]])
temp_d_forecast = np.concatenate([init,temp_d,forecast])
temp_d_low = np.concatenate([init, temp_d,low_conf])
temp_d_high = np.concatenate([init, temp_d,high_conf])

#undo the differencing:
temps_forecast = np.cumsum(temp_d_forecast)
temps_low = np.cumsum(temp_d_low)
temps_high = np.cumsum(temp_d_high)
future_times = np.concatenate([times,range(times[-1]+1,times[-1]+6)])

#construct plot to show forecasting
plt.subplot(211)
plt.plot(times,temps)
plt.plot(future_times[-6:], temps_forecast[-6:],color='g')
plt.plot(future_times[-6:], temps_high[-6:], color = 'r')
plt.plot(future_times[-6:], temps_low[-6:], color = 'r')
plt.title("Forecast Combined with Past data")

#Last plot is not super helpful, let's do a closeup
plt.subplot(212)
plt.plot(future_times[-6:], temps_forecast[-6:],color='g')
plt.plot(future_times[-6:], temps_high[-6:], color = 'r')
plt.plot(future_times[-6:], temps_low[-6:], color = 'r')
plt.title("2010-2015 Predictions")
plt.tight_layout()
plt.savefig("figs/forecasting.png")
```
![Forecast](/images/timeseries_analysis/forecasting.png)
\

Here we can see the forecasting that the model produced. Green represents the forecasted values, while the Red curves represent the high and low ends of the 95% confidence interval.

To help better understand the data, I now present them in tabular form

\vert Value/Year\vert  2011 \vert  2012 \vert  2013\vert  2014\vert  2015\vert 
\vert ----\vert ------------\vert  -----\vert -----\vert -----\vert -----\vert 
\vert Forecast\vert 0.523   \vert  0.516\vert 0.522 \vert 0.527\vert  0.532\vert 
\vert High CI\vert  0.724   \vert  0.935\vert 1.16\vert  1.393\vert  1.622\vert 
\vert Low CI\vert  0.323    \vert  0.098\vert -0.121\vert -0.340\vert  -0.558\vert 

That's all, folks!
