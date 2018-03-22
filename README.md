# Arbitrary distribution
Generating 100 samples from a neural network, creating a gaussian 
mixture.

This method is very simple as we just:
1. add noise in the network
2. make multiple predictions (by repeating the data)
3. use gaussian mixture loss (extension of square error)

The same network is used in all the following examples.

Initially an additional loss function was considered, in order
to force the noise into the network. However, it is not needed in 
this application since the gaussian mixture is created. In the 
case of regularizing using this method, it is required.

## Linear bimodal
Simple linear bimodal test case to get started.

* Blue points are observations
* Red points are predictions (or rather, the means of the gaussian 
  mixture)
![linear](linear.png)



## Quadratic bimodal
Required some changes to hyperparameters in order to converge
to something reasonable.
![quadratic](quadratic.png)

## Linear multimodal
Also sensitive to hyperparameters. This approach would require
a hyperparameter search for more difficult distributions (this 
is standard anyway but sensitivity might be greater than normal).
![multimodal](multimodal.png)

A proper hyperparameter search would likely yield better results.
