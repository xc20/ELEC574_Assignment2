# ELEC574_Assignment2

## Deliverable 1 - Making Sense of the Data

Comment on how each of the data could inform on which activity is happening. Use your plots as a reference. [15 points]

- According to the plot:
![](1.png)

- The activity "stationary (0) " can be indicated by the acceleration magnitude as well as barometer values. For instance, under the stationary status, the acceleration magnitude has very small fluctuations (almost "flat"). The barometer data also tends to be stable (or "flat") during the stationary activity, which is also the indicator. However, one thing we need to notice is, simply using barometer data's flat pattern to distinguish stationary won't work well, since this flat pattern also exhibits during the activity "walking-flat-surface (1)".

- The activity "walking-flat-surface (1) " is also indicated by the acceleration magnitude as well as barometer values but by different patterns. During "walking-flat-surface (1)", the acceleration magnitude starts to fluctuate but barometer values have almost no fluctuations. Further, even though the barometer values "stay flat", the level where they stay flat might change, see the example: the time stamp 512 - 1279 with "walking-flat-surface" (1) vs. other time stamps with "walking-flat-surface (1)"

- The activity "walking-up-stairs (2)" is indicated by a) the the acceleration magnitude's fluctuations, with the simlar level of flucations to "walking-flat-surface (1) " and b) linear dropping of barometer values.

- The activity "Walking-down-stairs (3)" is indicated by a) the the acceleration magnitude's fluctuations, with the simlar level of flucations to "walking-flat-surface (1) " and b) linear dropping of barometer values.

- 

Your task will be to compute mean and variance of acceleration for each segment. [14 points]

- Please find the implementation in the uploaded code.

Comment on how each of the features could inform on which activity is happening. Use your plots as a reference. [15 points]

## Deliverable 2 - Personal-dependent Model

Using the values in the confusion matrix to compute the precision, recall, and accuracy scores for each of the activities. [9 points]

## Deliverable 3 - Person-independent Model

Calculate the precision, recall and accuracy scores for all activities. Compare this with the results from Deliverable 2, comment on the differences. [12 points]

What are the top features? [10 points]

Think about whether these features make sense or not. [10 points] 

Discuss which other sensors and features help classifying these activities more accurately. [15 points]
