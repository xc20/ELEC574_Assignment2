# ELEC574_Assignment2

## Deliverable 1 - Making Sense of the Data

**Comment on how each of the data could inform on which activity is happening. Use your plots as a reference. [15 points]**

According to the plot below:
![](1.png)

- The activity "stationary (0) " can be indicated by the acceleration magnitude as well as barometer values. For instance, under the stationary status, the acceleration magnitude has very small fluctuations (almost "flat"). The barometer data also tends to be stable (or "flat") during the stationary activity, which is also the indicator. However, one thing we need to notice is, simply using barometer data's flat pattern to distinguish stationary won't work well, since this flat pattern also exhibits during the activity "walking-flat-surface (1)".

- The activity "walking-flat-surface (1) " is also indicated by the acceleration magnitude as well as barometer values but by different patterns. During "walking-flat-surface (1)", the acceleration magnitude starts to fluctuate, which might be captured by frequency domain features/variance features, but barometer values have almost no fluctuations. Further, even though the barometer values "stay flat", the level where they stay flat might change, see the example: the time stamp 512 - 1279 with "walking-flat-surface" (1) vs. other time stamps with "walking-flat-surface (1)".

- The activity "walking-up-stairs (2)" is indicated by: a) the acceleration magnitude's fluctuations, with the simlar level of flucations to "walking-flat-surface (1) " and b) the linear dropping of barometer values.

- The activity "Walking-down-stairs (3)" is indicated by: a) the acceleration magnitude's fluctuations, which are slightly larger than the acceleration flucations caused by "walking-up-stairs (2)" and "walking-flat-surface (1)"; b) almost the linear increasing of barometer values.

- The activity "Elevator-up (4)" is indicated by: a) the flat pattern of acceleration magnitude, and this flat pattern stays around the similiar level to the pattern caused by "stationary (0) ", which makes sense since elevator up involves no stepping movements; b) the linear dropping of barometer values.

- The activity "Running (5)" is indicated by: a) the extremely large fluctuations of the acceleration magnitude, compared to the fluctuations caused by other activities; b) the flat pattern of barometer values.

- The activity "Elevator-down (6)" is indicated by: a) the flat pattern of acceleration magnitude, and this flat pattern stays around the similiar level to the pattern caused by "stationary (0) " and "Elevator-up (4)", which also makes sense since elevator down does not involve stepping movements as well; b) the linear increasing of barometer values, as opposed to "Elevator-up (4)".

**Your task will be to compute mean and variance of acceleration for each segment. [14 points]**

Please find the implementation in the uploaded code.

**Comment on how each of the features could inform on which activity is happening. Use your plots as a reference. [15 points]**

According to the plots below:

![](2.png)

The mean acceleration magnitude and its variance features indicate the activity in the following ways: 
- "stationary (0) ": The mean acceleration magnitude feature has very small fluctuations, staying around a small constant number (i.e., the mean acceleration magnitude value is low during this activity) and the pattern is flat (i.e., less changes); the variance feature also has very small fluctuations, staying around 0 and the pattern is flat.
- "walking-flat-surface (1)": The mean acceleration magnitude feature starts to increase and fluctuate between 10 and 12 with the values higher than the feature in "stationary (0) " period; the variance feature also starts to increase and fluctuate, which is consistent with the severe acceleration fluctuation patterns we observe in the Question 1's plot above.
- "walking-up-stairs (2)": The mean feature has similar pattern to the case "walking-flat-surface (1)", fluctuating around a similar value; the variance feature becomes more stable compared to "walking-flat-surface (1)", and fluctuates around a number between the case "stationary (0) " and the case "walking-flat-surface (1)".
- "walking-down-stairs (3)": The mean acceleration magnitude feature fluctuates with the similar pattern to the case "walking-flat-surface (1)"; the variance feature values become larger and the curve becomes sharper with a peak period, which is consistent with our observation in Question 1's plot, where walking down stairs leads to high fluctuations of original acceleration magnitude data.
- "elevator-up (4)": The mean acceleration magnitude value becomes smaller, even back to the level of activity "stationary (0) " but still some fluctuations occur; the variance value also becomes smaller getting back to almost 0, which means very few fluctuations occur for the acceleration magnitude data during this activity.
- "running (5)": Both acceleration magnitude's mean and variance features increase significantly, which means that in this activity, the acceleration magnitude values are very high and change a lot, contributing to the "noisy" pattern shown in the original acceleration magnitude data.
- "elevator-down (6)": Both mean and variance of acceleration magnitude become smaller with minute fluctuations. The variance even moves back to 0.

![](3.png)

The acceleration magnitude power band features indicate the activity as follows: 
- "stationary (0) ": four equally-spaced band power features exhibit similar patterns - having the value around 0 and almost no fluctuation occurs.
- "walking-flat-surface (1)": four equally-spaced band power feature values become larger compared to the activity "stationary (0) " and slightly fluctuate with a few spikes. Band power can indicate frequency patterns so this observation makes sense.
- "walking-up-stairs (2)": four equally-spaced band power feature values are similar to their values in the activity "walking-flat-surface (1)" and fluctuates less compared to their patterns in the case "walking-flat-surface (1)".
- "walking-down-stairs (3)": four equally-spaced band power feature values become larger compared to their values in case "walking-flat-surface (1)" and "walking-up-stairs (2)" and fluctuate more. Looking back on the original acceleration magnitude plot in Question 1, this pattern makes sense - during this activity, the acceleration magnitude is very noisy and changes a lot with high frequency.
- "elevator-up (4)": four equally-spaced band power feature values become smaller near to 0 and stable. This is consistent with our observation that the original acceleration magnitude in this activity is not changing frequently (low frequency) and keeps relatively stable. 
- "running (5)": four equally-spaced band power feature values become larger and fluctuate a lot. This pattern makes sense since in running activity, the original acceleration magnitude changes frequently and significantly with high frequency, which is validated by this power band patterns.
- "elevator-down (6)": four equally-spaced band power feature values become smaller near to 0 and stable, with very few spikes. This can also be validated by the original acceleration magnitude plot, where original acceleration magnitude changes slightly and less frequently.

![](4.png)
The bar slope feature indicates the activity in the following ways: 
- "stationary (0) ", "walking-flat-surface (1)", "running (5)": The bar slope has very small fluctuations around 0. There might be sharp change during the transition from one actitivy to another (e.g., from activity stationary to walking-flat-surface).
- "elevator-up (4)": The bar slope feature is negative, and fluctuates more with a peak period.
- "elevator-down (6)": The bar slope feature is positive, and fluctuates more with a peak period.
- "walking-up-stairs (2)": The bar slope has very small fluctuations but around a negative number.
- "walking-down-stairs (3)": The bar slope has very small fluctuations but around a positive number.

## Deliverable 2 - Personal-dependent Model

**Using the values in the confusion matrix to compute the precision, recall, and accuracy scores for each of the activities. [9 points]**

The code for computing these metrics are integrated into five_fold_cross_validation function. See the code.

- Model accuracy score: 86.5%
- "stationary (0) ": precision 84.4%, recall 83.1%
- "walking-flat-surface (1)": precision 83.1%, recall 91.4%
- "walking-up-stairs (2)": precision 87.5%, recall 66.7%
- "walking-down-stairs (3)": precision 92.3%, recall 88.9%
- "elevator-up (4)": precision 90.0%, recall 84.4%
- "running (5)": precision 100.0%, recall 100.0%
- "elevator-down (6)": precision 88.9%, recall 88.9%

## Deliverable 3 - Person-independent Model

**Calculate the precision, recall and accuracy scores for all activities. Compare this with the results from Deliverable 2, comment on the differences. [12 points]**

**What are the top features? [10 points]**

**Think about whether these features make sense or not. [10 points]**

**Discuss which other sensors and features help classifying these activities more accurately. [15 points]**
