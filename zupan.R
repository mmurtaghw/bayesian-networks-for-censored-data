library(survival)
library(survminer)
library(dplyr)
library(condSURV)

#Using the myeloid Lukemia data built into R

glimpse(myeloid)

##First lets take a look at the data on a survival plot, with our event as a death.

surv_object <- Surv(time = myeloid$futime, event = myeloid$death)

fit1 <- survfit(surv_object ~ trt, data = myeloid, type="kaplan-meier")
ggsurvplot(fit1, data = myeloid, pval = TRUE)

#Now lets define an arbitrary T* that's the interval between 285 and 1200
#As per the paper, let's define any censored observation above this as a death, ie a failed outcome 0


myeloid_T <- myeloid %>%
  mutate(Tstar = ifelse(futime >= 285 & futime <= 1200 & death == 0,1,0)) %>%
  mutate(outcome = ifelse(futime >= 1200 & death == 0, 0, NA)) %>%
  mutate(outcome = ifelse(Tstar, 1, outcome)) %>%
  mutate(outcome = ifelse(death == 1, 0, outcome))

filterSplit <- myeloid_T %>%
  filter(futime <= 285) 

myeloid_T <- myeloid_T %>%
  filter(futime > 285 | futime < 285 & death == 1) %>%
  mutate(KMW = 1)

filterSplitPos <- filterSplit %>%
  mutate(outcome = 1)

filterSplitNeg <- filterSplit %>%
  mutate(outcome = 0)

KMW <- with(filterSplit, KMW(futime,death))

filterSplitNeg <- cbind(filterSplitNeg,KMW)

KMW <- 1 - KMW

filterSplitPos <- cbind(filterSplitPos,KMW)


myeloid_T <- rbind(myeloid_T, filterSplitNeg)

myeloid_T <- rbind(myeloid_T, filterSplitPos)

##Bayesian Network goes here

myeloid_T



