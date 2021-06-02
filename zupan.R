library(survival)
library(survminer)
library(dplyr)
library(condSURV)

#Using the myeloid Lukemia data built into R

glimpse(myeloid)

##First lets take a look at the data on a survival plot, with our event as a death.

surv_object <- Surv(time = jasa$futime, event = jasa$fustat)

fit1 <- survfit(surv_object ~ transplant, data = jasa, type="kaplan-meier")
ggsurvplot(fit1, data = jasa, pval = TRUE)

#Now lets define an arbitrary T* that's the interval between 285 and 1200
#As per the paper, let's define any censored observation above this as a death, ie a failed outcome 0


jasa_T <- jasa %>%
  mutate(Tstar = ifelse(futime >= 285 & futime <= 1200 & fustat == 0,1,0)) %>%
  mutate(outcome = ifelse(futime >= 1200 & fustat == 0, 0, NA)) %>%
  mutate(outcome = ifelse(Tstar, 1, outcome)) %>%
  mutate(outcome = ifelse(fustat == 1, 0, outcome))

filterSplit <- jasa_T %>%
  filter(futime <= 285) 

jasa_T <- jasa_T %>%
  filter(futime > 285 | futime < 285 & fustat == 1) %>%
  mutate(KMW = 1)

filterSplitPos <- filterSplit %>%
  mutate(outcome = 1)

filterSplitNeg <- filterSplit %>%
  mutate(outcome = 0)

KMW <- with(filterSplit, KMW(futime,fustat))

filterSplitNeg <- cbind(filterSplitNeg,KMW)

KMW <- 1 - KMW

filterSplitPos <- cbind(filterSplitPos,KMW)


jasa_T <- rbind(jasa_T, filterSplitNeg)

jasa_T <- rbind(jasa_T, filterSplitPos)

write.csv(jasa_T,"zupandata.csv")

##Bayesian Network goes here

