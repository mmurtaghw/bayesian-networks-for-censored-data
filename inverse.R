library(survival)
library(survminer)
library(dplyr)
library(condSURV)

#Using the myeloid Lukemia data built into R

glimpse(myeloid)


calcV <- function(input, eventTime, censorTime, tau){
  vArray = c()
  for (i in 1:nrow(input)){
    #print(censorTime)
    if(is.na(eventTime[i])){
      eventTime[i] = 100000000000
    }
    v = min(eventTime[i], censorTime[i])
    v = min(v, tau)
    if (min(tau,eventTime[i]) >  censorTime[i]){
      v = 0
    }
  
    vArray = append(v,vArray)
  }
  return(vArray)
}

v <- with(myeloid, calcV(myeloid, txtime, futime, 200))
##First lets take a look at the data on a survival plot, with our event as a death.

surv_object <- Surv(time = myeloid$futime, event = myeloid$death)

fit1 <- survfit(surv_object ~ trt, data = myeloid, type="kaplan-meier")

suma = summary(fit1, times = v)

out <-  data.frame(suma[6])

myeloid_N <- cbind(myeloid,out)
myeloid_N <- cbind(myeloid_N, v)

myeloid_N <- myeloid_N %>%
  mutate(weights = ifelse(v != 0,(1/surv), 0))