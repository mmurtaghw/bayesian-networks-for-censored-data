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


inverseWeights <- function(input, eventTime, censorTime, tau,isTreatment, isCensored){
  
  v <- calcV(input, eventTime, censorTime, tau)
  
  surv_object <- Surv(time = censorTime, event = isCensored)
  
  fit1 <- survfit(surv_object ~ isTreatment, data = input, type="kaplan-meier")
  
  suma = summary(fit1, times = v)
  
  out <-  data.frame(suma[6])
  
  input_N <- cbind(input,out)
  input_N <- cbind(input_N, v)  
  
  input_N <- input_N %>%
    mutate(weights = ifelse(v != 0,(1/surv), 0))
  
  return(input_N)
}

x <- with(myeloid,(inverseWeights(myeloid, txtime,futime,200, trt, death)))
