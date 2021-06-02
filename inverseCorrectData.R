library(survival)
library(survminer)
library(dplyr)
library(condSURV)
library(bnstruct)
library(qgcomp)
library(classInt)

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
x <- x %>% select(trt, sex, futime, death,txtime,crtime,rltime,weights) %>%  mutate(trt = ifelse(trt == "A",1,2)) %>% mutate (death = ifelse(death == 1, 2, 1))
x$sex <- as.numeric(x$sex)
x$trt <- as.numeric(x$trt)

x <- x[,-c(3, 5, 6,7,8) ]

headers <- names(x)
discreteness_vals <- c(TRUE, TRUE, TRUE)
second_try <- BNDataset(data=x, discreteness =discreteness_vals, variables =headers, node.sizes=c(2,2,2))
net <- learn.network(second_try, algo="sem")



show(net)

plot(net)



