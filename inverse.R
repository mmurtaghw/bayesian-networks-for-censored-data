library(survival)
library(survminer)
library(dplyr)
library(condSURV)
library(bnstruct)

#Using the myeloid Lukemia data built into R

glimpse(myeloid)
data <- jasa


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


inverseWeights <- function(input, eventTime, censorTime, tau, isCensored){
  
  v <- calcV(input, eventTime, censorTime, tau)
  
  surv_object <- Surv(time = censorTime, event = isCensored)
  
  fit1 <- survfit(surv_object ~ isCensored, data = input, type="kaplan-meier")
  
  suma = summary(fit1, times = v)
  
  out <-  data.frame(suma[6])
  
  input_N <- cbind(input,out)
  input_N <- cbind(input_N, v)  
  
  input_N <- input_N %>%
    mutate(weights = ifelse(v != 0,(1/surv), 0))
  
  return(input_N)
}


x <- with(data,(inverseWeights(data, transplant,futime, 750, fustat)))
write.csv(x,"bndata.csv")


########Select whatever variables you want after the weights are produced from here

x <- x %>% select(trt, sex, futime, death,txtime,crtime,rltime,weights) %>%  mutate(trt = ifelse(trt == "A",1,2)) %>% mutate (death = ifelse(death == 1, 2, 1))
x$sex <- as.numeric(x$sex)
x

write.csv(x,"bndata.csv")

headers <- names(x)
discreteness_vals <- c(TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE)
second_try <- BNDataset(x, discreteness_vals, headers, c(2,2,20,2,20,20,20,20))
net <- learn.network(second_try, algo="sem")
plot(net)
  