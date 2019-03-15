pcqr <- 
function(x, y, q, penalty = "MCP", lambda, nlambda = 100, lam_min = 1e-2, eps = 1e-3, maxstep = 1e3, gamma = 2, alpha = 1, 
            dfmax = NULL, user_lam = 1, eps0 = 1e-8, isbic = 1, nfold = 10)
{
    np <- dim(x)
    n <- np[1]
    p <- np[2]
    if (penalty == "LASSO") pen <- 1
    if (penalty == "MCP")   pen <- 2 
    if (penalty=="SCAD"){    
       gamma <- 3
       pen <- 3;
       }  
    if (gamma <= 1 & penalty=="MCP") stop("gamma must be greater than 1 for the MC penalty")
    if (gamma <= 2 & penalty=="SCAD") stop("gamma must be greater than 2 for the SCAD penalty")
    if (nlambda < 2) stop("nlambda must be at least 2")
    if (alpha <= 0) stop("alpha must be greater than 0; choose a small positive number instead")
    
    if (is.null(dfmax)) dfmax = p + 1
    if (missing(lam_min))
    {
        if (n>p) lam_min = 1e-2
        else lam_min = 0.1
    }
    if (missing(lambda)) lambda <- setuplambda(x, y, q = q, nlam = nlambda, lam_max = 1, lam_min = lam_min, alpha = alpha)
    if (missing(q)) q = c(1:19)/20
    fit <- .Call("CDMMSelect", as.numeric(x), as.integer(p), as.numeric(y), as.integer(pen), as.numeric(lambda), as.numeric(eps), 
        as.integer(maxstep), as.integer(gamma), as.numeric(alpha), as.integer(dfmax), as.integer(user_lam), as.numeric(eps0),
        as.numeric(q), as.integer(isbic), as.integer(nfold))
    betapath <- matrix(fit$betapath, nrow = p)
    beta0 <- matrix(fit$beta0, ncol = length(lambda))
    select.fit <- list(hatbeta = betapath[, fit$ind0], betapath = betapath, beta0 = beta0, df = fit$df, bic = fit$bic, 
                      loglikelih = fit$loglikelih, ind0 = fit$ind0, lambda = lambda, isbic = isbic, q = q, x = x, y = y)
    return(select.fit)
}

pqr <- 
function(x, y, q = 0.5, penalty = "MCP", lambda, nlambda = 100, lam_min = 1e-2, eps = 1e-3, maxstep = 1e3, gamma = 2, alpha = 1, 
            dfmax = NULL, user_lam = 1, eps0 = 1e-6, isbic = 1, nfold = 10)
{
    np <- dim(x)
    n <- np[1]
    p <- np[2]
    if (penalty == "LASSO") pen <- 1
    if (penalty == "MCP")   pen <- 2 
    if (penalty=="SCAD"){    
       gamma <- 3
       pen <- 3;
       }  
    if (gamma <= 1 & penalty=="MCP") stop("gamma must be greater than 1 for the MC penalty")
    if (gamma <= 2 & penalty=="SCAD") stop("gamma must be greater than 2 for the SCAD penalty")
    if (nlambda < 2) stop("nlambda must be at least 2")
    if (alpha <= 0) stop("alpha must be greater than 0; choose a small positive number instead")
    
    if (is.null(dfmax)) dfmax = p + 1
    
    if (missing(lam_min))
    {
        if (n>p) lam_min = 1e-2
        else lam_min = 0.1
    }
    if (missing(lambda)) lambda <- setuplambda(x, y, q = q, nlam = nlambda, lam_max = 1, lam_min = lam_min, alpha = alpha)
    fit <- .Call("CDMMSingleQuantile", as.numeric(x), as.integer(p), as.numeric(y), as.integer(pen), as.numeric(lambda), as.numeric(eps), 
        as.integer(maxstep), as.integer(gamma), as.numeric(alpha), as.integer(dfmax), as.integer(user_lam), as.numeric(eps0),
        as.numeric(q), as.integer(isbic), as.integer(nfold))
    betapath <- matrix(fit$betapath, nrow = p)
    beta0 <- matrix(fit$beta0, ncol = length(lambda))
    select.fit <- list(hatbeta = betapath[, fit$ind0], betapath = betapath, beta0 = beta0, df = fit$df, bic = fit$bic, 
                      loglikelih = fit$loglikelih, ind0 = fit$ind0, lambda = lambda, isbic = isbic, q = q, x = x, y = y)
    return(select.fit)
}


datasample <- function(n = n, p = p, beta0 = beta0, sigma = 1, distr = "normal", S = diag(rep(1, p)))
{
    if (distr == "normal" ) eps = rnorm(n)
    if (distr == "T-distr") eps = rt(n, 3)
    if (distr == "cauchy")
    {
        s <- 1
        m <- 0
        tmp <- pi*(2*runif(n)-1)
        eps <- tan(tmp)*s+m
    }
    if (distr == "double-exponential")
    {
        s <- 1
        mu <- 0
        tmp <- runif(n) - 1/2
        eps <- mu - s*sign(tmp)*log(1-2*abs(tmp))
    }
    if (distr == "T-normal-mixed") eps = sqrt(2)*rnorm(n)/2 + rt(n, 4)/2
    if (distr == "logistic-distribution") eps = log(tan(runif(n)))
    
    if(missing(S)) x <- matrix(rnorm(n*p), nrow = n)
    else
    {
        S.eigen <- eigen(S)
        Sroot <- S.eigen$vectors%*%diag(sqrt(S.eigen$values))%*%S.eigen$vectors
        x <- matrix(rnorm(n*p), nrow = n)%*%Sroot
    }
    y <- x%*%beta0 + sigma* eps
    data <- list(x = x, y = y)
    return(data)
}

setuplambda <- function(x, y, q, nlam = 20, lam_max = 1, lam_min = 1e-2, alpha = 1, eps0 = 1e-8)
{
    n <- length(y)
    M <- length(q)
    stdx <- sqrt(apply(x^2, 2, sum)) 
    qq <- quantile(y, probs = q)
    
    tmp.d = matrix(rep(1/(eps0+abs(y)),ncol(x)),nrow = n)
    stdx <- sqrt(apply(x^2*tmp.d, 2, sum)) 
    max.tmp <- max(colSums(x)/stdx)
    max.lam <- max(lam_max*max.tmp/alpha)
    if (lam_min==0) lambda <- c(exp(seq(log(max.lam),log(.001*max.lam/sqrt(p)),len=nlamb-1)),0)
    else lambda <- exp(seq(log(max.lam),log(lam_min*max.lam/sqrt(p)),len=nlam))
    return(lambda)       
}
