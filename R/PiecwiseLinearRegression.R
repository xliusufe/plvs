cqr <- function(x, y, q = 0.5, maxstep = 1e2, eps0 = 1e-8, eps = 1e-4)
{
    np = dim(x)
    n = np[1]
    p = np[2]
    fit <- .Call("CDMMQR", as.numeric(x), as.integer(p), as.numeric(y), as.numeric(q), as.integer(maxstep), as.numeric(eps0), as.numeric(eps))  
    results <- list(beta0 = fit$beta0, hatbeta = fit$beta, x = x, y = y)   
    return(results)   
}
