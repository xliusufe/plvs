# plvs
R package "plvs" for estimation of coefficients by ultra-high dimensional variable selection piecewise linear loss function. It is computationally efficient for quantile and composited quantile.

# Installation

    #install.packages("devtools")
    library(devtools)
    install_github("xliusufe/plvs")

# Usage

   - [x] [plvs-manual.pdf](https://github.com/xliusufe/plvs/blob/master/inst/plvs-manual.pdf) ---------- Details of the usage of the package.
# Example
    library(plvs)

    n = 200;p=10
    beta <- c(1, 2, 3, rep(0, p-3))
    q <- 0.5
    x <- matrix(rnorm(n*p), nrow = n)
    y <- x%*%beta + sqrt(3)*rnorm(n)
    fit <- cqr(x,y,q)
    fit$hatbeta
    fit$beta0
    
# References
 Liu, X., Jiang, H. and Shi, X. (2019). ultra-high dimensional variable selection piecewise linear loss function.. Manuscript.

# Development
This R package is developed by Xu Liu (liu.xu@sufe.edu.cn).
