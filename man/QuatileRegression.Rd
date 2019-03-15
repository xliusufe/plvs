\name{cqr}
\alias{cqr}
\Rdversion{1.0}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit usuall single quantile regression which is unpenalized.
}
\description{
This function coefficients for unpenalized composite quantile regression model, 
in which the Coordinate Descend and Minorization and Maximization (CDMM) algorithm is used.
}
\usage{
cqr <- function(x, y, q = 0.5, maxstep = 1e2, 
                              eps0 = 1e-8, eps = 1e-4)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
  A numeric design matrix for the model
}
  \item{y}{
  A numeric vector of responses
}
  \item{q}{
The \code{\eqn{q^{th}}} quantile, a scalar or vector with the value in \code{(0, 1)}. Default is \code{q = 0.5}.
}
  \item{maxstep}{
Maximum number of iterations.  Default is 100.
}
  \item{eps0}{The perturbation when MM algorithm is used. Default is \code{eps0=1e-8}.
}
  \item{eps}{Convergence threshhold.  The algorithm iterates until the
    relative change in any coefficient is less than \code{eps}.  Default
    is \code{.0001}.}
}
\details{
This function the estimator of \code{\eqn{\beta}} for unpenalized composite quantile regression model.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{hatbeta}{Estimator of \code{\eqn{\beta}}}
  \item{beta0 }{Intercept term which is numeric or vector dependent on input quatile \code{q}.}
  \item{...}{ other options for Composite Quantile }
}
\references{
Ultra-high dimensional variable selection piecewise linear loss function
}
\author{
Xu Liu, Hongmei Jiang and Xingjie Shi
}
%\note{}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{}
\examples{
# normal
n = 200;p=10
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(3)*rnorm(n)
fit <- cqr(x,y,q)
fit$hatbeta
fit$beta0

# T-distribution
n = 200;p=10
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + rt(n, 3)
fit <- cqr(x,y,q)
fit$hatbeta
fit$beta0

#logistic-distribution
n = 200;p=10
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + log(tan(runif(n)))
fit <- cqr(x,y,q)
fit$hatbeta
fit$beta0

#T-normal-mixed
n = 200;p=10
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(2)*rnorm(n)/2 + rt(n, 4)/2
fit <- cqr(x,y,q)
fit$hatbeta
fit$beta0
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ Quatile Regression }
\keyword{ Regression }
