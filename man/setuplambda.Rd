\name{setuplambda}
\alias{setuplambda}
\Rdversion{1.0}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Setup of the tuning parameter \code{\eqn{\lambda}}
}
\description{
This function sets the tuning parameter \code{\eqn{\lambda}}, which is used in \code{CompositeQuantile, SingleQuantile}. 
One can also set up \code{\eqn{\lambda}} by self.
}
\usage{
setuplambda <- function(x, y, q, nlam, lam_max = 1, lam_min = 1e-2, 
                        alpha = 1, eps0 = 1e-8)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
  A numeric design matrix for the model
}
  \item{y}{
  A numeric vector of responses
}
  \item{q}{The \code{\eqn{q^{th}}} quantile, a scalar or vector with the value in \code{(0, 1)}.}
  \item{nlam}{The number of tuning parameter \code{\eqn{\lambda}} to be setuped.}
  \item{lam_max}{A multiplier that times maximum \code{\eqn{\lambda}} which is selected by correlation.}
  \item{alpha}{Tuning parameter for the Mnet estimator which controls
    the relative contributions from the LASSO, MCP/SCAD penalty and the ridge,
    or L2 penalty.  \code{alpha=1} is equivalent to LASSO,, MCP/SCAD penalty,
    while \code{alpha=0} would be equivalent to ridge regression.
    However, \code{alpha=0} is not supported; \code{alpha} may be
    arbitrarily small, but not exactly 0.}
  \item{eps0}{the perturbation when MM algorithm is used. Default is \code{eps0=1e-8}.}
}
\details{
This function gives a sery of the tuning parameter \code{\eqn{\lambda}}, which is used in penalties.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{lambda}{Setup of the tuning parameter \code{\eqn{\lambda}}.}
}
\references{
Ultra-high dimensional variable selection piecewise linear loss function
}
\author{
Xu Liu, Hongmei Jiang and Xingjie Shi
}
%\note{}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
cqr, pqr, pcqr
}
\examples{
# normal
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(3)*rnorm(n)
lambda <- setuplambda(x,y,q)

# T-distribution
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + rt(n, 3)
lambda <- setuplambda(x,y,q)

#logistic-distribution
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + log(tan(runif(n)))
lambda <- setuplambda(x,y,q)

#T-normal-mixed
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(2)*rnorm(n)/2 + rt(n, 4)/2
lambda <- setuplambda(x,y,q)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ setuplambda }
\keyword{ lambda }
