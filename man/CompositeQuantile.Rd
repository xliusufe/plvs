\name{pcqr}
\alias{pcqr}
\Rdversion{1.0}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit the entire solution path for composite quantile regression based on three penalties \code{LASSO, MCP} and \code{SCAD}
}
\description{
This function selects the important variables for composite quantile regression model under ultra-high dimensional data, and simultaneously
estimate the corresponding parameters of the selected variables, in which the Coordinate Descend and Minorization and Maximization (CDMM) algorithm is used.
}
\usage{
pcqr(x, y, q , penalty = "MCP", lambda, nlambda, eps = 0.001, 
                 maxstep = 1000, gamma = 2, alpha = 1, dfmax = NULL, 
                 user_lam = 1, eps0 = 1e-8, isbic = 1, nfold = 10)
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
The \code{\eqn{q^{th}}} quantile, a scalar or vector with the value in \code{(0, 1)}. Default is \code{q = c(1:19)/20}.
}
  \item{penalty}{
LASSO, SCAD and MCP. Default is MCP
}
  \item{lambda}{A user-specified sequence of lambda values.  By default,
    a sequence of values of length \code{nlambda} is computed, equally
    spaced on the log scale.}
  \item{nlambda}{The length of lambda. Default is \code{100}.}
  \item{eps}{Convergence threshhold.  The algorithm iterates until the
    relative change in any coefficient is less than \code{eps}.  Default
    is \code{.001}.}
  \item{maxstep}{
Maximum number of iterations.  Default is 1000.
}
  \item{gamma}{The tuning parameter of the MCP/SCAD penalty (see details).}
  \item{alpha}{Tuning parameter for the Mnet estimator which controls
    the relative contributions from the LASSO, MCP/SCAD penalty and the ridge,
    or L2 penalty.  \code{alpha=1} is equivalent to LASSO, MCP/SCAD penalty,
    while \code{alpha=0} would be equivalent to ridge regression.
    However, \code{alpha=0} is not supported; \code{alpha} may be
    arbitrarily small, but not exactly 0.}
  \item{dfmax}{Upper bound for the number of nonzero coefficients.
    Default is no upper bound.  However, for large data sets,
    computational burden may be heavy for models with a large number of
    nonzero coefficients.}
  \item{user_lam}{
If given lambda? Defaul is FALSE.
}
  \item{eps0}{
The perturbation when MM algorithm is used. Default is \code{eps0=1e-8}.
}
  \item{isbic}{
Is BIC criteria used to select the tuning parameter \code{\eqn{\lambda}}? BIC \code{isbic=1}; CV \code{isbic=2}; AIC \code{isbic=3}.
}
  \item{nfold}{
How many fold to be used when cross-validation method is used to select the tuning parameter. Default is 10.
}
}
\details{
This function gives a series of solution path for composite quantile regression regression model, the corresponding degrees of freedom (df), and the log-likelihood value. 
Those values can be analysed in \code{plot.cv} and others. A tuning parameter is also selected by BIC (or CV and AIC) to gives
the estimator of \code{\eqn{\beta}}.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{hatbeta}{Estimator of \code{\eqn{\beta}}}
  \item{beta0 }{Intercept term which is numeric or vector dependent on input quatile \code{q}.}
  \item{betapath}{Solution path of \code{\eqn{\beta}}}
  \item{df}{Degrees of freedom}
  \item{bic}{\code{bic(\eqn{\lambda})} which is used to select the tuning parameter \code{\eqn{\lambda}} dependent on selecting criteria (BIC, CV or AIC)}
  \item{loglikelih}{Log-likelihood for each \code{\eqn{\lambda}}}
  \item{ind0}{Selected index of tuning parameter \code{\eqn{\lambda}}}
  \item{...}{ Other options for pcqr }
}
\references{
Ultra-high dimensional variable selection piecewise linear loss function
}
\author{
Xu Liu, Hongmei Jiang and Xingjie Shi
}
%\note{}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{pqr, plot.cv}
\examples{
# normal
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(3)*rnorm(n)
fit <- pcqr(x,y,q)
fit$ind0
fit$df[fit$ind0]
fit$hatbeta[abs(fit$hatbeta)>0]

# T-distribution
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + rt(n, 3)
fit <- pcqr(x,y,q)
fit$ind0
fit$df[fit$ind0]
fit$hatbeta[abs(fit$hatbeta)>0]

#logistic-distribution
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + log(tan(runif(n)))
fit <- pcqr(x,y,q)
fit$ind0
fit$df[fit$ind0]
fit$hatbeta[abs(fit$hatbeta)>0]

#T-normal-mixed
n = 200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- c(1:19)/20
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + sqrt(2)*rnorm(n)/2 + rt(n, 4)/2
fit <- pcqr(x,y,q)
fit$ind0
fit$df[fit$ind0]
fit$hatbeta[abs(fit$hatbeta)>0]
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ Composite Quantile }
\keyword{ Qauntile }
\keyword{ CDMM }
