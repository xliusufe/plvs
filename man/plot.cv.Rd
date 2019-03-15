\name{plot.cv}
\alias{plot.cv}
\Rdversion{1.0}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit the entire solution path for single quantile regression based on three penalties \code{LASSO, MCP} and \code{SCAD}
}
\description{
This function selects the important variables for single quantile regression model under ultra-high dimensional data, and simultaneously
estimate the corresponding parameters of the selected variables, in which the Coordinate Descend and Minorization and Maximization (CDMM) algorithm is used.
}
\usage{
plot.cv <- function(x, y, q = 0.5, fit = fit, isbic = 1)
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
  \item{fit}{
The fit results by \code{CompositeQuantile} or \code{SingleQuantile} including \code{lambda, ind0} and \code{bic}.
}
  \item{isbic}{
Is BIC criteria used to select the tuning parameter \code{\eqn{\lambda}}. BIC \code{isbic=1}; CV \code{isbic=2}; AIC \code{isbic=3}.
}
}
\details{
This function plots the \code{bic, aic or cv} corresponding to \code{\eqn{\beta}} to check whether the selected tuning parameter is correct or not.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{...}{ Plot the figure of \code{bic}, see the details of \code{bic} refering to \code{Composite Quantile} or \code{Single Quantile}}
}
\references{
Ultra-high dimensional variable selection piecewise linear loss function
}
\author{
Xu Liu, Hongmei Jiang and Xingjie Shi
}
%\note{}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{CompositeQuantile, SingleQuantile}
\examples{
n=200;p=20
beta <- c(1, 2, 3, rep(0, p-3))
q <- 0.5
x <- matrix(rnorm(n*p), nrow = n)
y <- x\%*\%beta + rnorm(rnorm(n))
plot.cv(x,y, q)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ plot.cv }
\keyword{ plot }
