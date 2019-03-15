plot.cv <- function(x, y, q = 0.5, fit = fit, isbic = 1)
{
    if(missing(fit)) fit<-pcqr(x, y, q=q, isbic=isbic)
    else 
    {
        q <- fit$q
        isbic <- fit$isbic
    } 
    bic <- fit$bic
    lambda <- fit$lambda
    ind0 <- fit$ind0
    xlab <- expression(lambda)
    if(isbic==1) ylab="bic"
    if(isbic==2) ylab="cv"
    plog.args<-list(x=lambda, y=bic, xlab=xlab, ylab=ylab, type="l")
    do.call("plot", plog.args)
    line.args <- list(x=rep(lambda[ind0], 2),y = c(0, max(lambda)), type="l", col= 1)
    do.call("matlines",line.args)
}
