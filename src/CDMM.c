#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <Rinternals.h>  

void SortQ(double *s, int l, int r)
{
    int i, j;
	double x;
    if (l < r)
    {
        i = l;
        j = r;
        x = s[i];
        while (i < j)
        {
            while(i < j && s[j] > x) j--; 
			if(i < j) s[i++] = s[j];
            while(i < j && s[i] < x) i++; 
			if(i < j) s[j--] = s[i];
        }
        s[i] = x;
        SortQ(s, l, i-1); 
        SortQ(s, i+1, r);
    }
}

void randunif(double *p, int n, double *r)
{
	int i,m;
    double s,u,v;
    s=65536.0; u=2053.0; v=13849.0;
    for (i=0; i<=n-1; i++)
      { *r=u*(*r)+v; m=(int)(*r/s);
        *r=*r-m*s; p[i]=*r/s;
      }
}

void QSortInd(double *s, int *ind, int l, int r)
{
    int i, j, m;
	double x;
    if (l < r)
    {
        i = l;
        j = r;
        x = s[i];
		m = ind[i];
        while (i < j)
        {
            while(i < j && s[j] > x) j--; 
			if(i < j) {s[i++] = s[j]; ind[i-1] = ind[j];}
            while(i < j && s[i] < x) i++; 
			if(i < j) {s[j--] = s[i]; ind[j+1] = ind[i];}
        }
        s[i] = x;
		ind[i] = m; 
        QSortInd(s, ind, l, i-1); 
        QSortInd(s, ind, i+1, r);
    }
}

void SampleQuantile(double *qr, int m, double *z, int n, double *q)
{
	double *zs=(double*)malloc(sizeof(double)*n);
	int i, ind;
	for(i=0;i<n;i++) zs[i] = z[i];
	SortQ(zs, 0, n-1);
	for(i=0;i<m;i++)
	{
		ind = floor(q[i]*n);
		if (ind!=n*q[i])
			qr[i] = zs[ind];
		else
			qr[i] = (zs[ind-1] + zs[ind])/2;
	}
	free(zs);
}

void Standarize(double *y, double *std, int n, int p, double *x, int flag)
{
	int i, j;
	double s, s1;
	for(j=0;j<p;j++)
	{
		s = 0; s1 = 0;
		for(i=0;i<n;i++) s += x[j*n+i];
		s = s/n;
		for(i=0;i<n;i++) s1  += x[j*n+i]*x[j*n+i];
		s1 = s1/n - s*s;
		if(flag)
			std[j] = sqrt(s1);
		else
			std[j] = sqrt(n*s1/(n-1));
		for(i=0;i<n;i++) y[j*n+i] = (x[j*n+i]-s)/std[j];
	}
}

void Loglikelih_path(double *loglikelih, double *df, double *beta0, double *beta, double *x, double *y, double *qq,
	int n, int p, int M, int L, int isalpha)
{
	int i,j,m,l, count;
	double s, s1;
	for(m=0;m<M;m++)
	{
		for(j=0;j<L;j++)
		{
			s = 0;
			for(i=0;i<n;i++)
			{
				s1 = 0;
				for(l=0;l<p;l++) s1 += x[l*n+i]*beta[j*p+l];
				s1 = y[i] - s1;
				if(isalpha==1) s1 = s1 - beta0[j*M+m];
				if(s1>=0)
					s += s1*qq[m];
				else
					s += s1*(qq[m] - 1); 
			}
			loglikelih[j] = loglikelih[j] + s;
		}
	}
	for(i=0;i<L;i++) loglikelih[i] = n*log(loglikelih[i]/M);

	for(i=0;i<L;i++)
	{
		count = 0;
		for(j=0;j<p;j++) 
			if(fabs(beta[i*p+j])>1e-15) count++;
		df[i] = count;
	}
}

void MMCDfitQuantile(double *beta0, double *beta, double *xx, double *y, int n, int p, int penalty, double *qq, double *lambda,
	int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user, int M, double eps0)
{
  int  active, lstart;
  int i,j, l, k, ll, step;
  double *beta_old, *beta0_old,  *residual_, **d1, **w, **residual, *std, *x;
  double d1sum, wsum1, *wsum2, tmp, z, v, converged, converged1, l1, l2;

  residual_ = (double*)malloc(sizeof(double)*n);
  beta_old = (double*)malloc(sizeof(double)*p);
  beta0_old = (double*)malloc(sizeof(double)*M);
  d1 = (double**)malloc(sizeof(double*)*n);
  w = (double**)malloc(sizeof(double*)*n);
  residual = (double**)malloc(sizeof(double*)*n);
  wsum2 = (double*)malloc(sizeof(double)*n);
  std = (double*)malloc(sizeof(double)*p);
  x = (double*)malloc(sizeof(double)*n*p);
  for(i=0;i<n;i++)
  {
	  d1[i] = (double*)malloc(sizeof(double)*M);
	  w[i] = (double*)malloc(sizeof(double)*M);
	  residual[i] = (double*)malloc(sizeof(double)*M);
  }
  for(i=0;i<p;i++)beta_old[i] = 0;
  for(i=0;i<M;i++)beta0_old[i] = 0;

  SampleQuantile(beta0_old, M, y, n, qq);
  if (user) lstart = 0;
  else lstart = 1;
  if (lstart==0) for(i=0;i<M;i++) beta0[i]=beta0_old[i];

  Standarize(x, std, n, p, xx, 1);// standarization for training data.

  /* Path */
  for (l=lstart;l<L;l++)
    {
		step = 0;
      if (l != 0) 
	  {
		  for (j=0;j<p;j++) beta_old[j] = beta[(l-1)*p+j];
		  for (j=0;j<M;j++) beta0_old[j] = beta0[(l-1)*M+j];
	  }
      while (step < max_iter)
	{
	  converged = 0;
	  step++;

	  /* Check dfmax */
	  active = 0;
	  for (j=0;j<p;j++) if (beta[l*p+j]!=0) active++;
	  if (active > dfmax)
	    {
	      for (ll=l;ll<L;ll++)
		{
		  for (j=0;j<p;j++) beta[ll*p+j] = R_NaReal;
		}
		  for(i=0;i<n;i++)
		  {
			  free(d1[i]);
			  free(w[i]);
			  free(residual[i]);
		  }
		  free(beta_old);
		  free(beta0_old);
		  free(residual);
		  free(residual_);
		  free(d1);
		  free(w);
		  free(wsum2);
		  free(std);
		  free(x);
	      return;
	    }

	  /* Covariates */
	  for(i=0;i<n;i++) 
	  {
		  tmp = 0;
		  for(j=0;j<p;j++) tmp += x[j*n+i]*beta_old[j];
		  residual_[i] = y[i] - tmp;// r = y-x*beta
	  }
	  for (i=0;i<M;i++)
	  {
		  d1sum=0;
		  wsum1=0;
		  for(j=0;j<n;j++)
		  {
			  tmp = residual_[j] - beta0_old[i]; // rk = r - beta0_old
			  d1[j][i] = (1-tmp/(eps0+fabs(tmp)))/2 - qq[i];
			  w[j][i] = 1/(eps0+fabs(tmp))/2;
			  d1sum += d1[j][i];
			  wsum1 += w[j][i];
		  }
		  beta0[l*M+i] = - d1sum/wsum1 + beta0_old[i];//  beta0(k,:) = - sum(d1) ./ sum(w) + beta0_old;
	  }
	  for(i=0;i<n;i++)
	  {
		  tmp=0;
		  for(j=0;j<M;j++)
		  {
			  tmp += w[i][j];// sum(w, 2)
			  residual[i][j] = - d1[i][j]/w[i][j] - beta0[l*M+j] + beta0_old[j];//residual = - d1./ w - repmat(beta0(k,:) - beta0_old, n, 1);
		  }
		  wsum2[i] = tmp/M;
	  }
		 
	  for (j=0;j<p;j++)
	    {
	      /* Calculate z */
	      z = 0, v = 0;
		  for(i=0;i<n;i++) v += pow(x[j*n+i],2)*wsum2[i];
		  v = v/n;
	      for (i=0;i<n;i++) 
		  {
			  tmp=0;
			  for(k=0;k<M;k++)
				  tmp += residual[i][k]*w[i][k];
			  z = z + x[j*n+i]*tmp;
		  }
		  z = z/n/M + v*beta_old[j];
	      /* Update beta_j */
		  l1 = lambda[l]*alpha; l2 = lambda[l]*(1-alpha);
		  if (penalty==1) 
		  {			  
			  if (z > l1) beta[l*p+j] = (z-l1)/(v*(1+l2));
			  if (z < -l1) beta[l*p+j] = (z+l1)/(v*(1+l2));
		  }
	      if (penalty==2)
		  {
			  double s = 0;
			  if (z > 0) s = 1;
			  else if (z < 0) s = -1;
			  if (fabs(z) <= l1) beta[l*p+j] = 0;
			  else if (fabs(z) <= gamma*l1*(1+l2)) beta[l*p+j] = s*(fabs(z)-l1)/(v*(1+l2-1/gamma));
			  else beta[l*p+j] = z/(v*(1+l2));
		  }
	      if (penalty==3) 
		  {
			  double s = 0;
			  if (z > 0) s = 1;
			  else if (z < 0) s = -1;
			  if (fabs(z) <= l1) beta[l*p+j] = 0;
			  else if (fabs(z) <= (l1*(1+l2)+l1)) beta[l*p+j] = s*(fabs(z)-l1)/(v*(1+l2));
			  else if (fabs(z) <= gamma*l1*(1+l2)) beta[l*p+j] = s*(fabs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2));
			  else beta[l*p+j] = z/(v*(1+l2));
		  }

	      /* Update r */
	      if (beta[l*p+j] != beta_old[j]) 
			  for (i=0;i<n;i++) 
			  {
				  tmp = (beta[l*p+j] - beta_old[j])*x[j*n+i];
				  for(k=0;k<M;k++)
					  residual[i][k] = residual[i][k] - tmp;
			  }
	    }

	  /* Check for convergence */
	  converged1 = 1;
	  for (j=0; j < p; j++)
	  {
		  if (beta[l*p+j]!=0 && beta_old[j]!=0)
		  {
			  if (fabs((beta[l*p+j]-beta_old[j])/beta_old[j]) > eps)
			  {
				  converged1 = 0;
				  break;
			  }
		  }
		  else if (beta[l*p+j]==0 && beta_old[j]!=0)
		  {
			  converged1 = 0;
			  break;
		  }
		  else if (beta[l*p+j]!=0 && beta_old[j]==0)
		  {
			  converged1 = 0;  
			  break;
		  }
	  }
	  if (converged1) {converged = 1; break;}

	  for (j=0;j<p;j++) beta_old[j] = beta[l*p+j];
	  for (j=0;j<M;j++) beta0_old[j] = beta0[l*M+j];
	}
    } //end out for
	for(j=0;j<L;j++)
		for(i=0;i<p;i++)
			beta[j*p+i] = beta[j*p+i]/std[i];

	for(i=0;i<n;i++)
	{
		free(d1[i]);
		free(w[i]);
		free(residual[i]);
	}
    free(beta_old);
    free(beta0_old);
    free(residual);
    free(residual_);
    free(d1);
    free(w);
    free(wsum2);
	free(std);
	free(x);
}


int CVSelectLambda(double *x, double *y, int n, int p, int penalty, double *qq, double *lambda,
	int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user,
	int M, double eps0, double *loglikelih, int nfold, int isalpha)
{
	double *xtest, *ytest, *xtrain, *ytrain, *beta0, *beta;
	int i, j, k, m, l, lag, tbeg, tend, ntrain, ind0, tmp1, *allind;
	double s, s1, min_log;
	lag = n/nfold;
	ntrain = n - lag;
	allind = (int*)malloc(sizeof(int)*n);
	for(i=0;i<n;i++) allind[i] = i;
	double r = y[0], *rx = (double*)malloc(sizeof(double)*n);
	randunif(rx, n, &r);
	QSortInd(rx, allind, 0, n-1);
	free(rx);

	xtrain = (double*)malloc(sizeof(double)*ntrain*p);
	xtest = (double*)malloc(sizeof(double)*lag*p);
	ytrain = (double*)malloc(sizeof(double)*ntrain);
	ytest = (double*)malloc(sizeof(double)*lag);

	beta0 = (double*)malloc(sizeof(double)*L*M);
	beta = (double*)malloc(sizeof(double)*L*p);

	// initialization with zeros 
	for(i=0;i<L;i++)
	{
		for(j=0;j<M;j++) beta0[i*M+j] = 0;
		for(j=0;j<p;j++) beta[i*p+j] = 0;
		loglikelih[i] = 0;
	}

	// runing iterations for nfold times//
	for(k=0;k<nfold;k++)
	{
		tbeg = k*lag;
		tend = (k+1)*lag;
		tmp1 = 0;
		for(i=tbeg;i<tend;i++)
		{
			ytest[tmp1] = y[allind[i]];
			for(j=0;j<p;j++)
				xtest[j*lag + tmp1] = x[j*n + allind[i]];
			tmp1++;
		}
		if(k==0)
		{
			tmp1 = 0;
			for(i=tend;i<n;i++)
			{
				ytrain[tmp1] = y[allind[i]];
				for(j=0;j<p;j++)
					xtrain[j*ntrain + tmp1] = x[j*n + allind[i]];
				tmp1++;
			}
		}
		else
		{
			for(i=tbeg-lag;i<tbeg;i++)
			{
				ytrain[i] = y[allind[i]];
				for(j=0;j<p;j++)
					xtrain[j*ntrain + i] = x[j*n + allind[i]];
			}
		}

		MMCDfitQuantile( beta0, beta, xtrain, ytrain, ntrain, p, penalty, qq,
			lambda, L, eps, max_iter, gamma, alpha, dfmax, user, M, eps0);

		// calculate loglikelihood//
		for(m=0;m<M;m++)
		{
			for(j=0;j<L;j++)
			{
				s = 0;
				for(i=0;i<lag;i++)
				{
					s1 = 0;
					for(l=0;l<p;l++) s1 += xtest[l*lag+i]*beta[j*p+l];
					s1 = ytest[i] - s1;
					if(isalpha==1) s1 = s1 - beta0[j*M+m];
					if(s1>=0)
						s += s1*qq[m];
					else
						s += s1*(qq[m] - 1); 
				}
				loglikelih[j] += s/M;
			}
		}

	}// end for nfold
	min_log = loglikelih[0];
	ind0 = 1;
	for(i=1;i<L;i++)
		if(min_log>loglikelih[i]) 
		{
			min_log = loglikelih[i];
			ind0 = i+1;
		}
	free(xtest);
	free(xtrain);
	free(ytest);
	free(ytrain);
	free(beta);
	free(beta0);
	return(ind0);
}

int BICSelectLambda(double *beta0, double *beta, double *x, double *y, int n, int p, int penalty, double *qq, 
	double *lambda, int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user,
	int M, double eps0, double *loglikelih, int isalpha, int isBIC, double *df, double *bic)
{
	int i, j, m, l, tmp1, count, ind0;
	double s, s1, min_bic;

	MMCDfitQuantile( beta0, beta, x, y, n, p, penalty, qq,
		lambda, L, eps, max_iter, gamma, alpha, dfmax, user, M, eps0);
	// calculate loglikelihood//
	for(m=0;m<M;m++)
	{
		for(j=0;j<L;j++)
		{
			s = 0;
			for(i=0;i<n;i++)
			{
				s1 = 0;
				for(l=0;l<p;l++) s1 += x[l*n+i]*beta[j*p+l];
				s1 = y[i] - s1;
				if(isalpha==1) s1 = s1 - beta0[j*M+m];
				if(s1>=0)
					s += s1*qq[m];
				else
					s += s1*(qq[m] - 1); 
			}
			loglikelih[j] = loglikelih[j] + s;
		}
	}
	for(i=0;i<L;i++) loglikelih[i] = n*log(loglikelih[i]/M);

	for(i=0;i<L;i++)
	{
		count = 0;
		for(j=0;j<p;j++) 
			if(fabs(beta[i*p+j])>1e-15) count++;
		df[i] = count;
	}
	tmp1 = log((double)n);
	if(isBIC>0)
		for(j=0;j<L;j++) bic[j] = loglikelih[j] + tmp1*df[j];
	else
		for(j=0;j<L;j++) bic[j] = loglikelih[j] -n + 2*df[j];

	min_bic = bic[0];
	ind0 = 1;
	for(i=1;i<L;i++)
		if(min_bic>bic[i]) 
		{
			min_bic = bic[i];
			ind0 = i+1;
		}
	return(ind0);
}

void MMCDfitQuantile0(double *beta0, double *beta, double *xx, double *y, int n, int p, int penalty, double qq, double *lambda,
	int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user, double eps0)
{
  int  active, lstart;
  int i,j, l, ll, step;
  double *beta_old, beta0_old=0,  *residual_, *d1, *w, *residual, *std, *x;
  double d1sum, wsum1, tmp, z, v, converged, converged1, l1, l2;

  residual_ = (double*)malloc(sizeof(double)*n);
  beta_old = (double*)malloc(sizeof(double)*p);
  d1 = (double*)malloc(sizeof(double)*n);
  w = (double*)malloc(sizeof(double)*n);
  residual = (double*)malloc(sizeof(double)*n);
  std = (double*)malloc(sizeof(double)*p);
  x = (double*)malloc(sizeof(double)*n*p);

  for(i=0;i<p;i++)beta_old[i] = 0;

  SampleQuantile(&beta0_old, 1, y, n, &qq);
  if (user) lstart = 0;
  else lstart = 1;
  if (lstart==0) beta0[0] = beta0_old;

  Standarize(x, std, n, p, xx, 1);// standarization for training data.

  /* Path */
  for (l=lstart;l<L;l++)
    {
		step = 0;
      if (l != 0) 
	  {
		  for (j=0;j<p;j++) beta_old[j] = beta[(l-1)*p+j];
		  beta0_old = beta0[l-1];
	  }
      while (step < max_iter)
	{
	  converged = 0;
	  step++;

	  /* Check dfmax */
	  active = 0;
	  for (j=0;j<p;j++) if (beta[l*p+j]!=0) active++;
	  if (active > dfmax)
	    {
	      for (ll=l;ll<L;ll++)
		{
		  for (j=0;j<p;j++) beta[ll*p+j] = R_NaReal;
		}
		  free(beta_old);
		  free(residual);
		  free(residual_);
		  free(d1);
		  free(w);
		  free(std);
		  free(x);
	      return;
	    }

	  /* Covariates */
	  for(i=0;i<n;i++) 
	  {
		  tmp = 0;
		  for(j=0;j<p;j++) tmp += x[j*n+i]*beta_old[j];
		  residual_[i] = y[i] - tmp;// r = y-x*beta
	  }

	 d1sum=0;
	  wsum1=0;
	  for(j=0;j<n;j++)
	  {
		  tmp = residual_[j] - beta0_old; // rk = r - beta0_old
		  d1[j] = (1-tmp/(eps0+fabs(tmp)))/2 - qq;
		  w[j] = 1/(eps0+fabs(tmp))/2;
		  d1sum += d1[j];
		  wsum1 += w[j];
	  }
	  beta0[l] = - d1sum/wsum1 + beta0_old;//  beta0(k,:) = - sum(d1) ./ sum(w) + beta0_old;
	
	  for(i=0;i<n;i++)
	  {
		  //residual[i] = - d1[i]/sqrt(w[i]) - beta0[l] + beta0_old;
		  residual[i] = - d1[i]/w[i] - beta0[l] + beta0_old;
	  }
		 
	  for (j=0;j<p;j++)
	    {
	      /* Calculate z */
	      z = 0, v = 0;
		  for(i=0;i<n;i++) 
		  {
			  tmp = x[j*n+i]*w[i];
			  v += tmp*x[j*n+i];		  
			  //z += x[j*n+i]*sqrt(w[i])*residual[i];
			  z += tmp*residual[i];
		  }
		  v = v/n;
		  z = z/n + v*beta_old[j];
	      /* Update beta_j */
		  l1 = lambda[l]*alpha; l2 = lambda[l]*(1-alpha);
		  if (penalty==1) 
		  {			  
			  if (z > l1) beta[l*p+j] = (z-l1)/(v*(1+l2));
			  if (z < -l1) beta[l*p+j] = (z+l1)/(v*(1+l2));
		  }
	      if (penalty==2)
		  {
			  double s = 0;
			  if (z > 0) s = 1;
			  else if (z < 0) s = -1;
			  if (fabs(z) <= l1) beta[l*p+j] = 0;
			  else if (fabs(z) <= gamma*l1*(1+l2)) beta[l*p+j] = s*(fabs(z)-l1)/(v*(1+l2-1/gamma));
			  else beta[l*p+j] = z/(v*(1+l2));
		  }
	      if (penalty==3) 
		  {
			  double s = 0;
			  if (z > 0) s = 1;
			  else if (z < 0) s = -1;
			  if (fabs(z) <= l1) beta[l*p+j] = 0;
			  else if (fabs(z) <= (l1*(1+l2)+l1)) beta[l*p+j] = s*(fabs(z)-l1)/(v*(1+l2));
			  else if (fabs(z) <= gamma*l1*(1+l2)) beta[l*p+j] = s*(fabs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2));
			  else beta[l*p+j] = z/(v*(1+l2));
		  }

	      /* Update r */
	      if (beta[l*p+j] != beta_old[j]) 
			  for (i=0;i<n;i++) 
				  residual[i] = residual[i] - (beta[l*p+j] - beta_old[j])*x[j*n+i];
	    }

	  /* Check for convergence */
	  converged1 = 1;
	  for (j=0; j < p; j++)
	  {
		  if (beta[l*p+j]!=0 && beta_old[j]!=0)
		  {
			  if (fabs((beta[l*p+j]-beta_old[j])/beta_old[j]) > eps)
			  {
				  converged1 = 0;
				  break;
			  }
		  }
		  else if (beta[l*p+j]==0 && beta_old[j]!=0)
		  {
			  converged1 = 0;
			  break;
		  }
		  else if (beta[l*p+j]!=0 && beta_old[j]==0)
		  {
			  converged1 = 0;  
			  break;
		  }
	  }
	  if (converged1) {converged = 1; break;}

	  for (j=0;j<p;j++) beta_old[j] = beta[l*p+j];
	  beta0_old = beta0[l];
	}// end while

    } //end out for

	for(j=0;j<L;j++)
		for(i=0;i<p;i++)
			beta[j*p+i] = beta[j*p+i]/std[i];

    free(beta_old);
    free(residual);
    free(residual_);
    free(d1);
    free(w);
	free(std);
	free(x);
}


int CVSelectLambda0(double *x, double *y, int n, int p, int penalty, double qq, double *lambda,
	int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user,
	double eps0, double *loglikelih, int nfold, int isalpha)
{
	double *xtest, *ytest, *xtrain, *ytrain, *beta0, *beta;
	int i, j, k, l, lag, tbeg, tend, ntrain, ind0, tmp1, *allind;
	double s, s1, min_log;
	lag = n/nfold;
	ntrain = n - lag;
	allind = (int*)malloc(sizeof(int)*n);
	for(i=0;i<n;i++) allind[i] = i;
	double r = y[0], *rx = (double*)malloc(sizeof(double)*n);
	randunif(rx, n, &r);
	QSortInd(rx, allind, 0, n-1);
	free(rx);

	xtrain = (double*)malloc(sizeof(double)*ntrain*p);
	xtest = (double*)malloc(sizeof(double)*lag*p);
	ytrain = (double*)malloc(sizeof(double)*ntrain);
	ytest = (double*)malloc(sizeof(double)*lag);

	beta0 = (double*)malloc(sizeof(double)*L);
	beta = (double*)malloc(sizeof(double)*L*p);

	// initialization with zeros 
	for(i=0;i<L;i++)
	{
		beta0[i] = 0;
		for(j=0;j<p;j++) beta[i*p+j] = 0;
		loglikelih[i] = 0;
	}

	// runing iterations for nfold times//
	for(k=0;k<nfold;k++)
	{
		tbeg = k*lag;
		tend = (k+1)*lag;
		tmp1 = 0;
		for(i=tbeg;i<tend;i++)
		{
			ytest[tmp1] = y[allind[i]];
			for(j=0;j<p;j++)
				xtest[j*lag + tmp1] = x[j*n + allind[i]];
			tmp1++;
		}
		if(k==0)
		{
			tmp1 = 0;
			for(i=tend;i<n;i++)
			{
				ytrain[tmp1] = y[allind[i]];
				for(j=0;j<p;j++)
					xtrain[j*ntrain + tmp1] = x[j*n + allind[i]];
				tmp1++;
			}
		}
		else
		{
			for(i=tbeg-lag;i<tbeg;i++)
			{
				ytrain[i] = y[allind[i]];
				for(j=0;j<p;j++)
					xtrain[j*ntrain + i] = x[j*n + allind[i]];
			}
		}

		MMCDfitQuantile0( beta0, beta, xtrain, ytrain, ntrain, p, penalty, qq,
			lambda, L, eps, max_iter, gamma, alpha, dfmax, user, eps0);

		// calculate loglikelihood//
		for(j=0;j<L;j++)
		{
			s = 0;
			for(i=0;i<lag;i++)
			{
				s1 = 0;
				for(l=0;l<p;l++) s1 += xtest[l*lag+i]*beta[j*p+l];
				s1 = ytest[i] - s1;
				if(isalpha==1) s1 = s1 - beta0[j];
				
				if(s1>=0)
					s += s1*qq;
				else
					s += s1*(qq - 1); 
			}
			loglikelih[j] += s;
		}

	}// end for nfold
	min_log = loglikelih[0];
	ind0 = 1;
	for(i=1;i<L;i++)
		if(min_log>loglikelih[i]) 
		{
			min_log = loglikelih[i];
			ind0 = i+1;
		}
	free(xtest);
	free(xtrain);
	free(ytest);
	free(ytrain);
	free(beta);
	free(beta0);
	return(ind0);
}

int BICSelectLambda0(double *beta0, double *beta, double *x, double *y, int n, int p, int penalty, double qq, 
	double *lambda, int L, double eps, int max_iter, double gamma, double alpha, int dfmax, int user,
	double eps0, double *loglikelih, int isalpha, int isBIC, double *df, double *bic)
{
	int i, j, l, tmp1, count, ind0;
	double s, s1, min_bic;

	MMCDfitQuantile0( beta0, beta, x, y, n, p, penalty, qq,
		lambda, L, eps, max_iter, gamma, alpha, dfmax, user, eps0);
	// calculate loglikelihood//

	for(j=0;j<L;j++)
	{
		s = 0;
		for(i=0;i<n;i++)
		{
			s1 = 0;
			for(l=0;l<p;l++) s1 += x[l*n+i]*beta[j*p+l];
			s1 = y[i] - s1;
			if(isalpha==1) s1 = s1 - beta0[j];
			if(s1>=0)
				s += s1*qq;
			else
				s += s1*(qq - 1); 
		}
		loglikelih[j] = loglikelih[j] + s;
	}
	for(i=0;i<L;i++) loglikelih[i] = n*log(loglikelih[i]);
	for(i=0;i<L;i++)
	{
		count = 0;
		for(j=0;j<p;j++) 
			if(fabs(beta[i*p+j])>1e-15) count++;
		df[i] = count;
	}
	tmp1 = log((double)n);
	if(isBIC>0)
		for(j=0;j<L;j++) bic[j] = loglikelih[j] + tmp1*df[j];
	else
		for(j=0;j<L;j++) bic[j] = loglikelih[j] -n + 2*df[j];

	min_bic = bic[0];
	ind0 = 1;
	for(i=1;i<L;i++)
		if(min_bic>bic[i]) 
		{
			min_bic = bic[i];
			ind0 = i+1;
		}
	return(ind0);
}

void CDMMfitQR(double *beta0, double *beta, double *xx, double *y, int n, int p, double *qq, double eps, int max_iter, int M, double eps0)
{
  int i,j, k, step;
  double *beta_old, *beta0_old,  *residual_, **d1, **w, **residual, *std, *x;
  double d1sum, wsum1, *wsum2, tmp = 0, z, v, converged;

  residual_ = (double*)malloc(sizeof(double)*n);
  beta_old = (double*)malloc(sizeof(double)*p);
  beta0_old = (double*)malloc(sizeof(double)*M);
  d1 = (double**)malloc(sizeof(double*)*n);
  w = (double**)malloc(sizeof(double*)*n);
  residual = (double**)malloc(sizeof(double*)*n);
  wsum2 = (double*)malloc(sizeof(double)*n);
  std = (double*)malloc(sizeof(double)*p);
  x = (double*)malloc(sizeof(double)*n*p);
  for(i=0;i<n;i++)
  {
	  d1[i] = (double*)malloc(sizeof(double)*M);
	  w[i] = (double*)malloc(sizeof(double)*M);
	  residual[i] = (double*)malloc(sizeof(double)*M);
  }
  for(i=0;i<p;i++) beta_old[i] = 0;
  for(i=0;i<M;i++) {beta0_old[i] = 0; beta0[i] = 0;}

  SampleQuantile(beta0_old, M, y, n, qq);

  Standarize(x, std, n, p, xx, 1);// standarization for training data.

  /* Path */
  step = 0;
  while (step < max_iter)
  {
	  converged = 0;
	  step++;

	  /* Covariates */
	  for(i=0;i<n;i++) 
	  {
		  tmp = 0;
		  for(j=0;j<p;j++) tmp += x[j*n+i]*beta_old[j];
		  residual_[i] = y[i] - tmp;// r = y-x*beta
	  }
	  for (i=0;i<M;i++)
	  {
		  d1sum=0;
		  wsum1=0;
		  for(j=0;j<n;j++)
		  {
			  tmp = residual_[j] - beta0_old[i]; // rk = r - beta0_old
			  d1[j][i] = (1-tmp/(eps0+fabs(tmp)))/2 - qq[i];
			  w[j][i] = 1/(eps0+fabs(tmp))/2;
			  d1sum += d1[j][i];
			  wsum1 += w[j][i];
		  }
		  beta0[i] = - d1sum/wsum1 + beta0_old[i];//  beta0(k,:) = - sum(d1) ./ sum(w) + beta0_old;
	  }
	  for(i=0;i<n;i++)
	  {
		  tmp=0;
		  for(j=0;j<M;j++)
		  {
			  tmp += w[i][j];// sum(w, 2)
			  residual[i][j] = - d1[i][j]/w[i][j] - beta0[j] + beta0_old[j];//residual = - d1./ w - repmat(beta0(k,:) - beta0_old, n, 1);
		  }
		  wsum2[i] = tmp/M;
	  }
		 
	  for (j=0;j<p;j++)
	    {
	      /* Calculate z and Update beta_j */
	      z = 0, v = 0;
		  for(i=0;i<n;i++) v += pow(x[j*n+i],2)*wsum2[i];
		  v = v/n;
	      for (i=0;i<n;i++) 
		  {
			  tmp=0;
			  for(k=0;k<M;k++)
				  tmp += residual[i][k]*w[i][k];
			  z = z + x[j*n+i]*tmp;
		  }
		  beta[j] = z/n/M/v + beta_old[j];     


	      /* Update r */
	      if (beta[j] != beta_old[j]) 
			  for (i=0;i<n;i++) 
			  {
				  tmp = (beta[j] - beta_old[j])*x[j*n+i];
				  for(k=0;k<M;k++)
					  residual[i][k] = residual[i][k] - tmp;
			  }
	    }

	  /* Check for convergence */
	  for(i=0;i<p;i++) tmp += fabs(beta[i] - beta_old[i]);
	  if (tmp<eps) 
	  {
		  converged = 1;
		  break;
	  }
	  for (j=0;j<p;j++) beta_old[j] = beta[j];
	  for (j=0;j<M;j++) beta0_old[j] = beta0[j];
	}
  for(i=0;i<p;i++)
	  beta[i] = beta[i]/std[i];

	for(i=0;i<n;i++)
	{
		free(d1[i]);
		free(w[i]);
		free(residual[i]);
	}
    free(beta_old);
    free(beta0_old);
    free(residual);
    free(residual_);
    free(d1);
    free(w);
    free(wsum2);
	free(std);
	free(x);
}

SEXP CDMMSelect(SEXP rx, SEXP rp, SEXP ry, SEXP rpenalty, SEXP rlambda, SEXP reps, SEXP rmax_iter, SEXP rgamma,
	SEXP ralpha, SEXP rdfmax, SEXP ruser, SEXP reps0, SEXP rqq, SEXP risbic, SEXP rnfold)
{
	int n, p, L, nfold, isbic, user, M, dfmax, penalty, max_iter, gamma, isalpha = 1;
	int *p_p, *nfold_p, *isbic_p, *user_p, *dfmax_p, *penalty_p, *max_iter_p, *gamma_p;
	double *x = REAL(rx), *y = REAL(ry), *lambda, *qq, *eps_p, *alpha_p, *eps0_p;
	double eps, alpha, eps0;
	SEXP rbeta0, rbetapath, rind0, rbic, rloglikelih, rdf, list, list_names;
	double *beta0, *betapath, *ind0, *bic, *loglikelih, *df;
	int ind00 = 1;
	L = length(rlambda);
	M = length(rqq);
	n = length(ry);
	p_p = INTEGER(rp);
	nfold_p = INTEGER(rnfold);
	isbic_p = INTEGER(risbic);
	user_p = INTEGER(ruser);
	max_iter_p = INTEGER(rmax_iter);
	gamma_p = INTEGER(rgamma);
	dfmax_p = INTEGER(rdfmax);
	penalty_p = INTEGER(rpenalty);
	p = p_p[0]; 
	nfold = nfold_p[0];
	isbic = isbic_p[0];
	user = user_p[0];
	max_iter = max_iter_p[0];
	gamma = gamma_p[0];
	dfmax = dfmax_p[0];
	penalty = penalty_p[0];
	
	lambda = REAL(rlambda);
	alpha_p = REAL(ralpha);
	eps_p = REAL(reps);
	eps0_p = REAL(reps0);
	qq = REAL(rqq);	
	alpha = alpha_p[0];
	eps = eps_p[0];
	eps0 =  eps0_p[0];


	PROTECT(rbeta0 = allocVector(REALSXP, M*L));
	PROTECT(rbetapath = allocVector(REALSXP, p*L));
	PROTECT(rind0 = allocVector(REALSXP, 1));
	PROTECT(rbic = allocVector(REALSXP, L));
	PROTECT(rdf = allocVector(REALSXP, L));
	PROTECT(rloglikelih = allocVector(REALSXP, L));
	beta0 = REAL(rbeta0);
	betapath = REAL(rbetapath);
	ind0 = REAL(rind0);
	bic = REAL(rbic);
	df = REAL(rdf);
	loglikelih = REAL(rloglikelih);
	
	 /* run the CV or BIC selection of lambda */
	 if(isbic==1) // BIC
	 {
		 ind00 = BICSelectLambda(beta0, betapath, x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, M, eps0, loglikelih, isalpha, 1, df, bic);
	 }
	 else if(isbic==2)// CV
	 {
		 ind00 = CVSelectLambda(x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, M, eps0, bic, nfold, isalpha);
		 
		 MMCDfitQuantile( beta0, betapath, x, y, n, p, penalty, qq,
			 lambda, L, eps, max_iter, gamma, alpha, dfmax, user, M, eps0);
		 
		 Loglikelih_path(loglikelih, df, beta0, betapath, x, y, qq, n, p, M, L, isalpha);
	 }
	 else if(isbic==3)// AIC
	 {
		 ind00 = BICSelectLambda(beta0, betapath, x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, M, eps0, loglikelih, isalpha, 0, df, bic);
	 }
	 ind0[0] = ind00;

	 char *names[6] = {"beta0", "betapath", "ind0", "df", "bic", "loglikelih"};
	 PROTECT(list_names = allocVector(STRSXP, 6));
	 int i;
	 for(i = 0; i < 6; i++)
		 SET_STRING_ELT(list_names, i,  mkChar(names[i]));
	 PROTECT(list = allocVector(VECSXP, 6)); 
	 SET_VECTOR_ELT(list, 0, rbeta0);
	 SET_VECTOR_ELT(list, 1, rbetapath);
	 SET_VECTOR_ELT(list, 2, rind0);
	 SET_VECTOR_ELT(list, 3, rdf);
	 SET_VECTOR_ELT(list, 4, rbic);
	 SET_VECTOR_ELT(list, 5, rloglikelih);    
	 setAttrib(list, R_NamesSymbol, list_names); 
	 
	 UNPROTECT(8);  
	 return(list);
}

SEXP CDMMSingleQuantile(SEXP rx, SEXP rp, SEXP ry, SEXP rpenalty, SEXP rlambda, SEXP reps, SEXP rmax_iter, SEXP rgamma,
	SEXP ralpha, SEXP rdfmax, SEXP ruser, SEXP reps0, SEXP rqq, SEXP risbic, SEXP rnfold)
{
	int n, p, L, nfold, isbic, user, dfmax, penalty, max_iter, gamma, isalpha = 1;
	int *p_p, *nfold_p, *isbic_p, *user_p, *dfmax_p, *penalty_p, *max_iter_p, *gamma_p;
	double *x = REAL(rx), *y = REAL(ry), *lambda, *qq_p, *eps_p, *alpha_p, *eps0_p;
	double eps, alpha, eps0, qq;
	SEXP rbeta0, rbetapath, rind0, rbic, rloglikelih, rdf, list, list_names;
	double *beta0, *betapath, *ind0, *bic, *loglikelih, *df;
	int ind00 = 1;
	L = length(rlambda);
	n = length(ry);
	p_p = INTEGER(rp);
	nfold_p = INTEGER(rnfold);
	isbic_p = INTEGER(risbic);
	user_p = INTEGER(ruser);
	max_iter_p = INTEGER(rmax_iter);
	gamma_p = INTEGER(rgamma);
	dfmax_p = INTEGER(rdfmax);
	penalty_p = INTEGER(rpenalty);
	p = p_p[0]; 
	nfold = nfold_p[0];
	isbic = isbic_p[0];
	user = user_p[0];
	max_iter = max_iter_p[0];
	gamma = gamma_p[0];
	dfmax = dfmax_p[0];
	penalty = penalty_p[0];


	
	lambda = REAL(rlambda);
	alpha_p = REAL(ralpha);
	eps_p = REAL(reps);
	eps0_p = REAL(reps0);
	qq_p = REAL(rqq);	
	alpha = alpha_p[0];
	eps = eps_p[0];
	eps0 =  eps0_p[0];
	qq = qq_p[0];


	PROTECT(rbeta0 = allocVector(REALSXP, L));
	PROTECT(rbetapath = allocVector(REALSXP, p*L));
	PROTECT(rind0 = allocVector(REALSXP, 1));
	PROTECT(rbic = allocVector(REALSXP, L));
	PROTECT(rdf = allocVector(REALSXP, L));
	PROTECT(rloglikelih = allocVector(REALSXP, L));
	beta0 = REAL(rbeta0);
	betapath = REAL(rbetapath);
	ind0 = REAL(rind0);
	bic = REAL(rbic);
	df = REAL(rdf);
	loglikelih = REAL(rloglikelih);
	
	 /* run the CV or BIC selection of lambda */
	 if(isbic==1) // BIC
	 {
		 ind00 = BICSelectLambda0(beta0, betapath, x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, eps0, loglikelih, isalpha, 1, df, bic);
	 }
	 else if(isbic==2)// CV
	 {
		 ind00 = CVSelectLambda0(x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, eps0, bic, nfold, isalpha);
		 
		 MMCDfitQuantile0( beta0, betapath, x, y, n, p, penalty, qq,
			 lambda, L, eps, max_iter, gamma, alpha, dfmax, user, eps0);
		 
		 //Loglikelih_path0(loglikelih, df, beta0, betapath, x, y, qq, n, p, L, isalpha);
		 Loglikelih_path(loglikelih, df, beta0, betapath, x, y, &qq, n, p, 1, L, isalpha);
	 }
	 else if(isbic==3)// AIC
	 {
		 ind00 = BICSelectLambda0(beta0, betapath, x, y, n, p, penalty, qq, lambda, L, eps, max_iter, 
			 gamma, alpha, dfmax, user, eps0, loglikelih, isalpha, 0, df, bic);
	 }
	 ind0[0] = ind00;

	 char *names[6] = {"beta0", "betapath", "ind0", "df", "bic", "loglikelih"};
	 PROTECT(list_names = allocVector(STRSXP, 6));
	 int i;
	 for(i = 0; i < 6; i++)
		 SET_STRING_ELT(list_names, i,  mkChar(names[i]));
	 PROTECT(list = allocVector(VECSXP, 6)); 
	 SET_VECTOR_ELT(list, 0, rbeta0);
	 SET_VECTOR_ELT(list, 1, rbetapath);
	 SET_VECTOR_ELT(list, 2, rind0);
	 SET_VECTOR_ELT(list, 3, rdf);
	 SET_VECTOR_ELT(list, 4, rbic);
	 SET_VECTOR_ELT(list, 5, rloglikelih);    
	 setAttrib(list, R_NamesSymbol, list_names); 
	 
	 UNPROTECT(8);  
	 return(list);
}

SEXP CDMMQR(SEXP rx, SEXP rp, SEXP ry, SEXP rqq, SEXP rmax_iter, SEXP reps0, SEXP reps)
{
	int n, p, M, max_iter;
	int *p_p, *max_iter_p;
	double *x = REAL(rx), *y = REAL(ry), *qq = REAL(rqq), *eps_p, *eps0_p;
	double eps, eps0;
	SEXP rbeta0, rbeta, list, list_names;
	double *beta0, *beta;
	M = length(rqq);
	n = length(ry);
	p_p = INTEGER(rp);
	max_iter_p = INTEGER(rmax_iter);
	p = p_p[0]; 
	max_iter = max_iter_p[0];
	
	eps_p = REAL(reps);
	eps0_p = REAL(reps0);
	eps = eps_p[0];
	eps0 =  eps0_p[0];


	PROTECT(rbeta0 = allocVector(REALSXP, M));
	PROTECT(rbeta = allocVector(REALSXP, p));
	beta0 = REAL(rbeta0);
	beta = REAL(rbeta);
	CDMMfitQR(beta0, beta, x, y, n, p, qq, eps, 100, M, eps0);

	 char *names[2] = {"beta0", "beta"};
	 PROTECT(list_names = allocVector(STRSXP, 2));
	 int i;
	 for(i = 0; i < 2; i++)
		 SET_STRING_ELT(list_names, i,  mkChar(names[i]));
	 PROTECT(list = allocVector(VECSXP, 2)); 
	 SET_VECTOR_ELT(list, 0, rbeta0);
	 SET_VECTOR_ELT(list, 1, rbeta);   
	 setAttrib(list, R_NamesSymbol, list_names); 
	 
	 UNPROTECT(4);  
	 return(list);
}
