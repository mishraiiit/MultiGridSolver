%% 

A = mmread('../../matrices/matvf3dSky30.mtx');
size(A)
nrows = size(A, 1);

b = rand(nrows,1);

P = mmread('../../matrices/matvf3dSky30promatrix.mtx');
size(P)
%[P, ind] = agtwolev(A, 2, 1);

Ac = P' * A * P;

setup.type = 'nofill';
[L,U] = ilu(A,setup);

%precon_solve = @(x) ( P * (Ac \ (P'*x)) + (diag(diag(A)) \ x));
precon_solve = @(x) ( P * (Ac \ (P'*x)) + (U \ (L \ x)) );

maxit = 1000;
tol = 1e-10;

%[x,flag,relres,iter] = pcg(A, b, tol, maxit);
[x,flag,relres,iter] = pcg(A, b, tol, maxit, precon_solve);
iter
