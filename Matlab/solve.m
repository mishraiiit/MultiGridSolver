%% 

A = mmread('../matrices/poisson10000.mtx');
size(A)
nrows = size(A, 1);

b = rand(nrows,1);

P = mmread('../matrices/poisson10000promatrix.mtx');
tic
%[P, ind] = agtwolev(A, 2, 0, 2, 10);
toc

size(P)

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
