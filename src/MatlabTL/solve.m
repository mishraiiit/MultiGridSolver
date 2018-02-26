%% 

gridx = 100;
gridy = 100;
A = gallery('poisson', gridx);
nrows = gridx*gridy;

b = rand(nrows,1);

load pro_matrix.dat
P = spconvert(pro_matrix);
%[P, ind] = agtwolev(A, 2, 1);

Ac = P' * A * P;

setup.type = 'nofill';
[L,U] = ilu(A,setup);

precon_solve = @(x) ( P * (Ac \ (P'*x)) + (diag(diag(A)) \ x));
%precon_solve = @(x) ( P * (Ac \ (P'*x)) + (U \ (L \ x)) );

maxit = 1000;
tol = 1e-10;

%[x,flag,relres,iter] = pcg(A, b, tol, maxit);
[x,flag,relres,iter] = pcg(A, b, tol, maxit, precon_solve);
iter
