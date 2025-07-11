clc, clear all
filename = 'CSky3d30'; kappa = 10; npass = 2; l = 2; symm = 0; kappa_symm = 8;
A = mmread(strcat('../../matrices/', filename, '.mtx'));
nrows = size(A, 1); b = rand(nrows,1);

%% Construct coarse grid matrix
if symm == 0 % if unsymmetric
tic, [P, ind] = agtwolev(A, l, symm, npass, kappa); toc
else % if symmetric
tic, [P, ind] = agtwolev(A, l, symm, npass, kappa_unsymm); toc
end    
Ac = P' * A * P;
setup.type = 'nofill';[L,U] = ilu(A,setup);
%% Try preconditioners: try taking either of M1 or M2 in any order
M1 = @(x) (P*(Ac\(P'*x)));
M2 = @(x) (U\(L\x));
%M2 = @(x) (diag(diag(A)) \ x);
%M2 = @(x)(triu(A) \ ((tril(A) \ x) ./ diag(A)));

%% Try additive preconditioners
%precon_solve = @(x) ( M1(x) + ( M2(x) ));
%precon_solve = @(x) ( M1(x) + (M2(x)));

%% Try multiplicative preconditioner
precon_solve = @(x) (M1(x) + M2(x) - (M2((A*(M1(x))))));

%% Now solve it with PCG
maxit = 100; tol = 1e-10;
%[x,flag,relres,iter] = pcg(A, b, tol, maxit);
if symm == 1
    [x,flag,relres,iter] = pcg(A, b, tol, maxit, precon_solve);
else
    [x,flag,relres,iter] = bicgstab(A, b, tol, maxit, precon_solve);
end
fprintf('relres= %g\nIterations = %d\nFlag = %d\n', relres, floor(iter), flag)