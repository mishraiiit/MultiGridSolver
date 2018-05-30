%% 
function [] = solve(filename, ktg, npass, tou)

%filename = 'Poisson2d400'
%ktg = 8
%npass = 2
%tou = 4

A = mmread(strcat('../matrices/', filename, '.mtx'));

%size(A)
%nrows = size(A, 1);
%b = rand(nrows,1);

%P = mmread('../matrices/poisson10000promatrix.mtx');
tic
[P, ind] = agtwolev(A, 2, 0, npass, ktg, -0.5, tou);
toc
size(P)
mmwrite(strcat('../matrices/', filename, 'promatrix.mtx'), P);

%Ac = P' * A * P;
%setup.type = 'nofill';
%[L,U] = ilu(A,setup);

%precon_solve = @(x) ( P * (Ac \ (P'*x)) + (diag(diag(A)) \ x));
%precon_solve = @(x) ( P * (Ac \ (P'*x)) + (U \ (L \ x)) );

%maxit = 1000;
%tol = 1e-10;

%[x,flag,relres,iter] = pcg(A, b, tol, maxit);
%[x,flag,relres,iter] = pcg(A, b, tol, maxit, precon_solve);
%iter
