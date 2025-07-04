! This file is part of AGMG software package,
! Release 3.0 built on "Jul  4 2011"
!
!    AGMG is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    AGMG is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    along with AGMG.  If not, see <http://www.gnu.org/licenses/>.
!
! Up-to-date copies of the AGMG package can be obtained
! from the Web pages <http://homepages.ulb.ac.be/~ynotay/dagmg>.
!
! You can acknowledge, citing references [1] [2], and [3], the contribution
! of this package in any scientific publication dependent upon it use.
!
! [1] Y. Notay, An aggregation-based algebraic multigrid method,
!    Electronic Transactions on Numerical Analysis, vol. 37, pp. 123-146, 2010
!
! [2] A. Napov and Y. Notay, An algebraic multigrid method with guaranteed
!    convergence rate, Report GANMN 10-03, Universite Libre de Bruxelles,
!    Brussels, Belgium, 2010.
!
! [3] Y. Notay, Aggregation-based algebraic multigrid for convection-diffusion
!    equations, Report GANMN 11-01, Universite Libre de Bruxelles, Brussels,
!    Belgium, 2011.
!
! See the accompanying userguide for more details on how to use the software,
! and the README file for installation instructions.
!
!-----------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!! PARAMETERS DEFINITON -  may be tuned by expert users
!-----------------------------------------------------------------------
  MODULE dag2l_mem
    SAVE
!
! INTEGER
!
!  maxlev  is the maximal number of levels
!          (should be large enough - much larger than needed is armless).
!  real_len is the length of 1 REAL(kind(0.0d0)) in byte
!        (used only to display information on memory usage).
!  nsmooth  is the number of pre- and post-smoothing steps;
!  smoothtype indicates which smoother use:
!    if smoothtype==1, the smoothing scheme is
!        pre-smoothing: Forward Gauss-Seidel, then Backward GS, then Fwd GS,etc
!       post-smoothing: Backward Gauss-Seidel, then Forward GS, then Bwd GS,etc
!                       (nsmooth sweeps for both pre- and post-smoothing)
!    if smoothtype==1, ILU(0) smoother is used
!  nstep   is the maximum number of coarsening steps
!          nstep<0 means that coarsening is stopped only according to
!          the matrix size, see parameter maxcoarsesize.
!  nlvcyc  is the number of coarse levels (from bottom) on which V-cycle
!          formulation is enforced (Rmk: K-cycle always allowed at first
!          coarse level).
!  npass   is the maximal number of pairwise aggregation passes for each
!          coarsening step, according to the algorithms in [2,3].
!  maxcoarsesize: when the size of the coarse grid matrix is less than or
!                 equal to maxcoarsesize*N^(1/3),  it is factorized exactly
!                 and coarsening is stopped;
!         maxcoarsesizeslow: in case of slow coarsening,
!                 exact factorization can be performed when the size of
!                 the coarse grid matrix is less than or equal to
!                 maxcoarsesizeslow*N^(1/3).
    INTEGER, PARAMETER :: maxlev=40, real_len=8
    INTEGER, PARAMETER :: nsmooth=1, smoothtype=1, nstep=-1, nlvcyc=0
    INTEGER, PARAMETER :: maxcoarsesize=0,maxcoarsesizeslow=0
    INTEGER :: npass=2
!
! REAL
!
!  trspos is a threshold: if a row has a positive offidiagonal entry larger
!         than trspos times the diagonal entry, the corresponding node is
!         transferred unaggregated to the coarse grid
!  kaptg_ is the threshold used to accept or not a tentative aggregate
!         when applying the coarsening algorithms from [2,3];
!         kaptg_blocdia is used for control based on bloc diagonal smoother [2];
!         kaptg_dampJac is used for control based on Jacobi smoother [3].
!  checkdd is the threshold to keep outside aggregation nodes where
!         the matrix is strongly diagonally dominant (based on mean of row
!         and column);
!         In fact, AGMG use the maximum of |checkdd| and of the default value
!            according to kaptg_ as indicated in [2,3]
!            (hence |checkdd| < 1 ensures that one uses this default value)
!         checkdd <0 : consider |checkdd|, but base the test on minus the
!                sum of offdiagonal elements, without taking the absolute value.
!  targetcoarsefac is the target coarsening factor (parameter tau in the main
!         coarsening algorithm in [2,3]): futher pairwise aggregation passes
!         are omitted once the number of nunzero entries has been reduced by a
!         factor of at least targetcoarsefac.
!  fracnegrcsum: if, at some level, more than fracnegrcsum*n nodes have
!         negative mean row and column sum, then the aggregation algorithm
!         of [2,3] is modified, exchanging all diagonal entries for the mean
!         row and column sum (that is, the algorithm is applied to a
!         modified matrix with mean row and colum sum enforced to be zero)
    REAL(kind(0.0d0)) :: trspos=0.45
    REAL(kind(0.0d0)) :: kaptg_blocdia=8, kaptg_dampJac=10
    REAL(kind(0.0d0)) :: checkdd=-0.5
    REAL(kind(0.0d0)) :: targetcoarsefac=4
    REAL(kind(0.0d0)) :: fracnegrcsum=0.25
!!!!!!!!!!!!!!!!!!!! END of PARAMETERS DEFINITION -----------------
! Internal variables declaration
!
    TYPE InnerData
       REAL(kind(0.0d0)), DIMENSION(:), POINTER :: a
       INTEGER, DIMENSION(:), POINTER :: ja
       INTEGER, DIMENSION(:), POINTER :: ia
       INTEGER, DIMENSION(:), POINTER :: il
       INTEGER, DIMENSION(:), POINTER :: iu
       REAL(kind(0.0d0)), DIMENSION(:), POINTER :: p
       INTEGER, DIMENSION(:), POINTER :: idiag
       INTEGER, DIMENSION(:), POINTER :: ind
       INTEGER, DIMENSION(:), POINTER :: iext
       INTEGER, DIMENSION(:), POINTER :: ilstout
       INTEGER, DIMENSION(:), POINTER :: lstout
       INTEGER, DIMENSION(:), POINTER :: ilstin
       INTEGER, DIMENSION(:), POINTER :: lstin
    END TYPE InnerData
!
    TYPE(InnerData) :: dt(maxlev)
    REAL(kind(0.0d0)), ALLOCATABLE :: scald(:)
    INTEGER :: nn(maxlev),kstat(2,maxlev)=0,innermax(maxlev)
    INTEGER :: nlev,nwrkcum,iout,nrst
    INTEGER :: maxcoarset=maxcoarsesize,maxcoarseslowt=maxcoarsesizeslow
    REAL(kind(0.0d0)) :: memi=0.0,memax=0.0,memr=0.0,mritr,rlenilen
    REAL(kind(0.0d0)) :: wcplex(4),fracnz(maxlev)
    LOGICAL :: spd,wfo,wff,allzero,trans,transint,zerors
    REAL(kind(0.0d0)), PARAMETER :: cplxmax=3.0, xsi=0.6d0
    REAL(kind(0.0d0)), PARAMETER :: repsmach=SQRT(EPSILON(1.0d0))
    INTEGER :: nlc(2),nlcp(2),nlc1(2),icum
    REAL(kind(0.0d0)) :: ngl(2),nglp(2),nlctot(2),ngl1(2),ngltot(2)
    INTEGER, PARAMETER :: IRANK=-9999
    REAL(kind(0.0d0)) :: checkddJ, checkddB
    INTEGER :: nb_trs
  END MODULE dag2l_mem
!-----------------------------------------------------------------------
! END of Internal variables declaration
!-----------------------------------------------------------------------
SUBROUTINE dag2l_twolev(l,n,a,ja,ia,ind,sym,npas,kap,checkd,targetcf  &
                        ,fracnzrs,trspo,iprint)
  USE dag2l_mem
  IMPLICIT NONE
  INTEGER :: l, l1, n, ja(*), ia(n+1), ind(n), sym, npas, iprint, nc, i, k
  REAL(kind(0.0d0)) :: a(*), kap, checkd, targetcf, fracnzrs, trspo
  l1=1
  IF (l.NE.1) l1=2
  spd=sym.EQ.1
  npass=min(npas,10)
  kaptg_blocdia=kap
  kaptg_dampJac=kap
  targetcoarsefac=targetcf
  fracnegrcsum=fracnzrs
  trspos=trspo
  checkdd=checkd
  checkddJ=MAX(ABS(checkdd),kaptg_dampJac/(kaptg_dampJac-2))
  checkddB=MAX(ABS(checkdd),(kaptg_blocdia+1)/(kaptg_blocdia-1))
  nb_trs=floor(kap+0.99)
    wfo=.TRUE.
    iout=iprint
    IF (iprint <= 0) THEN
       iout=6
       wfo=.FALSE.
    ELSE IF (iprint == 5) THEN
       iout=6
    END IF
    wff=wfo
    !
    ! Find pointers to diagonal elements
    !
       ALLOCATE(dt(l1)%idiag(n+1))
       DO i=1,n
          DO k=ia(i),ia(i+1)-1
             IF (ja(k) .EQ. i) THEN
                dt(l1)%idiag(i)=k
                EXIT
             END IF
          END DO
       END DO
  CALL dag2l_aggregation(l1,n,a,ja,ia,nc)
  IF (nc > 0) THEN
    ind(1:n)=dt(l1)%ind(1:n)
    DEALLOCATE(dt(l1+1)%a,dt(l1+1)%ja,dt(l1+1)%ia,dt(l1+1)%idiag      &
              ,dt(l1)%ind)
  ELSE
    ind(1:n)=0
  END IF
  DEALLOCATE(dt(l1)%idiag)
  RETURN
END SUBROUTINE dag2l_twolev
  SUBROUTINE dag2l_aggregation(l,n,a,ja,ia,nc)
!
!  Perform aggregation of level l matrix with partial
!  ordering of the rows specified in (n,a,ja,ia,idiag,iext)
!  (iext significant only with the parallel version)
!
!  Output:
!     nc : number of aggregates (= number of coarse unknowns)
!     dt(l)%ind(1:n): ind(i) is the index of the aggregates to which
!                     i belongs; ind(i)=0 iff i has been kept outside
!                     aggregation (see checkdd).
!     P: associated prologation matrix;
!        P(i,j)=1 iff ind(i)=j and P(i,j)=0 otherwise (i=1,n ; j=1,nc).
!
!     Corresponding coarse grid matrix stored with partially ordered rows
!       in dt(l+1)%a, dt(l+1)%ja, dt(l+1)%ia, dt(l+1)%idiag, dt(l+1)%iext
!                                   (both these latter with length nc+1).
!
!  note: storage not optimal: dt(l+1)%a, dt(l+1)%ja greater than needed
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: l,n,nc
    INTEGER :: ja(*),ia(n+1)
    REAL (kind(0.0d0)) :: a(*)
    INTEGER :: ierr,i,j,k,maxdg,np,kpass,nzc,m1,ndd,nzp,isize,nddp,npass1,nz
    LOGICAL :: skipass
    !
    INTEGER, POINTER, DIMENSION(:) :: jan,ian,idiagn,iextn,ind2,lcg,lcgn
    REAL(kind(0.0d0)), POINTER, DIMENSION(:) :: an
    REAL(kind(0.0d0)), POINTER, DIMENSION(:) :: sinn
    !
    INTEGER, POINTER, DIMENSION(:) :: jap,iap,idiagp,iextp,lcg1
    REAL(kind(0.0d0)), POINTER, DIMENSION(:) :: ap
    REAL(kind(0.0d0)), POINTER, DIMENSION(:) :: sip
    !
    INTEGER, ALLOCATABLE, DIMENSION(:) :: ldd,iw,iperm,riperm
    REAL(kind(0.0d0)), ALLOCATABLE, DIMENSION(:) :: si1,w
    REAL(kind(0.0d0)), ALLOCATABLE, DIMENSION(:) :: wc
    !
    CHARACTER (len=80) :: lout
    !
    IF (l .EQ. 1) THEN
       IF (wfo) THEN
          WRITE (lout,901) IRANK
          call mexPrintf(lout//char(10))
       END IF
       IF (wff) THEN
          IF (spd) THEN
             WRITE (lout,906)
             call mexPrintf(lout//char(10))
          ELSE IF ( .NOT. transint) THEN
             WRITE (lout,908)
             call mexPrintf(lout//char(10))
          END IF
          IF (.not.spd) then
             WRITE (lout,902) 'Jacobi',kaptg_dampJac,checkddJ
             call mexPrintf(lout//char(10))
          ELSE
             WRITE (lout,902) 'BlockD',kaptg_blocdia,checkddB
             call mexPrintf(lout//char(10))
          END IF
          IF (checkdd < 0) THEN
             WRITE (lout,904)
             call mexPrintf(lout//char(10))
          END IF
          WRITE(lout,903) npass,targetcoarsefac
          call mexPrintf(lout//char(10))
          WRITE (lout,905) trspos
          call mexPrintf(lout//char(10))
       END IF
    END IF
    !
    IF (l .EQ. 1) THEN
       ALLOCATE(si1(n),ind2(n),iperm(n),riperm(n))
       memi=memi+4*n+1
       memr=memr+n
       CALL dag2l_setCMK(n,ja,ia,dt(l)%idiag,dt(l)%iext,riperm,iperm)
    ELSE
       ALLOCATE(si1(n),ind2(n),iperm(n))
       memi=memi+2*n
       memr=memr+n
       iperm(1:n)=1
    END IF
    memax=MAX(memax,memr+memi*rlenilen)
    !
    call dag2l_prepareagg(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,ind2   &
            ,iperm,si1,ndd)
    !
       IF (wfo) THEN
          IF (zerors) THEN
             WRITE (lout,907) IRANK
             call mexPrintf(lout//char(10))
          END IF
          WRITE(lout,910) IRANK,(i,i=0,nb_trs)
          call mexPrintf(lout//char(10))
907       FORMAT(i3,'* Too much nodes with neg. mean row & col. sum:',  &
           ' diag. modified for aggregation')
910       FORMAT(i3,'* DomD  NPC /',20(i2,2x))
       END IF
    IF (ndd .EQ. n) THEN
       nc=0
       nzc=0
       DEALLOCATE(si1,iperm,ind2)
       memi=memi-2*n
       memi=memi-2*n
       memr=memr-n
       IF (l .EQ. 1) THEN
          DEALLOCATE(riperm)
          memi=memi-n
       END IF
       GOTO 999
    END IF
    !
    ALLOCATE(ldd(ndd),lcg(2*(n-ndd)))
    memi=memi+2*n-ndd
    memax=MAX(memax,memr+memi*rlenilen)
    !
    IF (dble(n) .GT. targetcoarsefac*(n-ndd)) THEN
       !  skip first pairwise pass if the coarsening seems sufficiently
       !  fast based on ndd only
       skipass=.TRUE.
       !  but add one more pass to compensate in case of need
       npass1=npass+1
    ELSE
       skipass=.FALSE.
       npass1=npass
    END IF
    !
    !  perform initial pairwise aggregation; output in ind2 and lcg
    !
    IF (l > 1) THEN
       IF (spd) THEN
          CALL dag2l_findpairs_SI(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1  &
               ,ind2,lcg,nc,ndd,ldd,skipass                              &
               ,iperm)
       ELSE
          CALL dag2l_findpairs_GI(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1  &
               ,ind2,lcg,nc,ndd,ldd,skipass                              &
               ,iperm)
       END IF
       DEALLOCATE(iperm)
       memi=memi-n
    ELSE
       IF (spd) THEN
          CALL dag2l_findpairs_SI1(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1  &
               ,ind2,lcg,nc,ndd,ldd,skipass                               &
               ,riperm,iperm)
       ELSE
          CALL dag2l_findpairs_GI1(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1  &
               ,ind2,lcg,nc,ndd,ldd,skipass                               &
               ,riperm,iperm)
       END IF
       DEALLOCATE(iperm,riperm)
       memi=memi-2*n
    END IF
    nz=ia(n+1)-1
    !
    ! Form aggregated matrix in (an,jan,ian)
    ! the new matrix has at most the following number of nonzero:
    !  nz (the previous number) - ndd - 2*(n-ndd-nc)
    !         indeed: from rows listed in ldd, at least ndd
    !         diagonal entries are removed;
    !         (n-ndd-nc) is the number of pairs formed,
    !         and each of them condensate at least 3 entries in a
    !         single diagonal entry
    IF (npass1.GT.1) THEN
       isize=nc
    ELSE
       isize=1
    END IF
    ALLOCATE(an(nz-2*(n-nc)+ndd),jan(nz-2*(n-nc)+ndd)             &
         ,ian(nc+1),idiagn(nc+1),sinn(isize),wc(nc),iw(2*nc))
    memi=memi+nz-2*(n-nc)+ndd+2*(nc+1)+2*nc+isize
    memr=memr+nz-2*(n-nc)+ndd+nc
    memax=MAX(memax,memr+memi*rlenilen)
    CALL dag2l_setcg(n,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1,ind2,lcg      &
         ,nc,an,jan,ian,idiagn,iextn,sinn,npass1.GT.1,maxdg,iw,iw(nc+1),wc)
    DEALLOCATE(wc,iw)
    memi=memi-2*nc
    memr=memr-nc
    nzc=ian(nc+1)-1
    IF (dble(nz).GT.targetcoarsefac*nzc .OR. npass1.LE.1) THEN
       DEALLOCATE(si1,sinn,lcg)
       memi=memi-2*(n-ndd)
       memr=memr-n-isize
       dt(l)%ind  => ind2
       dt(l+1)%a    => an
       dt(l+1)%ja   => jan
       dt(l+1)%ia   => ian
       dt(l+1)%idiag=> idiagn
       NULLIFY(ind2,an,jan,ian,idiagn)
       GOTO 999
    END IF
    !
    DEALLOCATE(ind2)
    memi=memi-n
    !
    lcg1 => lcg
    NULLIFY(lcg)
    m1=1
    !
    !
    ! Allocated pointers at this stage: an, jan, ian, idiagn, sinn, lcg1
    !                                             (lcg accessed via lcg1)
    ! Allocated arrays: si1, ldd
    !
    DO kpass=2,npass1
       m1=2*m1
       np  = nc
       nzp = nzc
       ap     => an
       jap    => jan
       iap    => ian
       idiagp => idiagn
       iextp  => iextn
       sip    => sinn
       NULLIFY(an,jan,ian,idiagn,sinn)
       !
       ALLOCATE(lcg(2*np),ind2(np),w(maxdg),iw(maxdg))
       memi=memi+maxdg+3*np
       memr=memr+maxdg
       memax=MAX(memax,memr+memi*rlenilen)
       !
       !   perform further pairwise aggregation; output in ind2 and lcg
       !
       ind2(1:np)=-1
       IF (spd) THEN
          CALL dag2l_findpairs_SF(np,ap,jap,iap,idiagp,iextp,sip  &
               ,ind2,lcg,nc                                        &
               ,m1,lcg1,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1,w,iw)
       ELSE
          CALL dag2l_findpairs_GF(np,ap,jap,iap,idiagp,iextp,sip     &
               ,ind2,lcg,nc                                           &
               ,m1,lcg1,a,ja,ia,dt(l)%idiag,dt(l)%iext,si1,w,iw)
       END IF
       DEALLOCATE(w,iw)
       memi=memi-maxdg
       memr=memr-maxdg
       IF (kpass.NE.npass1) THEN
          isize=nc
       ELSE
          isize=1
          DEALLOCATE(si1)
          memr=memr-n
       END IF
       !
       ! Form aggregated matrix in (an,jan,ian,idiagn,iextn)
       ! the new matrix has at most the following number of nonzero:
       !  nzp (the previous number) - 2*(np-nc)
       !         indeed: np-nc is the number of pairs formed,
       !         and each of them condensate at least 3 entries in a
       !         single diagonal entry
       !
       !  below we reallocate an,ja,ian,idiagn; the memory they were
       !        pointing to can be accessed via ap,jap,iap,idiagp
       ALLOCATE(an(nzp-2*(np-nc)),jan(nzp-2*(np-nc))                 &
            ,ian(nc+1),idiagn(nc+1),sinn(isize),wc(nc),iw(2*nc))
       memi=memi+nzp-2*(np-nc)+2*(nc+1)+2*nc+isize
       memr=memr+nzp-2*(np-nc)+nc
       memax=MAX(memax,memr+memi*rlenilen)
       !
       ! Allocated pointers at this stage: an,jan,ian,idiagn,sinn,(iextn)
       !                                   ap,jap,iap,idiagp,sip,(iextp)
       !                                   lcg1,lcg,ind2
       ! Allocated arrays: si1,ldd,wc,iw
       !
       CALL dag2l_setcg(np,ap,jap,iap,idiagp,iextp,sip,ind2,lcg,nc,an     &
            ,jan,ian,idiagn,iextn,sinn,kpass.NE.npass1,maxdg,iw,iw(nc+1),wc)
       memi=memi-SIZE(jap)-2*(np+1)-np-2*nc
       memr=memr-SIZE(ap)-SIZE(sip)-nc
       DEALLOCATE(ap,jap,iap,idiagp,sip,ind2,wc,iw)
       !
       ! Allocated pointers at this stage: an,jan,ian,idiagn,(iextn)
       !                                   sinn,lcg1,lcg
       ! Allocated arrays: si1,ldd
       !
       ! Form new list of aggregates from previou one and the pairwise step
       !   just performed
       ALLOCATE(lcgn(2*m1*nc))
       memi=memi+2*m1*nc
       memax=MAX(memax,memr+memi*rlenilen)
       CALL dag2l_lcgmix(nc,m1,lcg1,lcg,lcgn)
       memi=memi-SIZE(lcg)-SIZE(lcg1)
       DEALLOCATE(lcg,lcg1)
       lcg1 => lcgn
       NULLIFY(lcgn)
       !
       ! Allocated pointers at this stage: an,jan,ian,idiagn,sinn,lcg1,(iextn)
       ! Allocated arrays: si1,ldd
       !  -- same as before entering the do loop --
       nzc=ian(nc+1)-1
       IF ( kpass.NE.npass1 .AND. dble(nz).GT.targetcoarsefac*nzc ) THEN
          DEALLOCATE(si1)
          memr=memr-n
          EXIT
       END IF
    END DO
    !
    memr=memr-SIZE(sinn)
    DEALLOCATE(sinn)
    !
    ALLOCATE(dt(l)%ind(n))
    memi=memi+n
    memax=MAX(memax,memr+memi*rlenilen)
    CALL dag2l_setind(nc,ndd,ldd,lcg1,2*m1,dt(l)%ind)
    memi=memi-ndd-SIZE(lcg1)
    DEALLOCATE(lcg1,ldd)
    !
    ! Allocated pointers at this stage: an,jan,ian,idiagn,(iextn)
    ! Allocated arrays: -
    !
       dt(l+1)%a    => an
       dt(l+1)%ja   => jan
       dt(l+1)%ia   => ian
       dt(l+1)%idiag=> idiagn
       NULLIFY(an,jan,ian,idiagn)
    !
999 CONTINUE
    !
    RETURN
901 FORMAT(i3,'*SETUP: Coarsening by multiple pairwise aggregation')
902 FORMAT('****       Quality threshold (',A6,'):',f6.2, &
         ' ;  Strong diag. dom. trs:',f5.2)
903 FORMAT('****         Maximal number of passes:',i3,     &
         '  ; Target coarsening factor:',f5.2)
904 FORMAT('****           Diag. dom. checked w.r.t. sum of offdiag', &
          ' (no absolute vaues)')
905 FORMAT('****',22x,'Threshold for rows with large pos. offdiag.:',f5.2)
906 FORMAT('****  Rmk: Setup performed assuming the matrix symmetric')
908 FORMAT('****  Rmk: Setup performed for the transpose of the input matrix')
  END SUBROUTINE dag2l_aggregation
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_setCMK(n,ja,ia,idiag,iext,riperm,iperm)
    !
    !     compute CMK permutation
    !
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),riperm(*),iperm(n)
    LOGICAL :: exc
    INTEGER :: i,j,jj,jk,jd,k,kk,j1,j2,i1,i2,ijs,ijs1,ijs2,dg,mindg,kdim
    !
    !
    ! First find starting node of minimal degree
    ! while already numbering nodes whith degree one
    ! (i.e., only the diagonal element is present in the row);
    ! store the opposite of the degree in iperm
    mindg=n+1
    jj=1
    i2=1
    DO i = 1,n
       dg=ia(i+1)-ia(i)
       IF (dg .GT. 1) THEN
          iperm(i)=-dg
          IF (dg.LT.mindg) THEN
             mindg=dg
             jj=i
          END IF
       ELSE
          riperm(i2)=i
          iperm(i)=i2
          ! one more node is numbered: increase i2
          i2=i2+1
       END IF
    ENDDO
    !
    ijs=0
    i1=i2
15  CONTINUE
    !
    ! Start with next node of minimum degree: jj
    riperm(i2)=jj
    iperm(jj)=i2
    !
    ! in what follows: i2 is the number of nodes already numbered
    !    and i1-1 the number of nodes whose neighbors have been processed
    DO WHILE (i1.LE.i2 .AND. i2.LT.n)
       !
       ! Number neighbors of riperm(i1)
       !
       i=riperm(i1)
       ijs1=i2+1
       !
       j1 =ia(i)
       jd=idiag(i)
       j2 = ia (i+1)-1
       DO kk = j1,jd-1
          j=ja(kk)
          IF (iperm(j) .LT. 0) THEN
             ! one more node is numbered: increase i2
             i2=i2+1
             riperm(i2)=j
          END IF
       ENDDO
       DO kk = jd+1,j2
          j=ja(kk)
          IF (iperm(j) .LT. 0) THEN
             ! one more node is numbered: increase i2
             i2=i2+1
             riperm(i2)=j
          END IF
       ENDDO
       !
       ijs2=i2
       ! Sort just found nodes such that value stored in iperm
       !  will be in descending order; since this is the oposite of
       !  the degree, degrees will be in ascending order
       exc=.TRUE. .AND. ijs2.GT.ijs1
       DO WHILE(exc)
          exc=.FALSE.
          DO kk=ijs1+1,ijs2
             IF( iperm(riperm(kk)) .GT. iperm(riperm(kk-1)) )THEN
                j=riperm(kk)
                riperm(kk)=riperm(kk-1)
                riperm(kk-1)=j
                exc=.TRUE.
             END IF
          END DO
       END DO
       DO kk=ijs1,ijs2
          iperm(riperm(kk))=kk
       END DO
       !
       ! Done with riperm(i1): proceed with next node
       i1=i1+1
    END DO
    IF (i2 .LT. n) THEN
       !
       ! Not all nodes numbered during first pass
       !   (may happen if the matrix is reducible)
       ! Find a new starting node with minimal degree and proceed
       jj=0
       DO WHILE (jj .EQ. 0)
          ijs=ijs+1
          IF (ijs .GT. n) THEN
             mindg=mindg+1
             ijs=1
          END IF
          ijs1=ijs
          IF (iperm(ijs1).LT.0 .AND. ia(ijs1+1)-ia(ijs1).EQ.mindg) &
               jj=ijs1
       END DO
       i2=i2+1
       ! Repeat previous step
       GOTO 15
    END IF
    !
    RETURN
  END SUBROUTINE dag2l_setCMK
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_prepareagg(n,a,ja,ia,idiag,iext,ind2,iperm,si,ndd)
    !
    !  Performs some setup before aggregation:
    !     detects nodes to be kept outside because of strong
    !     diagonal dominance, as well as nodes to be transferred
    !     as is to the coarse grid because of strong positive
    !     coupling in the row or column
    !
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind2(n),iperm(n)
    INTEGER :: ndd
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) , TARGET :: si(n)
    REAL(kind(0.0d0)) :: checkddl,oda,odm,ods,vald
    INTEGER :: i,j,jj,jk,jd,k,kk,j1,j2,i1,i2,ijs,ijs1,ijs2,dg,kdim,nnegrcs
    REAL(kind(0.0d0)), POINTER, DIMENSION(:) :: odmax,odabs,osi
    !
    IF (.NOT.spd) THEN
       checkddl=checkddJ
    ELSE
       checkddl=checkddB
    END IF
    !
    IF (.NOT.spd) THEN
       !
       ! Matrix is not symmetric: gather for each node
       !   information about elements in column:
       !   sum, maximum and, if needed, sum of abs value
       !
       IF (checkdd > 0) THEN
          ALLOCATE(odmax(n),odabs(n))
          memi=memi+2*n
          odabs(1:n)=0.0d0
       ELSE
          ALLOCATE(odmax(n))
          memi=memi+n
       END IF
       osi => si(1:n)
       si(1:n)=0.0d0
       odmax(1:n)=0.0d0
       memax=MAX(memax,memr+memi*rlenilen)
       DO i=n,1,-1
          j =ia(i)
          jd=idiag(i)
          jj = ia (i+1)-1
          DO k = j,jd-1
             jk=ja(k)
             osi(jk)=osi(jk)+a(k)
             odmax(jk)=max(odmax(jk),a(k))
             IF (checkdd > 0) odabs(jk)=odabs(jk)+abs(a(k))
          ENDDO
          DO k = jd+1,jj
             jk=ja(k)
             osi(jk)=osi(jk)+a(k)
             odmax(jk)=max(odmax(jk),a(k))
             IF (checkdd > 0) odabs(jk)=odabs(jk)+abs(a(k))
          ENDDO
       ENDDO
    END IF
    !
    ! Compute the mean row and column sum,
    !  and check if the node will participate to aggregation
    !
    ndd=0
    nnegrcs=0
    !
    DO i=1,n
       j1 =ia(i)
       jd=idiag(i)
       j2 = ia (i+1)-1
       vald = a(jd)
       odm=0.0d0
       oda=0.0d0
       ods=0.0d0
       DO kk = j1,jd-1
          ods=ods+a(kk)
          odm=max(odm,a(kk))
          IF (checkdd > 0) oda=oda+abs(a(kk))
       ENDDO
       DO kk = jd+1,j2
          ods=ods+a(kk)
          odm=max(odm,a(kk))
          IF (checkdd > 0) oda=oda+abs(a(kk))
       ENDDO
       !
       ! Update with info from columns if needed
       IF (.NOT.spd) THEN
          ods=(osi(i)+ods)/2
          odm=max(odm,odmax(i))
          IF (checkdd > 0) oda=(oda+odabs(i))/2
       END IF
       !
       ! Count the number of nodes with negative mean row and column sum
       IF ((vald+ods) .LT. -repsmach) nnegrcs=nnegrcs+1
       !
       ! Now check the category of the node and fill si(i)
       !
       si(i)=-ods
       IF ( (checkdd.GT.0 .AND. vald.GT.checkddl*oda)     &
            .OR. (checkdd.LT.0 .AND. vald.GT.checkddl*abs(ods)) ) THEN
          !
          ! Node to be kept outside the aggregation because of strong diagonal
          !   dominance: set ind2(i) to 0
          ind2(i)=0
          ndd=ndd+1
       ELSE
          ! Normal node: set ind2(i) to negative
          ind2(i)=-1
          !
          ! iperm(i) set to zero for nodes that are to be transeferred as is
          ! to the coarse grid
          IF (odm .GT. trspos*vald) iperm(i)=0
       ENDIF
    END DO
    !
    ! set zerors according to nnegrcs and fracnegrcsum
    !
    zerors=.FALSE.
    IF (nnegrcs .GT. fracnegrcsum*n) THEN
       zerors=.TRUE.
       ndd=0
       ind2(1:n)=-1
    END IF
    !
    ! Release local memory if needed
    IF (.NOT.spd) THEN
       IF (checkdd > 0) THEN
          DEALLOCATE(odmax,odabs)
          memi=memi-2*n
       ELSE
          DEALLOCATE(odmax)
          memi=memi-n
       END IF
    END IF
    RETURN
  END SUBROUTINE dag2l_prepareagg
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_GF(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,m1,lcg1,a1,ja1,ia1,idiag1,iext1,si1,rtent,jtent)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: further pairwise aggregation for general matrices [3].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: m1,ja1(*),ia1(n+1),idiag1(n),iext1(*),jtent(*),lcg1(m1,n)
    REAL(kind(0.0d0)) :: a1(*)
    REAL(kind(0.0d0)) :: si1(n),rtent(*)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be negative
!
!         ja1(*),ia1(n+1),idiag1(n): unaggregated matrix, in which the
!                threshold condition has to be checked
!         si1(n): si in that matrix
!
!         m1    : maximal number of nodes in aggregates from previous passes
!
!         lcg1(m1,n): list of aggregates from previous passes
!           lcg1(2,i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding corase grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!
!  rtent, jtent are working arrays of size >= the maximal degree of a node in
!               input matrix (a,ja,ia)
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER, PARAMETER :: nddl=0
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (lcg1(2,isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ntentleft=0
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(lcg1(2,j).EQ.0 .OR. ind(j).GE.0) CYCLE
          !   search for the corresponding entry in transposed matrix
          kk=0
          IF (i .LT. idiag(isel)) THEN
             j2=ia(j+1)-1
             DO jk=idiag(j)+1,j2
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ELSE
             DO jk=ia(j),idiag(j)-1
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ENDIF
          vals=-a(i)/2
          IF(kk .NE. 0) vals=vals-a(kk)/2
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
             eta1=2*si(isel)
             eta2=2*si(j)
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
             eta1=2*a(idiag(isel))
             eta2=2*a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
          ELSE
             del1=rsi+2*sig1
          END IF
          IF (sig2 > 0.0d0) THEN
             del2=rsj
          ELSE
             del2=rsj+2*sig2
          END IF
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=((eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=(eta1*eta2)/(eta1+eta2)
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_dampJac) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is possibly
          !    an acceptable pair; check if it is the best one
          !      and update the list of possible second choices in rtent, jtent
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          ntentleft=ntentleft+1
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. j.LT.ipair)  GOTO 9
          rtent(ntentleft)=tent
          jtent(ntentleft)=j
          CYCLE
9         CONTINUE
          rtent(ntentleft)=val
          jtent(ntentleft)=ipair
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       IF (ipair .EQ. 0) GOTO 25
20     CONTINUE
       !       Perform check in unagregated matrix, considering possible pairs
       !       in right order
       CALL dag2l_checktentagg_GF(isel,ipair           &
            ,m1,lcg1,a1,ja1,ia1,idiag1,iext1,si1,val,acc)
       !
       IF (.NOT.acc) THEN
          !     Tentative pair has been rejected, check for the next one, if any
          ipair = 0
          IF (ntentleft .GT.0) THEN
             i=1
             j=1
             DO WHILE (i .LE. ntentleft)
                IF (jtent(j).GT.0) THEN
                   tent=rtent(j)
                   IF (ipair.EQ.0) GOTO 22
                   IF (16*(tent-val).LT.-1) GOTO 22
                   IF (16*(tent-val).LT.1 .AND. j.LT.ipair) GOTO 22
                   GOTO 23
22                 CONTINUE
                   val=tent
                   ipair=jtent(j)
                   ijtent=j
23                 CONTINUE
                   i=i+1
                END IF
                j=j+1
             END DO
             ntentleft=ntentleft-1
             jtent(ijtent)=0
             GOTO 20
          END IF
       END IF
       !
25     CONTINUE
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_GF
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_checktentagg_GF                          &
       (isel,ipair,m1,lcg1,a,ja,ia,idiag,iext,si,val,acc)
!
!  Check the quality in the original matrix of an aggregate formed
!   by grouping two aggregates from previous pass(es)
!
!  INPUT: see dag2l_findpairs_GF
!
!  OUTPUT: acc(logical): acc=.true. if the new aggregate can be accepted and
!                        acc=.false. otherwise
!
    USE dag2l_mem
    IMPLICIT NONE
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(*),  val
    INTEGER :: m1, isel, ipair, lcg1(m1,*), ia(*), ja(*), idiag(*), iext(*)
    LOGICAL :: acc
!
    INTEGER, PARAMETER :: mm=max(2**(10),8)
    REAL(kind(0.0d0)) :: W(mm,mm), sig(mm), AGe(mm), v(mm)
    REAL(kind(0.0d0)) :: alpha, alp, tmp, beta, f1, f2
    INTEGER :: j,jj,k,l,m,info, setdim1, setdim, l2, k2
    INTEGER :: set(mm), l1, wdthT
    REAL(kind(0.0d0)) :: T
    LOGICAL :: exc
    REAL(kind(0.0d0)) ::  bndmum1m1,dbndmum1,umdbndmum1
    bndmum1m1=1.0d0/(kaptg_dampJac-1.0d0)
    dbndmum1=2.0d0/kaptg_dampJac
    umdbndmum1=1.0d0-dbndmum1
!
! Find indices of submatrix to be extracted and store them in set(1:setdim)
    IF (m1.eq.2) THEN
       IF (lcg1(2,isel) .LT. 0) THEN
          IF (lcg1(2,ipair) .LT. 0) THEN
             set(1)=lcg1(1,isel)
             set(2)=lcg1(1,ipair)
             setdim=2
          ELSE
             set(1)=lcg1(1,isel)
             set(2)=lcg1(1,ipair)
             set(3)=lcg1(2,ipair)
             setdim=3
          END IF
          l1=1
       ELSE
          IF (lcg1(2,ipair) .LT. 0) THEN
             set(1)=lcg1(1,isel)
             set(2)=lcg1(2,isel)
             set(3)=lcg1(1,ipair)
             setdim=3
          ELSE
             set(1)=lcg1(1,isel)
             set(2)=lcg1(2,isel)
             set(3)=lcg1(1,ipair)
             set(4)=lcg1(2,ipair)
             setdim=4
          END IF
          l1=2
       END IF
    ELSE
       l1=m1
       IF (lcg1(m1,isel).LT.0) l1=-lcg1(m1,isel)
       set(1:l1)=lcg1(1:l1,isel)
       l2=m1
       IF (lcg1(m1,ipair).LT.0) l2=-lcg1(m1,ipair)
       set(l1+1:l1+l2)=lcg1(1:l2,ipair)
       setdim=l1+l2
    END IF
!
! Sort indices in set(1:setdim) in increasing order
!   (makes susequent processing faster)
    exc=.TRUE.
    DO WHILE(exc)
       exc=.FALSE.
       DO l=2,SetDim
          IF( set(l)<set(l-1) )THEN
             jj=set(l)
             set(l)=set(l-1)
             set(l-1)=jj
             exc=.TRUE.
          END IF
       END DO
    END DO
!
! Extract submatrix of (the symmetric part of) input matrix and store it in W
! Store in AGe the mean row and column sum of input matrix for related indices
! (this is also the row sum of A_G)
    DO j=1,SetDim
       jj=Set(j)
       sig(j)=si(jj)
       IF (zerors) THEN
          W(j,j)=sig(j)
          AGe(j)=0.0d0
       ELSE
          W(j,j)=a(idiag(jj))
          AGe(j)=W(j,j)-sig(j)
       END IF
       l2=j+1
       DO l=l2,SetDim
          W(j,l)=0.0d0
          W(l,j)=0.0d0
       END DO
       k2=ia(jj+1)-1
       DO k=idiag(jj)+1,k2
          DO l=l2,SetDim
             m=Set(l)
             IF(ja(k)==m)THEN
                alpha=a(k)/2
                W(j,l)=alpha
                W(l,j)=alpha
                EXIT
             END IF
          END DO
       END DO
       DO k=ia(jj),idiag(jj)-1
          DO l=1,j-1
             m=Set(l)
             IF(ja(k)==m)THEN
                alpha=a(k)/2
                W(j,l)=W(j,l)+alpha
                W(l,j)=W(j,l)
                EXIT
             END IF
          END DO
       END DO
    END DO
!
    DO j=1,SetDim
   !       Set sig(j) equal to minus the sum of connections that are
   !       external to the tentative aggregate; that is, add the internal
   !       offdiagonal entries to current sig(j)
       DO k=1,SetDim
          IF (j.ne.k) THEN
             sig(j)=sig(j)+W(j,k)
          END IF
       ENDDO
   !       Heuristic if sig(j) is negative: set Sigma_G(j)=abs(sig(j))
   !         Hence need to correct the row sum of A_G stored in AGe
       IF (sig(j) < 0.0d0)  AGe(j)=AGe(j)+2*sig(j)
   !
   !       Store diagonal element in v
       v(j)=W(j,j)
   !                          2
   !       Now set W = A_G - --- D_G
   !                         kap
   !
   !       Then W contains the matrix that need to be nonnegative definite to
   !       accept the aggregate according to the theory in [3],
   !       up to the rank one correction that is added below
   !
   !        Offdiagonal entrie of W alraedy OK;
   !        for the diagonal ones, note that A_G(j,j)=W(j,j)-abs(sig(j))
   !        and D_G(j,j)=W(j,j), hence the rule below with umdbndmum1=1-2/kap
   !
       W(j,j)=umdbndmum1*W(j,j)-abs(sig(j))
   !       beta <- quadratic form (1^T * D_G * 1) ; [ 1 = vector of all ones ]
   !       alp is positive  if and only if AGe(j) is zero for all j;
   !              that is, if and only if A_G has 1 in its null space
       IF (j .eq. 1) THEN
          beta=v(j)
          alp=abs(AGe(j))
       ELSE
          beta=beta+v(j)
          alp=max(alp,abs(AGe(j)))
       END IF
    END DO
!
! Eventually, add to W the rank one correction
!                2
!       -------------------  D_G * 1 * 1^T * D_G
!       kap*(1^T * D_G * 1)
!
    beta=dbndmum1/beta
    DO j=1,SetDim
       DO k=1,SetDim
          W(j,k)=W(j,k)+beta*v(j)*v(k)
       END DO
    END DO
!
! Now, the rule is to accept the tentative aggregate if and only if
! W(setdim,setdim) is nonnegative definite
! Two cases:
!     alp == 0 , then W is singular, hence we check the positive
!                definitenes of W(setdim-1,setdim-1)
!     alp > 0  , then W is normally not singular and we require thus that
!                it is positive definite
!
    IF (alp.LT.repsmach*beta) THEN
       SetDim1=SetDim-1
    ELSE
       SetDim1=SetDim
    END IF
!
! Perform a Choleky factorization of W(setdim1,setdim1) and check
! for negative pivots; code expended for small dimensions (usual case)
!
    acc=.FALSE.
! acc has just be defined to FALSE; hence RETURN statement means rejection
!
    SELECT CASE (SetDim1)
    CASE (1)
       GOTO 11
    CASE (2)
       GOTO 12
    CASE (3)
       GOTO 13
    CASE (4)
       GOTO 14
    CASE (5)
       GOTO 15
    CASE (6)
       GOTO 16
    CASE (7)
       GOTO 17
    CASE (8)
       GOTO 18
    CASE DEFAULT
       CALL DPOTRF('U',SetDim1,W,mm,info)
       IF (info .NE. 0) RETURN
       GOTO 10
    END SELECT
18  CONTINUE
    IF (W(8,8) .LE. 0.0d0) RETURN
    W(7,7) = W(7,7) - (W(7,8)/W(8,8)) * W(7,8)
    T = W(6,8)/W(8,8)
    W(6,7) = W(6,7) - T * W(7,8)
    W(6,6) = W(6,6) - T * W(6,8)
    T = W(5,8)/W(8,8)
    W(5,7) = W(5,7) - T * W(7,8)
    W(5,6) = W(5,6) - T * W(6,8)
    W(5,5) = W(5,5) - T * W(5,8)
    T = W(4,8)/W(8,8)
    W(4,7) = W(4,7) - T * W(7,8)
    W(4,6) = W(4,6) - T * W(6,8)
    W(4,5) = W(4,5) - T * W(5,8)
    W(4,4) = W(4,4) - T * W(4,8)
    T = W(3,8)/W(8,8)
    W(3,7) = W(3,7) - T * W(7,8)
    W(3,6) = W(3,6) - T * W(6,8)
    W(3,5) = W(3,5) - T * W(5,8)
    W(3,4) = W(3,4) - T * W(4,8)
    W(3,3) = W(3,3) - T * W(3,8)
    T = W(2,8)/W(8,8)
    W(2,7) = W(2,7) - T * W(7,8)
    W(2,6) = W(2,6) - T * W(6,8)
    W(2,5) = W(2,5) - T * W(5,8)
    W(2,4) = W(2,4) - T * W(4,8)
    W(2,3) = W(2,3) - T * W(3,8)
    W(2,2) = W(2,2) - T * W(2,8)
    T = W(1,8)/W(8,8)
    W(1,7) = W(1,7) - T * W(7,8)
    W(1,6) = W(1,6) - T * W(6,8)
    W(1,5) = W(1,5) - T * W(5,8)
    W(1,4) = W(1,4) - T * W(4,8)
    W(1,3) = W(1,3) - T * W(3,8)
    W(1,2) = W(1,2) - T * W(2,8)
    W(1,1) = W(1,1) - T * W(1,8)
17  CONTINUE
    IF (W(7,7) .LE. 0.0d0) RETURN
    W(6,6) = W(6,6) - (W(6,7)/W(7,7)) * W(6,7)
    T = W(5,7)/W(7,7)
    W(5,6) = W(5,6) - T * W(6,7)
    W(5,5) = W(5,5) - T * W(5,7)
    T = W(4,7)/W(7,7)
    W(4,6) = W(4,6) - T * W(6,7)
    W(4,5) = W(4,5) - T * W(5,7)
    W(4,4) = W(4,4) - T * W(4,7)
    T = W(3,7)/W(7,7)
    W(3,6) = W(3,6) - T * W(6,7)
    W(3,5) = W(3,5) - T * W(5,7)
    W(3,4) = W(3,4) - T * W(4,7)
    W(3,3) = W(3,3) - T * W(3,7)
    T = W(2,7)/W(7,7)
    W(2,6) = W(2,6) - T * W(6,7)
    W(2,5) = W(2,5) - T * W(5,7)
    W(2,4) = W(2,4) - T * W(4,7)
    W(2,3) = W(2,3) - T * W(3,7)
    W(2,2) = W(2,2) - T * W(2,7)
    T = W(1,7)/W(7,7)
    W(1,6) = W(1,6) - T * W(6,7)
    W(1,5) = W(1,5) - T * W(5,7)
    W(1,4) = W(1,4) - T * W(4,7)
    W(1,3) = W(1,3) - T * W(3,7)
    W(1,2) = W(1,2) - T * W(2,7)
    W(1,1) = W(1,1) - T * W(1,7)
16  CONTINUE
    IF (W(6,6) .LE. 0.0d0) RETURN
    W(5,5) = W(5,5) - (W(5,6)/W(6,6)) * W(5,6)
    T = W(4,6)/W(6,6)
    W(4,5) = W(4,5) - T * W(5,6)
    W(4,4) = W(4,4) - T * W(4,6)
    T = W(3,6)/W(6,6)
    W(3,5) = W(3,5) - T * W(5,6)
    W(3,4) = W(3,4) - T * W(4,6)
    W(3,3) = W(3,3) - T * W(3,6)
    T = W(2,6)/W(6,6)
    W(2,5) = W(2,5) - T * W(5,6)
    W(2,4) = W(2,4) - T * W(4,6)
    W(2,3) = W(2,3) - T * W(3,6)
    W(2,2) = W(2,2) - T * W(2,6)
    T = W(1,6)/W(6,6)
    W(1,5) = W(1,5) - T * W(5,6)
    W(1,4) = W(1,4) - T * W(4,6)
    W(1,3) = W(1,3) - T * W(3,6)
    W(1,2) = W(1,2) - T * W(2,6)
    W(1,1) = W(1,1) - T * W(1,6)
15  CONTINUE
    IF (W(5,5) .LE. 0.0d0) RETURN
    W(4,4) = W(4,4) - (W(4,5)/W(5,5)) * W(4,5)
    T = W(3,5)/W(5,5)
    W(3,4) = W(3,4) - T * W(4,5)
    W(3,3) = W(3,3) - T * W(3,5)
    T = W(2,5)/W(5,5)
    W(2,4) = W(2,4) - T * W(4,5)
    W(2,3) = W(2,3) - T * W(3,5)
    W(2,2) = W(2,2) - T * W(2,5)
    T = W(1,5)/W(5,5)
    W(1,4) = W(1,4) - T * W(4,5)
    W(1,3) = W(1,3) - T * W(3,5)
    W(1,2) = W(1,2) - T * W(2,5)
    W(1,1) = W(1,1) - T * W(1,5)
14  CONTINUE
    IF (W(4,4) .LE. 0.0d0) RETURN
    W(3,3) = W(3,3) - (W(3,4)/W(4,4)) * W(3,4)
    T = W(2,4)/W(4,4)
    W(2,3) = W(2,3) - T * W(3,4)
    W(2,2) = W(2,2) - T * W(2,4)
    T = W(1,4)/W(4,4)
    W(1,3) = W(1,3) - T * W(3,4)
    W(1,2) = W(1,2) - T * W(2,4)
    W(1,1) = W(1,1) - T * W(1,4)
13  CONTINUE
    IF (W(3,3) .LE. 0.0d0) RETURN
    W(2,2) = W(2,2) - (W(2,3)/W(3,3)) * W(2,3)
    T = W(1,3)/W(3,3)
    W(1,2) = W(1,2) - T * W(2,3)
    W(1,1) = W(1,1) - T * W(1,3)
12  CONTINUE
    IF (W(2,2) .LE. 0.0d0) RETURN
    W(1,1) = W(1,1) - (W(1,2)/W(2,2)) * W(1,2)
11  CONTINUE
    IF (W(1,1) .LE. 0.0d0) RETURN
10  CONTINUE
!
!   all test passed: accept the aggregate
    acc=.TRUE.
!
    RETURN
  END SUBROUTINE dag2l_checktentagg_GF
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_SF(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,m1,lcg1,a1,ja1,ia1,idiag1,iext1,si1,rtent,jtent)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: further pairwise aggregation for symm. matrices [2].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: m1,ja1(*),ia1(n+1),idiag1(n),iext1(*),jtent(*),lcg1(m1,n)
    REAL(kind(0.0d0)) :: a1(*)
    REAL(kind(0.0d0)) :: si1(n),rtent(*)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be negative
!
!         ja1(*),ia1(n+1),idiag1(n): unaggregated matrix, in which the
!                threshold condition has to be checked
!         si1(n): si in that matrix
!
!         m1    : maximal number of nodes in aggregates from previous passes
!
!         lcg1(m1,n): list of aggregates from previous passes
!           lcg1(2,i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding corase grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!
!  rtent, jtent are working arrays of size >= the maximal degree of a node in
!               input matrix (a,ja,ia)
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER, PARAMETER :: nddl=0
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (lcg1(2,isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ntentleft=0
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(lcg1(2,j).EQ.0 .OR. ind(j).GE.0) CYCLE
          vals=-a(i)
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
             eta1=rsi+2*sig1
          ELSE
             del1=rsi+2*sig1
             eta1=rsi
          END IF
          IF (eta1 < 0.0d0) CYCLE
          IF (sig2 > 0.0d0) THEN
             del2=rsj
             eta2=rsj+2*sig2
          ELSE
             del2=rsj+2*sig2
             eta2=rsj
          END IF
          IF (eta2 < 0.0d0) CYCLE
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=(vals+(eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=vals+(eta1*eta2)/(eta1+eta2)
             IF (vals < 0.0d0) CYCLE
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_blocdia) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is possibly
          !    an acceptable pair; check if it is the best one
          !      and update the list of possible second choices in rtent, jtent
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          ntentleft=ntentleft+1
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. j.LT.ipair)  GOTO 9
          rtent(ntentleft)=tent
          jtent(ntentleft)=j
          CYCLE
9         CONTINUE
          rtent(ntentleft)=val
          jtent(ntentleft)=ipair
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       IF (ipair .EQ. 0) GOTO 25
20     CONTINUE
       !       Perform check in unagregated matrix, considering possible pairs
       !       in right order
       CALL dag2l_checktentagg_SF(isel,ipair           &
            ,m1,lcg1,a1,ja1,ia1,idiag1,iext1,si1,val,acc)
       !
       IF (.NOT.acc) THEN
          !     Tentative pair has been rejected, check for the next one, if any
          ipair = 0
          IF (ntentleft .GT.0) THEN
             i=1
             j=1
             DO WHILE (i .LE. ntentleft)
                IF (jtent(j).GT.0) THEN
                   tent=rtent(j)
                   IF (ipair.EQ.0) GOTO 22
                   IF (16*(tent-val).LT.-1) GOTO 22
                   IF (16*(tent-val).LT.1 .AND. j.LT.ipair) GOTO 22
                   GOTO 23
22                 CONTINUE
                   val=tent
                   ipair=jtent(j)
                   ijtent=j
23                 CONTINUE
                   i=i+1
                END IF
                j=j+1
             END DO
             ntentleft=ntentleft-1
             jtent(ijtent)=0
             GOTO 20
          END IF
       END IF
       !
25     CONTINUE
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_SF
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_checktentagg_SF                          &
       (isel,ipair,m1,lcg1,a,ja,ia,idiag,iext,si,val,acc)
!
!  Check the quality in the original matrix of an aggregate formed
!   by grouping two aggregates from previous pass(es)
!
!  INPUT: see dag2l_findpairs_SF
!
!  OUTPUT: acc(logical): acc=.true. if the new aggregate can be accepted and
!                        acc=.false. otherwise
!
    USE dag2l_mem
    IMPLICIT NONE
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(*),  val
    INTEGER :: m1, isel, ipair, lcg1(m1,*), ia(*), ja(*), idiag(*), iext(*)
    LOGICAL :: acc
!
    INTEGER, PARAMETER :: mm=max(2**(10),8)
    REAL(kind(0.0d0)) :: W(mm,mm), sig(mm), AGe(mm), v(mm)
    REAL(kind(0.0d0)) :: alpha, alp, tmp, beta, f1, f2
    INTEGER :: j,jj,k,l,m,info, setdim1, setdim, l2, k2
    INTEGER :: set(mm), l1, wdthT
    REAL(kind(0.0d0)) :: T
    LOGICAL :: exc
    REAL(kind(0.0d0)) ::  bndmum1m1,dbndmum1,umdbndmum1
    bndmum1m1=1.0d0/(kaptg_blocdia-1.0d0)
    dbndmum1=2.0d0/kaptg_blocdia
    umdbndmum1=1.0d0-dbndmum1
!
! Find indices of submatrix to be extracted and store them in set(1:setdim)
    IF (m1.eq.2) THEN
       IF (lcg1(2,isel) .LT. 0) THEN
          IF (lcg1(2,ipair) .LT. 0) THEN
             set(1)=lcg1(1,isel)
             set(2)=lcg1(1,ipair)
             setdim=2
          ELSE
             set(1)=lcg1(1,isel)
             set(2)=lcg1(1,ipair)
             set(3)=lcg1(2,ipair)
             setdim=3
          END IF
          l1=1
       ELSE
          IF (lcg1(2,ipair) .LT. 0) THEN
             set(1)=lcg1(1,isel)
             set(2)=lcg1(2,isel)
             set(3)=lcg1(1,ipair)
             setdim=3
          ELSE
             set(1)=lcg1(1,isel)
             set(2)=lcg1(2,isel)
             set(3)=lcg1(1,ipair)
             set(4)=lcg1(2,ipair)
             setdim=4
          END IF
          l1=2
       END IF
    ELSE
       l1=m1
       IF (lcg1(m1,isel).LT.0) l1=-lcg1(m1,isel)
       set(1:l1)=lcg1(1:l1,isel)
       l2=m1
       IF (lcg1(m1,ipair).LT.0) l2=-lcg1(m1,ipair)
       set(l1+1:l1+l2)=lcg1(1:l2,ipair)
       setdim=l1+l2
    END IF
!
! Sort indices in set(1:setdim) in increasing order
!   (makes susequent processing faster)
    exc=.TRUE.
    DO WHILE(exc)
       exc=.FALSE.
       DO l=2,SetDim
          IF( set(l)<set(l-1) )THEN
             jj=set(l)
             set(l)=set(l-1)
             set(l-1)=jj
             exc=.TRUE.
          END IF
       END DO
    END DO
!
! Extract submatrix of (the symmetric part of) input matrix and store it in W
! Store in AGe the mean row and column sum of input matrix for related indices
! (this is also the row sum of A_G)
    DO j=1,SetDim
       jj=Set(j)
       sig(j)=si(jj)
       IF (zerors) THEN
          W(j,j)=sig(j)
          AGe(j)=0.0d0
       ELSE
          W(j,j)=a(idiag(jj))
          AGe(j)=W(j,j)-sig(j)
       END IF
       l2=j+1
       DO l=l2,SetDim
          W(j,l)=0.0d0
          W(l,j)=0.0d0
       END DO
       k2=ia(jj+1)-1
       DO k=idiag(jj)+1,k2
          DO l=l2,SetDim
             m=Set(l)
             IF(ja(k)==m)THEN
                alpha=a(k)
                W(j,l)=alpha
                W(l,j)=alpha
                EXIT
             END IF
          END DO
       END DO
    END DO
!
    DO j=1,SetDim
   !       Set sig(j) equal to minus the sum of connections that are
   !       external to the tentative aggregate; that is, add the internal
   !       offdiagonal entries to current sig(j)
       DO k=1,SetDim
          IF (j.ne.k) THEN
             sig(j)=sig(j)+W(j,k)
          END IF
       ENDDO
   !       Heuristic if sig(j) is negative: set Sigma_G(j)=abs(sig(j))
   !         Hence need to correct the row sum of A_G stored in AGe
       IF (sig(j) < 0.0d0)  AGe(j)=AGe(j)+2*sig(j)
   !
   !       Set W = A_G ; offidagonal entries OK, need only to correct
   !       the diagonal
   !
       W(j,j)=W(j,j)-abs(sig(j))
   !
   !                    kap             1
   !       Now set W = ------ A_G  -  -----  M_G
   !                   kap-1          kap-1
   !
   !       Then W contains the matrix that need to be nonnegative definite to
   !       accept the aggregate according to the theory in [2],
   !       up to the rank one correction that is added below
   !
   !        Offdiagonal entrie of W alraedy OK [rule: offdiag(A_G)=ofdiag(MG)];
   !        for the diagonal ones, note that M_G(j,j)=A_G(j,j)+2*abs(sig(j)),
   !        hence the rule below with bndmum1m1=1/(kap-1)
   !
       tmp=2*abs(sig(j))
       W(j,j)=W(j,j)-bndmum1m1*tmp
   !
   !       Store M_G*1 = rowsum(A_G)+2*sig in v ; [ 1 = vector of all ones ];
   !       beta <- quadratic form (1^T * D_G * 1)
   !       alp is positive  if and only if AGe(j) is zero for all j;
   !              that is, if and only if A_G has 1 in its null space
       v(j)=tmp+AGe(j)
       IF (j .eq. 1) THEN
          beta=v(j)
          alp=abs(AGe(j))
       ELSE
          beta=beta+v(j)
          alp=max(alp,abs(AGe(j)))
       END IF
    END DO
!
! Eventually, add to W the rank one correction
!                   1
!       -------------------------  M_G * 1 * 1^T * M_G
!       (kap-1)*(1^T * M_G * 1)
!
    beta=bndmum1m1/beta
    DO j=1,SetDim
       DO k=1,SetDim
          W(j,k)=W(j,k)+beta*v(j)*v(k)
       END DO
    END DO
!
! Now, the rule is to accept the tentative aggregate if and only if
! W(setdim,setdim) is nonnegative definite
! Two cases:
!     alp == 0 , then W is singular, hence we check the positive
!                definitenes of W(setdim-1,setdim-1)
!     alp > 0  , then W is normally not singular and we require thus that
!                it is positive definite
!
    IF (alp.LT.repsmach*beta) THEN
       SetDim1=SetDim-1
    ELSE
       SetDim1=SetDim
    END IF
!
! Perform a Choleky factorization of W(setdim1,setdim1) and check
! for negative pivots; code expended for small dimensions (usual case)
!
    acc=.FALSE.
! acc has just be defined to FALSE; hence RETURN statement means rejection
!
    SELECT CASE (SetDim1)
    CASE (1)
       GOTO 11
    CASE (2)
       GOTO 12
    CASE (3)
       GOTO 13
    CASE (4)
       GOTO 14
    CASE (5)
       GOTO 15
    CASE (6)
       GOTO 16
    CASE (7)
       GOTO 17
    CASE (8)
       GOTO 18
    CASE DEFAULT
       CALL DPOTRF('U',SetDim1,W,mm,info)
       IF (info .NE. 0) RETURN
       GOTO 10
    END SELECT
18  CONTINUE
    IF (W(8,8) .LE. 0.0d0) RETURN
    W(7,7) = W(7,7) - (W(7,8)/W(8,8)) * W(7,8)
    T = W(6,8)/W(8,8)
    W(6,7) = W(6,7) - T * W(7,8)
    W(6,6) = W(6,6) - T * W(6,8)
    T = W(5,8)/W(8,8)
    W(5,7) = W(5,7) - T * W(7,8)
    W(5,6) = W(5,6) - T * W(6,8)
    W(5,5) = W(5,5) - T * W(5,8)
    T = W(4,8)/W(8,8)
    W(4,7) = W(4,7) - T * W(7,8)
    W(4,6) = W(4,6) - T * W(6,8)
    W(4,5) = W(4,5) - T * W(5,8)
    W(4,4) = W(4,4) - T * W(4,8)
    T = W(3,8)/W(8,8)
    W(3,7) = W(3,7) - T * W(7,8)
    W(3,6) = W(3,6) - T * W(6,8)
    W(3,5) = W(3,5) - T * W(5,8)
    W(3,4) = W(3,4) - T * W(4,8)
    W(3,3) = W(3,3) - T * W(3,8)
    T = W(2,8)/W(8,8)
    W(2,7) = W(2,7) - T * W(7,8)
    W(2,6) = W(2,6) - T * W(6,8)
    W(2,5) = W(2,5) - T * W(5,8)
    W(2,4) = W(2,4) - T * W(4,8)
    W(2,3) = W(2,3) - T * W(3,8)
    W(2,2) = W(2,2) - T * W(2,8)
    T = W(1,8)/W(8,8)
    W(1,7) = W(1,7) - T * W(7,8)
    W(1,6) = W(1,6) - T * W(6,8)
    W(1,5) = W(1,5) - T * W(5,8)
    W(1,4) = W(1,4) - T * W(4,8)
    W(1,3) = W(1,3) - T * W(3,8)
    W(1,2) = W(1,2) - T * W(2,8)
    W(1,1) = W(1,1) - T * W(1,8)
17  CONTINUE
    IF (W(7,7) .LE. 0.0d0) RETURN
    W(6,6) = W(6,6) - (W(6,7)/W(7,7)) * W(6,7)
    T = W(5,7)/W(7,7)
    W(5,6) = W(5,6) - T * W(6,7)
    W(5,5) = W(5,5) - T * W(5,7)
    T = W(4,7)/W(7,7)
    W(4,6) = W(4,6) - T * W(6,7)
    W(4,5) = W(4,5) - T * W(5,7)
    W(4,4) = W(4,4) - T * W(4,7)
    T = W(3,7)/W(7,7)
    W(3,6) = W(3,6) - T * W(6,7)
    W(3,5) = W(3,5) - T * W(5,7)
    W(3,4) = W(3,4) - T * W(4,7)
    W(3,3) = W(3,3) - T * W(3,7)
    T = W(2,7)/W(7,7)
    W(2,6) = W(2,6) - T * W(6,7)
    W(2,5) = W(2,5) - T * W(5,7)
    W(2,4) = W(2,4) - T * W(4,7)
    W(2,3) = W(2,3) - T * W(3,7)
    W(2,2) = W(2,2) - T * W(2,7)
    T = W(1,7)/W(7,7)
    W(1,6) = W(1,6) - T * W(6,7)
    W(1,5) = W(1,5) - T * W(5,7)
    W(1,4) = W(1,4) - T * W(4,7)
    W(1,3) = W(1,3) - T * W(3,7)
    W(1,2) = W(1,2) - T * W(2,7)
    W(1,1) = W(1,1) - T * W(1,7)
16  CONTINUE
    IF (W(6,6) .LE. 0.0d0) RETURN
    W(5,5) = W(5,5) - (W(5,6)/W(6,6)) * W(5,6)
    T = W(4,6)/W(6,6)
    W(4,5) = W(4,5) - T * W(5,6)
    W(4,4) = W(4,4) - T * W(4,6)
    T = W(3,6)/W(6,6)
    W(3,5) = W(3,5) - T * W(5,6)
    W(3,4) = W(3,4) - T * W(4,6)
    W(3,3) = W(3,3) - T * W(3,6)
    T = W(2,6)/W(6,6)
    W(2,5) = W(2,5) - T * W(5,6)
    W(2,4) = W(2,4) - T * W(4,6)
    W(2,3) = W(2,3) - T * W(3,6)
    W(2,2) = W(2,2) - T * W(2,6)
    T = W(1,6)/W(6,6)
    W(1,5) = W(1,5) - T * W(5,6)
    W(1,4) = W(1,4) - T * W(4,6)
    W(1,3) = W(1,3) - T * W(3,6)
    W(1,2) = W(1,2) - T * W(2,6)
    W(1,1) = W(1,1) - T * W(1,6)
15  CONTINUE
    IF (W(5,5) .LE. 0.0d0) RETURN
    W(4,4) = W(4,4) - (W(4,5)/W(5,5)) * W(4,5)
    T = W(3,5)/W(5,5)
    W(3,4) = W(3,4) - T * W(4,5)
    W(3,3) = W(3,3) - T * W(3,5)
    T = W(2,5)/W(5,5)
    W(2,4) = W(2,4) - T * W(4,5)
    W(2,3) = W(2,3) - T * W(3,5)
    W(2,2) = W(2,2) - T * W(2,5)
    T = W(1,5)/W(5,5)
    W(1,4) = W(1,4) - T * W(4,5)
    W(1,3) = W(1,3) - T * W(3,5)
    W(1,2) = W(1,2) - T * W(2,5)
    W(1,1) = W(1,1) - T * W(1,5)
14  CONTINUE
    IF (W(4,4) .LE. 0.0d0) RETURN
    W(3,3) = W(3,3) - (W(3,4)/W(4,4)) * W(3,4)
    T = W(2,4)/W(4,4)
    W(2,3) = W(2,3) - T * W(3,4)
    W(2,2) = W(2,2) - T * W(2,4)
    T = W(1,4)/W(4,4)
    W(1,3) = W(1,3) - T * W(3,4)
    W(1,2) = W(1,2) - T * W(2,4)
    W(1,1) = W(1,1) - T * W(1,4)
13  CONTINUE
    IF (W(3,3) .LE. 0.0d0) RETURN
    W(2,2) = W(2,2) - (W(2,3)/W(3,3)) * W(2,3)
    T = W(1,3)/W(3,3)
    W(1,2) = W(1,2) - T * W(2,3)
    W(1,1) = W(1,1) - T * W(1,3)
12  CONTINUE
    IF (W(2,2) .LE. 0.0d0) RETURN
    W(1,1) = W(1,1) - (W(1,2)/W(2,2)) * W(1,2)
11  CONTINUE
    IF (W(1,1) .LE. 0.0d0) RETURN
10  CONTINUE
!
!   all test passed: accept the aggregate
    acc=.TRUE.
!
    RETURN
  END SUBROUTINE dag2l_checktentagg_SF
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_GI(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,nddl,ldd,skipass                                      &
    ,ipc)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: initial pairwise aggregation (next levels) for general matrices [3].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: ldd(nddl),nddl
    LOGICAL :: skipass
    INTEGER :: ipc(n)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be nonnegative, and equal to
!             zero only at nodes that are to be kept outside the aggregation
!
!         nddl  : number of nodes i such that ind(i)=0
!
!         skipass: if true, pairwise aggregation is skipped; i.e.,
!               the coarsening only removes nodes i such that ind(i)==0
!
!         ipc: ipc(i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding coarse grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!                  ind(i)=0 means that i does not belong to any pair
!                           and is listed to ldd
!                           (detected from ind(i)=0 on input)
!          ldd(nddl): list of nodes such that ind(i)=0 on input & output
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! First check if isel has to be kept outside aggregation
       IF (ind(isel) .EQ. 0) THEN
          idd=idd+1
          nmark=nmark+1
          ldd(idd)=isel
          CYCLE
       END IF
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (ipc(isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ! Skip pairwise aggregation is skipass==true
       IF (skipass) THEN
          lcg(2,nc) = -1
          CYCLE
       END IF
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(ipc(j).EQ.0 .OR. ind(j).GE.0) CYCLE
          !   search for the corresponding entry in transposed matrix
          kk=0
          IF (i .LT. idiag(isel)) THEN
             j2=ia(j+1)-1
             DO jk=idiag(j)+1,j2
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ELSE
             DO jk=ia(j),idiag(j)-1
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ENDIF
          vals=-a(i)/2
          IF(kk .NE. 0) vals=vals-a(kk)/2
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
             eta1=2*si(isel)
             eta2=2*si(j)
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
             eta1=2*a(idiag(isel))
             eta2=2*a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
          ELSE
             del1=rsi+2*sig1
          END IF
          IF (sig2 > 0.0d0) THEN
             del2=rsj
          ELSE
             del2=rsj+2*sig2
          END IF
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=((eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=(eta1*eta2)/(eta1+eta2)
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_dampJac) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is
          !    an acceptable pair; check if it is the best one
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. j.LT.ipair)  GOTO 9
          CYCLE
9         CONTINUE
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
       IF (skipass) THEN
          WRITE(lout,223) IRANK,nddl,npc
       ELSE
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
       ENDIF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_GI
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_SI(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,nddl,ldd,skipass                                      &
    ,ipc)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: initial pairwise aggregation (next levels) for symm. matrices [2].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: ldd(nddl),nddl
    LOGICAL :: skipass
    INTEGER :: ipc(n)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be nonnegative, and equal to
!             zero only at nodes that are to be kept outside the aggregation
!
!         nddl  : number of nodes i such that ind(i)=0
!
!         skipass: if true, pairwise aggregation is skipped; i.e.,
!               the coarsening only removes nodes i such that ind(i)==0
!
!         ipc: ipc(i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding coarse grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!                  ind(i)=0 means that i does not belong to any pair
!                           and is listed to ldd
!                           (detected from ind(i)=0 on input)
!          ldd(nddl): list of nodes such that ind(i)=0 on input & output
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! First check if isel has to be kept outside aggregation
       IF (ind(isel) .EQ. 0) THEN
          idd=idd+1
          nmark=nmark+1
          ldd(idd)=isel
          CYCLE
       END IF
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (ipc(isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ! Skip pairwise aggregation is skipass==true
       IF (skipass) THEN
          lcg(2,nc) = -1
          CYCLE
       END IF
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(ipc(j).EQ.0 .OR. ind(j).GE.0) CYCLE
          vals=-a(i)
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
             eta1=rsi+2*sig1
          ELSE
             del1=rsi+2*sig1
             eta1=rsi
          END IF
          IF (eta1 < 0.0d0) CYCLE
          IF (sig2 > 0.0d0) THEN
             del2=rsj
             eta2=rsj+2*sig2
          ELSE
             del2=rsj+2*sig2
             eta2=rsj
          END IF
          IF (eta2 < 0.0d0) CYCLE
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=(vals+(eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=vals+(eta1*eta2)/(eta1+eta2)
             IF (vals < 0.0d0) CYCLE
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_blocdia) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is
          !    an acceptable pair; check if it is the best one
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. j.LT.ipair)  GOTO 9
          CYCLE
9         CONTINUE
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
       IF (skipass) THEN
          WRITE(lout,223) IRANK,nddl,npc
       ELSE
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
       ENDIF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_SI
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_GI1(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,nddl,ldd,skipass                                      &
    ,riperm,iperm)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: initial pairwise aggregation (top level) for general matrices [3].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: ldd(nddl),nddl
    LOGICAL :: skipass
    INTEGER :: iperm(n),riperm(n)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be nonnegative, and equal to
!             zero only at nodes that are to be kept outside the aggregation
!
!         nddl  : number of nodes i such that ind(i)=0
!
!         skipass: if true, pairwise aggregation is skipped; i.e.,
!               the coarsening only removes nodes i such that ind(i)==0
!
!         iperm(n),riperm(n): CMK and reverse CMK permutation, respectively
!           iperm(i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding coarse grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!                  ind(i)=0 means that i does not belong to any pair
!                           and is listed to ldd
!                           (detected from ind(i)=0 on input)
!          ldd(nddl): list of nodes such that ind(i)=0 on input & output
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       isel=riperm(ijs)
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! First check if isel has to be kept outside aggregation
       IF (ind(isel) .EQ. 0) THEN
          idd=idd+1
          nmark=nmark+1
          ldd(idd)=isel
          CYCLE
       END IF
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (iperm(isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ! Skip pairwise aggregation is skipass==true
       IF (skipass) THEN
          lcg(2,nc) = -1
          CYCLE
       END IF
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(iperm(j).EQ.0 .OR. ind(j).GE.0) CYCLE
          !   search for the corresponding entry in transposed matrix
          kk=0
          IF (i .LT. idiag(isel)) THEN
             j2=ia(j+1)-1
             DO jk=idiag(j)+1,j2
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ELSE
             DO jk=ia(j),idiag(j)-1
                IF (ja(jk) .EQ. isel) THEN
                   kk=jk
                   EXIT
                END IF
             END DO
          ENDIF
          vals=-a(i)/2
          IF(kk .NE. 0) vals=vals-a(kk)/2
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
             eta1=2*si(isel)
             eta2=2*si(j)
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
             eta1=2*a(idiag(isel))
             eta2=2*a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
          ELSE
             del1=rsi+2*sig1
          END IF
          IF (sig2 > 0.0d0) THEN
             del2=rsj
          ELSE
             del2=rsj+2*sig2
          END IF
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=((eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=(eta1*eta2)/(eta1+eta2)
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_dampJac) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is
          !    an acceptable pair; check if it is the best one
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. iperm(j).LT.iperm(ipair))  GOTO 9
          CYCLE
9         CONTINUE
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
       IF (skipass) THEN
          WRITE(lout,223) IRANK,nddl,npc
       ELSE
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
       ENDIF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_GI1
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_findpairs_SI1(n,a,ja,ia,idiag,iext,si,ind,lcg,nc     &
    ,nddl,ldd,skipass                                      &
    ,riperm,iperm)
! Performs pairwise aggregation according to Algorithms 4.2 and 4.3 in [2,3],
! with some heuristic enhancements to cover non diagonally  dominant matrices.
!
! Version: initial pairwise aggregation (top level) for symm. matrices [2].
!
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),lcg(2,*),nc
    REAL(kind(0.0d0)) :: a(*)
    REAL(kind(0.0d0)) :: si(n)
    INTEGER :: ldd(nddl),nddl
    LOGICAL :: skipass
    INTEGER :: iperm(n),riperm(n)
!
!  INPUT: n,ja(*),ia(n+1),idiag(n): aggregated matrix from previous pass
!           (CSR format with partial ordering of each row: lower
!            triangular part, next diag., and eventually upper triagular part;
!            idiag point to the diagonal element)
!
!         si(n) : vector s or \tile{s} in Algorithm 4.2 or 4.3 of [2,3]
!
!         ind(n): on input, entries in ind should be nonnegative, and equal to
!             zero only at nodes that are to be kept outside the aggregation
!
!         nddl  : number of nodes i such that ind(i)=0
!
!         skipass: if true, pairwise aggregation is skipped; i.e.,
!               the coarsening only removes nodes i such that ind(i)==0
!
!         iperm(n),riperm(n): CMK and reverse CMK permutation, respectively
!           iperm(i) = 0 means that node i has to be transferred unaggregated
!                        to the coarse grid, and lcg(2,j) set to 0
!                        for the corresponding coarse grid node j
!
!         zerors: if true, diagonal entries are not taken into account;
!                 instead, the matrix is treated as it would have mean
!                 column and row sum equal to zero
!
!  OUTPUT: nc   : number of pairs formed
!
!          lcg(2,nc): list of aggregates;
!                     lcg(2,i)=0 : lcg(1,i) is a singleton that was
!                                  forced to be transferred unaggregated
!                     lcg(2,i)=-1: lcg(1,i) is a singleton because no valid
!                                  pair has been found
!                     lcg(2,i)>0 : (lcg(1,i),lcg(2,i)) form the ith pair
!
!          ind(n): ind(i)=j means that i belongs to aggregate j
!                  ind(i)=0 means that i does not belong to any pair
!                           and is listed to ldd
!                           (detected from ind(i)=0 on input)
!          ldd(nddl): list of nodes such that ind(i)=0 on input & output
!
!----------------
! Local variables
!
    REAL(kind(0.0d0)) :: val,vals,valp,tent,rsi,rsj,epsr
    REAL(kind(0.0d0)) :: del1,del2,eta1,eta2,del12,sig1,sig2,rnd,vald
    INTEGER :: mindg,i,j,jj,k,kk,jk,isel,dg,ipair,nmark,idd
    INTEGER :: i1,i2,i3,ijs,ntentleft,npc,itrs,ijtent,j2
    LOGICAL :: acc
    INTEGER :: save_trs(nb_trs)
    CHARACTER (len=80) :: lout
    save_trs=0
    idd=0
    nmark=0
    nc=0
    ijs=1
    npc=0
    DO WHILE (nmark.LT.n)
       isel=ijs
       isel=riperm(ijs)
       ijs=ijs+1
       !
       ! Node isel has been selected
       !
       ! First check if isel has to be kept outside aggregation
       IF (ind(isel) .EQ. 0) THEN
          idd=idd+1
          nmark=nmark+1
          ldd(idd)=isel
          CYCLE
       END IF
       !
       ! Check if isel has already been processed
       IF (ind(isel) .GE. 0) CYCLE
       !
       ! A new aggregate is formed that contains isel
       nc = nc + 1
       lcg(1,nc) = isel
       nmark = nmark+1
       ind(isel) = nc
       ipair = 0
       !
       ! Check if isel has to be transferred unaggregated
       IF (iperm(isel) .EQ. 0) THEN
          lcg(2,nc) = 0
          npc=npc+1
          CYCLE
       END IF
       ! Skip pairwise aggregation is skipass==true
       IF (skipass) THEN
          lcg(2,nc) = -1
          CYCLE
       END IF
       !
       !  Try to form a pair with isel: follow list of neighbors
       !
       i2=ia(isel+1)-1
       DO i = ia(isel),i2
          !  CYCLE means: reject this neighbor, proceed with next one
          IF (i .EQ. idiag(isel)) CYCLE
          j = ja (i)
          !  check if j is available to form a pair
          IF(iperm(j).EQ.0 .OR. ind(j).GE.0) CYCLE
          vals=-a(i)
          IF (zerors) THEN
             rsi=0.0d0
             rsj=0.0d0
          ELSE
             rsi=-si(isel)+a(idiag(isel))
             rsj=-si(j)+a(idiag(j))
          END IF
          sig1=si(isel)-vals
          sig2=si(j)-vals
          !
          !   CYCLE instructions below: pair rejected because A_G is not
          !   nonnegative definite
          !
          !  Heuristic if sigj is negative: set Sigma_G(j)=abs(sigj) (j=1,2)
          IF (sig1 > 0.0d0) THEN
             del1=rsi
             eta1=rsi+2*sig1
          ELSE
             del1=rsi+2*sig1
             eta1=rsi
          END IF
          IF (eta1 < 0.0d0) CYCLE
          IF (sig2 > 0.0d0) THEN
             del2=rsj
             eta2=rsj+2*sig2
          ELSE
             del2=rsj+2*sig2
             eta2=rsj
          END IF
          IF (eta2 < 0.0d0) CYCLE
          IF (vals > 0.0d0) THEN
             epsr=repsmach*vals
             IF (ABS(del1) < epsr .AND. ABS(del2) < epsr) THEN
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del1) < epsr) THEN
                IF (del2 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE IF (ABS(del2) < epsr) THEN
                IF (del1 < -epsr) CYCLE
                valp=1.0d0+(eta1*eta2)/(vals*(eta1+eta2))
             ELSE
                del12=del1+del2
                IF (del12 < -epsr) CYCLE
                valp=vals+del1*del2/del12
                IF (valp < 0.0d0) CYCLE
                valp=(vals+(eta1*eta2)/(eta1+eta2))/valp
             END IF
          ELSE
             IF (del1 .LE. 0.0d0 .OR. del2 .LE. 0.0d0) CYCLE
             valp=vals+del1*del2/(del1+del2)
             IF (valp < 0.0d0) CYCLE
             vals=vals+(eta1*eta2)/(eta1+eta2)
             IF (vals < 0.0d0) CYCLE
             valp=vals/valp
          END IF
          !   main threshold test
          IF (valp > kaptg_blocdia) CYCLE
          !
          !    A_G is nonneagtive definite and passed the corresponding
          !    "quality" threshold test: (isel,j) is
          !    an acceptable pair; check if it is the best one
          !
          tent=valp
          IF (ipair.EQ.0) GOTO 10
          IF (16*(tent-val).LT.-1) GOTO 9
          IF (16*(tent-val).LT.1 .AND. iperm(j).LT.iperm(ipair))  GOTO 9
          CYCLE
9         CONTINUE
10        CONTINUE
          ipair = j
          val = tent
       ENDDO
       !
       IF (ipair .EQ. 0) THEN
       ! no valid pair found: isel left alone in aggregate nc
          lcg(2,nc) = -1
       ELSE
       ! pair nc is formed with isel and ipair
          lcg(2,nc) = ipair
          ind(ipair) = nc
          nmark = nmark+1
          itrs=min(nb_trs,FLOOR(val)+1)
          save_trs(itrs)=save_trs(itrs)+1
       END IF
    ENDDO
    IF (wfo) THEN
       i1=0
       DO i2=1,nb_trs
          i1=i1+save_trs(i2)
       END DO
       IF (skipass) THEN
          WRITE(lout,223) IRANK,nddl,npc
       ELSE
          IF (i1 == 0) THEN
               WRITE(lout,224) IRANK,nddl,npc
          ELSE
               WRITE(lout,222) IRANK,nddl,npc,      &
               (((save_trs(i2)*100)/i1),i2=1,nb_trs),9999999,i1*100/(n-nddl)
          END IF
       ENDIF
    ENDIF
    call mexPrintf(lout//char(10))
222 FORMAT (i3,'*',2i5,' *',20(1x,i3))
223 FORMAT (i3,'*',2i5,' * -- pass skipped --')
224 FORMAT (i3,'*',2i5,' * -- no aggregation at this pass --')
    RETURN
  END SUBROUTINE dag2l_findpairs_SI1
!------------------------------------------------------------------
  SUBROUTINE dag2l_lcgmix(nc,m,lcg1,lcg,lcgn)
    INTEGER :: nc,m,lcg1(m,*),lcg(2,*),lcgn(2*m,*),i,l,l1,l2
    IF (m.eq.2) THEN
       DO i=1,nc
          IF(lcg(2,i) .EQ. 0) THEN
             lcgn(1,i)=lcg1(1,lcg(1,i))
             lcgn(2,i)=0
             lcgn(4,i)=-1
          ELSE IF(lcg(2,i) .LT. 0) THEN
             IF (lcg1(2,lcg(1,i)) .LT. 0) THEN
                lcgn(1,i)=lcg1(1,lcg(1,i))
                lcgn(2,i)=-1
                lcgn(4,i)=-1
             ELSE
                lcgn(1,i)=lcg1(1,lcg(1,i))
                lcgn(2,i)=lcg1(2,lcg(1,i))
                lcgn(4,i)=-2
             END IF
          ELSE
             IF (lcg1(2,lcg(1,i)) .LT. 0) THEN
                IF (lcg1(2,lcg(2,i)) .LT. 0) THEN
                   lcgn(1,i)=lcg1(1,lcg(1,i))
                   lcgn(2,i)=lcg1(1,lcg(2,i))
                   lcgn(4,i)=-2
                ELSE
                   lcgn(1,i)=lcg1(1,lcg(1,i))
                   lcgn(2,i)=lcg1(1,lcg(2,i))
                   lcgn(3,i)=lcg1(2,lcg(2,i))
                   lcgn(4,i)=-3
                END IF
             ELSE
                IF (lcg1(2,lcg(2,i)) .LT. 0) THEN
                   lcgn(1,i)=lcg1(1,lcg(1,i))
                   lcgn(2,i)=lcg1(2,lcg(1,i))
                   lcgn(3,i)=lcg1(1,lcg(2,i))
                   lcgn(4,i)=-3
                ELSE
                   lcgn(1,i)=lcg1(1,lcg(1,i))
                   lcgn(2,i)=lcg1(2,lcg(1,i))
                   lcgn(3,i)=lcg1(1,lcg(2,i))
                   lcgn(4,i)=lcg1(2,lcg(2,i))
                END IF
             END IF
          END IF
       END DO
    ELSE
       DO i=1,nc
          IF(lcg(2,i) .EQ. 0) THEN
             lcgn(1,i)=lcg1(1,lcg(1,i))
             lcgn(2,i)=0
             lcgn(2*m,i)=-1
          ELSE
             lcgn(2,i)=-1
             l1=m
             IF (lcg1(m,lcg(1,i)).LT.0) l1=-lcg1(m,lcg(1,i))
             lcgn(1:l1,i)=lcg1(1:l1,lcg(1,i))
             IF(lcg(2,i) .LT. 0) THEN
                l=l1
             ELSE
                l2=m
                IF (lcg1(m,lcg(2,i)).LT.0) l2=-lcg1(m,lcg(2,i))
                lcgn(l1+1:l1+l2,i)=lcg1(1:l2,lcg(2,i))
                l=l1+l2
             END IF
             IF(l .LT. 2*m) lcgn(2*m,i)=-l
          END IF
       END DO
    END IF
    RETURN
  END SUBROUTINE dag2l_lcgmix
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_setind(nc,ndd,ldd,lcg,m,ind)
    INTEGER :: nc,m,lcg(m,*),nll,ldd(ndd),ind(*),i,k,l
    DO i=1,ndd
       ind(ldd(i))=0
    END DO
    DO i=1,nc
       l=m
       IF (lcg(m,i) .LT. 0) l=-lcg(m,i)
       DO k=1,l
          ind(lcg(k,i))=i
       END DO
    END DO
    RETURN
  END SUBROUTINE dag2l_setind
!-----------------------------------------------------------------------
  SUBROUTINE dag2l_setcg(n,a,ja,ia,idiag,iext,si,ind,lcg           &
       ,nc,a2,ja2,ia2,idiag2,iext2,si2,ysi,maxdg,iw1,iw2,w)
    USE dag2l_mem
    IMPLICIT NONE
    INTEGER :: n,ja(*),ia(n+1),idiag(n),iext(*),ind(n),icp(n),nc,lcg(2,nc)
    INTEGER :: ja2(*),ia2(n+1),idiag2(n),iext2(*),maxdg,iw1(nc),iw2(nc)
    REAL(kind(0.0d0)) :: a(*),a2(*),w(nc),vald
    REAL(kind(0.0d0)) :: si(n),si2(*),sii
    LOGICAL :: ysi
    INTEGER :: nz,nzu,i,jj,jc,jcol,ki,kb,kf,jpos
    !
    nz = 0
    iw1(1:nc)=0
    maxdg=0
    ia2(1)=1
    DO i = 1,nc
       sii=0.0d0
       vald=0.0d0
       nzu=0
       DO ki= 1,2
          jj = lcg(ki,i)
          IF (ki.EQ.1 .OR. jj.GT.0) THEN
             IF (ysi) sii=sii+si(jj)
             kf = ia(jj+1) - 1
             DO kb = ia(jj),kf
                jc = ja(kb)
                jcol = ind(jc)
                IF (jcol .GT. 0) THEN
                   IF (jcol .LT. i) THEN
                      jpos = iw1(jcol)
                      IF (jpos.EQ.0) THEN
                         nz = nz+1
                         ja2(nz) = jcol
                         iw1(jcol) = nz
                         a2(nz) = a(kb)
                      ELSE
                         a2(jpos) = a2(jpos) + a(kb)
                      ENDIF
                   ELSE IF (jcol .GT. i) THEN
                      jpos = iw1(jcol)
                      IF (jpos.EQ.0) THEN
                         nzu = nzu+1
                         iw2(nzu) = jcol
                         iw1(jcol) = nzu
                         w(nzu) = a(kb)
                      ELSE
                         w(jpos) = w(jpos) + a(kb)
                      ENDIF
                   ELSE
                      vald=vald+a(kb)
                      IF (ysi .AND. jc.NE.jj) sii=sii+a(kb)
                   ENDIF
                ENDIF
             ENDDO
          ENDIF
       ENDDO
       nz=nz+1
       a2(nz)=vald
       idiag2(i)=nz
       ja2(nz)=i
       a2(nz+1:nz+nzu)=w(1:nzu)
       ja2(nz+1:nz+nzu)=iw2(1:nzu)
       nz=nz+nzu
       maxdg=max(maxdg,nz-ia2(i))
       DO kb = ia2(i), nz
          iw1(ja2(kb))=0
       ENDDO
       IF (ysi) si2(i)=sii
       ia2(i+1)=nz+1
    ENDDO
    RETURN
  END SUBROUTINE dag2l_setcg
!-----------------------------------------------------------------------
