module fitting
contains
 
      function polyfit(vx, vy, d)
        implicit none
        integer, intent(in)                   :: d
        integer, parameter                    :: dp = selected_real_kind(15, 307)
        real(dp), dimension(d+1)              :: polyfit
        real(dp), dimension(:), intent(in)    :: vx, vy
 
        real(dp), dimension(:,:), allocatable :: X
        real(dp), dimension(:,:), allocatable :: XT
        real(dp), dimension(:,:), allocatable :: XTX
 
        integer :: i, j
 
        integer     :: n, lda, lwork
        integer :: info
        integer, dimension(:), allocatable :: ipiv
        real(dp), dimension(:), allocatable :: work
 
        n = d+1
        lda = n
        lwork = n
 
        allocate(ipiv(n))
        allocate(work(lwork))
        allocate(XT(n, size(vx)))
        allocate(X(size(vx), n))
        allocate(XTX(n, n))
 
        ! prepare the matrix
        do i = 0, d
        do j = 1, size(vx)
          X(j, i+1) = vx(j)**i
        end do
        end do
 
        XT  = transpose(X)
        XTX = matmul(XT, X)
 
        ! calls to LAPACK subs DGETRF and DGETRI
        call DGETRF(n, n, XTX, lda, ipiv, info)
        if ( info /= 0 ) then
        print *, "problem"
        return
        end if
        call DGETRI(n, XTX, lda, ipiv, work, lwork, info)
        if ( info /= 0 ) then
        print *, "problem"
        return
        end if
 
        polyfit = matmul( matmul(XTX, XT), vy)
 
        deallocate(ipiv)
        deallocate(work)
        deallocate(X)
        deallocate(XT)
        deallocate(XTX)
 
      end function
 
end module


SUBROUTINE forcfortran(SF,x_range,y_range,matrix_z,a_1_2,point_2,a_1_4,b_1_4,grid,k)
      use fitting
      IMPLICIT NONE
      !f2py intent(in) SF,point_1,a_1_2,point_2,point_4,a_1_4,b_1_4
      !f2py intent(in) x_range
      !f2py intent(in) y_range
      !f2py intent(in) rawdata,matrix_z
      !f2py intent(out) grid,k
      INTEGER :: SF,n,m,mm,nn,SFs,i,j,k
      Real(KIND=8) :: x,y
      Real(KIND=8) :: a_1_2,a_1_4,b_1_4
      Real(KIND=4) :: t
      Real, Dimension (:) :: x_range,y_range,point_2
      Real matrix_z(:,:),grid(10*10000,3)
      Real, Dimension(2) :: mn
      !Real, Dimension(3) :: list
      !Real, Dimension(10000,10000) :: grid_data
      m = size(x_range)
      n = size(y_range)
      mn = size(matrix_z)
      SFs = 2*SF+1
      k=1
      DO mm = 1,m
        DO nn = 1,n
          x=x_range(mm)
          y=y_range(nn)
          If(y-x<=-0.1)then
              If(y-x-a_1_2>=-0.01)then
                  If(y-a_1_4*x-b_1_4<=-0.01)then
                      If(y-point_2(2)>=-0.01)then
                          k=1
                          DO i = 1,SFs
                          DO j = 1,SFs
                            If(m-SF+i<=mn(1)-1)then
                                If(n-SF+j<=mn(2)-1)then
                                    grid(k,:)=(/x_range(mm-SF+i),y_range(nn-SF+i),matrix_z(mm-SF+i,nn-SF+j)/)
                                    k=k+1
                                end If
                             end If
                          ENDDO
                          ENDDO
                          t=polyfit(grid(:,1),grid(:,2),2)
                          !grid_data(mm,nn)=grid
                      end If
                  end If
              end If
          end If
        ENDDO
      ENDDO

      
END SUBROUTINE forcfortran

