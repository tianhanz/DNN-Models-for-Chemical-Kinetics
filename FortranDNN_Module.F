*define FortarnDNN Module
      module fdnn
      integer, dimension(5) :: n_layer=(/9, 1600, 800, 400, 9/)
      integer::sample_num=1, epoch=5280, fileunit=20200
      double precision:: delta_t=1e-6
      ! define mean and std 
      double precision, allocatable :: Xmu(:,:), Xstd(:,:), 
     &                                 Ymu(:,:), Ystd(:,:)
      ! define input/output & weight，bias
      double precision, allocatable :: w(:,:),  b(:,:),
     &                                 matrix(:, :), input(:, :), 
     &                                 output(:, :)

      type:: dnn_layer_type
          double precision, allocatable :: w(:,:),b(:,:),x(:,:)
      end type
      type:: dnn_network_type
          type(dnn_layer_type), allocatable :: layer(:)
      end type
      type(dnn_network_type) :: dnn_h2

      contains

      subroutine readtxt(matrix,filename,fileunit,row,col)
            double precision,allocatable::matrix(:,:)
            character(len = 40) :: filename
            integer::row,col,i, fileunit
            filename = trim(filename)
            do i=1,row
              open(unit=fileunit,file=filename)
              read(fileunit,*) matrix(i,:)
            enddo
            close(fileunit)
      end subroutine readtxt


      subroutine readtxts()
      character(len = 40) :: filename
      ! read Xmu and Xstd
      write(filename, "(A12,I4,A8)") 'Model/model', epoch, '/Xmu.txt'
      call readtxt(Xmu,filename,fileunit,size(Xmu,1),size(Xmu,2))
      write(filename, "(A12,I4,A9)") 'Model/model', epoch, '/Xstd.txt'
      call readtxt(Xstd,filename,fileunit,size(Xstd,1),size(Xstd,2))
      write(filename, "(A12,I4,A8)") 'Model/model', epoch, '/Ymu.txt'
      call readtxt(Ymu,filename,fileunit,size(Ymu,1),size(Ymu,2))
      write(filename, "(A12,I4,A9)") 'Model/model', epoch, '/Ystd.txt'
      call readtxt(Ystd,filename,fileunit,size(Ystd,1),size(Ystd,2))
      ! read weight and bias
      do j = 1, size(n_layer) - 1
          allocate(w(n_layer(j+1),n_layer(j)))
          write(filename, "(A12,I4,A4,I1,A11)") 'Model/model', 
     & epoch, '/fc.',2*(j-1),'.weight.txt'
          call readtxt(w,filename,fileunit,size(w,1),size(w,2))
          allocate(b(n_layer(j+1), 1))
          write(filename, "(A12,I4,A4,I1,A9)") 'Model/model', 
     & epoch, '/fc.',2*(j-1),'.bias.txt'
          call readtxt(b,filename,fileunit,size(b,1),size(b,2))
          dnn_h2%layer(j)%w = w
          dnn_h2%layer(j)%b = b
          DEALLOCATE(w, b)
      enddo
      end subroutine readtxts

      subroutine initialization()
        allocate(input(n_layer(1),sample_num))
        allocate(output(n_layer(size(n_layer)),sample_num))
        allocate(Xmu(n_layer(1),1))
        allocate(Xstd(n_layer(1),1))
        allocate(Ymu(n_layer(size(n_layer)),1))
        allocate(Ystd(n_layer(size(n_layer)),1))
        allocate(dnn_h2%layer(size(n_layer)))
        do j = 1, size(n_layer) - 1
            allocate(dnn_h2%layer(j)%w(n_layer(j+1), n_layer(j)))
            allocate(dnn_h2%layer(j)%b(n_layer(j+1), 1))
            allocate(dnn_h2%layer(j)%x(n_layer(j), sample_num))
        enddo
        allocate(dnn_h2%layer(size(n_layer))%
     &      x(n_layer(size(n_layer)), sample_num))
        call readtxts()
      end subroutine initialization

      subroutine finalize()
        deallocate(xmu, xstd, ymu, ystd)
        DEALLOCATE(input, output)
        do j = 1, size(n_layer) - 1
            deallocate(dnn_h2%layer(j)%w, dnn_h2%layer(j)%b)
        enddo
        do j = 1, size(n_layer)
            deallocate(dnn_h2%layer(j)%x)
        enddo
        deallocate(dnn_h2%layer)
      end subroutine finalize


      function netOneStep(input,epoch,n_layer,sample_num)
        ! See modelclass.py line 100: netOneStep(*args)
        ! given input state, return ouput state after delta_t=1e-6s
        ! input: array; the input state composed by [T,P,Y_i]
        ! sample_num: the number of inputs,could be integer >=1
        integer,intent(in)::n_layer(5), sample_num, epoch
        double precision, allocatable:: input(:,:)
        double precision, allocatable:: input_normed(:,:),input_bct(:,:)
        double precision, allocatable:: output_normed(:,:)
        double precision, allocatable:: netOneStep(:,:)

        input_bct=input
        input_bct(3:,:) = BCT(input_bct(3:,:))

        input_normed=(input_bct-spread(Xmu(:,1),2,size(input,2)))
     & / spread(Xstd(:,1),2,size(input,2)) ! normalization
        output_normed=netForward(input_normed,epoch,n_layer,sample_num)
        netOneStep = output_normed 
     & * spread(Ystd(:,1),2,size(output_normed,2))
     & + spread(Ymu(:,1),2,size(output_normed,2))
        netOneStep = netOneStep*delta_t + input_bct  
        netOneStep(3:,:)=IBCT(netOneStep(3:,:))
      end function netOneStep


      function netForward(input,epoch,n_layer,sample_num) 
        integer,intent(in)::n_layer(5),sample_num,epoch
        double precision, allocatable:: input(:,:),netForward(:,:)
        double precision, allocatable:: w(:,:),  b(:,:), tmp(:, :)
        ! network forward oneStep
        ! DNN matrix operation
        dnn_h2%layer(1)%x = input
        do j=1,size(n_layer)-1
            dnn_h2%layer(j+1)%x = 
     &          matmul(dnn_h2%layer(j)%w, dnn_h2%layer(j)%x) + 
     &          spread(dnn_h2%layer(j)%b(:,1),2, sample_num)
            if (j.lt.size(n_layer)-1) then 
                dnn_h2%layer(j+1)%x = GELU(dnn_h2%layer(j+1)%x) 
            end if
        end do
        netForward=dnn_h2%layer(size(n_layer))%x
      end function netForward


      subroutine saveOutput(result,sample_num)
          double precision, allocatable::result(:,:)
          integer::sample_num, fileunit=20200
          open(unit=fileunit, file='output.txt' ,position='append')
          do k=1,sample_num
           write(fileunit,'(9e24.16)') result(:,k)
          end do
        close(fileunit)
      end subroutine saveOutput


      pure function GELU(x) result(res)
        ! Gaussian activation function.
        ! See modelclass.py line 27 : class MyGELU(), same definition
        double precision, intent(in) :: x(:,:)
        double precision :: res(size(x,1),size(x,2))
        double precision,parameter::Pi=3.141592653600000
        res = 0.5*x*(1+tanh(sqrt(2/Pi)*(x+0.044715*x**3)))
      end function GELU


      pure function BCT(x) result(res)
        ! Box-Cox Transformation
        double precision, intent(in) :: x(:,:)
        double precision :: res(size(x,1),size(x,2))
        double precision,parameter::lambda=0.1
        res = (x**lambda-1)/lambda
      end function BCT


      pure function IBCT(x) result(res)
        ! Inverse Box-Cox Transformation
        double precision, intent(in) :: x(:,:)
        double precision :: res(size(x,1),size(x,2))
        double precision,parameter::lambda=0.1
        res = (lambda*x+1)**(1/lambda)
      end function IBCT
      end module fdnn
