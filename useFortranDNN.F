* demo for DNN prediction using Fortran
* given input=[T,P(atm),Y_i]
* return ouptut=[T,P(atm),Y_i] after delta_t=1e-6
* chemical mechnism: Evans-Schexnayder(1980) kinetic mechanism
      program main
      use fdnn
      sample_num = 1 ! number of data points to be predicted by DNN
      call initialization()
      input = 0
      input(1,1) = 1300 ! temperature 
      input(2,1) = 1 ! pressure
      input(7,1) = 2.263540069710073543e-01 ! O2
      input(8,1) = 2.852238752756739576e-02 ! H2
      input(9,1) = 7.451236055014253123e-01 ! N2  
      output = netOneStep(input,epoch,n_layer,sample_num)
      write(*,*) output
      end program
