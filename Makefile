lenet: lenet.cu cnnfunc.cu
	nvcc lenet.cu cnnfunc.cu -lineinfo -o lenet
	nvcc lenet_cpu.cu cnnfunc.cu -lineinfo -o lenet_cpu
