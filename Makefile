lenet: lenet.cu cnnfunc.cu
	nvcc lenet.cu cnnfunc.cu -o lenet
	nvcc lenet_cpu.cu cnnfunc.cu -o lenet_cpu