lenet: lenet.cu cnnfunc.cu lib/convolution/convolution.cu lib/relu/relu.cu lib/maxpooling/maxpooling.cu lib/classifier/classifier.cu -o test
	nvcc lenet.cu cnnfunc.cu lib/convolution/convolution.cu lib/relu/relu.cu lib/maxpooling/maxpooling.cu lib/classifier/classifier.cu -o test -o lenet
	nvcc lenet_cpu.cu cnnfunc.cu lib/convolution/convolution.cu lib/relu/relu.cu lib/maxpooling/maxpooling.cu lib/classifier/classifier.cu -o test -o lenet_cpu
