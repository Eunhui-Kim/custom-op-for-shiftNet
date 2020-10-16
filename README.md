# custom-op for active shift layer
custom-op for active shift layer in tensorflow 1.15 v. 

This code is based on original active shift layer paper open code which is in ASL URL https://github.com/jyh2986/Active-Shift-TF .

Active Shift Layer (ASL)
The active shift layer is
 - Uses depthwise shift
 - new shift parameters for each channel
 - New shift parameters(alpha, beta) are learnable

# pre-requisite
Note that this code is tested only in the environment decribed below. Mismatched versions does not guarantee correct execution.
 - Ubuntu kernel ver. 4.15.0-117-generic #118~16.04.1
 - Tensorflow 1.15.3
 - Cuda 10.0
 - g++ 7.5.0
 - python 3.7

[Experience share for Tensorflow installation]
I modified the ASL URL according to the guide in custom-op which is in https://github.com/tensorflow/custom-op .
Since I expect efficient operation, I installed tensorflow 1.15.3 gpu version as source-level.
(I also tested pip install tensorflow-gpu==1.15.3, however it's slower than source-level installation,
 in my case, it can make 3 times acceleration for training. 
 In imagenet dataset, for ResNet50 model of 256 batch, it takes about 3 weeks on pip installed V100 2-GPU server, 
 while it takes about 1 week on source-level installed RTX2080 2-GPU server.)


# Build
If the environment is the same as pre-requisite, you can install directly in artifacts directory.
by command 'pip install tensorflow_custom_ops-0.0.1-cp37-cp37m-linux_x86_64.whl tensorflow==1.15.3'
Since I want to use tensorpack with active-shift-layer, 
if you have a plan to use tensorflow 1.x version, then you have to point out the tensorflow version after whl name.

# Testing
If it installed correctly, you can call the python library as follows.
python
>> from tensorflow_active_shift.python.ops import active_shift2d_op

If you can call the library, then you can test the operation by using test function.
In the path, 
./custom-op/tensorflow_active_shift/python/ops
 python test_forward_ASL.py
 python test_gradient_ASL.py
 
 While you run the test, you can see the 3 OK sign.
 Since compatibility issue, one might skipped.
 However, it's ok.
 

 
