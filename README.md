# Adobe SDK New GPU Setup

 _Create an AE/PR plugin which uses CUDA/OpenCL/Metal_

*You will want to make sure you are using the newest version of the sdk*

## Preparing the environment
#### Before editing or compiling code, we need to prepare our computer to compile GPU plugins

### Windows
1. Install <a href="https://www.boost.org/">BOOST</a>. (Run bootstrap.bat and run ./b2)
2. Install the <a href="https://developer.nvidia.com/cudadownloads">CUDA SDK</a>. (AE 18.2 currently uses CUDA 10.1 update 2)
3. Environment variables
    * CUDA_SDK_BASE_PATH (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1)
    * BOOST_BASE_PATH (C:\boost_1_71_0)
4. Change .cu file settings
    * Open "SDK_Invert_ProcAmp" Project
    * Right-click ""SDK_Invert_ProcAmp.cu" 
    * "Properties"
    * "Custom Build Tool" -> "General" -> Edit the "Command Line"
    * Change the version number after "/MSVC/" to match the version on your computer (ex "/MSVC/14.12.25827")

### Mac
1. Download and install <a href="https://www.boost.org/">BOOST</a> to appear like: "/usr/local/Cellar/boost/1.78.0/include"
2. Open "SDK_Invert_ProcAmp" Project
3. In the "Project" and "Target" "Build Settings", add the BOOST root path to the property "Header Search Paths" (ex "/usr/local/Cellar/boost_1_78_0")

#### Now you can save and use this project file as a template for any plugin you want to port to be a GPU accelerated one

#### What is cool about this setup, is we can write all the GPU code in C++ (CUDA), and it will automatically compile for OpenCL and Metal as well!

## Understanding the code

#### Although most things are already setup, it's important to know what's been added to make the GPU acceleration work, so you may make changes if you wish

### .h file
```
// here we are just including important files, depending if you're compiling on Win/Mac/M1
#if _WIN32
#include <CL/cl.h>
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "SDK_Invert_ProcAmp_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
	#include <Windows.h>
#endif
```

```
// this is some metal specific stuff
#if HAS_METAL
	/*
	 ** Plugins must not rely on a host autorelease pool.
	 ** Create a pool if autorelease is used, or Cocoa convention calls, such as Metal, might internally autorelease.
	 */
	struct ScopedAutoreleasePool
	{
		ScopedAutoreleasePool()
		:  mPool([[NSAutoreleasePool alloc] init])
		{
		}
	
		~ScopedAutoreleasePool()
		{
			[mPool release];
		}
	
		NSAutoreleasePool *mPool;
	};
#endif 
```

```
// this is 1 of the two structs defined in the header
{
	float mBrightness;
	float mContrast;
	float mHueCosSaturation;
	float mHueSinSaturation;
} InvertProcAmpParams;
```

### .cpp file
```
// if we're using CUDA we need to include this
#if HAS_CUDA
	#include <cuda_runtime.h>
	// SDK_Invert_ProcAmp.h defines these and are needed whereas the cuda_runtime ones are not.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "SDK_Invert_ProcAmp.h"

// important code for OpenCL
inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	} else {
		// set a breakpoint here to pick up OpenCL errors.
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}
#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

// here we are defining our pixel processing function that will be in the CUDA (.cu) file
extern void Exposure_CUDA(
	float const *src,
	float *dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height);
```

#### If you're compiling for AE, you will want to include this outflag
```out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;```

```
// more required metal code (but continuing to note the required stuff)
#if HAS_METAL
	PF_Err NSError2PFErr(NSError *inError)
	{
		if (inError)
		{
			return PF_Err_INTERNAL_STRUCT_DAMAGED;  //For debugging, uncomment above line and set breakpoint here
		}
		return PF_Err_NONE;
	}
#endif //HAS_METAL


// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
    // here are two of the openCL objects, if you want, you could just use 1
	cl_kernel invert_kernel;
	cl_kernel procamp_kernel;
};

#if HAS_METAL
	struct MetalGPUData
	{
		id<MTLComputePipelineState>invert_pipeline;
		id<MTLComputePipelineState>procamp_pipeline;
	};
#endif
```

#### GPUDeviceSetup
##### This is where we setup any of the 3 types of GPUs
* CUDA just adds an outflag
* OpenCL sets up these properties "InvertColorKernel" and "ProcAmp2Kernel"
```
// You can add or remove these at will based on your project
if (!err) {
			cl_gpu_data->invert_kernel = clCreateKernel(program, "InvertColorKernel", &result);
			CL_ERR(result);
		}

		if (!err) {
			cl_gpu_data->procamp_kernel = clCreateKernel(program, "ProcAmp2Kernel", &result);
			CL_ERR(result);
		}
```
* Metal does quite a bit, setting up
    * Names: 
    ```
    NSString *invert_name = [NSString stringWithCString:"InvertColorKernel" encoding:NSUTF8StringEncoding];
    ```
    * Properties:
    ```metal_data->invert_pipeline = [device
    newComputePipelineStateWithFunction:invert_function error:&error];
				err = NSError2PFErr(error);
    ```
    * And the same previous outflag


#### GPUDeviceSetdown
##### This is where the GPU memory is handled and disposed of

#### Don't worry, we're almost to editing the GPU code and modifying the pixels themselves (the fun part)

#### Entry point function
##### Here we have a new PF_Cmd_SMART_RENDER_GPU case
##### This will now detect when a GPU is ready to be used (usually in File -> Project Settings -> Video Rendering and Effects). We are sending both the CPU and GPU to the same SmartRender() function, however, we will pass a bool as to whether or not we're using GPU
##### In SmartRender(), we then branch to either SmartRenderCPU() or SmartRenderGPU() depending on the incoming bool

#### SmartRenderGPU
##### Here is just a bunch more setup, including creating a few GPU worlds, "src_mem", "dst_mem", and "im_mem". 
##### As per the usual SmartFX setup, we also instantiate our pre-defined structs, so we can send things like UI Parameter data, down into our pixel modifying functions
##### Following this are 3 seconds, 1 for OpenCL, 1 for CUDA, and 1 for Metal. There are some important bits here, especially if you plan to modify the properties in any way.

#### OpenCL
```
// Set the arguments
// This is setting up the openCL arguments based on the struct data we're using
// You will want this to be in the right order according to your GPU function argument order
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_im_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &invert_params.mHeight));

// This is where we actually launch the OpenCL operations
// Launch the kernel
size_t threadBlock[2] = { 16, 16 };
size_t grid[2] = { RoundUp(invert_params.mWidth, threadBlock[0]), RoundUp(invert_params.mHeight, threadBlock[1])};

CL_ERR(clEnqueueNDRangeKernel(
								 (cl_command_queue)device_info.command_queuePV,
								 cl_gpu_dataP->invert_kernel,
								 2,
								 0,
							     grid,
							     threadBlock,
							        0,
							      0,
							      0));
```
##### The above OpenCL code appears twice in the example project, 1 for each of the GPU functions this effect runs. The main thing to make sure you remember is:
1. Setup each of your GPU arguments in order ```CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_src_mem));```
2. Launch the OpenCL kernel and make sure you change/update ```cl_gpu_dataP->invert_kernel```


#### CUDA
##### Compared to the others, this is cake
##### Simply call your function (defined in this file and the .cu)
##### Be sure to provide the right arguments in the right order
```
Exposure_CUDA(
				(const float *)src_mem,
				(float *)dst_mem,
				invert_params.mSrcPitch,
				invert_params.mDstPitch,
				invert_params.m16f,
				invert_params.mWidth,
				invert_params.mHeight);
```

_A side note on messing around with these function calls... Be sure you double check your input and output worlds. src_mem means input world. im_mem means intermediate world. dst_mem means output world. Too many times I have spent hours changing lines of code, only to realise I am using "im_mem" as my output world, when the final destination should be "dst_mem"_

#### Metal
##### Overall, the metal code is a bit unfamiliar to me, but here are the important parts
```
// Params/Arguments Setup
id<MTLBuffer> procamp_param_buffer = [[device newBufferWithBytes:&procamp_params
						length:sizeof(ProcAmp2Params)
						options:MTLResourceStorageModeManaged] autorelease];

// Defining Metal worlds
id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
id<MTLBuffer> im_metal_buffer = (id<MTLBuffer>)im_mem;
id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;


// thread setup before we run
MTLSize threadsPerGroup1 = {[metal_dataP->invert_pipeline threadExecutionWidth], 16, 1};
			MTLSize numThreadgroups1 = {DivideRoundUp(invert_params.mWidth, threadsPerGroup1.width), DivideRoundUp(invert_params.mHeight, threadsPerGroup1.height), 1};

// actual metal execution
[computeEncoder setComputePipelineState:metal_dataP->invert_pipeline];
			[computeEncoder setBuffer:src_metal_buffer offset:0 atIndex:0];
			[computeEncoder setBuffer:im_metal_buffer offset:0 atIndex:1];
			[computeEncoder setBuffer:invert_param_buffer offset:0 atIndex:2];
			[computeEncoder dispatchThreadgroups:numThreadgroups1 threadsPerThreadgroup:threadsPerGroup1];
```
##### Once again, with the above code, there are 2 versions of the same functions, one for each of the GPU functions that are run
##### Now let's do the fun stuff


### CUDA .cu file
##### There are 2 functions in here, Exposure_CUDA() and ProcAmp_CUDA(). Let's take a look at Exposure to keep it simple
```
// this is where all of those previous arguments we setup come in
// we now call the actual CUDA function "InvertColorKernel" (I didn't change the name to match Exposure)
void Exposure_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    // this is basically just calling another function, using the above thread/block count, and passing along our src, dst, and arguments
	InvertColorKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height);

	cudaDeviceSynchronize();
}
```
#### The good stuff
```
// this is the function definition
// each argument is cast to a certain data type using the formatting
// ((castType)(varName))
// you can keep adding them without commas
// You only need a comma after your destination world and the last argument
// There is an extra argument ((uint2)(inXY)(KERNEL_XY))) which is required
GF_KERNEL_FUNCTION(InvertColorKernel,
	((const GF_PTR(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

		pixel.x = fmax(fmin(1.0f, pixel.x), 0.0f);
		pixel.y = fmax(fmin(1.0f, pixel.y), 0.0f);
		pixel.z = fmax(fmin(1.0f, pixel.z), 0.0f);
		pixel.w = fmax(fmin(1.0f, pixel.w), 0.0f);

		pixel.x = 1.0 - pixel.x;
		pixel.y = 1.0 - pixel.y;
		pixel.z = 1.0 - pixel.z;

		WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}
```
#### Quickly modifying this function, we can pretend our UI actually is bringing in an exposure value, and do simple, but GPU accelerated functions

```
GF_KERNEL_FUNCTION(InvertColorKernel,
	((const GF_PTR(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((float)(exposure))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

		pixel.x = fmax(fmin(1.0f, pixel.x), 0.0f);
		pixel.y = fmax(fmin(1.0f, pixel.y), 0.0f);
		pixel.z = fmax(fmin(1.0f, pixel.z), 0.0f);
		pixel.w = pixel.w;

		WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}
```
#### A few final notes/tips
```
pixel.x == B
pixel.y == G
pixel.z == R
pixel.w == A
```

*in16f is important, and when we write output pixels, will automagically cast the values to the right 8/16/32bpc range*
