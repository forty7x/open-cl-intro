
#include <cl.h>
#include <cl_platform.h>
#include <iostream>
#include <vector>


int main() {

    cl_int err;
    // Initialize data
    float a = 5.0f;
    float b = 10.0f;
    float c = 0.0f;

    // The OpenCL kernel code will be input as a string of characters
    const char* kernelSource =
        "__kernel void add_floats(float a, float b, __global float* c) { \n"
        "    c[0] = a + b; \n"
        "} \n";

    // Get the number of platforms available
    // A platform is a specific vendor's implementation of the OpenCL specification.
    // Since Nvidia's openCL SDK is installed on my computer, it will detect just 1 platform on my computer
    // So if a computer has nvidia and AMD cards installed, thats 2 platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(1, nullptr, &num_platforms);
    if (err < 0) {
        perror("Cant find any platforms");
        exit(1);
    }

    // The cl_platform_id structure represents a platform
    // So for just 1 platform, we will have 1 cl_platform_id structure 
    std::vector<cl_platform_id> platforms(num_platforms);

    // Identify a platform
    // first argument in the function is the max number of platforms you want to detect
    err = clGetPlatformIDs(1, platforms.data(), nullptr);
    if (err < 0) {
        perror("Cant find any platforms");
        exit(1);
    }


    // Get the number of available openCL devices for the first and only platform in our case
    // By setting either of the last two arguments to NULL, we can use the clGetDeviceIDs function 
    // in two different ways. One will populate an array with device IDs, and the other will
    // give us the number of devices detected
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 3, nullptr, &num_devices);
    if (err < 0) {
        perror("No device found");
        exit(1);
    }

    // Once we know how many devices we have detected, we get the device IDs using the same function
    // setting the last argument to NULL
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err < 0) {
        perror("Cant find any device IDs");
        exit(1);
    }

    // Get the queried device name and print it
    char device_name[1024];
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Device name: " << device_name << std::endl;



    // Create an OpenCL context and command queue for the chosen device.
    cl_context context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, nullptr);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    // Create memory buffers on the device for the result
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);


    // Create a program from our kernel code
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);

    // Build program
    clBuildProgram(program, 1, &devices[0], nullptr, nullptr, nullptr);

    // Create the openCL kernel
    // 2nd argument will be the name of the kernel function
    cl_kernel kernel = clCreateKernel(program, "add_floats", nullptr);

    // Set the arguments for the kernel code.
    clSetKernelArg(kernel, 0, sizeof(float), &a);
    clSetKernelArg(kernel, 1, sizeof(float), &b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);

    // Execute the kernel code
    size_t work_units_per_kernel = 1;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &work_units_per_kernel, nullptr, 0, nullptr, nullptr);

    // Read the results from the device back to the host.
    clEnqueueReadBuffer(commandQueue, result_buffer, CL_TRUE, 0, sizeof(float), &c, 0, nullptr, nullptr);

    // Clean up memory
    clReleaseMemObject(result_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    std::cout << "Result is :" << c << std::endl;

    return 0;
}
