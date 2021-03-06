// conv_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
extern "C"
{
	#include <mkl.h>
}


int main()
{
	std::cout << "Start executing \n";
	// Initialize needed variables
	int status = 0;
	VSLCorrTaskPtr task;
	constexpr int inputs_shape[3] = { 68,52,100 };
	constexpr int weights_shape[3] = { 5,4,4 };
	constexpr int outputs_shape[3] = {64,48,96 };

	MKL_INT Rmin[] = { 0, 0 };
	MKL_INT Rmax[] = { inputs_shape[0] + weights_shape[0] - 1, inputs_shape[1] + weights_shape[1] - 1 };
	MKL_INT mode = VSL_CORR_MODE_AUTO;
	MKL_INT rank = 3;
	MKL_INT inputs_stride[3] = { 1,1,1 };
	MKL_INT weights_stride[3] = { 1,1,1 };
	MKL_INT outputs_stride[3] = { 1,64,48*64 };
	
	MKL_INT start[3] = {-63, -47, -95 };
	std::cout << "Succesfully initialized variables \n";
	std::vector<float> inputs(inputs_shape[0]  * inputs_shape[1]  * inputs_shape[2],1.0f);
	std::vector<float> weights(weights_shape[0] * weights_shape[1] * weights_shape[2],1.0f);
	//Set first value to other value for testing purposes
	weights[0, 0, 0] = 6.f;
	std::vector<float> outputs(outputs_shape[0] * outputs_shape[1] * outputs_shape[2] + 1000,777.0f);
	
	std::cout << "Succesfully initialized arrays \n";
	
	// Time code
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	//Create task descriptor
	status = vslsCorrNewTask(&task, mode, rank, inputs_shape, weights_shape, outputs_shape);
	if (status != VSL_STATUS_OK)
	{
		std::cout << "ERROR: creation of job failed, exit with " << std::to_string(status) << "\n";
		return 1;
	}
	std::cout << "Succesfully created task descriptor \n";

	//Set the value of the parameter start correctly
	status = vslCorrSetStart(task, start);
	if (status != VSL_STATUS_OK)
	{
		std::cout << "ERROR: setting of starting value failed, exit with " << std::to_string(status) << "\n";
		return 1;
	}
	std::cout << "Succesfully set start value \n";

	//Execute task
	status = vslsCorrExec(task, inputs.data(), inputs_stride, weights.data(), weights_stride, outputs.data(), outputs_stride);
	//status = vslsConvExec(task, inputs.data(), NULL, weights.data(), NULL, outputs.data(), NULL);
	if (status != VSL_STATUS_OK) 
	{
		std::cout << "ERROR: failed to calculate cross-correlation, exit with "<< std::to_string(status) << "\n";
		return 1;
	}

	//Delete task
	status = vslCorrDeleteTask(&task);
	if (status != VSL_STATUS_OK)
	{
		std::cout << "ERROR: failed to delete task object, exit with  " << std::to_string(status) << "\n";
		return 1;
	}

    std::cout << "Passed!\n";

	// End timing & print time difference
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000. << "[seconds]" << std::endl;
}