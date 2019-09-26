// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"
#include "Grid.h"
#include "diff_U.h"
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
extern "C" 
{
#include <netcdf.h>
}

//Return this error code in event of a problem related to netCDF
static constexpr int NC_ERR = 2;

//Define status code for netCDF functions
static int retval = 0;

//Define IDs for netCDF-files, dimensions, variables, and settings ...
static constexpr int ndim = 4;
//for reading:
static int ncid_reading = 0;
static int varid_u = 0;
static int varid_v = 0;
static int varid_w = 0;
static size_t count_u[ndim] = {}; //initialize fixed arrays to 0
static size_t count_v[ndim] = {};
static size_t count_w[ndim] = {};
static size_t start_reading[ndim] = {};
//for storage:
static int ncid_store = 0;
static int tstep_dimid = 0;
static int zc_dimid = 0;
static int zhc_dimid = 0;
static int yc_dimid = 0;
static int yhc_dimid = 0;
static int xc_dimid = 0;
static int xhc_dimid = 0;
static int dimids_ut[ndim] = {}; 
static int dimids_vt[ndim] = {};
static int dimids_wt[ndim] = {};
static size_t count_ut[ndim] = {};
static size_t count_vt[ndim] = {};
static size_t count_wt[ndim] = {};
static int varid_tstep = 0;
static int varid_zc = 0;
static int varid_zhc = 0;
static int varid_yc = 0;
static int varid_yhc = 0;
static int varid_xc = 0;
static int varid_xhc = 0;
static int varid_ut = 0;
static int varid_vt = 0;
static int varid_wt = 0;
static size_t start_writing[ndim] = {};

inline static int nc_error_print(int e)
{
	std::cerr << "Error: " << nc_strerror(e);
	exit(NC_ERR);
}

int main()
{
    // For now, hard-code command-line inputs user (Windows)
	//std::string grid_filenc = "M:\\My Documents\\Machine learning projects\\SURFsara project\\Scripts\\Training data\\training_data.nc";
	//std::string var_filepath = "M:\\My Documents\\Machine learning projects\\SURFsara project\\Scripts\\Neural network\\Variables_MLP11\\";
	//std::string training_file = "M:\\My Documents\\Machine learning projects\\SURFsara project\\Scripts\\Training data\\training_data.nc";
	//std::string inference_file = "M:\\My Documents\\Machine learning projects\\SURFsara project\\Scripts\\Neural network\\Variables_MLP11\\inference_reconstructed_field_manual_cpp.nc";
	
	// For now, hard-code command-line inputs user (Linux)
	std::string grid_filenc = "/projects/1/flowsim/simulation1/lesscoarse/training_data.nc";
	std::string var_filepath = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP11/";
	std::string training_file = "/projects/1/flowsim/simulation1/lesscoarse/training_data.nc";
	std::string inference_file = "/home/robinst/microhh/cases/moser600/git_repository/Neural Network/predictions_real_data_MLP11/inference_reconstructed_field_manual_cpp.nc";
	
	constexpr bool store_variables = true;

	//Instantiate grid object, which extract the grid information from the specified nc-file (but for now is still hard-coded)
	Grid grid(grid_filenc);

	//Calculate height differences ASSUMING a second order numerical scheme AND an equidistant vertical grid!!!
	int dz_length = grid.m_ktot + 2;
	std::vector<float> dzi(dz_length,0);
	std::vector<float> dzhi(dz_length, 0);
	for (int k = 0; k < dz_length; ++k)
	{
		dzi[k]  = grid.m_ktot / grid.m_zsize;
		dzhi[k] = grid.m_ktot / grid.m_zsize;
	}

	// Define time steps for inference
	int tstart = 27;
	int tend = 30;
	int nt = tend - tstart;
	
	// initialize dynamically allocated arrays.
	//
	std::vector<float> u(grid.m_kcells*grid.m_jcells*grid.m_icells,0.0f);
	//NOTE: on purpose one smaller in xh-direction than stored in nc-file, compensate for this when reading flow fields from nc-file.
	std::vector<float> v(grid.m_kcells*grid.m_jcells*grid.m_icells, 0.0f);
	//NOTE: on purpose one smaller in yh-direction than stored in nc-file
	std::vector<float> w(grid.m_khcells*grid.m_jcells*grid.m_icells, 0.0f);
	std::vector<float> ut(grid.m_ktot*grid.m_jtot*grid.m_itot,0.0f);
	std::vector<float> vt(grid.m_ktot*grid.m_jtot*grid.m_itot, 0.0f);
	std::vector<float> wt(grid.m_khtot*grid.m_jtot*grid.m_itot, 0.0f);

	// Open nc-file  for reading
	if ((retval = nc_open(training_file.c_str(), NC_NOWRITE, &ncid_reading)))
	{
		nc_error_print(retval);
	}

	// Get the varids of the variables based on their names
	if ((retval = nc_inq_varid(ncid_reading, "uc", &varid_u)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(ncid_reading, "vc", &varid_v)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(ncid_reading, "wc", &varid_w)))
	{
		nc_error_print(retval);
	}

	// Define settings such that each iteration one time step is read from the nc-file, 
	// and the ghostcells in the xh- and yh-direction are discared.
	count_u[0] = 1;
	count_u[1] = grid.m_kcells;
	count_u[2] = grid.m_jcells;
	count_u[3] = grid.m_icells;
	count_v[0] = 1;
	count_v[1] = grid.m_kcells;
	count_v[2] = grid.m_jcells;
	count_v[3] = grid.m_icells;
	count_w[0] = 1;
	count_w[1] = grid.m_khcells;
	count_w[2] = grid.m_jcells;
	count_w[3] = grid.m_icells;

	// Define nc-file for storage when required, and store already time steps and coordinates
	if (store_variables)
	{
		// Create netCDF-file, overwrite if it already exists
		if ((retval = nc_create(inference_file.c_str(), NC_CLOBBER, &ncid_store)))
		{
			nc_error_print(retval);
		}
		// Define the dimensions. NetCDF will hand back an ID for each
		if ((retval = nc_def_dim(ncid_store, "tstep", nt, &tstep_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "zc", grid.m_ktot, &zc_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "zhc", grid.m_khtot, &zhc_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "yc", grid.m_jtot, &yc_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "yhc", grid.m_jtot, &yhc_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "xc", grid.m_itot, &xc_dimid)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_dim(ncid_store, "xhc", grid.m_itot, &xhc_dimid)))
		{
			nc_error_print(retval);
		}
		// Define the dimensions for each variable. Aggregrate the corresponding dimension indices
		// 1) ut
		dimids_ut[0] = tstep_dimid;
		dimids_ut[1] = zc_dimid;
		dimids_ut[2] = yc_dimid;
		dimids_ut[3] = xhc_dimid;
		// 2) vt
		dimids_vt[0] = tstep_dimid;
		dimids_vt[1] = zc_dimid;
		dimids_vt[2] = yhc_dimid;
		dimids_vt[3] = xc_dimid;
		// 3) wt
		dimids_wt[0] = tstep_dimid;
		dimids_wt[1] = zhc_dimid;
		dimids_wt[2] = yc_dimid;
		dimids_wt[3] = xc_dimid;

		//Define the variables
		if ((retval = nc_def_var(ncid_store, "tstep", NC_INT, 1,
			&tstep_dimid, &varid_tstep)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "zc", NC_FLOAT, 1,
			&zc_dimid, &varid_zc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "zhc", NC_FLOAT, 1,
			&zhc_dimid, &varid_zhc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "yc", NC_FLOAT, 1,
			&yc_dimid, &varid_yc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "yhc", NC_FLOAT, 1,
			&yhc_dimid, &varid_yhc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "xc", NC_FLOAT, 1,
			&xc_dimid, &varid_xc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "xhc", NC_FLOAT, 1,
			&xhc_dimid, &varid_xhc)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "ut", NC_FLOAT, ndim,
			dimids_ut, &varid_ut)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "vt", NC_FLOAT, ndim,
			dimids_vt, &varid_vt)))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_def_var(ncid_store, "wt", NC_FLOAT, ndim,
			dimids_wt, &varid_wt)))
		{
			nc_error_print(retval);
		}

		//End define mode for nc-file
		if ((retval = nc_enddef(ncid_store)))
		{
			nc_error_print(retval);
		}

		//Make vector with time steps for storage
		std::vector<int> tstep(nt, 0);
		for (int t = 0; t < nt; ++t)
		{
			tstep[t] = tstart + t;
		}

		// Implement setttings to store in each iteration below 1 time step
		count_ut[0] = 1;
		count_ut[1] = grid.m_ktot;
		count_ut[2] = grid.m_jtot;
		count_ut[3] = grid.m_itot;
		count_vt[0] = 1;
		count_vt[1] = grid.m_ktot;
		count_vt[2] = grid.m_jtot;
		count_vt[3] = grid.m_itot;
		count_wt[0] = 1;
		count_wt[1] = grid.m_khtot;
		count_wt[2] = grid.m_jtot;
		count_wt[3] = grid.m_itot;

		//Store time steps and coordinates
		if ((retval = nc_put_var_int(ncid_store, varid_tstep, &tstep[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_zc, &grid.m_zc[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_zhc, &grid.m_zhc[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_yc, &grid.m_yc[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_yhc, &grid.m_yhc[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_xc, &grid.m_xc[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_put_var_float(ncid_store, varid_xhc, &grid.m_xhc[0])))
		{
			nc_error_print(retval);
		}
	}

	//Instantiate Network class to make predictions
	Network MLP(var_filepath);

	//Start time loop, loop over all flow fields
	int counter = 0;
	for (int t = tstart; t < tend; ++t)
	//for (int t = 0; t < 1; ++t) // FOR TESTING PURPOSES ONLY!
	{
		//Set first start value to counter such that the next time step is read (and stored if required)
		start_reading[0] = tstart + counter;
		start_writing[0] = counter;

		// Extract flow fields of time step stored in nc-file, make shape implicitly consistent with arrays in MicroHH
		if ((retval = nc_get_vara_float(ncid_reading, varid_u, start_reading, count_u, &u[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_get_vara_float(ncid_reading, varid_v, start_reading, count_v, &v[0])))
		{
			nc_error_print(retval);
		}
		if ((retval = nc_get_vara_float(ncid_reading, varid_w, start_reading, count_w, &w[0])))
		{
			nc_error_print(retval);
		}

		// Undo normalisation of flow fields in training file to be consistent with MicroHH
		// NOTE: HERE IT IS ASSUMED THAT THE U* USED DURING TRAINING IS THE SAME AS THE ONE USED DURING INFERENCE!!!
		for (int i = 0; i < (grid.m_kcells*grid.m_jcells*grid.m_icells); ++i)
		{
			u[i] = MLP.m_utau_ref * u[i];
			v[i] = MLP.m_utau_ref * v[i];
		}
		for (int i = 0; i < (grid.m_khcells*grid.m_jcells*grid.m_icells); ++i)
		{
			w[i] = MLP.m_utau_ref * w[i];
		}

		//Time code
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		//Call diff_U
		diff_U(u.data(), v.data(), w.data(), dzi.data(), dzhi.data(), ut.data(), vt.data(), wt.data(), grid, MLP);
		//End timing & print time difference
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000. << "[seconds]" << std::endl;

		//Store output variables in netCDF-file when required
		if (store_variables)
		{
			//Write output variables to nc-file
			if ((retval = nc_put_vara_float(ncid_store, varid_ut, start_writing, count_ut, &ut[0])))
			{
				nc_error_print(retval);
			}
			if ((retval = nc_put_vara_float(ncid_store, varid_vt, start_writing, count_vt, &vt[0])))
			{
				nc_error_print(retval);
			}
			if ((retval = nc_put_vara_float(ncid_store, varid_wt, start_writing, count_wt, &wt[0])))
			{
				nc_error_print(retval);
			}
		}
		//Add 1 to counter
		counter += 1;

		//Set tendencies back to 0 for next iteration
		for (int i = 0; i < grid.m_ktot*grid.m_jtot*grid.m_itot; ++i)
		{
			ut[i] = 0;
			vt[i] = 0;
		}
		for (int i = 0; i < grid.m_khtot*grid.m_jtot*grid.m_itot; ++i)
		{
			wt[i] = 0;
		}
	}

	//Close opened nc-files
	if (store_variables)
	{
		if ((retval = nc_close(ncid_store)))
		{
			nc_error_print(retval);
		}
	}
	if((retval = nc_close(ncid_reading)))
	{
		nc_error_print(retval);
	}

	return 0;
}
