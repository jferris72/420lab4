/*
    Test the result stored in "data_output" against a serial implementation.

    -----
    Compiling:
    Include "Lab4_IO.c" to compile. Set the macro "LAB4_EXTEND" defined in the "Lab4_IO.c" file to include the extended functions
    $ gcc serialtester.c Lab4_IO.c -o serialtester -lm 

    -----
    Return values:
    0      result is correct
    1      result is wrong
    2      problem size does not match
    253    no "data_output" file
    254    no "data_input" file
*/
#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "timer.h"
#include "Lab4_IO.h"


#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

#define THRESHOLD 0.0001

int main (int argc, char* argv[]){
    struct node *nodehead;
    int nodecount;
    int *num_in_links, *num_out_links;
    double *r, *r_pre, *r_global, *r_pre_global;
    int i, j;
    double damp_const;
    int iterationcount = 0;
    int collected_nodecount;
    double *collected_r;
    double cst_addapted_threshold;
    double error;
	double start, end;
    FILE *fp;
	int my_rank, comm_size, nodes_per_proc;
	MPI_Comm comm;
	
	MPI_Init(NULL, NULL);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(comm, &my_rank);	

    if (get_node_stat(&nodecount, &num_in_links, &num_out_links)) return 254;

    // Calculate the result
    if (node_init(&nodehead, num_in_links, num_out_links, 0, nodecount)) return 254;

	nodes_per_proc = nodecount / comm_size;

	r = malloc(nodes_per_proc * sizeof(double));
	//r_pre = malloc(nodes_per_proc * sizeof(double));
	for ( i = 0; i < nodes_per_proc; ++i)
	    r[i] = 1.0 / nodecount;
	r_pre_global = malloc(nodecount * sizeof(double));
	r_global = malloc(nodecount * sizeof(double));
	damp_const = (1.0 - DAMPING_FACTOR) / nodecount;

	GET_TIME(start);
    // CORE CALCULATION
    while(1) {

        vec_cp(r_global, r_pre_global, nodecount);

        for ( i = 0; i < nodes_per_proc; ++i){
            r[i] = 0;
            for ( j = 0; j < nodehead[my_rank*nodes_per_proc + i].num_in_links; ++j)
                r[i] += r_pre_global[nodehead[my_rank*nodes_per_proc + i].inlinks[j]] / 
										num_out_links[nodehead[my_rank*nodes_per_proc + i].inlinks[j]];
            r[i] *= DAMPING_FACTOR;
            r[i] += damp_const;
        }

		MPI_Allgather(r, nodes_per_proc, MPI_DOUBLE, r_global, nodes_per_proc, MPI_DOUBLE, comm);
		//MPI_Gather(r_pre, nodes_per_proc, MPI_DOUBLE, r_pre_global, nodecount, MPI_DOUBLE, 0, comm);

		if (rel_error(r_global, r_pre_global, nodecount) < EPSILON) {
			break;
		}

		if(my_rank == 0) {
			++iterationcount;
		}

    }

	GET_TIME(end);
	if(my_rank == 0) {
		Lab4_saveoutput(r_global, nodecount, end-start);
	}

    // post processing
    node_destroy(nodehead, nodecount);

	MPI_Finalize();
    free(num_in_links); free(num_out_links);

    free(r); free(r_pre); free(collected_r);
  
}
