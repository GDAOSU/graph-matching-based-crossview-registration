/*
 * FILE: simpsolib.h, v.1.7.1, 4/28/2014
 * Author: Tomas V. Arredondo
 *
 * SimGALib: A simple yet flexible PSO implementation in C++.
 *
 * DISCLAIMER: No liability is assumed by the author for any use made
 * of this program.
 * DISTRIBUTION: Any use may be made of this program, as long as the
 * clear acknowledgment is made to the author in code and runtime executables
 */
#ifndef SIMPSOLIB_H
#define SIMPSOLIB_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "simtstlib.h"

namespace simpsolib
{
using namespace std;

// Early declaration
class EvalFN;

// Currently implemented optimization methods:
// Particle Swarm Optimization (Kennedy and Eberhart 1995)
// there are simplified versions where phy_g=0, omega=0, particle to update chosen at random (MOL, Pedersen 2009)
double run_pso(EvalFN eval, int number_runs, int pso_pop_size, int pso_number_iters,
            float phi_p, float phi_g, float omega, bool rand_particle_upd_flag, std::vector<double> &x_best);

// **********************************
// Default simulation values here
#define POPULATION_SIZE 100
#define NUM_DIMS_UNDEF -1
#define NUM_ITERATIONS 100
// Default simulation values here
// **********************************

class Organism
{
public:
    Organism()
    {
        num_dims=NUM_DIMS_UNDEF;
        best_value=0.0;
        value=0.0;
        cout << "Error: Called Organism constructor without parameters\n" << endl;
    }

    Organism(int tmp_num_dims)
    {
        num_dims=tmp_num_dims;
        position = vector<double>(num_dims);
        velocity = vector<double>(num_dims);
        best_position = vector<double>(num_dims);
        best_value=0.0;
        value=0.0;
    }
    friend ostream & operator<<(ostream& output, const Organism& p);
    friend class Population;
    vector<double> position;
    vector<double> velocity;
    double value;
    vector<double> best_position;
    double best_value;
    int num_dims;           // Number of dimensions in optimization
};

// Early declaration
class Population_data;

struct eval_fn_data
{
	int *image;
	int rows, cols;
	float *pts;
	int num_pts;
	float dx, dy;
};

class EvalFN
{
    friend class Population;
public:
    int num_parms;
    vector<double> lower_range;
    vector<double> upper_range;
    double (*eval_fn)(std::vector<double>, int *, int, int, float *, float, float, int);
    char szName[256];
	eval_fn_data *data;
    EvalFN(): num_parms(0), eval_fn(0)
    {
        ;
    }
    EvalFN(char *tmp_name, int tmp_num_parms, vector<double> tmp_lower_range, vector<double> tmp_upper_range, eval_fn_data *input_data, 
		double (*tmp_eval_fn)(std::vector<double>, int *, int, int, float *, float, float, int))
    {
        strcpy(szName, tmp_name);
        num_parms=tmp_num_parms;

        lower_range=vector<double> (num_parms);
        upper_range=vector<double> (num_parms);

        lower_range=tmp_lower_range;
        upper_range=tmp_upper_range;

        eval_fn=tmp_eval_fn;
		data = input_data;
    }
    double evaluate(vector<double> position);
};

class Population
{
public:
    int population_size;
    int num_iterations;
    EvalFN evaluator;
    vector<double> overall_best_position;
    double overall_best_value;
    int num_dims;
    float phi_p;
    float phi_g;
    float omega;
    bool rand_particle_upd_flag;

public:
    vector<Organism *> pool;
    void create();
    void destroy();
    void display();
    void evaluate();
    void update_vel();
    void update_pos();

    Population()
    {
        population_size=POPULATION_SIZE;
        num_iterations=NUM_ITERATIONS;
        num_dims=NUM_DIMS_UNDEF;
        overall_best_value=0.0;
    }
    Population(int tmp_num_dims)
    {
        // This population constructor is passed the number of genes per Organism
        population_size=POPULATION_SIZE;
        num_iterations=NUM_ITERATIONS;
        num_dims=tmp_num_dims;
        overall_best_value=0.0;
    }
    ~Population()
    {
        ;
    }
    void setEvalFN(EvalFN tmp_evaluator)
    {
        evaluator=tmp_evaluator;
    }
    void setNumIters(int tmp_numiters)
    {
        num_iterations=tmp_numiters;
    }
    int getNumIters() const
    {
        return num_iterations;
    }
    void setSize(int tmp_size)
    {
        population_size=tmp_size;
    }
    int getSize() const
    {
        return population_size;
    }
    int getNumDims() const
    {
        return num_dims;
    }
    float getPhiG() const
    {
        return phi_g;
    }
    void setPhiG(float tmp_phi_g)
    {
        phi_g=tmp_phi_g;
    }
    float getPhiP() const
    {
        return phi_p;
    }
    void setPhiP(float tmp_phi_p)
    {
        phi_p=tmp_phi_p;
    }
    float getOmega() const
    {
        return omega;
    }
    void setOmega(float tmp_omega)
    {
        omega=tmp_omega;
    }
    bool getRandPartUpdFlag() const
    {
        return rand_particle_upd_flag;
    }
    void setRandPartUpdFlag(bool tmp_rand_particle_upd_flag)
    {
        rand_particle_upd_flag=tmp_rand_particle_upd_flag;
    }

};

class Population_data
{
public:
    int           max_index;
    //Organism      max_organism;
    double        max_value;
    int           min_index;
    //Organism      min_organism;
    double        min_value;
    double        avg_value;
    double        sum_values;

    void clear_pop_data()
    {
        max_index=0;
        max_value=0;
        min_index=0;
        min_value=0;
        avg_value=0;
        sum_values=0;
    }
    void evaluate_population_info(Population * pop);
    void display_population_stats();
};




} // namespace simpsolib


#endif // SIMPSOLIB_H
