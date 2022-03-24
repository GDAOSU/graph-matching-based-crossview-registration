/*
 * FILE: simpsolib.cpp, v.1.7.1, 4/28/2014
 * Author: Tomas V. Arredondo
 *
 * SimPSOLib: A simple yet flexible PSO implementation in C++.
 *
 * DISCLAIMER: No liability is assumed by the author for any use made
 * of this program.
 * DISTRIBUTION: Any use may be made of this program, as long as the
 * clear acknowledgment is made to the author in code and runtime executables
 */

#include "simpsolib.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdarg.h>
#include <stdlib.h>
#include <omp.h>


using namespace std;
using namespace simpsolib;
using namespace simtstlib;

int function_cnt = 0;  // function eval counter

// seed random number generator for ran2
long semran2 = -1000;

void simpsolib::Population::evaluate()
{
    // evaluate fitness
	std::vector<double> fn_values(pool.size());
#pragma omp parallel for
	for (int i = 0; i < pool.size(); i++)
	{
		fn_values[i] = evaluator.evaluate(pool[i]->position);
	}

	int i = 0;
	for (std::vector<Organism*>::iterator it_pool = pool.begin(); it_pool != pool.end(); ++it_pool)
	{
		double fn_value = fn_values[i];
		(*it_pool)->value = fn_value;

		if (fn_value >= (*it_pool)->best_value)
		{
			(*it_pool)->best_position = (*it_pool)->position;
			(*it_pool)->best_value = fn_value;
		}

		if (fn_value >= overall_best_value)
		{
			overall_best_position = (*it_pool)->position;
			overall_best_value = fn_value;
		}

		i++;
		function_cnt++;
	}
}


void simpsolib::Population_data::evaluate_population_info(Population *pop)
{

    int pop_size=pop->getSize();
    clear_pop_data();
    int cnt=0;

    min_value=(*(pop->pool[0])).value;
    min_index=0;

    for (std::vector<Organism*>::iterator it_pool = pop->pool.begin(); it_pool != pop->pool.end(); ++it_pool)

    {
        sum_values+=(*it_pool)->value;

        if ((*it_pool)->value > max_value)
        {
            max_value=(*it_pool)->value;
            // max_organism=(*(pop->pool[i]));
            max_index=cnt;
        }
        if ((*it_pool)->value < min_value)
        {
            min_value=(*it_pool)->value;
            // min_organism=(*(pop->pool[i]));
            min_index=cnt;
        }
        cnt++;
    }
    if (sum_values)
        avg_value=sum_values/pop_size;
    else
        avg_value=0;

}

void simpsolib::Population_data::display_population_stats()
{
    vector<int>::const_iterator it;
#if 0
    for (it = max_organism.genes.begin(); it != max_organism.genes.end(); it++)
    {
        std::cout << "fittest member ["<< max_index << "]";
        std::cout << *it << " ";
        std::cout << std::endl;
    }
#endif

    std::cout << "fittest member ["<< max_index << "]" << endl;
    std::cout << "fittest member value: "<< max_value << endl;

#if 0
    for (it = min_organism.genes.begin(); it != min_organism.genes.end(); it++)
    {
        std::cout << "least fit member ["<< min_index << "]:";
        std::cout << *it << " ";
        std::cout << std::endl;
    }

    std::cout << "least fit member value: "<< min_value;
    std::cout << std::endl;
#endif

    std::cout << "pop avg value: "<< avg_value;
    std::cout << std::endl;
}

void simpsolib::Population::destroy()
{

    // destroy population
    for ( vector<Organism*>::iterator i = pool.begin(); i != pool.end(); ++i )
    {
        delete *i;
    }
    // empty the container
    pool.clear();
}

void simpsolib::Population::create()
{
    pool.resize(population_size);

    for (int i=0; i < population_size; i++)
        pool[i]=new Organism(num_dims);

    // initialize population
    for (int i = 0; i < population_size; i++)
    {
        for (int j = 0; j < num_dims; j++)
        {
            (*pool[i]).position[j]=ran2((float)evaluator.lower_range[j],(float)evaluator.upper_range[j]);
        }
    }

}

void simpsolib::Population::display()
{
    for (int i = 0; i < population_size; i++)
    {
        std::cout << "member ["<< i << "]";
        // std::cout << pool[i]; TODO
        std::cout << std::endl;
        std::cout << "value: "<< (*pool[i]).value << endl;
        // std::cout << "double_genes: "<< pool[i].double_genes << endl;
    }
}

#if 0
void simpsolib::Population::display_fittest()
{
    for (int i = 0; i < population_size; i++)
    {
        std::cout << "member ["<< i << "]";
        std::cout << pool[i];
        std::cout << std::endl;
        std::cout << "value: "<< pool[i].value << endl;
        // std::cout << "double_genes: "<< pool[i].double_genes << endl;
    }
}
#endif

double simpsolib::EvalFN::evaluate(vector<double> position)
{
    return ((*eval_fn)(position, data->image, data->rows, data->cols, data->pts, data->dx, data->dy, data->num_pts));
}

#if 0
std::cout << "---------- Unsorted GA Population (press enter) ------" << std::endl << flush;
//   std::cin.get(temp);
for (int i = 0; i < population_size; i++)
{
    std::cout << "member ["<< i << "]" << endl;
    std::cout << "member index:" << pool[i].index << endl;
    std::cout << pool[i];
    std::cout << std::endl;
    std::cout << "value: "<< pool[i].value << endl;
    //std::cout << "double_genes: "<< pool[i].double_genes << endl;
}
std::cout << flush;
char temp2;
std::cin.get(temp2);
#endif

void simpsolib::Population::update_vel()
{
    double r_p;
    double r_g;

    // evaluate value
    for (std::vector<Organism*>::iterator it_pool = pool.begin(); it_pool != pool.end(); ++it_pool)
    {
        r_p=ran2(&(semran2));
        r_g=ran2(&(semran2));

        for (int i=0; i< num_dims; i++) // Shi, Eberhart (1998, 2001)
        {
            (*it_pool)->velocity[i] = omega*((*it_pool)->velocity[i]) + phi_p*r_p*((*it_pool)->best_position[i] - (*it_pool)->position[i]) + phi_g*r_g*(overall_best_position[i] - (*it_pool)->position[i]);

            // Limit velocity in each dimension to full dynamic range in search space (Simplifying PSO, Pedersen 2009)
            if (fabs((*it_pool)->velocity[i]) > (evaluator.upper_range[i] - evaluator.lower_range[i]))
            {
                (*it_pool)->velocity[i]=ran2(&(semran2))*(evaluator.upper_range[i] - evaluator.lower_range[i]);
            }

        }
    }

}

void simpsolib::Population::update_pos()
{
    // evaluate value
    for (std::vector<Organism*>::iterator it_pool = pool.begin(); it_pool != pool.end(); ++it_pool)
    {
        for (int i=0; i< num_dims; i++) // Shi, Eberhart (1998, 2001)
        {
            (*it_pool)->position[i] = (*it_pool)->position[i] + (*it_pool)->velocity[i] ;

            // Limit velocity in each dimension to full dynamic range in search space (Simplifying PSO, Pedersen 2009)
            if ((*it_pool)->position[i] > (evaluator.upper_range[i]))
            {
                (*it_pool)->position[i]=evaluator.upper_range[i];
            }
            else if ((*it_pool)->position[i] < (evaluator.lower_range[i]))
            {
                (*it_pool)->position[i]=evaluator.lower_range[i];
            }
        }
    }
}
//-----------------------------------------------------------------------------
// Optimization Methods
//-----------------------------------------------------------------------------
double simpsolib::run_pso(EvalFN eval, int number_runs, int pso_pop_size, int pso_number_iters,
                       float phi_p, float phi_g, float omega, bool rand_particle_upd_flag, std::vector<double> &x_best)
{
    clock_t start,end,diff=0;
    start=clock();

    // Run params
    int nRun=0;
    double nRunsAvgFitness=0;
    double nRunsMaxFitness=0;
    double nRunsMinFitness=999999999;
	std::vector<double> nRunFitness(number_runs);
    double nRunsAvgFitnessVariance=0;
    double nRunsAvgFitnessSTD=0;
    function_cnt=0;

    for (int i=0; i < number_runs ; i++)
        nRunFitness[i]=0;

    // Set the population parameters: number of dimensions and other parms
    Population pop(eval.num_parms);

    // Set pop parameters
    pop.setEvalFN(eval);
    Population_data pop_info;
    pop.setSize(pso_pop_size);
    pop.setNumIters(pso_number_iters);

    srand(clock());

    for (nRun=0; nRun < number_runs; nRun++)
    {
        pop.create(); // instantiate (new) the population
        pop.evaluate();
        pop_info.evaluate_population_info(&pop);

        for (int i=1; i < pop.getNumIters() ; i++)
        {
            pop.update_vel();
            pop.update_pos();

            pop.evaluate();
            pop_info.evaluate_population_info(&pop);
        }

        // Add up numbers for each Run
        if (pop_info.max_value > nRunsMaxFitness)
        {
            nRunsMaxFitness = pop_info.max_value;
        }
        if (pop_info.min_value < nRunsMinFitness)
        {
            nRunsMinFitness = pop_info.min_value;
        }

        nRunFitness[nRun]=pop_info.avg_value;
        nRunsAvgFitness+=pop_info.avg_value;

        pop.destroy();  // del the population
    }

    end=clock();
    diff=end-start;

    // Calculate RUN results
    nRunsAvgFitness=nRunsAvgFitness/number_runs;

    for (int i=0; i<nRun; i++)
    {
        nRunsAvgFitnessVariance += pow(nRunsAvgFitness-nRunFitness[i],2);
    }
    nRunsAvgFitnessVariance=nRunsAvgFitnessVariance/number_runs;
    nRunsAvgFitnessSTD=pow(nRunsAvgFitnessVariance, 0.5);

	x_best = pop.overall_best_position;
	return pop.overall_best_value;
}
