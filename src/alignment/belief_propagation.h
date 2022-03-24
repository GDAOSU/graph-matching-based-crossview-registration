#ifndef BELIEF_PROPAGATION_H
#define BELIEF_PROPAGATION_H

#include <list>
#include <vector>

#define BELIEF_COMPARE std::less

class BeliefPropagation{
 public:
  typedef float EnergyType;
  typedef float EnergyTermType;
  typedef int   LabelID;
  typedef int   SiteID;

	typedef EnergyTermType (*SmoothBeliefFn)(SiteID s1, SiteID s2, LabelID l1, LabelID l2);
	typedef EnergyTermType (*DataBeliefFn)(SiteID s, LabelID l);
	typedef EnergyTermType (*SmoothBeliefFnExtra)(SiteID s1, SiteID s2, LabelID l1, LabelID l2,void *);
	typedef EnergyTermType (*DataBeliefFnExtra)(SiteID s, LabelID l,void *);

  BeliefPropagation(SiteID num_sites, LabelID num_labels);
  virtual ~BeliefPropagation();

	SiteID  num_sites() const { return num_sites_; };
	LabelID num_labels() const { return num_labels_; };

  void    SetVerbose(int level, const char* dir);

  void    Propagation(int iteration_number);
  EnergyTermType* LabelBeliefs(SiteID site) const;
  EnergyType ComputeEnergy();

 public:
	struct DataBeliefFunctor;      // use this class to pass a functor to setDataBelief
	struct SmoothBeliefFunctor;    // use this class to pass a functor to setSmoothBelief

	void SetDataBelief(DataBeliefFn fn);
	void SetDataBelief(DataBeliefFnExtra fn, void *extraData);
	void SetDataBelief(EnergyTermType *dataArray);
	void SetDataBelief(SiteID s, LabelID l, EnergyTermType e);
	void SetDataBeliefFunctor(DataBeliefFunctor* f);
	struct DataBeliefFunctor {
		virtual EnergyTermType Compute(SiteID s, LabelID l) = 0;
	};

	void SetSmoothBelief(SmoothBeliefFn fn);
	void SetSmoothBelief(SmoothBeliefFnExtra fn, void *extraData);
	void SetSmoothBelief(LabelID l1, LabelID l2, EnergyTermType e);
	void SetSmoothBelief(EnergyTermType *smoothArray);
	void SetSmoothBeliefFunctor(SmoothBeliefFunctor* f);
	struct SmoothBeliefFunctor {
		virtual EnergyTermType Compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2) = 0;
	};

	struct DataBeliefFnFromArray {
		DataBeliefFnFromArray(EnergyTermType* theArray, LabelID num_labels)
			: array_(theArray), num_labels_(num_labels){}
		EnergyTermType Compute(SiteID s, LabelID l){return array_[s*num_labels_+l];}
	private:
		const EnergyTermType* const array_;
		const LabelID num_labels_;
	};

	struct DataBeliefFnFromFunction {
		DataBeliefFnFromFunction(DataBeliefFn fn): fn_(fn){}
		EnergyTermType Compute(SiteID s, LabelID l){return fn_(s,l);}
	private:
		const DataBeliefFn fn_;
	};

	struct DataBeliefFnFromFunctionExtra {
		DataBeliefFnFromFunctionExtra(DataBeliefFnExtra fn,void *extraData): fn_(fn),extra_data_(extraData){}
		EnergyTermType Compute(SiteID s, LabelID l){return fn_(s,l,extra_data_);}
	private:
		const DataBeliefFnExtra fn_;
		void *extra_data_;
	};

	struct SmoothBeliefFnFromArray {
		SmoothBeliefFnFromArray(EnergyTermType* theArray, LabelID num_labels)
			: array_(theArray), num_labels_(num_labels){}
		EnergyTermType Compute(SiteID , SiteID , LabelID l1, LabelID l2){return array_[l1*num_labels_+l2];}
	private:
		const EnergyTermType* const array_;
		const LabelID num_labels_;
	};

	struct SmoothBeliefFnFromFunction {
		SmoothBeliefFnFromFunction(SmoothBeliefFn fn)
			: fn_(fn){}
		EnergyTermType Compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2){return fn_(s1,s2,l1,l2);}
	private:
		const SmoothBeliefFn fn_;
	};

	struct SmoothBeliefFnFromFunctionExtra {
		SmoothBeliefFnFromFunctionExtra(SmoothBeliefFnExtra fn,void *extraData)
			: fn_(fn),extra_data_(extraData){}
		EnergyTermType Compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2){return fn_(s1,s2,l1,l2,extra_data_);}
	private:
		const SmoothBeliefFnExtra fn_;
		void *extra_data_;
	};

	struct SmoothBeliefFnPotts {
		EnergyTermType Compute(SiteID, SiteID, LabelID l1, LabelID l2){return l1 != l2 ? (EnergyTermType)1 : (EnergyTermType)0;}
	};

	template <typename UserFunctor> void SpecializeDataBeliefFunctor(const UserFunctor f);
	template <typename UserFunctor> void SpecializeSmoothBeliefFunctor(const UserFunctor f);
  template <typename DataBeliefT> void SetupDataBeliefs(SiteID* activeSites, SiteID size, LabelID label, EnergyTermType* label_belief);
  template <typename DataBeliefT> EnergyTermType GetDataBelief(SiteID s, LabelID l);
  template <typename SmoothBeliefT> EnergyTermType GetSmoothBelief(SiteID, SiteID, LabelID, LabelID);

	template <typename Functor> static void DeleteFunctor(void* f) { delete reinterpret_cast<Functor*>(f); }

 protected:
	// returns a pointer to the neighbors of a site and the weights
	virtual void GiveNeighborInfo(SiteID site, SiteID *numSites, SiteID **neighbors, EnergyTermType **weights)=0;
	virtual void FinalizeNeighbors() = 0;

  void UpdateLabeling();

 public:
  void SetupLabelBeliefs();
  void 		RunOnePropagation();

 protected:
  LabelID 			num_labels_;
  SiteID 				num_sites_;

  void* 				data_belief_func_;
  void* 				smooth_belief_func_;

  void					(*data_belief_func_delete_)(void* );
  void					(*smooth_belief_func_delete_)(void* );
  void					(BeliefPropagation::*setup_data_beliefs_func_)(SiteID*,SiteID,LabelID ,EnergyTermType*);
  EnergyTermType (BeliefPropagation::*get_data_belief_)(SiteID, LabelID);
  EnergyTermType (BeliefPropagation::*get_smooth_belief_)(SiteID, SiteID, LabelID, LabelID);
	EnergyTermType* data_belief_individual_;
	EnergyTermType* smooth_belief_individual_;

  EnergyTermType* label_belief_individual_;

  LabelID*			labeling_;

  int 					verbose_;
  char 					work_directory_[512];
};

class BeliefPropagationGeneralGraph : public BeliefPropagation{
 public:
  BeliefPropagationGeneralGraph(SiteID num_sites, LabelID num_labels);
  virtual ~BeliefPropagationGeneralGraph();

 public:
	void SetNeighbors(SiteID site1, SiteID site2, EnergyTermType weight=1, bool undirected = true);
  void SetAllNeighbors(SiteID *numNeighbors,SiteID **neighborsIndexes, EnergyTermType **neighborsWeights);

 protected: 
	virtual void GiveNeighborInfo(SiteID site, SiteID *numSites, SiteID **neighbors, EnergyTermType **weights);
	virtual void FinalizeNeighbors();

 protected:
  bool need_to_delete_neighbors_;
  bool need_to_finish_neighbors_;

	struct Neighbor{
		SiteID  node_;
		EnergyTermType weight_;
    Neighbor(SiteID& node, EnergyTermType& weight)
        : node_(node), weight_(weight){}
	};
  std::vector< std::list<Neighbor> > neighbors_;
  SiteID*  num_neighbors_;
  SiteID** neighbors_indices_;
  EnergyTermType** neighbors_weights_;
};

#endif
