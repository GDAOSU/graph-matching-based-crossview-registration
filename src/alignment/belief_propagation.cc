#include "belief_propagation.h"

#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <queue>

#define BP_USE_OPENMP

class BPException {
 public:
	const char* message;
	BPException( const char* m ): message(m) { }
	void Report(){
    printf("\n[BeliefProp]: %s\n",message);
    exit(1);
}
};

inline void HandleError(const char* message){
  throw BPException(message);
}

BeliefPropagation::BeliefPropagation(SiteID num_sites, LabelID num_labels)
    : data_belief_func_(nullptr)
    , smooth_belief_func_(nullptr)
    , data_belief_func_delete_(nullptr)
    , smooth_belief_func_delete_(nullptr)
    , data_belief_individual_(nullptr)
    , smooth_belief_individual_(nullptr)
    , label_belief_individual_(nullptr)
    , labeling_(new LabelID[num_sites])
{
  assert(num_sites>0&&num_labels>1);
  num_labels_ = num_labels;
  num_sites_ = num_sites;
  work_directory_[0] = 0;
}
BeliefPropagation::~BeliefPropagation(){
	if ( data_belief_func_delete_ ) data_belief_func_delete_(data_belief_func_);
	if ( smooth_belief_func_delete_ ) smooth_belief_func_delete_(smooth_belief_func_);
  if ( data_belief_individual_ ) delete[] data_belief_individual_;
  if ( smooth_belief_individual_ ) delete[] smooth_belief_individual_;
  if ( label_belief_individual_ ) delete[] label_belief_individual_;
  if ( labeling_ ) delete[] labeling_;
}

void BeliefPropagation::SetVerbose(int level, const char *dir)
{
  verbose_ = level;
  if(dir) strcpy(work_directory_,dir);
}
//-------------------------------------------------------------------

template <typename UserFunctor>
void BeliefPropagation::SpecializeDataBeliefFunctor(const UserFunctor f) {
	if ( data_belief_func_delete_ )
		data_belief_func_delete_(data_belief_func_);
	if ( data_belief_individual_ )
	{
		delete [] data_belief_individual_;
		data_belief_individual_ = nullptr;
	}
	data_belief_func_ = new UserFunctor(f);
	data_belief_func_delete_      = &BeliefPropagation::DeleteFunctor<UserFunctor>;
  get_data_belief_ = &BeliefPropagation::GetDataBelief<UserFunctor>;
}

template <typename UserFunctor>
void BeliefPropagation::SpecializeSmoothBeliefFunctor(const UserFunctor f) {
	if ( smooth_belief_func_delete_ )
		smooth_belief_func_delete_(smooth_belief_func_);
	if ( smooth_belief_individual_ )
	{
		delete [] smooth_belief_individual_;
		smooth_belief_individual_ = 0;
	}
	smooth_belief_func_ = new UserFunctor(f);
	smooth_belief_func_delete_    = &BeliefPropagation::DeleteFunctor<UserFunctor>;
  get_smooth_belief_ = &BeliefPropagation::GetSmoothBelief<UserFunctor>;
}

//-------------------------------------------------------------------

void BeliefPropagation::SetDataBelief(DataBeliefFn fn){
  SpecializeDataBeliefFunctor(DataBeliefFnFromFunction(fn));
}
void BeliefPropagation::SetDataBelief(DataBeliefFnExtra fn, void *extraData){
  SpecializeDataBeliefFunctor(DataBeliefFnFromFunctionExtra(fn,extraData));
}
void BeliefPropagation::SetDataBelief(EnergyTermType *dataArray){
  SpecializeDataBeliefFunctor(DataBeliefFnFromArray(dataArray, num_labels_));
}
void BeliefPropagation::SetDataBelief(SiteID s, LabelID l, EnergyTermType e) {
	if ( !data_belief_individual_ )
	{
		EnergyTermType* table = new EnergyTermType[num_sites_*num_labels_];
		memset(table, 0, num_sites_*num_labels_*sizeof(EnergyTermType));
		SpecializeDataBeliefFunctor(DataBeliefFnFromArray(table, num_labels_));
		data_belief_individual_ = table;
	}
	data_belief_individual_[s*num_labels_ + l] = e;
}
void BeliefPropagation::SetDataBeliefFunctor(DataBeliefFunctor* f) {
	if ( data_belief_func_delete_ )
		data_belief_func_delete_(data_belief_func_);
	if ( data_belief_individual_ )
	{
		delete [] data_belief_individual_;
    data_belief_individual_ = nullptr;
	}
	data_belief_func_ = f;
	data_belief_func_delete_    = nullptr;
  setup_data_beliefs_func_ = &BeliefPropagation::SetupDataBeliefs<DataBeliefFunctor>;
  get_data_belief_ = &BeliefPropagation::GetDataBelief<DataBeliefFunctor>;
}

void BeliefPropagation::SetSmoothBelief(SmoothBeliefFn fn){
  SpecializeSmoothBeliefFunctor(SmoothBeliefFnFromFunction(fn));
}
void BeliefPropagation::SetSmoothBelief(SmoothBeliefFnExtra fn, void *extraData){
  SpecializeSmoothBeliefFunctor(SmoothBeliefFnFromFunctionExtra(fn,extraData));
}
void BeliefPropagation::SetSmoothBelief(EnergyTermType *smoothArray){
  SpecializeSmoothBeliefFunctor(SmoothBeliefFnFromArray(smoothArray, num_labels_));
}
void BeliefPropagation::SetSmoothBelief(LabelID l1, LabelID l2, EnergyTermType e) {
	if ( !smooth_belief_individual_ )
	{
		EnergyTermType* table = new EnergyTermType[num_labels_*num_labels_];
		memset(table, 0, num_labels_*num_labels_*sizeof(EnergyTermType));
		SpecializeSmoothBeliefFunctor(SmoothBeliefFnFromArray(table, num_labels_));
		smooth_belief_individual_ = table;
	}
	smooth_belief_individual_[l1*num_labels_ + l2] = e;
}
void BeliefPropagation::SetSmoothBeliefFunctor(SmoothBeliefFunctor* f) {
	if ( smooth_belief_func_delete_ )
		smooth_belief_func_delete_(smooth_belief_func_);
	if ( smooth_belief_individual_ )
	{
		delete [] smooth_belief_individual_;
    smooth_belief_individual_ = nullptr;
	}
	smooth_belief_func_ = f;
	smooth_belief_func_delete_    = nullptr;
  get_smooth_belief_ = &BeliefPropagation::GetSmoothBelief<SmoothBeliefFunctor>;
}

template <typename DataBeliefT>
void BeliefPropagation::SetupDataBeliefs(SiteID* activeSites, SiteID size, LabelID label, EnergyTermType* label_belief){
	DataBeliefT* dc = (DataBeliefT*)data_belief_func_;
  for(SiteID i=0; i<size; ++i){
    label_belief[activeSites[i]*num_labels_+label] = dc->Compute(activeSites[i],label);
  }
}

template <typename DataBeliefT>
BeliefPropagation::EnergyTermType BeliefPropagation::GetDataBelief(SiteID s, LabelID l){
	DataBeliefT* dc = (DataBeliefT*)data_belief_func_;
  return dc->Compute(s,l);
}
template <typename SmoothBeliefT>
BeliefPropagation::EnergyTermType BeliefPropagation::GetSmoothBelief(SiteID s1, SiteID s2, LabelID l1, LabelID l2){
  SmoothBeliefT* sc = (SmoothBeliefT*)smooth_belief_func_;
  return sc->Compute(s1,s2,l1,l2);
}

//-------------------------------------------------------------------

void BeliefPropagation::SetupLabelBeliefs(){
  assert(get_data_belief_!=nullptr);
  if(label_belief_individual_==nullptr){
    label_belief_individual_ = new EnergyTermType[num_labels_*num_sites_];
    if(!label_belief_individual_) HandleError("Not enough memory.");
  }
  // std::vector<SiteID> active_site;
  // for(int i=0; i<num_sites_; ++i){
  //   active_site[i] = i;
  // }

  // for(LabelID i=0; i<num_labels_; ++i){
  //   this->setup_data_beliefs_func_( active_site.data(), active_site.size(), i, label_belief_individual_);
  // }
  EnergyTermType* belief = label_belief_individual_;

#ifdef BP_USE_OPENMP
#pragma omp parallel for
#endif
  for(SiteID s=0; s<num_sites_; ++s){
    for(LabelID l=0; l<num_labels_; ++l){
      belief[s*num_labels_+l] = (this->*get_data_belief_)(s,l);
    }
  }
  FinalizeNeighbors();
  UpdateLabeling();
}

void BeliefPropagation::RunOnePropagation(){
  FinalizeNeighbors();
  EnergyTermType* belief = new EnergyTermType[num_sites_*num_labels_];
  if(belief==nullptr) HandleError("Not enough memory.");
  memcpy(belief, label_belief_individual_, sizeof(EnergyTermType)*num_sites_*num_labels_);

#ifdef BP_USE_OPENMP
#pragma omp parallel for
#endif
  for(SiteID s1=0; s1<num_sites_; ++s1){
    SiteID num_neighbors = 0;
    SiteID* neighbor_indices = nullptr;
    EnergyTermType* neighbor_weights = nullptr;
    GiveNeighborInfo(s1, &num_neighbors, &neighbor_indices, &neighbor_weights);
    for(LabelID l1=0; l1<num_labels_; ++l1){
      EnergyTermType sum = 0;
      EnergyTermType sumw = 0;
      for(int i=0; i<num_neighbors; ++i){
        SiteID s2 = neighbor_indices[i];
        EnergyTermType c = 0;
        EnergyTermType cw = 0;
        for(LabelID l2=0; l2<num_labels_; ++l2){
          auto t = (this->*get_smooth_belief_)(s1,s2,l1,l2);
          c += t*label_belief_individual_[s2*num_labels_+l2];
          cw+=t;
        }
        sum += c;
        sumw += cw;
      }
      if(sumw==(EnergyTermType)0.0)  sum = 0;
      else sum /= sumw;
      belief[s1*num_labels_+l1] += sum;
    }
    EnergyTermType maxc = belief[s1*num_labels_+0];
    for(LabelID l1=1; l1<num_labels_; ++l1){
      if(maxc<belief[s1*num_labels_+l1]) maxc = belief[s1*num_labels_+l1];
    }
    if(maxc>1e-5){
      for(LabelID l1=0; l1<num_labels_; ++l1){
        belief[s1*num_labels_+l1] /= maxc;
      }
    }
  }
  delete[] label_belief_individual_;
  label_belief_individual_ = belief;
  UpdateLabeling();
}

typedef std::pair<BeliefPropagation::EnergyTermType, int> SortType;
typedef std::priority_queue< SortType, std::vector<SortType>, BELIEF_COMPARE<SortType> > SortQueue;

bool SaveLabelBeliefs(const char* filepath, BeliefPropagation::EnergyTermType* beliefs, BeliefPropagation::SiteID num_sites, BeliefPropagation::LabelID num_labels){
  FILE* fp = fopen(filepath,"w");
  if(fp==nullptr) return false;
  for(size_t i=0; i<num_sites; ++i){
    fprintf(fp,"%zu\n",i);
    SortQueue queue;
    for(size_t j=0; j<num_labels; ++j){
      queue.push(std::make_pair(*beliefs++,j));
    }
    for(size_t j=0; j<num_labels; ++j){
      auto& t = queue.top();
      fprintf(fp,"%3d\t%lf\n",t.second,t.first);
      queue.pop();
    }
  }
  fclose(fp);
  return true;
}

void BeliefPropagation::Propagation(int iteration_number){
  SetupLabelBeliefs();
  if(work_directory_[0]!=0&&verbose_>2){
    char* ps = work_directory_+strlen(work_directory_);
    strcpy(ps,"/bp0.txt");
    SaveLabelBeliefs(work_directory_, label_belief_individual_, num_sites_, num_labels_);
    std::cout << "Saving belief propagation[0] to " << work_directory_ << " ... " << "OK" << std::endl;
    *ps = 0;
  }
  EnergyType energy = ComputeEnergy();
  EnergyType diff(energy);
  unsigned i(0);
  unsigned nIncreases(0), nTotalIncreases(0);
  while(true){
    EnergyType last_energy = energy;
    RunOnePropagation();

    energy = ComputeEnergy();
    diff = -(last_energy-energy);
    if(verbose_>2){
      std::cout << "\t[" << i << "] e: " << last_energy << "\td: " << diff << std::endl;
    }
    if (++i >= iteration_number || diff == EnergyType(0))
      break;
    if (diff < EnergyType(0)) {
      ++nTotalIncreases;
      if (++nIncreases > 1)
        break;
    } else {
      if (nTotalIncreases > 2)
        break;
      nIncreases = 0;
    }

    if(work_directory_[0]!=0&&verbose_>2){
      char* ps = work_directory_+strlen(work_directory_);
      sprintf(ps,"/bp%d.txt",i);
      SaveLabelBeliefs(work_directory_, label_belief_individual_, num_sites_, num_labels_);
      std::cout << "Saving belief propagation[" << i << "] to " << work_directory_ << " ... " << "OK" << std::endl;
      *ps = 0;
    }
  }
}

BeliefPropagation::EnergyTermType* BeliefPropagation::LabelBeliefs(SiteID site) const{
  assert(label_belief_individual_!=nullptr);
  return label_belief_individual_+site*num_labels_;
}

void BeliefPropagation::UpdateLabeling(){
  BELIEF_COMPARE<EnergyTermType> compare;
#ifdef BP_USE_OPENMP
#pragma omp parallel for
#endif
  for(SiteID s=0; s<num_sites_; ++s){
    EnergyTermType* belief = label_belief_individual_+s*num_labels_;
    EnergyTermType c = *belief++;
    LabelID i = 0;
    for(LabelID l=1; l<num_labels_; ++l, belief++){
      if(compare(c,*belief)){
        c = *belief;
        i = l;
      }
    }
    labeling_[s] = i;
  }
}

BeliefPropagation::EnergyType BeliefPropagation::ComputeEnergy(){
  EnergyType energy = 0;

#ifdef BP_USE_OPENMP
#pragma omp parallel for reduction(+:energy)
#endif
  for(SiteID s=0; s<num_sites_; ++s){
    energy += (this->*get_data_belief_)(s,labeling_[s]);
  }

#ifdef BP_USE_OPENMP
#pragma omp parallel for reduction(+:energy)
#endif
  for(SiteID s1=0; s1<num_sites_; ++s1){
    SiteID num_neighbors = 0;
    SiteID* neighbor_indices = nullptr;
    EnergyTermType* neighbor_weights = nullptr;
    GiveNeighborInfo(s1, &num_neighbors, &neighbor_indices, &neighbor_weights);
    for(SiteID n=0; n<num_neighbors; ++n){
      SiteID s2 = neighbor_indices[n];
      if(s2 > s1)
        energy += neighbor_weights[n]*(this->*get_smooth_belief_)(s1,s2,labeling_[s1],labeling_[s2]);
    }
  }
  return energy;
}
//-------------------------------------------------------------------

BeliefPropagationGeneralGraph::BeliefPropagationGeneralGraph(SiteID num_sites, LabelID num_labels)
    : BeliefPropagation(num_sites,num_labels)
    , need_to_delete_neighbors_(true)
    , need_to_finish_neighbors_(true)
    , num_neighbors_(nullptr)
    , neighbors_indices_(nullptr)
    , neighbors_weights_(nullptr)
{
  neighbors_.resize(BeliefPropagation::num_sites());
}
BeliefPropagationGeneralGraph::~BeliefPropagationGeneralGraph(){
  if(num_neighbors_&&need_to_delete_neighbors_){
    for(SiteID i=0; i<num_sites_; ++i){
      if(num_neighbors_[i]!=0){
        delete[] neighbors_indices_[i];
        delete[] neighbors_weights_[i];
      }
    }

    delete[] num_neighbors_;
    delete[] neighbors_indices_;
    delete[] neighbors_weights_;
  }
}

void BeliefPropagationGeneralGraph::SetNeighbors(SiteID site1, SiteID site2, EnergyTermType weight, bool undirected)
{

	assert( site1 < num_sites_ && site1 >= 0 && site2 < num_sites_ && site2 >= 0);
	assert( need_to_finish_neighbors_ );

  neighbors_[site1].push_back(Neighbor(site2,weight));
  if(undirected) neighbors_[site2].push_back(Neighbor(site1,weight));
}
void BeliefPropagationGeneralGraph::SetAllNeighbors(SiteID *numNeighbors,SiteID **neighborsIndexes, EnergyTermType **neighborsWeights)
{
  need_to_delete_neighbors_ = false;
  need_to_finish_neighbors_ = false;
	if ( num_neighbors_ != nullptr )
		HandleError("Already set up neighborhood system.");
  num_neighbors_ = numNeighbors;
  neighbors_indices_ = neighborsIndexes;
  neighbors_weights_ = neighborsWeights;
}

void BeliefPropagationGeneralGraph::FinalizeNeighbors(){
  if(!need_to_finish_neighbors_) return;
  need_to_finish_neighbors_ = false;

  num_neighbors_ = new SiteID[num_sites_];
  neighbors_indices_ = new SiteID*[num_sites_];
  neighbors_weights_ = new EnergyTermType*[num_sites_];

  if(!num_neighbors_||!neighbors_indices_||!neighbors_weights_) HandleError("Not enough memory.");

#ifdef BP_USE_OPENMP
#pragma omp parallel for
#endif
  for(SiteID site=0; site<num_sites_; ++site){
    auto& neighbors = neighbors_[site];
    if(neighbors.size()>0){
      auto sz = neighbors.size();
      num_neighbors_[site] = sz;
      SiteID* indices = new SiteID[sz];
      EnergyTermType* weights = new EnergyTermType[sz];
      neighbors_indices_[site] = indices;
      neighbors_weights_[site] = weights;
      for(auto it=neighbors.begin(); it!=neighbors.end(); ++it){
        auto& neigbor = *it;
        *indices++ = it->node_;
        *weights++ = it->weight_;
      }
    }else num_neighbors_[site] = 0;
  }
  neighbors_.clear();
}

void BeliefPropagationGeneralGraph::GiveNeighborInfo(SiteID site, SiteID *numSites, SiteID **neighbors, EnergyTermType **weights)
{
  assert(num_neighbors_);
  (*numSites)  =  num_neighbors_[site];
  (*neighbors) = neighbors_indices_[site];
  (*weights)   = neighbors_weights_[site];
}

