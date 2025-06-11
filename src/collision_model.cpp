#include "collision_model.h"
#include "math_const.h"
#include "comm.h"
#include "math.h"
#include "torch/torch.h"
#include "error.h"
#include <fstream>

using namespace SPARTA_NS;
using namespace MathConst;
using namespace torch::indexing;

#define MAXLINE 1024

MLCollideModel::MLCollideModel(SPARTA *sparta) : Pointers(sparta){}

MLCollideModel::~MLCollideModel(){}

// Below needed to load models saved from python.

std::vector<char> MLCollideModel::get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void MLCollideModel::load_weights(std::string pt_pth) {
  std::vector<char> f = this->get_the_bytes(pt_pth);
  c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(f).toGenericDict();

  const torch::OrderedDict<std::string, at::Tensor>& model_params = get_implementation()->named_parameters();
  std::vector<std::string> param_names;
  for (auto const& w : model_params) {
    param_names.push_back(w.key());
  }

  torch::NoGradGuard no_grad;
  for (auto const& w : weights) {
      std::string name = w.key().toStringRef();
      at::Tensor param = w.value().toTensor();

      if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()){
        model_params.find(name)->copy_(param);
      } else {
        std::cout << name << " does not exist among model parameters." << std::endl;
      };

  }
}

int MLCollideModel::train_this_step(int step, int training){
  if (training == NO || training == OFFLINE) return 0;
  return (step % train_params.train_every == 0) && (step < train_params.train_max);
}

void  MLCollideModel::save_to_disk(torch::Tensor inputs, torch::Tensor outputs, int step){
    // Save data to disk
      auto pickled = torch::pickle_save(inputs);
      std::string filename_in = "out/input_" + std::to_string(comm->me) + "_" + std::to_string(step);
      std::ofstream fins(filename_in, std::ios::out | std::ios::binary);
      fins.write(pickled.data(), pickled.size());
      fins.close();

      pickled = torch::pickle_save(outputs);
      std::string filename_out = "out/out_" + std::to_string(comm->me) + "_" + std::to_string(step);
      std::ofstream fout(filename_out, std::ios::out | std::ios::binary);
      fout.write(pickled.data(), pickled.size());
      fout.close();
}

int MLCollideModel::requires_data(){
  return ( training_data.outputs.size() / training_data.num_outputs< train_params.len_data );
}


void  MLCollideModel::read_train_params(){
  const char* fname = "in.training_params";
  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open DMS parameter file %s",fname);
    error->one(FLERR,str);
  }
  int REQWORDS = 9;
  char **words = new char*[REQWORDS]; // one extra word in cross-species lines
  char line[MAXLINE];
  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(REQWORDS,line,words);
    if (nwords < REQWORDS) 
      error->one(FLERR,"Incorrect line format in DMS parameter file");
    
    train_params.width = atoi(words[0]);
    train_params.train_every = atoi(words[1]);
    train_params.train_max = atoi(words[2]);
    train_params.epochs=atoi(words[3]);
    train_params.len_data=atoi(words[4])/ comm->nprocs;
    train_params.LR=atof(words[5]);
    train_params.A = atof(words[6]); 
    train_params.B = atof(words[7]);
    train_params.C = 1.; 
    train_params.batch_size = atoi(words[8]);
    train_params.e_ref = 40;
    train_params.b_ref = 2;
  }
  delete [] words;
  fclose(fp);
}


void MLCollideModel::train(int step, int training){
    if (!train_this_step(step, training)){
    return;
  } else {
    int N_data = MIN( train_params.len_data, training_data.outputs.size() / training_data.num_outputs);
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    torch::Tensor inputs, outputs;
    // Gather training data to one process. TODO: should be a subcommunicator

    // OUTPUTS
    int *counts = new int[comm->nprocs];
    int nelements = (int)training_data.outputs.size();
    // Each process tells the root how many elements it holds
    MPI_Gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, world);

    // Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[comm->nprocs];
    // Displacement for the first chunk of data - 0
    for (int i = 0; i < comm->nprocs; i++)
      disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

    // Place to hold the gathered data
    // Allocate at root only
    double *data_out = NULL;
    if (comm->me == 0)
      // disps[size-1]+counts[size-1] == total number of elements
      data_out = new double[disps[comm->nprocs-1]+counts[comm->nprocs-1]];
    // Collect everything into the root
    MPI_Gatherv(training_data.outputs.data(), nelements, MPI_DOUBLE,
                data_out, counts, disps, MPI_DOUBLE, 0, world);

    // INPUTS

    // Gather training data to one process. TODO: should be a subcommunicator
    counts = new int[comm->nprocs];
    nelements = (int)training_data.features.size();
    // Each process tells the root how many elements it holds
    MPI_Gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, world);

    // Displacements in the receive buffer for MPI_GATHERV
    disps = new int[comm->nprocs];
    // Displacement for the first chunk of data - 0
    for (int i = 0; i < comm->nprocs; i++)
      disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

    // Place to hold the gathered data
    // Allocate at root only
    double *data_inputs = NULL;
    if (comm->me == 0)
      // disps[size-1]+counts[size-1] == total number of elements
      data_inputs = new double[disps[comm->nprocs-1]+counts[comm->nprocs-1]];
    // Collect everything into the root
    MPI_Gatherv(training_data.features.data(), nelements, MPI_DOUBLE,
                data_inputs, counts, disps, MPI_DOUBLE, 0, world);

    MPI_Allreduce(MPI_IN_PLACE, &N_data, 
              1,
              MPI_INT,
              MPI_SUM, world);
  
    if (comm->me == 0){

      inputs = torch::from_blob(data_inputs, {N_data, training_data.num_features}, options).to(device);
      outputs = torch::from_blob(data_out, {N_data, training_data.num_outputs}, options).to(device);
      save_to_disk(inputs, outputs, step);
      get_implementation()->to(device);
      for(int l=0;l<train_params.epochs;l++){
        torch::Tensor shuffled_indices = torch::randperm(N_data, torch::TensorOptions().dtype(at::kLong));

        outputs = outputs.index({shuffled_indices});
        inputs = inputs.index({shuffled_indices});
        update_LR(total_epochs);
        
        double total_loss=0.;
        int batch_size = train_params.batch_size;
        for (int p=0; (p+batch_size)<N_data+1; p=p+batch_size) {
          Slice slice(p, p+batch_size);
          
          torch::Tensor loss = get_loss(inputs.index({slice}), outputs.index({slice}));
          
          loss.backward();

          total_loss=total_loss + *loss.data_ptr<double>();
          optimizer_step(); // this method should zero the gradients too
          
        }

        // Report training info for the epoch
        std::string filename = "out/training_" + std::to_string(comm->me);
        std::ofstream outfile;

        outfile.open(filename, std::ios_base::app); 
        outfile <<  step << ", " << l << ", " << comm->me << ", " << N_data << ", " <<  total_loss << ", " << std::endl;
        outfile.close();
        total_epochs++;
      }
    } 
  MPI_Barrier(world);

  // Delete training data so that it can be rebuilt for next training step.
  training_data.features.clear();
  training_data.outputs.clear();

  get_implementation()->to(torch::kCPU); // Move back to CPU since collisions happen there. Evaluate this later.
  broadcast_weights();
  
  if (comm->me == 0) {
    save_weights(step);
  }
  }
}

int MLCollideModel::wordparse(int maxwords, char *line, char **words)
{
  int nwords = 1;
  char * word;

  words[0] = strtok(line," \t\n");
  while ((word = strtok(NULL," \t\n")) != NULL && nwords < maxwords) {
    words[nwords++] = word;
  }
  return nwords;
}