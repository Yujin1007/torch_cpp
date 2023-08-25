#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <chrono>
using namespace std;

class Net : public torch::nn::Module
{
public:
  Net(int64_t input_size, int64_t hidden_size, int64_t output_size)
      : linear1(input_size, hidden_size),
        linear2(hidden_size, hidden_size),
        linear3(hidden_size, output_size)
  {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
    register_module("linear3", linear3);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(linear1(x));
    x = torch::relu(linear2(x));
    x = linear3(x);
    return x;
  }
  void setWeights()
  {
    torch::load(linear1, "../weight/fc1.pt");
    torch::load(linear2, "../weight/fc2.pt");
    torch::load(linear3, "../weight/fc3.pt");
    // cout<<linear1->bias<<endl;
  }

private:
  torch::nn::Linear linear1;
  torch::nn::Linear linear2;
  torch::nn::Linear linear3;
};

int main(int argc, char **argv)
{
  ifstream inputFile(argv[1]);
  vector<vector<float>> data;
  string line;
  while (getline(inputFile, line))
  {
    vector<float> row;
    istringstream iss(line);
    float value;
    while (iss >> value)
    {
      row.push_back(value);
      if (iss.peek() == ',')
        iss.ignore();
    }

    data.push_back(row);
  }
  inputFile.close();

  ifstream inputFile2(argv[2]);
  vector<vector<float>> target;

  while (getline(inputFile2, line))
  {
    vector<float> row;
    istringstream iss(line);
    float value;
    while (iss >> value)
    {
      row.push_back(value);
      if (iss.peek() == ',')
        iss.ignore();
    }

    target.push_back(row);
  }
  inputFile2.close();

  int64_t input_size = 139;
  int64_t hidden_size = 256;
  int64_t output_size = 6;

  torch::manual_seed(123); // Set a random seed for reproducibility

  // Create a neural network
  Net net(input_size, hidden_size, output_size);

  net.setWeights();
  net.to(torch::kCPU);

  // torch::nn::MSELoss loss_fn;
  // torch::optim::SGD optimizer(net.parameters(), /*lr=*/0.01);
  torch::TensorOptions options_(torch::kFloat) s;
  clock_t start, finish;
  double duration;

  std::chrono::steady_clock::time_point st_start_time;
  
  
  
  for (size_t i = 0; i < 10; ++i)
  {
    torch::Tensor input = torch::from_blob(data[i].data(), {139}, options_);

    torch::Tensor answer = torch::from_blob(target[i].data(), {6}, options_);
    start = clock();

    st_start_time = std::chrono::steady_clock::now();

    double control_time_real_ = 0.0;
    auto output = net.forward(input);
    // cout<<input<<endl;
    control_time_real_ = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - st_start_time).count();
  
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f sec\n", duration);
    control_time_real_ = control_time_real_ / 1000;
  cout << "all : " << control_time_real_ << "ms" << endl
       << endl;

    // cout<<answer<<endl<<endl;
    // cout<<output<<endl;
  }
  return 0;
}