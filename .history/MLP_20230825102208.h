#pragma once
#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <time.h>

using namespace std;
using namespace Eigen;

class MLP
{
public:
    MLP();
    virtual ~MLP();

    VectorXd Fc_layer1(VectorXd x);
    VectorXd Fc_layer2(VectorXd x);
    VectorXd Fc_layer3(VectorXd x);
    VectorXd MLP_layer(VectorXd x);
    void state_update(VectorXd x);
    void setup_weight(const char weight_hh[], const char weigh_hi[], const char bias_hh[], const char bias_hi[], const char fcWeight1[], const char fcbias1[]);


private:
    void Initialize();
    VectorXd V_ReLU(VectorXd x);

    ifstream weight;

    double weight0[256][139];     //FC1_weight
    double weight1[256];        //FC1_bias 
    double weight2[256][256];   //FC2_weight
    double weight3[256];        //FC2_bias
    
    double weight4[6][256];     //FC3_weight
    double weight5[6];          //FC3_bias


    int features, hidden_units, fc_node, num_classes, timewindow;

    VectorXd x;
    MatrixXd x_buffer;
    VectorXd _Fcb1, _Fcb2, _Fcb3;
    MatrixXd _FcW1, _FcW2, _FcW3, buffer;

    VectorXd Fc1_output, output1;
    VectorXd Fc2_output, output2;
    VectorXd Fc3_output, output3;
    VectorXd ones_2;

    VectorXd  FC1_out, FC2_out, FC3_out;
    ofstream ffout;
};
#endif