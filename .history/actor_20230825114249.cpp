#include "Actor.h"
#include <fstream>

Actor::Actor()
{
    Initialize();
}
Actor::~Actor()
{
    cout<<"Actor Memory delete"<<endl;
}


void Actor::setup_weight(const char FC1_weight[], const char FC1_bias[], const char FC2_weight[], const char FC2_bias[], const char FC3_weight[], const char FC3_bias[])
{
    weight.open(FC1_weight); // W_ih
    for(int r=0; r<hidden_units; r++)
    {
        for(int c=0; c<output_size; c++)
        {
            weight>>weight0[r][c];    
        }    
    }
    weight.close();
    weight.open(FC1_bias); // W_hh
    for(int r=0; r<hidden_units; r++)
    {
        weight>>weight1[r];      
    }
    weight.close();
    weight.open(FC2_weight);// b_ih
    for(int r=0; r<hidden_units; r++)
    {   
        for(int c=0; c<hidden_units; c++)
        { 
            weight>>weight2[r][c];    
        }
    }
    weight.close();
    weight.open(FC2_bias);// b_hh
    for(int r=0; r<hidden_units; r++)
    {
        weight>>weight3[r];    
    }
    weight.close();
    weight.open(FC3_weight);// W_fc
    for(int r=0; r<output_size; r++)
    {
        for(int c=0; c<hidden_units; c++)
        {
            weight>>weight4[r][c];    
        }    
    }
    weight.close();
    weight.open(FC3_bias);// b_fc
    for(int r=0; r<output_size; r++)
    {
        weight>>weight5[r];      
    }
    weight.close();
 
    //Fc1
    for (int i = 0; i < hidden_units; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            _FcW1(i,j) = weight0[i][j]; 
        }
        
    }
    for (int i = 0; i < hidden_units; i++)
    {
        _Fcb1(i) = weight1[i];
    }
    //Fc2
    for (int i = 0; i < hidden_units; i++)
    {
        for (int j = 0; j < hidden_units; j++)
        {
            _FcW2(i,j) = weight2[i][j]; 
        }
    }
    for (int i = 0; i < hidden_units; i++)
    {
        _Fcb2(i) = weight3[i];
    }

    //Fc3
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < hidden_units; j++)
        {
            _FcW3(i,j) = weight4[i][j]; 
        }
    }
    for (int i = 0; i < output_size; i++)
    {
        _Fcb3(i) = weight5[i];
    }

    cout << "pass setup weights"<<endl;

}
VectorXd Actor::Fc_layer1(VectorXd x)
{  
    Fc1_output = (_FcW1 * x) + _Fcb1;
    output1 = V_ReLU(Fc1_output);
    return output1;
}
VectorXd Actor::Fc_layer2(VectorXd x)
{  
    Fc2_output = (_FcW2 * x) + _Fcb2;
    output2 = V_ReLU(Fc2_output);
   
    return output2;
}
VectorXd Actor::Fc_layer3(VectorXd x)
{  
    Fc3_output = (_FcW3 * x) + _Fcb3;
    output3 = (Fc3_output);
   
    return output3;
}
VectorXd Actor::Forward(VectorXd x)
{   
    FC1_out = Fc_layer1(x);
    FC2_out = Fc_layer2(FC1_out);
    FC3_out = Fc_layer3(FC2_out);
    return  FC3_out ;
}

void Actor::Initialize()
{       
    input_size = 139; // tau1,4 둘다 예측할 때
    output_size = 6;
    hidden_units = 256;

    x.setZero(input_size);

    _FcW1.setZero(hidden_units, input_size);
    _Fcb1.setZero(hidden_units);
    _FcW2.setZero(hidden_units, hidden_units);
    _Fcb2.setZero(hidden_units);
    _FcW3.setZero(output_size, hidden_units);
    _Fcb3.setZero(output_size);

    Fc1_output.setZero(hidden_units);
    output1.setZero(hidden_units);

    Fc2_output.setZero(hidden_units);
    output2.setZero(hidden_units);

    Fc3_output.setZero(output_size);
    output3.setZero(output_size);

}
VectorXd Actor::V_ReLU(VectorXd x)
{
    VectorXd _x, _relu_x;
    if (x.rows() > 2)
    {
        _relu_x.setZero(x.rows());
        _x.setZero(x.rows());
        _x = x;
        for (int i = 0; i < x.rows(); i++)
        {
            if(_x(i) < 0)
            {
                _relu_x(i) = 0;
            }
            else
            {
                _relu_x(i) = _x(i);
            }
        }
    }
    else
    {
        _relu_x.setZero(x.rows());
        _x.setZero(x.rows());
        _x = x;
        for (int i = 0; i < x.rows(); i++)
        {
            if(_x(i) < 0)
            {
                _relu_x(i) = 0;
            }
            else
            {
                _relu_x(i) = _x(i);
            }
        }
    }
    return _relu_x;
}