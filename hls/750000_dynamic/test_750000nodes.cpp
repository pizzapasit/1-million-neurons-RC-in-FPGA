#include "test_750000nodes.h"



float GetNumberOfDigits (unsigned i)
{
    return i > 0 ? (int) log10 ((double) i) + 1 : 1;
}

void my_srand(unsigned long seed)
{
    my_rand_state = seed;
}

float my_rand()
{
    my_rand_state = (my_rand_state * 1103515245 + 12345) % 2147483648;
    int digits = GetNumberOfDigits(my_rand_state);
    float answer = my_rand_state/pow(10,digits-3);
    return answer;
}

double generate_static_double_indices(int num, int i, int j, bool flag)
{
    int randomNumber = num;
    int answer = (randomNumber + (256*i)+(1024*j));
    int digits = GetNumberOfDigits(answer);
    //string stringAnswer = to_string(answer);
    int lastDigit = 0;
    if(flag == true)
    {
        lastDigit = lastDigit+ (answer % 10);
        double finalAnswer = lastDigit;
        return finalAnswer;
    }
    else{
        lastDigit =  lastDigit+ ((answer % 100)-(answer % 10));
        double finalAnswer = lastDigit;
        return finalAnswer;
    }

}


float generate_static_number(int num, int i, int j)
{
    int randomNumber = num;
    int answer = (randomNumber * (110+i+j) + 12345) % 213;
    int digits = GetNumberOfDigits(answer);
    float finalAnswer = answer/pow(10,digits);
    return finalAnswer;
}

const float alpha = 0.05;
const int dataRow = 850;
const int testRow = 150;
const int featuresNumber = 8;
const int stateNumber = 10;
const int connectivityNumber = 3;
const int numberOfNode = 750000;

float inputMatrix[dataRow][featuresNumber];
float testMatrix[testRow][featuresNumber];
float stateVector[stateNumber];
float weightMatrix[numberOfNode][connectivityNumber];
float finalInputVector[stateNumber + featuresNumber];
bool flag = true;
float weightLinear[featuresNumber];

  
float labelTrain[dataRow];
float labelTest[testRow];
float prediction[testRow]; 


void ESN()
{

    int *p1;
    //p1 = (int*)malloc(sizeof(int));
    //int res1 = ((int64_t)p1%10)+20;
    int res1 = 26;

    int *p2;
    //p2 = (int*)malloc(sizeof(int));
    //int res2 = ((int64_t)p2%10)+276;
    int res2 = 27;

    //std::cout<<res2<<endl;
    int *p3;
    //p3 = (int*)malloc(sizeof(int));
    //int res3 = ((int64_t)p3%10)+20;
    int res3 = 28;

    int *p4;
    //p4 = (int*)malloc(sizeof(int));
    //int res4 = ((int64_t)p4%10)+20;
    int res4 = 29;

 
    for (int i = 0; i < stateNumber; i++)
    {
        float randomVar = generate_static_number(res1,i,1);
        stateVector[i] = randomVar;
    }


    for(int i=0;i < stateNumber+featuresNumber; i++)
    {
        if(i < stateNumber)
        {
            finalInputVector[i] = stateVector[i];
        }
        if(i >= stateNumber)
        {
            finalInputVector[i] = inputMatrix[0][i-stateNumber];
        }
    }

        /*
    for (int i = 0; i < numberOfNode; i++)
    {
        for (int j = 0; j < connectivityNumber; j++)
        {
            float randomVar = generate_static_number(res3,i,j);
            weightMatrix[i][j] = randomVar*10;
        }

    }*/


    for (int i = 0; i < featuresNumber; i++)
        {
            float r = generate_static_number(res4,i,2);
            weightLinear[i] = r*10;
        }

    for(int epoch = 0; epoch < 1; epoch++)
    {
        float weightOutput[numberOfNode];
        for(int i = 0;i < numberOfNode;i++)
        {
            float sum = 0;
            for(int j = 0;j < connectivityNumber; j++)
            {
                int element = generate_static_double_indices(res2,i,j,false);
                float weight = generate_static_double_indices(res2,i,j,true);
                float answer = finalInputVector[element]*weight;
               
                sum = sum+answer;
            }
            weightOutput[i] = sum;
        }
        if(epoch < dataRow-1)
        {
            for(int i = 0; i < stateNumber;i++)
            {
                finalInputVector[i] = 1/(1 + std::exp(-weightOutput[i]));
            }
            for(int i = stateNumber; i < stateNumber+featuresNumber;i++)
            {
                finalInputVector[i] = inputMatrix[epoch+1][i-stateNumber];
            }
            
        }
        float dataForTraining[featuresNumber];
        float sum = 0;
        for(int i=0; i<= featuresNumber;i++)
        {

            dataForTraining[i] = 1/(1 + std::exp(-weightOutput[i]));
            sum = sum + (weightLinear[i]*dataForTraining[i]);
        }
        if(labelTrain[epoch]< sum)
        {
           int cols = sizeof(dataForTraining)/sizeof(dataForTraining[0]);
           for(int i=0; i<= featuresNumber;i++)
            {
                weightLinear[i] = weightLinear[i]-alpha*dataForTraining[cols-i]*labelTrain[i];
            }
        }
        else
        {
            int cols = sizeof(dataForTraining)/sizeof(dataForTraining[0]);
            for(int i=0; i<= featuresNumber;i++)
            {
                weightLinear[i] = weightLinear[i]+alpha*dataForTraining[cols-i]*labelTrain[i];
            }
        }
        
        


    }




    for(int i=0;i < featuresNumber;i++)
    {
        std::cout<<weightLinear[i]<<endl;
    }

 
}
