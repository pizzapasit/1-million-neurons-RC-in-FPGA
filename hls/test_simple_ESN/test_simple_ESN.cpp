#include "test_header.h"

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

float generate_static_number()
{
    float answer = 0;
    return answer;
}

const float alpha = 0.005;
const int dataRow = 850;
const int testRow = 150;
const int featuresNumber = 8;
const int stateNumber = 10;
const int connectivityNumber = 100;
const int numberOfNode = 200;

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



    for (int i = 0; i < stateNumber; i++)
    {
        float randomVar = my_rand();
        stateVector[i] = randomVar;
    }
    for (int i = 1; i < dataRow; i++)
    {
        for (int j = 0; j < featuresNumber+1; j++)
        {
            float randomVar = my_rand();
            inputMatrix[i][j] = randomVar*10;
        }
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

    for (int i = 0; i < numberOfNode; i++)
    {
        for (int j = 0; j < connectivityNumber; j++)
        {
            float randomVar = round(my_rand());
            float r = static_cast <float> (my_rand()) / static_cast <float> (RAND_MAX);
            weightMatrix[i][j] = randomVar*r*10;
        }

    }
    for (int i = 0; i < featuresNumber; i++)
        {
            float r = static_cast <float> (my_rand()) / static_cast <float> (RAND_MAX);
            weightLinear[i] = r*10;
        }

    for(int epoch = 0; epoch < 100; epoch++)
    {
        float weightOutput[numberOfNode];
        for(int i = 0;i < numberOfNode;i++)
        {
            float sum = 0;
            for(int j = 0;j < connectivityNumber; j++)
            {
                float answer = weightMatrix[i][j]*finalInputVector[j];
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

