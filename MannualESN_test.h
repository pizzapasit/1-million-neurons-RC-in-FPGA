#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>
using namespace std;

static unsigned long my_rand_state = 1;

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
    //std::cout<<answer<<endl;
    string stringAnswer = to_string(answer);
    string lastDigit = "0";
    if(flag == true)
    {
        lastDigit = lastDigit+"."+stringAnswer[stringAnswer.length()-1];
        double finalAnswer = std::stod(lastDigit);
        return finalAnswer;
    }
    else{
        lastDigit = lastDigit+"."+stringAnswer[stringAnswer.length()-2];
        double finalAnswer = std::stod(lastDigit)*10;
        return finalAnswer;
    }
    //B.append(A[0]);    
    //std::cout<<A[0]<<endl;
    //std::cout<<B<<endl;
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
const int connectivityNumber = 4;
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

    int *p1;
    p1 = (int*)malloc(sizeof(int));
    int res1 = ((int64_t)p1%10)+20;

    int *p2;
    p2 = (int*)malloc(sizeof(int));
    int res2 = ((int64_t)p2%10)+276;
    //std::cout<<(int)p1<<endl;
    //std::cout<<seconds<<endl;
    //float rand_num = my_rand()*seconds;
    std::cout<<res2<<endl;
    int *p3;
    p3 = (int*)malloc(sizeof(int));
    int res3 = ((int64_t)p3%10)+20;

    int *p4;
    p4 = (int*)malloc(sizeof(int));
    int res4 = ((int64_t)p4%10)+20;

    /*std::cout<<res2<<endl;
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<2;j++)
        {
            std::cout<<generate_static_double_indices(res2,i,j,true)<<endl;
        }
    }*/

    //generate_static_number(res,i,j)

    /*string fname = "Concrete_Data_New.csv";
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    fstream file (fname, ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();
            stringstream str(line);
            while(getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    else
    {
        cout<<"Could not open the file\n";
        flag = false;
    }

    */
    //content.size()
    //j<content[i].size()
    /*std::cout<<content[1].size()<<endl;
    for(int i=1;i<testRow+1;i++)
    {
        for(int j=0;j<content[i].size();j++)
        {
            cout<<content[i][j]<<" ";
        }
        cout<<"\n";
    }*/
    
     // For print a binary matrix.
     
    /*std::cout << "Binary Matrix" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            std::cout << random_int(0, 1) << " ";
        }
        std::cout << std::endl;
    }*/

    //generate state vector
    for (int i = 0; i < stateNumber; i++)
    {
        float randomVar = generate_static_number(res1,i,1);
        stateVector[i] = randomVar;
        //std::cout << stateVector[i] << '\t';
    }

    //generate data matrix
    /*for (int i = 1; i < dataRow; i++)
    {
        for (int j = 0; j < featuresNumber+1; j++)
        {
            float randomVar = generate_static_number(res2,i,j);
            inputMatrix[i][j] = randomVar*10;

        }
    }*/
    
    /*for (int i = 0; i < dataRow; i++)
    {
        for (int j = 0; j < featuresNumber+1; j++)
        {
            std::cout<<inputMatrix[i][j]<<'\t';
        }
        std::cout<<labelTrain[i];
        std::cout<<endl;
    }*/

    //initialize the first input vector (state+data)
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

    /*for(int i=0; i<stateNumber+featuresNumber;i++)
    {
        std::cout<<finalInputVector[i]<<endl;
    }*/

    //generate weight matrix
    for (int i = 0; i < numberOfNode; i++)
    {
        for (int j = 0; j < connectivityNumber; j++)
        {
            //float randomVar = round(my_rand());
            //float r = static_cast <float> (my_rand()) / static_cast <float> (RAND_MAX);
            float randomVar = generate_static_number(res3,i,j);
            weightMatrix[i][j] = randomVar*10;
            //std::cout<<weightMatrix[i][j]<<'\t';
        }
        //std::cout<<endl;
    }

    //std::cout<<"aaaaa"<<endl;
    //generate weight linear
    for (int i = 0; i < featuresNumber; i++)
        {
            float r = generate_static_number(res4,i,2);
            weightLinear[i] = r*10;
           //std::cout<<weightLinear[i]<<endl;
        }

    for(int epoch = 0; epoch < 100; epoch++)
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
                //float getWeight = 
                //float answer = weightMatrix[i][j]*finalInputVector[j];
               
                sum = sum+answer;
                /*if(sum > 2000)
                {
                    std::cout<<"weight : "<<weightMatrix[i][j]<<endl;
                    std::cout<<"input vector : "<<finalInputVector[j]<<endl;
                }*/
            }
            weightOutput[i] = sum;
            /*if(weightOutput[i] > 2000)
            {
                std::cout<<"output : "<<weightOutput[i]<<endl;
            }*/
        }
        if(epoch < dataRow-1)
        {
            for(int i = 0; i < stateNumber;i++)
            {
                finalInputVector[i] = 1/(1 + std::exp(-weightOutput[i]));
                //finalInputVector[i] = weightOutput[i];
            }
            for(int i = stateNumber; i < stateNumber+featuresNumber;i++)
            {
                finalInputVector[i] = inputMatrix[epoch+1][i-stateNumber];
                //setting up for next state vector
            }
            
        }
        float dataForTraining[featuresNumber];
        float sum = 0;
        for(int i=0; i<= featuresNumber;i++)
        {
            //std::cout<<"weightoutput "<<weightOutput[i]<<endl;
            //std::cout<<"weightlinear "<<weightLinear[i]<<endl;
            dataForTraining[i] = 1/(1 + std::exp(-weightOutput[i]));
            //dataForTraining[i] = weightOutput[i];
            //std::cout<<"datafortraining "<<dataForTraining[i]<<endl;
            sum = sum + (weightLinear[i]*dataForTraining[i]);
            //std::cout<<"sum "<<sum<<endl;
        }
        if(labelTrain[epoch]< sum)
        {
           int cols = sizeof(dataForTraining)/sizeof(dataForTraining[0]);
           //dataForTraining[i] = 1/(1 + std::exp(-weightOutput[cols-i]));
           for(int i=0; i<= featuresNumber;i++)
            {
                weightLinear[i] = weightLinear[i]-alpha*dataForTraining[cols-i]*labelTrain[i];
            }
        }
        else
        {
            //std::cout<<"BBBB "<<labelTrain[epoch]<<" >= "<<sum<<endl;
            int cols = sizeof(dataForTraining)/sizeof(dataForTraining[0]);
            //weightLinear.row(0) = weightLinear - (alpha*labelMatrix.coeff(epoch, 0)*dataMatrix.row(epoch));
            for(int i=0; i<= featuresNumber;i++)
            {
                weightLinear[i] = weightLinear[i]+alpha*dataForTraining[cols-i]*labelTrain[i];
            }
        }
        //int rows = sizeof(finalInputVector)/sizeof(finalInputVector[0]);
        // cols = sizeof(finalInputVector[0])/sizeof(finalInputVector[0][0]);

        //std::cout<<rows<<endl;
        //std::cout<<cols<<endl;
        
        


    }
    /*for(int i=0; i < dataRow;i++)
    {
        std::cout<<labelTrain[i]<<endl;
    }*/



    for(int i=0;i < featuresNumber;i++)
    {
        std::cout<<weightLinear[i]<<endl;
    }

    //test phase

    /*
    int count = 0;
    for(int i=dataRow;i<dataRow+testRow;i++)
    {
        for (int j = 0; j < featuresNumber+1; j++)
        {
            try
            {
                if(j == featuresNumber)
                {
                    labelTest[count] = std::stof(content[i][j]);
                }
                else
                {
                    testMatrix[count][j] = std::stof(content[i][j]);
                }
            }
            catch(exception e)
            {
                std::cout<<"Not Number"<<endl;
                std::cout<<content[i][j]<<endl;
            }
        }
        count = count + 1;
    }
    */

        //int rows = sizeof(testMatrix)/sizeof(testMatrix[0]);
        //int cols = sizeof(testMatrix[0])/sizeof(testMatrix[0][0]);

        //std::cout<<rows<<endl;
        //std::cout<<cols<<endl;

        /*int rows = sizeof(labelTest)/sizeof(labelTest);
        int cols = sizeof(labelTest)/sizeof(labelTest[0]);

        std::cout<<rows<<endl;
        std::cout<<cols<<endl;*/

    /*
    float sumError = 0;
    for(int i=0;i<testRow;i++)
    {
        int sum = 0;
        for(int j=0;j<featuresNumber;j++)
        {
            sum = sum+(weightLinear[i]*testMatrix[i][j]);
        }
        prediction[i] = sum; 
        sumError = sumError + (prediction[i] - labelTest[i]);
        //std::cout<<"Comparision : "<< prediction[i]<<" "<<labelTest[i]<<endl;
        //std::cout<<"Result : "<< sumError<<endl;
    }

    std::cout<<"sum error : "<< sumError<<endl;
    std::cout<<"RSE : "<< sqrt(pow(sumError,2)/ testRow)<<endl;
	*/

}
