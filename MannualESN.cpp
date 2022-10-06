#include <iostream>
#include <cmath>

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;



int main(void)
{

    float alpha = 0.005;
    int dataRow = 850;
    int testRow = 150;
    //int featuresNumber = 30;
    int featuresNumber = 8;
    //int stateNumber = 70;
    int stateNumber = 10;
    int connectivityNumber = 100;
    int numberOfNode = 200;
    float inputMatrix[dataRow][featuresNumber];
    float testMatrix[testRow][featuresNumber];
    float stateVector[stateNumber];
    float weightMatrix[numberOfNode][connectivityNumber];
    float finalInputVector[stateNumber + featuresNumber];
    bool flag = true;
    float weightLinear[featuresNumber];
    int seed = 16;
  
    float labelTrain[dataRow];
    float labelTest[testRow];
    float prediction[testRow]; 

    string fname = "Concrete_Data_New.csv";
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
        float randomVar = ((float)rand() / (float)RAND_MAX);
        stateVector[i] = randomVar;
        //std::cout << stateVector[i] << '\t';
    }

    //generate data matrix
    for (int i = 1; i < dataRow; i++)
    {
        for (int j = 0; j < featuresNumber+1; j++)
        {
            //float randomVar = ((float)rand() / (float)RAND_MAX);
            //inputMatrix[i][j] = randomVar*10;
            //std::cout << inputMatrix[i][j] << '\t';
            try
            {
                if(j == featuresNumber)
                {
                    //std::cout<<content[i][j];
                    labelTrain[i-1] = std::stof(content[i][j]);
                    //std::stof(str);
                }
                else
                {
                    inputMatrix[i-1][j] = std::stof(content[i][j]);
                }
            }
            catch(exception e)
            {
                std::cout<<"Not Number"<<endl;
                std::cout<<content[i][j]<<endl;
            }
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


    //generate weight matrix
    for (int i = 0; i < numberOfNode; i++)
    {
        for (int j = 0; j < connectivityNumber; j++)
        {
            float randomVar = round(((float)rand() / (float)RAND_MAX));
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            weightMatrix[i][j] = randomVar*r*10;
        }
    }

    //generate weight linear
    for (int i = 0; i < featuresNumber; i++)
        {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            weightLinear[i] = r*10;
        }

    for(int epoch = 0; epoch < 100; epoch++)
    {
        float weightOutput[numberOfNode];
        for(int i = 0;i < numberOfNode;i++)
        {
            float sum = 0;
            int count = 0;
            for(int j = 0;j < connectivityNumber; j++)
            {
                int randCon = round(generate_random_connection(seed, i, j)*100)+1;
                if(count > 0)
                {
                    float answer = weightMatrix[i][randCon]*finalInputVector[j];
                }
                else{
                    float answer = weightMatrix[i][j]*finalInputVector[j];
                    count = count +1;
                } 
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
                //setting up for next state vector
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

        //int rows = sizeof(testMatrix)/sizeof(testMatrix[0]);
        //int cols = sizeof(testMatrix[0])/sizeof(testMatrix[0][0]);

        //std::cout<<rows<<endl;
        //std::cout<<cols<<endl;

        /*int rows = sizeof(labelTest)/sizeof(labelTest);
        int cols = sizeof(labelTest)/sizeof(labelTest[0]);

        std::cout<<rows<<endl;
        std::cout<<cols<<endl;*/
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


    return 0;
}
