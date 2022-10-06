#include <iostream>
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
using namespace std;

double Logistic(double x)
{
    float answer = 1/(1+std::exp(-x));
    return answer;
}

double LinearRegression(double input, double weight)
{
    float answer = input*weight;
    return answer;
}

double floorValue(double input)
{
    float answer = round(input);
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    return answer*r;
}

/*double trainingUpdateIncrease(double prediction, double target)
{
    //w = w + alpha*y_train[r]*x
    float alpha = 0.05;
    float answer = 0;
    return answer;
}

double trainingUpdateDecrease(double prediction, double target)
{
    //w = w - alpha*y_train[r]*x
    float alpha = 0.05;
    float answer = 0;
    return answer;
}*/

int main(void)
{
    float alpha = 0.05;
    int dataRow = 1000;
    int featuresNumber = 30;
    int stateNumber = 70;
    int connectivityNumber = 100;
    int numberOfNode = 1000000;

    //float dataMatrix[dataRow][featuresNumber];
    //float stateVector[stateNumber];
    //float weightMatrix[numberOfNode][connectivityNumber];
    //float InputVector[stateNumber + featuresNumber];

    Eigen::MatrixXd dataMatrix = Eigen::MatrixXd(dataRow,featuresNumber);
    dataMatrix.setRandom();
    dataMatrix = 10*dataMatrix;

    Eigen::MatrixXd labelMatrix = Eigen::MatrixXd(dataRow,1);
    labelMatrix.setRandom();




    Eigen::MatrixXd weightMatrix = Eigen::MatrixXd(numberOfNode,connectivityNumber);
    weightMatrix.setRandom();
    weightMatrix << weightMatrix.unaryExpr(&floorValue);


    Eigen::MatrixXd stateVector = Eigen::MatrixXd(1,stateNumber);
    stateVector.setRandom();

    Eigen::MatrixXd weightLinear = Eigen::MatrixXd(1,featuresNumber);
    weightLinear.setRandom();

    std::cout<<weightLinear<<endl;


    for(int epoch = 0; epoch < dataRow; epoch++)
    {
       
        Eigen::MatrixXd temp1(1,stateNumber + featuresNumber);
        temp1 << stateVector, dataMatrix.row(epoch);
        Eigen::MatrixXd inputVector(stateNumber + featuresNumber,1);
        inputVector << temp1.transpose();
        Eigen::MatrixXd dotProduct(numberOfNode,1);
        dotProduct = weightMatrix * inputVector;
        Eigen::MatrixXd temp0(featuresNumber,1);
        temp0.col(0) = dotProduct.bottomRows(featuresNumber); 
        Eigen::MatrixXd temp2(featuresNumber,1);
        temp2 << temp0.unaryExpr(&Logistic);
        Eigen::MatrixXd temp6(numberOfNode,1);
        temp6 << dotProduct.unaryExpr(&Logistic);
        Eigen::MatrixXd prediction(1,1);
        prediction = weightLinear * temp2;
        
        if(prediction.coeff(0, 0) < labelMatrix.coeff(epoch, 0))
        {
            Eigen::MatrixXd weightLinearTemp = Eigen::MatrixXd(1,numberOfNode);
            weightLinear.row(0) = weightLinear + (alpha*labelMatrix.coeff(epoch, 0)*dataMatrix.row(epoch));
        }
        else if (prediction.coeff(0, 0) > labelMatrix.coeff(epoch, 0))
        {
            weightLinear.row(0) = weightLinear - (alpha*labelMatrix.coeff(epoch, 0)*dataMatrix.row(epoch));
        }

        Eigen::MatrixXd temp3(1,numberOfNode);
        temp3 << temp6.transpose();
        stateVector.row(0) = temp3.rightCols(stateNumber);

        

    }




    std::cout<<weightLinear<<endl;
    std::cout<<numberOfNode<<endl;

    return 0;
}
