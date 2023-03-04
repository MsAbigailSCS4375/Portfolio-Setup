/* Author:      Abigail Smith
 * NETID :      ARS190011
 * Course:      CS 4375.004
 * Professor:   Dr. Mazidi
 * TA:          Ouyang Xu
 * Date:        03/04/2023
 *
 * Purpose: This program was created for assignment "C++ Algorithms from Scratch" and is for part 1
 * of the assignment. This program reads in the provided "Titanic.csv" file and performs logistic
 * regression to predict survived based on sex.
 *
 * To find Logistic Regression, this code uses the Logistic Regression formula
 * p(x) = 1 / (1 + e^-(w0 + w1x) by first calculating the odds, log odds, probability of survival
 * based on sex, and the coefficients w0 and w1 before using the test subset to create a confusion
 * matrix.
 */

// Libraries
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <math.h>
#include <chrono>
using namespace std;

// Variables
string line;
ifstream in_file;
ofstream log_file;
ofstream exploration_file;
string index_in, pclass_in, survived_in, sex_in, age_in;
const int MAX_LEN = 1046;
vector<int> indexObs(MAX_LEN);
vector<int> pclass(MAX_LEN);
vector<int> survived(MAX_LEN);
vector<int> sex(MAX_LEN);
vector<int> age(MAX_LEN);
time_t ttime = time(0);
float data_matrix_by_weight[800][1];
float prob_vector[1][800];
float error[1][800];
string CSV_pathname = ""; // please update to correct path
string log_pathname = ""; // please update to correct path
string data_exploration = ""; // please update to correct path


// Functions
void openCSV();
void readCSV();
void logistic_regression();
void matrix_mult(float data_matrix[800][2], float weights[2][1]);
double p_function(double, double, double);
double p_2_function(double z);

int main() {
    // Creating runtime log
    log_file.open(log_pathname);
    log_file.clear();   // clearing any previous content
    exploration_file.open(data_exploration);
    log_file.clear();

    cout << "RUNNING CS4375_DataExploration.cpp..." << endl;
    log_file << ctime(&ttime) << "\nRUNNING CS4375_DataExploration.cpp..." << endl;
    exploration_file << ctime(&ttime) << endl;

    // Opening and reading in CSV data
    openCSV();
    readCSV();

    // Performing logistic regression from read in data
    logistic_regression();
    return 0;
}

// Opening CSV file
void openCSV(){
    // opening CSV file
    log_file << "\tAttempting to open " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << endl;
    in_file.open(CSV_pathname);

    // verifying file was open correctly
    if(!in_file.is_open()){
        // file was not open correctly
        cout << "ERROR: File " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " could not be opened. Exiting." << endl;
        log_file << "ERROR: File " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " could not be opened. Exiting." << endl;
        log_file.close();
        exit(0);
    }

    // file was opened correctly, prompting user
    log_file << "\t\t" << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) <<" was opened successfully." << endl;
    return;
}

// Reading in CSV file
void readCSV(){
    log_file << "\tReading " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << endl;

    // verifying file was open correctly
    if(!in_file.is_open()){
        // file was not open correctly
        cout << "ERROR: File " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " could not be opened. Exiting." << endl;
        log_file << "ERROR: File " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " could not be opened. Exiting." << endl;
        log_file.close();
        exit(0);
    }

    // reading in column headings
    getline(in_file, line);
    log_file << line << endl;

    // reading in CSV columns into vectors
    int numObservations = 0;
    while(in_file.good()){
        getline(in_file, index_in, ',');        // reading in rm column with ',' as delimiter
        getline(in_file, pclass_in, ',');      // reading in medv column with '\n' as delimiter
        getline(in_file, survived_in, ',');    // reading in medv column with '\n' as delimiter
        getline(in_file, sex_in, ',');         // reading in medv column with '\n' as delimiter
        getline(in_file, age_in, '\n');         // reading in medv column with '\n' as delimiter

        // writing read contents to vector
        indexObs.at(numObservations) = stof(pclass_in);
        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }

    // setting vector size to correct number of observations
    indexObs.resize(numObservations);
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    log_file << "\t\t" <<  CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " was read successfully." << endl;

    // closing file
    log_file << "\tClosing file " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << "." << endl;
    in_file.close();
    return;
}

// Performs logistic regression
void logistic_regression(){
    // Creating train and test subsets (based on assignment description)
    int trainSize = 800;
    int testSize = MAX_LEN - trainSize;

    // Instantiating vectors for train and test data sets
    vector<int> train_sex(trainSize);
    vector<int> train_survived(trainSize);
    vector<int> test_sex(testSize);
    vector<int> test_survived(testSize);

    // Selecting observations for train
    for(int i = 0; i < trainSize; i++){
        train_sex[i] = sex[i];
        train_survived[i] = survived[i];
    }

    // Selecting observations for test
    for(int i = trainSize + 1; i < MAX_LEN; i++){
        test_sex[i - trainSize] = sex[i];
        test_survived[i - trainSize] = survived[i];
    }

    // Preparing data
    // Counting number of female and male passengers in total
    int count_male = 0;
    int count_female = 0;
    for(int i = 0; i < sex.size(); i++){
        if(sex[i] == 1){
            count_male++;
        } else {
            count_female++;
        }
    }

    // Counting number of female and male passengers in train subset
    int train_count_male = 0;
    int train_count_female = 0;
    for(int i = 0; i < train_sex.size(); i++){
        if(train_sex[i] == 1){
            train_count_male++;
        } else {
            train_count_female++;
        }
    }

    // Writing data exploration to data exploration file
    exploration_file << "Count of total passengers: " << count_male + count_female << "\n\tMale passengers: " << count_male << "\n\tFemale passengers: " << count_female << endl;
    exploration_file << "Percentages of total passengers:\n" << "\t% of male passengers: " << ((double)count_male / (double)(count_male + count_female))*100 << endl;
    exploration_file << "\t% of female passengers: " << ((double)count_female / (double)(count_male + count_female))*100 << endl;
    exploration_file << "Count of passengers (from train subset): " << train_count_male + train_count_female << "\n\tMale passengers: " << train_count_male << "\n\tFemale passengers: " << train_count_female << endl;
    exploration_file << "Percentages of passengers (from train subset):\n" << "\t% of male passengers: " << ((double)train_count_male / (double)(train_count_male + train_count_female))*100 << endl;
    exploration_file << "\t% of female passengers: " << ((double)train_count_female / (double)(train_count_male + train_count_female))*100 << endl;

    // Counting the total number of survived vs dies passengers based on sex
    int count_male_surv = 0;
    int count_male_died = 0;
    int count_female_surv = 0;
    int count_female_died = 0;
    for(int i = 0; i < sex.size(); i++){
        if(sex[i] == 1 && survived[i] == 1){
            count_male_surv++;
        } else if(sex[i] == 1 && survived[i] == 0){
            count_male_died++;
        } else if(sex[i] == 0 && survived[i] == 1){
            count_female_surv++;
        } else {
            count_female_died++;
        }
    }

    // Counting the number of survived vs dies passengers based on sex in train subset
    int train_count_male_surv = 0;
    int train_count_male_died = 0;
    int train_count_female_surv = 0;
    int train_count_female_died = 0;
    for(int i = 0; i < train_sex.size(); i++){
        if(train_sex[i] == 1 && train_survived[i] == 1){
            train_count_male_surv++;
        } else if(train_sex[i] == 1 && train_survived[i] == 0){
            train_count_male_died++;
        } else if(train_sex[i] == 0 && train_survived[i] == 1){
            train_count_female_surv++;
        } else {
            train_count_female_died++;
        }
    }

    // Writing data exploration to data exploration file
    exploration_file << "Exploring count of survived vs died passengers based on sex (from train subset):\n" <<"\tCount of male passengers that survived: " << train_count_male_surv << endl;
    exploration_file << "\tCount of male passengers that died: " << train_count_male_died << "\n\tCount of female passengers that survived: " << train_count_female_surv <<endl;
    exploration_file << "\tCount of female passengers that died: " << train_count_female_died << endl;

    // Starting timer to measure training time (by assignment description)
    auto start = chrono::high_resolution_clock::now();

    // METHOD 1
    // Finding odds
    // Formula from slide deck "Chapter 06 - Logistic Regression" slide 07.
    double odds_fem_surv = (double)(train_count_female_surv)/(double)(train_count_female_died);
    double odds_fem_died = (double)(train_count_female_died)/(double)(train_count_female_surv);
    double odds_male_surv = (double)(train_count_male_surv)/(double)(train_count_male_died);
    double odds_male_died = (double)(train_count_male_died)/(double)(train_count_male_surv);

    // Writing data exploration to data exploration file
    exploration_file << "\nFrom train subset:\nOdds of..." << endl;
    exploration_file << "\tfemale and survived: " << odds_fem_surv << endl;
    exploration_file << "\tfemale and died: " << odds_fem_died << endl;
    exploration_file << "\tmale and survived: " << odds_male_surv << endl;
    exploration_file << "\tmale and died: " << odds_male_died << endl;

    // Finding log odds
    // Formula from slide deck "Chapter 06 - Logistic Regression" slide 13.
    double log_odds_fem_surv = log(odds_fem_surv);
    double log_odds_fem_died = log(odds_fem_died);
    double log_odds_male_surv = log(odds_male_surv);
    double log_odds_male_died = log(odds_male_died);

    // Writing data exploration to data exploration file
    exploration_file << "Log odds of..." << endl;
    exploration_file << "\tfemale and survived: " << log_odds_fem_surv << endl;
    exploration_file << "\tfemale and died: " << log_odds_fem_died << endl;
    exploration_file << "\tmale and survived: " << log_odds_male_surv << endl;
    exploration_file << "\tmale and died: " << log_odds_male_died << endl;

    // Finding probability
    // Formula from slide deck "Chapter 06 - Logistic Regression" slide 8.
    double prob_fem_surv = (double)odds_fem_surv/(1+odds_fem_surv);
    double prob_fem_died = (double)odds_fem_died/(1+odds_fem_died);
    double prob_male_surv = (double)odds_male_surv/(1+odds_male_surv);
    double prob_male_died = (double)odds_male_died/(1+odds_male_died);

    // Writing data exploration to data exploration file
    exploration_file << "Probability of..." << endl;
    exploration_file << "\tfemale and survived: " << prob_fem_surv << endl;
    exploration_file << "\tfemale and died: " << prob_fem_died << endl;
    exploration_file << "\tmale and survived: " << prob_male_surv << endl;
    exploration_file << "\tmale and died: " << prob_male_died << endl;

    // Finding w0 and w1 for logistic function (logistic function = 1 / (1 + e^-(w0+w1x)
    // Formula from slide deck "Chapter 06 - Logistic Regression" slide 13
    double w0 = log(prob_fem_surv / (1 - prob_fem_surv));   // intercept
    double w1 = log((prob_male_surv / (1 - prob_male_surv)) / (prob_fem_surv / (1 - prob_fem_surv))); // slope/coefficient for sex; odds of male over odds of female

    // Writing data exploration to data exploration file
    exploration_file << "Finding components for logistic function:" << endl;
    exploration_file << "\tw0 = " << w0 << endl;
    exploration_file << "\tw1 = " << w1 << endl;

    cout << "Method 1 (Not Utilizing Pseudocode)" << endl;

    for(int i = 0; i < 25; i++){
        cout << "-";
    }
    cout << endl;

    cout << "Coefficients:" << "\n\tw0 = " << w0 << "\n\tw1 = " << w1 << endl;

    for(int i = 0; i < 25; i++){
        cout << "-";
    }
    cout << endl;

    // Ending timer since analysis on training is complete
    auto end = chrono::high_resolution_clock::now();
    // Printing elapsed time
    chrono::duration<double> elapsed_seconds = end - start;

    // Using found components to complete confusion matrix by test subset
    double curr_prob = 0.0;
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    int true_negative = 0;
    for(int i = 0; i < test_sex.size(); i++){
        // Using current observation in test
        curr_prob = p_function(test_sex[i], w0, w1);
        // checking if the passenger survived (based on curr_prob)
        if(curr_prob >= 0.5){
            // checking if passenger actually survived
            if(test_survived[i] == 1) {
                true_negative++;
                // the passenger actually died
            } else {
                false_negative++;
            }
            // the passenger died (based on curr_prob)
        } else {
            // checking if the passenger actually died
            if(test_survived[i] == 0){
                true_positive++;
                // the passenger actually survived
            } else {
                false_positive++;
            }
        }
    }

    for(int i = 0; i < 30; i++){
        cout << "-";
    }
    cout << endl;

    // Printing confusion matrix
    cout << "Confusion Matrix:" << endl;
    cout << "\tpred\t0\t1" << endl;
    cout << "\t0   \t" << true_positive << "\t" << false_positive << endl;
    cout << "\t1   \t" << false_negative << "\t" << true_negative << endl;

    for(int i = 0; i < 30; i++){
        cout << "-";
    }
    cout << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    // Finding accuracy, sensitivity, and specificity
    // Formula from slide deck "Chapter 06 - Logistic Regression" slides 23 and 25
    double accuracy = ((double)(true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative));
    double sensitivity = ((double)(true_positive) / (true_positive + false_negative));
    double specificity = ((double)(true_negative) / (true_negative + false_positive));

    cout << "Accuracy, sensitivity, and specificity:" << endl;
    cout << "\tAccuracy: " << accuracy << endl;
    cout << "\tSensitivity: " << sensitivity << endl;
    cout << "\tSpecificity: " << specificity << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    cout << "Run time for algorithm:\n\t" << (chrono::duration_cast<chrono::nanoseconds>(elapsed_seconds)).count() << " ns" << endl;

    // METHOD 2
    float prob_vector[800][1];
    float error[800][1];
    float transposed_data_matrix[2][800];
    float transposed_data_matrix_by_error[2][1];
    float learning_rate = 0.001;
    float weight[2][1];
    weight[0][0] = 1.0; //w0
    weight[1][0] = 1.0; // w1

//    cout << "weight[0][0] = " << weight[0][0] << endl;
//    cout << "weight[1][0] = " << weight[1][0] << endl;

    float data_matrix[800][2];
    // Writing first column as 1 and second column as passenger sex
    for(int i = 0; i < 800; i++){
        data_matrix[i][0] = 1.0;
        data_matrix[i][1] = (float)train_sex[i];
    }

//    cout << "\ndata_matrix[0][0] = " << data_matrix[0][0] << " data_matrix[1][0] = " << data_matrix[1][0]<< endl;
//    cout << "data_matrix[1][0] = " << data_matrix[1][0] << " data_matrix[1][1] = " << data_matrix[1][1] << endl;
//    cout << "data_matrix[2][0] = " << data_matrix[2][0] << " data_matrix[2][1] = " << data_matrix[2][1] << endl;


    vector<float> labels(800);
    // Writing labels
    for(int i = 0; i < 800; i++){
        labels[i] = train_survived[i];
    }

//    cout << "\nlabels[0] = " << labels[0] << endl;
//    cout << "labels[1] = " << labels[1] << endl;
//    cout << "labels[2] = " << labels[2] << endl;

//    cout << "Performing opimiziation: " << endl;

    auto start2 = chrono::high_resolution_clock::now();

    int numIterations = 500000;
    for(int m = 0; m < numIterations; m++){
        matrix_mult(data_matrix, weight);

//        cout << "\ndata_matrix_by_weight[0][0] = " << data_matrix_by_weight[0][0] << endl;
//        cout << "data_matrix_by_weight[1][0] = " << data_matrix_by_weight[1][0] << endl;
//        cout << "data_matrix_by_weight[2][0] = " << data_matrix_by_weight[2][0] << endl;

        // Calculating prob_vector
        for(int i = 0; i < 800; i++){
            prob_vector[0][i] = p_2_function(data_matrix_by_weight[i][0]);
        }

//        cout << "\nprob_vector[0][0] = " << prob_vector[0][0] << endl;
//        cout << "prob_vector[0][1] = " << prob_vector[0][1] << endl;
//        cout << "prob_vector[0][2] = " << prob_vector[0][2] << endl;

        // Calculating error
        for(int i = 0; i < 800; i++){
            error[i][0] = labels[i] - prob_vector[0][i];
        }

//        cout << "\nerror[0][0] = " << error[0][0] << endl;
//        cout << "error[1][0] = " << error[1][0] << endl;
//        cout << "error[2][0] = " << error[2][0] << endl;

        // Making data_matrix_by_weight transposed
        for(int i = 0; i < 800; i++){
            transposed_data_matrix[0][i] = data_matrix[i][0];
            transposed_data_matrix[1][i] = data_matrix[i][1];
        }

//        cout << "\ntransposed_data_matrix[0][0] = " << transposed_data_matrix[0][0] << " transposed_data_matrix[1][0] = " << transposed_data_matrix[1][0] << endl;
//        cout << "transposed_data_matrix[0][1] = " << transposed_data_matrix[0][1] << " transposed_data_matrix[1][1] = " << transposed_data_matrix[1][1] << endl;
//        cout << "transposed_data_matrix[0][2] = " << transposed_data_matrix[0][2] << " transposed_data_matrix[1][2] = " << transposed_data_matrix[1][2] << endl;

        // Perfoming matrix multiplcation on transposed data matrix and error matrix
        transposed_data_matrix_by_error[0][0] = 0.0;
        transposed_data_matrix_by_error[0][1] = 0.0;
        for(int i = 0; i < 800; i++){
            for(int j = 0; j < 2; j++){
                transposed_data_matrix_by_error[0][j] += transposed_data_matrix[j][i] * error[i][0];
            }
        }

//        cout << "\ntransposed_data_matrix_by_error[0][0] = " << transposed_data_matrix_by_error[0][0] << endl;
//        cout << "transposed_data_matrix_by_error[0][1] = " << transposed_data_matrix_by_error[0][1] << endl;

        // Finding new weights
        weight[0][0] = weight[0][0] + (learning_rate * transposed_data_matrix_by_error[0][0]);
        weight[1][0] = weight[1][0] + (learning_rate * transposed_data_matrix_by_error[0][1]);

        if(m == 2){
            exploration_file << "\t50 iterations\n\t\tw0 = " << weight[0][0] << endl;
            exploration_file << "\t\tw1 = " << weight[1][0] << endl;
        } else if(m == 500){
            exploration_file << "\t500 iterations\n\t\tw0 = " << weight[0][0] << endl;
            exploration_file << "\t\tw1 = " << weight[1][0] << endl;
        } else if(m == 5000){
            exploration_file << "\t5000 iterations\n\t\tw0 = " << weight[0][0] << endl;
            exploration_file << "\t\tw1 = " << weight[1][0] << endl;
        } else if(m == 50000){
            exploration_file << "\t50000 iterations\n\t\tw0 = " << weight[0][0] << endl;
            exploration_file << "\t\tw1 = " << weight[1][0] << endl;
        } else if(m == 500000){
            exploration_file << "\t500000 iterations\n\t\tw0 = " << weight[0][0] << endl;
            exploration_file << "\t\tw1 = " << weight[1][0] << endl;
        }
//        cout << "\nupdated weight[0][0] (w0) = " << weight[0][0] << endl;
//        cout << "updated weight[1][0] (w1) = " << weight[1][0] << endl;
    }

    auto end2 = chrono::high_resolution_clock::now();
    cout << "\nMethod 2 (Utilizing Pseudocode)" << endl;
    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    cout << "Coefficients:" << "\n\tw0 = " << weight[0][0] << "\n\tw1 = " << weight[1][0] << endl;
    cout << "\t(found after " << numIterations << " iterations)" << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    // Using found components to complete confusion matrix by test subset
    curr_prob = 0.0;
    true_positive = 0;
    false_positive = 0;
    false_negative = 0;
    true_negative = 0;
    for(int i = 0; i < test_sex.size(); i++){
        // Using current observation in test
        curr_prob = p_function(test_sex[i], w0, w1);
        // checking if the passenger survived (based on curr_prob)
        if(curr_prob >= 0.5){
            // checking if passenger actually survived
            if(test_survived[i] == 1) {
                true_negative++;
                // the passenger actually died
            } else {
                false_negative++;
            }
            // the passenger died (based on curr_prob)
        } else {
            // checking if the passenger actually died
            if(test_survived[i] == 0){
                true_positive++;
                // the passenger actually survived
            } else {
                false_positive++;
            }
        }
    }

    for(int i = 0; i < 30; i++){
        cout << "-";
    }
    cout << endl;

    // Printing confusion matrix
    cout << "Confusion Matrix:" << endl;
    cout << "\tpred\t0\t1" << endl;
    cout << "\t0   \t" << true_positive << "\t" << false_positive << endl;
    cout << "\t1   \t" << false_negative << "\t" << true_negative << endl;

    for(int i = 0; i < 30; i++){
        cout << "-";
    }
    cout << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    // Finding accuracy, sensitivity, and specificity
    // Formula from slide deck "Chapter 06 - Logistic Regression" slides 23 and 25
    accuracy = ((double)(true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative));
    sensitivity = ((double)(true_positive) / (true_positive + false_negative));
    specificity = ((double)(true_negative) / (true_negative + false_positive));

    cout << "Accuracy, sensitivity, and specificity:" << endl;
    cout << "\tAccuracy: " << accuracy << endl;
    cout << "\tSensitivity: " << sensitivity << endl;
    cout << "\tSpecificity: " << specificity << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    chrono::duration<double> elapsed_seconds2 = end2 - start2;
    cout << "Run time for algorithm:\n\t" << (chrono::duration_cast<chrono::nanoseconds>(elapsed_seconds2)).count() << " ns" << endl;

}

// Performs matrix multiplication
void matrix_mult(float data_matrix[800][2], float weights[2][1]){
    // data_matrix_by_weight is 800x1
    for(int i = 0; i < 800; i++){
        data_matrix_by_weight[i][0] = 0.0;
    }

    for(int i = 0; i < 800; i++) {
        for(int j = 0; j < 1; j++) {
            for(int k = 0; k < 2; k++) {
                data_matrix_by_weight[i][j] += data_matrix[i][k] * weights[k][j];
            }
        }
    }
}

// Solving for p gives the logisitc function
// Formula from slide deck "Chapter 06 - Logistic Regression" slide 13.
double p_function(double x, double w0, double w1){
    return 1 / (1 + exp(-(w0 + w1*x)));
}

double p_2_function(double z){
    return 1 / (1 + exp(-z));
}