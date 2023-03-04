/* Author:      Abigail Smith
 * NETID :      ARS190011
 * Course:      CS 4375.004
 * Professor:   Dr. Mazidi
 * TA:          Ouyang Xu
 * Date:        03/04/2023
 *
 * Purpose: This program was created for assignment "C++ Algorithms from Scratch" and is for part 2
 * of the assignment. This program reads in the provided "Titanic.csv" file and performs naive bayes
 * to predict survived based on sex, age, and passenger class.
 *
 * To find Naive Bayes, this code uses the Naive Bayes formula (uppercase PI)^(D)_(i=1){p(X_i|Y)
 * by first finding the posterior (by prior, likelihood, and marginal), conditional distributions,
 * and conditional probabilities before using the test subset to create a confusion matrix.
 */

// Libraries
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

// Variables
string line;
ifstream in_file;
ofstream log_file;
ofstream exploration_file;
string index_in, pclass_in, survived_in, sex_in, age_in;
const int MAX_LEN = 1046;
vector<int> index(MAX_LEN);
vector<int> pclass(MAX_LEN);
vector<int> survived(MAX_LEN);
vector<int> sex(MAX_LEN);
vector<double> age(MAX_LEN);
time_t ttime = time(0);
string CSV_pathname = ""; // please update to correct path
string log_pathname = ""; // please update to correct path
string data_exploration = ""; // please update to correct path

// Function Declarations
void openCSV();
void readCSV();
void naive_bayes();
double Gaussian_age(double age, double stdev, double mean);
double get_prob(int sex, double age, int pclass, double mean, double stdev, double conditional_prior_surv_or_died,
                double conditional_prob_female, double p_class_1_conditional, double p_class_2_conditional,
                double p_class_3_conditional);

int main() {
    // Creating runtime log
    log_file.open(log_pathname);
    log_file.clear();   // clearing any previous content
    exploration_file.open(data_exploration);
    log_file.clear();

    cout << "RUNNING Naive Bayes.cpp..." << endl;
    log_file << ctime(&ttime) << "\nRUNNING Naive Bayes.cpp..." << endl;
    exploration_file << ctime(&ttime) << endl;

    // Opening and reading in CSV data
    openCSV();
    readCSV();

    // Calculating Naive Bayes from read in data
    naive_bayes();

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
        index.at(numObservations) = stof(pclass_in);
        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stod(age_in);

        numObservations++;
    }

    // setting vector size to correct number of observations
    index.resize(numObservations);
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

// Calculates Naive Bayes
void naive_bayes(){
    // Creating test and train subsets (sizes based on assignment description)
    int trainSize = 800;
    int testSize = MAX_LEN - trainSize;

    // Instantiating vectors for train and test data sets
    vector<int> train_sex(trainSize);
    vector<int> train_survived(trainSize);
    vector<int> train_pclass(trainSize);
    vector<double> train_age(trainSize);
    vector<int> test_sex((int)testSize);
    vector<int> test_survived((int)testSize);
    vector<int> test_pclass(testSize);
    vector<double> test_age(testSize);

    // Selecting observations for train
    for(int i = 0; i < trainSize; i++){
        train_sex[i] = sex[i];
        train_survived[i] = survived[i];
        train_pclass[i] = pclass[i];
        train_age[i] = age[i];
    }

    // Selecting observations for test
    for(int i = trainSize + 1; i < 1046; i++){
        test_sex[i - trainSize] = sex[i];
        test_survived[i - trainSize] = survived[i];
        test_pclass[i - trainSize] = pclass[i];
        test_age[i - trainSize] = age[i];
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
    exploration_file << "\tCount of total passengers: " << count_male + count_female << "\n\tMale passengers: " << count_male << "\n\tFemale passengers: " << count_female << endl;
    exploration_file << "Percentages of total passengers:\n" << "\t% of male passengers: " << ((double)count_male / (double)(count_male + count_female))*100 << endl;
    exploration_file << "\t% of female passengers: " << ((double)count_female / (double)(count_male + count_female))*100 << endl;
    exploration_file << "Count of passengers (from train subset): " << train_count_male + train_count_female << "\n\tMale passengers: " << train_count_male << "\n\tFemale passengers: " << train_count_female << endl;
    exploration_file << "Percentages of passengers (from train subset):\n" << "\t% of male passengers: " << ((double)train_count_male / (double)(train_count_male + train_count_female))*100 << endl;
    exploration_file << "\t% of female passengers: " << ((double)train_count_female / (double)(train_count_male + train_count_female))*100 << endl;

    // Counting the number of survived vs dies passengers based on sex in train subset
    int train_count_male_surv = 0;
    int train_count_male_died = 0;
    int train_count_female_surv = 0;
    int train_count_female_died = 0;
    for(int i = 0; i < train_sex.size(); i++){
        // Checking if male and survived
        if(train_sex[i] == 1 && train_survived[i] == 1){
            train_count_male_surv++;
        // Checking if male and died
        } else if(train_sex[i] == 1 && train_survived[i] == 0){
            train_count_male_died++;
        // Checking if female and survived
        } else if(train_sex[i] == 0 && train_survived[i] == 1){
            train_count_female_surv++;
        // Checking if female and died
        } else {
            train_count_female_died++;
        }
    }

    // Writing data exploration to data exploration file
    exploration_file << "\nFrom train subset:" << "\nCount passengers died vs survived based on sex:" << "\n\tfemale and survived: " << train_count_female_surv << "\n\tfemale and died: " << train_count_female_died << endl;
    exploration_file << "\tmale and survived: " << train_count_male_surv << "\n\tmale and died: " << train_count_male_died << endl;

    // Counting the number of survived vs dies passengers based on passenger class in train subset
    int train_count_pclass_1 = 0;
    int train_count_pclass_2 = 0;
    int train_count_pclass_3 = 0;
    int train_count_pclass_1_surv = 0;
    int train_count_pclass_1_died = 0;
    int train_count_pclass_2_surv = 0;
    int train_count_pclass_2_died = 0;
    int train_count_pclass_3_surv = 0;
    int train_count_pclass_3_died = 0;
    for(int i = 0; i < trainSize; i++){
        // Checking the passenger class type
        if(train_pclass[i] == 1){
            // Checking if the passenger died
            if(train_survived[i] == 0){
                train_count_pclass_1_died++;
                train_count_pclass_1++;
            } else {
                train_count_pclass_1_surv++;
                train_count_pclass_1++;
            }
        } else if(train_pclass[i] == 2){
            // Checking if the passenger died
            if(train_survived[i] == 0){
                train_count_pclass_2_died++;
                train_count_pclass_2++;
            } else {
                train_count_pclass_2_surv++;
                train_count_pclass_2++;
            }
        } else if(train_pclass[i] == 3){
            // Checking if the passenger died
            if(train_survived[i] == 0){
                train_count_pclass_3_died++;
                train_count_pclass_3++;
            } else {
                train_count_pclass_3_surv++;
                train_count_pclass_3++;
            }
        }
    }

    // Writing data exploration to data exploration file
    exploration_file << "Count passengers died vs survived based on passenger class:" << "\n\tpassenger class 1 and survived: " << train_count_pclass_1_surv << "\n\tpassenger class 1 and died: " << train_count_pclass_1_died << endl;
    exploration_file << "\tpassenger class 2 and survived: " << train_count_pclass_2_surv << "\n\tpassenger class 2 and died: " << train_count_pclass_2_died << endl;
    exploration_file << "\tpassenger class 3 and survived: " << train_count_pclass_3_surv << "\n\tpassenger class 3 and died: " << train_count_pclass_3_died << endl;

    // Printing A-Priori Probabilities (Priors for Survived)
    for(int i = 0; i < 50; i++){
        cout << "-";
    }

    // Starting timer to measure training time (by assignment description)
    auto start = chrono::high_resolution_clock::now();

    // Finding prior(s) for posterior
    // Formula from slide deck "Chapter 7 Probability Distributions and Naive Bayes" slide 18 and textbook pg. 126
    double prior_positive_survived = (double)(train_count_female_surv + train_count_male_surv) / 800;
    double prior_negative_survived = (double)(train_count_female_died + train_count_male_died) / 800;
    printf("\nA-priori probabilities (priors for survived):\n\tSurvived: %f\n\tDied: %f\n", prior_positive_survived, prior_negative_survived);

    for(int i = 0; i < 50; i++){
        cout << "-";
    }
    cout << endl;

    // Finding conditional distributions
    // Formula from slide deck "Chapter 7 Probability Distributions and Naive Bayes" slide and textbook pg. 126-127
    double conditional_fem_surv = (double)train_count_female_surv / (train_count_female_surv + train_count_male_surv);
    double conditional_male_surv = (double)train_count_male_surv / (train_count_female_surv + train_count_male_surv);
    double conditional_fem_died = (double)train_count_female_died / (train_count_female_died + train_count_male_died);
    double conditional_male_died = (double)train_count_male_died / (train_count_female_died + train_count_male_died);

    // Writing data exploration to data exploration file
    exploration_file << "Conditional distributions for..." << endl;
    exploration_file << "\tfemale and survived: " << conditional_fem_surv << "\n\tfemale and died: " << conditional_fem_died << endl;
    exploration_file << "\tmale and survived: " << conditional_male_surv << "\n\tmale and died: " << conditional_male_died << endl;

    for(int i = 0; i < 80; i++){
        cout << "-";
    }
    cout << endl;

    // Printing conditional probabilities for survived based on sex
    printf("\tConditional Probabilities (Likelihood Tables):\n\t\t\t\tSex\n\t\t\t\tFemale:\t\tMale:");
    printf("\nSurvived\tDied:\t\t%f\t%f\n\t\tSurvived:\t%f\t%f\n", conditional_fem_died, conditional_male_died, conditional_fem_surv, conditional_male_surv);

    // Finding conditional distributions
    // Formula from slide deck "Chapter 7 Probability Distributions and Naive Bayes" slide and textbook pg. 126-127
    double conditional_pclass_1_surv = (double)train_count_pclass_1_surv / (train_count_pclass_1_surv + train_count_pclass_2_surv + train_count_pclass_3_surv);
    double conditional_pclass_1_died = (double)train_count_pclass_1_died / (train_count_pclass_1_died + train_count_pclass_2_died + train_count_pclass_3_died);
    double conditional_pclass_2_surv = (double)train_count_pclass_2_surv / (train_count_pclass_1_surv + train_count_pclass_2_surv + train_count_pclass_3_surv);
    double conditional_pclass_2_died = (double)train_count_pclass_2_died / (train_count_pclass_1_died + train_count_pclass_2_died + train_count_pclass_3_died);
    double conditional_pclass_3_surv = (double)train_count_pclass_3_surv / (train_count_pclass_1_surv + train_count_pclass_2_surv + train_count_pclass_3_surv);
    double conditional_pclass_3_died = (double)train_count_pclass_3_died / (train_count_pclass_1_died + train_count_pclass_2_died + train_count_pclass_3_died);

    // Writing data exploration to data exploration file
    exploration_file << "Conditional distributions for..." << endl;
    exploration_file << "\tpassenger class 1 and survived: " << conditional_pclass_1_surv << "\n\tpassenger class 1 and died: " << conditional_pclass_1_died << endl;
    exploration_file << "\tpassenger class 2 and survived: " << conditional_pclass_2_surv << "\n\tpassenger class 2 and died: " << conditional_pclass_2_died << endl;
    exploration_file << "\tpassenger class 3 and survived: " << conditional_pclass_3_surv << "\n\tpassenger class 3 and died: " << conditional_pclass_3_died << endl;

    // Printing conditional probabilities for survived based on sex
    printf("\n\n\t\t\t\tPassenger Class:\n\t\t\t\t1:\t\t2:\t\t3:");
    printf("\nSurvived\tDied:\t\t%f\t%f\t%f\n\t\tSurvived:\t%f\t%f\t%f", conditional_pclass_1_died, conditional_pclass_2_died, conditional_pclass_3_died, conditional_pclass_1_surv, conditional_pclass_2_surv, conditional_pclass_3_surv);

    // Finding mean age for survived and died
    double sum_surv = 0.0;
    double sum_died = 0.0;
    int count_surv = 0;
    int count_died = 0;
    for(int i = 0; i < trainSize; i++){
        if(train_survived[i] == 0){
            sum_died = sum_died + train_age[i];
            count_died = count_died + 1;
        } else {
            sum_surv = sum_surv + train_age[i];
            count_surv = count_surv + 1;
        }
    }

    double mean_died = (double)sum_died/count_died;
    double mean_surv = (double)sum_surv/count_surv;

    // Finding standard error for age based on survived and died
    double stdev_surv = 0.0;
    double stdev_died = 0.0;
    for(int i = 0; i < trainSize; i++) {
        if(train_survived[i] == 0) {
            stdev_died += pow((train_age[i] - mean_died), 2);
        } else {
            stdev_surv += pow((train_age[i] - mean_surv), 2);
        }
    }
    stdev_surv = sqrt(stdev_surv / (count_surv - 1));
    stdev_died = sqrt(stdev_died / (count_died - 1));

    // Ending timer since analysis on training is complete
    auto end = chrono::high_resolution_clock::now();
    // Printing elapsed time
    chrono::duration<double> elapsed_seconds = end - start;

    // Writing data exploration to data exploration file
    exploration_file << "Conditional distributions for..." << endl;
    exploration_file << "\tAge (Survived) (Mean): " << mean_surv << "\n\tAge (Survived) (Standard Deviation): " << stdev_surv << endl;
    exploration_file << "\tAge (Died) (Mean): " << mean_died << "\n\tAge (Died) (Standard Deviation): " << stdev_died << endl;

    // Printing conditional probabilities for survived based on sex
    printf("\n\n\t\t\t\tAge\n\t\t\t\tMean:\t\tStandard Deviation:\nSurvived:\tDied:\t\t%f\t%f\n\t\tSurvived:\t%f\t%f\n", (double)sum_died/count_died, stdev_died, (double)sum_surv/count_surv, stdev_surv);

    for(int i = 0; i < 80; i++){
        cout << "-";
    }
    cout << endl;

    // Using found components to complete confusion matrix by test subset
    double curr_prob_surv = 0.0;
    double curr_prob_died = 0.0;
    int true_negative = 0;
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    for(int i = 0; i < test_sex.size(); i++){
        // Finding naive bayes from found components
        curr_prob_died = get_prob(test_sex[i], test_age[i], test_pclass[i], mean_died, stdev_died, prior_negative_survived,
            conditional_fem_died, conditional_pclass_1_died, conditional_pclass_2_died, conditional_pclass_3_died);
        curr_prob_surv = get_prob(test_sex[i], test_age[i], test_pclass[i], mean_surv, stdev_surv, prior_positive_survived, conditional_fem_surv,
            conditional_pclass_1_surv, conditional_pclass_2_surv, conditional_pclass_3_surv);
        if(curr_prob_surv >= curr_prob_died){
            // checking if passenger actually survived
            if(test_survived[i] == 1) {
                true_negative++;
                // the passenger actually died
            } else {
                false_negative++;
            }
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

    // Printing Confusion Matrix
    for(int i = 0; i < 20; i++){
        cout << "-";
    }

    cout << "\nConfusion Matrix" << endl;
    cout << "pred\t0\t1" << endl;
    cout << "0   \t" << true_positive << "\t" << false_positive << endl;
    cout << "1   \t" << false_negative << "\t" << true_negative << endl;

    for(int i = 0; i < 20; i++){
        cout << "-";
    }
    cout << endl;

    // Finding accuracy, sensitivity, and specificity
    // Formula from slide deck "Chapter 06 - Logistic Regression" slides 23 and 25
    double accuracy = ((double)(true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative));
    double sensitivity = ((double)(true_positive)/(true_positive + false_negative));
    double specificity = ((double)(true_negative)/(true_negative+false_positive));

    for(int i = 0; i < 45; i++){
        cout << "-";
    }

    cout << endl;

    cout << "Accuracy, sensitivity, and specificity:" << endl;
    cout << "\tAccuracy: " << accuracy << endl;
    cout << "\tSensitivity: " << sensitivity << endl;
    cout << "\tSpecificity: " << specificity << endl;

    for(int i = 0; i < 45; i++){
        cout << "-";
    }
    cout << endl;

    // Printing run time for algorithm:
    cout << "Run time for algorithm:\n\t" << (chrono::duration_cast<chrono::nanoseconds>(elapsed_seconds)).count() << " ns" << endl;
}

// Finding Gaussian (normal distribution) (likeihood)
// Formula from slide deck "Chapter 07 Probability Distributions and Naive Bayes" slide 14, textbook pg. 134, and textbook pg. 143
double Gaussian_age(double age, double stdev, double mean) {
    double e = 2.71828;
    double pi = 3.14159;
    double gaussian = (1.0 / (sqrt(2 * pi) * stdev)) * pow(e, -((age - mean) * (age - mean) / (2 * stdev * stdev)));
    return gaussian;

}

// Finding probability
// Formula from slide deck "Gaussian Naive Bayes" slide 22 and textbook pg. 136
double get_prob(int sex, double age, int pclass, double mean, double stdev, double conditional_prior_surv_or_died,
                double conditional_prob_female, double p_class_1_conditional, double p_class_2_conditional,
                double p_class_3_conditional){
    double curr_prob = 1.0;

    // First multiply the already found prior with 1.
    curr_prob *= conditional_prior_surv_or_died;

    // finding prob for died
    // Using sex first
    if(sex == 0){
        // female passenger
        curr_prob *= conditional_prob_female;
    } else {
        // male passenger
        curr_prob *= (1 - conditional_prob_female);
    }

    // Using passenger class
    if(pclass == 1){
        // low clas passenger
        curr_prob *= p_class_1_conditional;
    } else if(pclass == 2){
        // mid class passenger
        curr_prob *= p_class_2_conditional;
    } else {
        // high class passenger
        curr_prob *= p_class_3_conditional;
    }

    curr_prob = curr_prob * Gaussian_age(age, stdev, mean);

    return curr_prob;
}
