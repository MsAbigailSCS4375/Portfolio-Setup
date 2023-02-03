// Author:      Abigail Smith
// NetID:       ARS190011
// Date:        02/04/2023
// Course:      CS 4375.004
// Professor:   Dr. Mazidi
/*
 * Summary:
 *  This program is for CS 4375.004 assignment "Portfolio: C++ Data Exploration".
 *  The purpose of this program is to read a CSV file into 2 vectors, find their sum, mean, median, range,
 *  covariance, and correlation.
 *
 *  NOTE: The code to read in a CSV file is from the provided code in the assignment description.
 */

// Libraries
#include <iostream>
#include <fstream>      // file reading and writing log
#include <vector>       // creating vectors
#include <algorithm>    // finding max and min in vector
#include <cmath>        // finding sqrt and power
#include <string>       // for getting file name
#include <ctime>        // for log
using namespace std;

// Variables
string CSV_pathname = "C:\\Boston.csv";    // PLEASE fill in with correct pathname
string log_pathname = "C:\\log_C++_Data_Exploration.txt";  // PLEASE fill in with correct pathname
ifstream in_file;
ofstream log_file;
string line;                    // used for reading in CSV
string rm_in, medv_in;          // used for reading in CSV
const int MAX_LEN = 1000;       // used for vectors holding CSV columns
vector<double> rm(MAX_LEN);
vector<double> medv(MAX_LEN);
time_t ttime = time(0);   // used for log

// Functions
void openCSV();
void readCSV();
double findSum(vector<double>);
double findMean(vector<double>);
double findMedian(vector<double>);
double findRange(vector<double>);
double findCovariance(vector<double>, vector<double>);
double findCorrelation(vector<double>, vector<double>);
vector<double> insertionSort(vector<double> vector1);

int main() {
    // Creating runtime log
    log_file.open(log_pathname);
    log_file.clear();   // clearing any previous content

    cout << "RUNNING CS4375_DataExploration - main.cpp..." << endl;
    log_file << ctime(&ttime) << "\nRUNNING CS4375_DataExploration - main.cpp..." << endl;

    // Opening and reading CSV
    openCSV();
    readCSV();

    // Finding sum of CSV vectors
    cout << "Finding the sum of vector rm and vector medv" << endl;
    cout << "\tSum of rm:   " << findSum(rm) << endl;
    cout << "\tSum of medv: " << findSum(medv) << endl;

    // Finding mean of CSV vectors
    cout << "Finding the mean of vector rm and vector medv" << endl;
    cout << "\tMean of rm:   " << findMean(rm) << endl;
    cout << "\tMean of medv: " << findMean(medv) << endl;

    // Finding median of CSV vectors
    cout << "Finding the median of vector rm and vector medv" << endl;
    cout << "\tMedian of rm:   " << findMedian(rm) << endl;
    cout << "\tMedian of medv: " << findMedian(medv) << endl;

    // Finding range of CSV vectors
    cout << "Finding the range of vector rm and vector medv" << endl;
    cout << "\tRange of rm:   " << findRange(rm) << endl;
    cout << "\tRange of medv: " << findRange(medv) << endl;

    // Finding covariance of CSV vectors
    cout << "Finding the covariance of vector rm and vector medv" << endl;
    cout << "\tCovariance is: " << findCovariance(rm, medv) << endl;

    // Finding correlation of CSV vectors
    cout << "Finding the correlation of vector rm and vector medv" << endl;
    cout << "\tCorrelation is: " << findCorrelation(rm, medv) << endl;

    // Closing runtime log
    log_file << "FINISHED CS4375_DataExploration - main.cpp. Terminating." << endl;
    log_file.close();
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

    // reading in CSV columns into vectors
    int numObservations = 0;
    while(in_file.good()){
        getline(in_file, rm_in, ',');       // reading in rm column with ',' as delimiter
        getline(in_file, medv_in, '\n');    // reading in medv column with '\n' as delimiter

        // writing read contents to vector
        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    // setting vector size to correct number of observations
    rm.resize(numObservations);
    medv.resize(numObservations);

    log_file << "\t\t" <<  CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << " was read successfully." << endl;

    // closing file
    log_file << "\tClosing file " << CSV_pathname.substr(CSV_pathname.find_last_of("\\") + 1) << "." << endl;
    in_file.close();
    return;
}

// Finding the sum of all elements in the provided vector
double findSum(vector<double> curr_vector){
    double sum = 0.0;

    // traversing provided vector
    for(int i = 0; i < curr_vector.size(); i++){
        sum = sum + curr_vector[i];
    }

    return sum;
}

// Finding the mean of elements in the provided vector
double findMean(vector<double> curr_vector){
    double mean = 0.0;

    mean = findSum(curr_vector) / curr_vector.size();

    return mean;
}

// Finding the median of elements in the provided vector
double findMedian(vector<double> curr_vector){
    double median = 0.0;

    // creating a duplicate vector and sorting all values from smallest to largest
    vector<double> sorted_curr_vector = insertionSort(curr_vector);

    // if the size of the sorted vector is even, select the middle element, else select the middle element + 1
    if((sorted_curr_vector.size() % 2) == 0){
        // the size is even
        median = sorted_curr_vector[sorted_curr_vector.size() / 2];
    } else {
        // the size is odd
        median = sorted_curr_vector[(sorted_curr_vector.size() / 2) + 1];
    }

    return median;
}

// Utilizing insertion sort to sort the provided vector
vector<double> insertionSort(vector<double> curr_vector) {
    int curr_index = 0;
    double curr_index_value = 0.0;

    // iterating through all elements in provided vector
    for(int i = 1; i < curr_vector.size(); i++){
        curr_index = i;
        // while the current element is smaller than its previous element, swap elements
        while(curr_index > 0 && curr_vector[curr_index] < curr_vector[curr_index-1]){
            curr_index_value = curr_vector[curr_index];
            curr_vector[curr_index] = curr_vector[curr_index-1];
            curr_vector[curr_index-1] = curr_index_value;
            curr_index--;
        }
    }

    return curr_vector;
}

// Finding the range of the elements in the provided vector
double findRange(vector<double> curr_vector){
    double range = 0.0;

    // calling insertion sort to sort a duplicate of vector
    vector<double> sorted_vector = insertionSort(curr_vector);

    // finding min and max element
    double min = sorted_vector[0];
    double max = sorted_vector[sorted_vector.size()-1];

    range = max - min;

    return range;
}

// Finding covariance of the elements in the provided vectors
double findCovariance(vector<double> first_vector, vector<double> second_vector) {
    double covariance = 0.0;
    double first_vector_mean = findMean(first_vector);
    double second_vector_mean = findMean(second_vector);
    double numerator = 0.0;

    // finding numerator for covariance formula
    for (int i = 0; i < first_vector.size(); i++) {
        numerator = numerator + ((first_vector[i] - first_vector_mean) * (second_vector[i] - second_vector_mean));
    }

    // from formula in textbook
    covariance = numerator / (first_vector.size() - 1);

    return covariance;
}

// Finding correlation of the elements in the provided vectors
double findCorrelation(vector<double> first_vector, vector<double> second_vector){
    double correlation = 0.0;
    double covariance = findCovariance(first_vector, second_vector);
    double first_vector_mean = findMean(first_vector);
    double second_vector_mean = findMean(second_vector);
    double first_vector_standard_deviation = 0.0;
    double second_vector_standard_deviation = 0.0;
    double first_vector_internal_sum = 0.0;     // used to calc standard deviation
    double second_vector_internal_sum = 0.0;    // used to calc standard deviation

    // finding the standard deviation for the first and second vector
    for(int i = 0; i < first_vector.size(); i++){
        first_vector_internal_sum = first_vector_internal_sum + pow(first_vector[i] - first_vector_mean, 2);
        second_vector_internal_sum = second_vector_internal_sum + pow(second_vector[i] - second_vector_mean, 2);
    }

    // calculating standard deviations
    first_vector_standard_deviation = sqrt( first_vector_internal_sum / first_vector.size());
    second_vector_standard_deviation = sqrt(second_vector_internal_sum / second_vector.size());

    // calculating correlation from covariance and standard deviations
    correlation = covariance / (first_vector_standard_deviation * second_vector_standard_deviation);

    return correlation;
}