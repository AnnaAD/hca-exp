#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath> // For std::sqrt
#include <cstdint>
#include <timer.h>
#include <omp.h>

double distance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    // Ensure vectors have the same length
    if (vec1.size() != vec2.size()) {
        // Handle error, e.g., throw an exception or return a special value
        return -1.0; // Or throw std::invalid_argument("Vectors must have the same length");
    }

    double dotProduct = 0.0;
    double magnitude1 = 0.0;
    double magnitude2 = 0.0;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    // Handle cases where one or both magnitudes are zero to avoid division by zero
    if (magnitude1 == 0.0 || magnitude2 == 0.0) {
        return 1.0; // Or handle as appropriate for your application (e.g., 0 for identical zero vectors)
    }

    double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

    // Cosine distance is 1 - cosine similarity
    return 1.0 - cosineSimilarity;
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <csv_file> <rows> <cols>\n";
        return 1;
    }

    std::string filename = argv[1];
    int rows = std::atoi(argv[2]);
    int cols = std::atoi(argv[3]);

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return 1;
    }

    std::vector<std::vector<double>> data(rows, std::vector<double>(cols));
    std::string line;
    std::getline(infile, line);
    std::cout << "headers: " << line << "\n";
    int row = 1;

    while (std::getline(infile, line) && row < rows) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',') && col < cols) {
            try {
                data[row][col] = std::stod(cell);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Non-numeric value in CSV at row " << row << ", col " << col << "\n";
                return 1;
            }
            ++col;
        }

        if (col != cols) {
            std::cerr << "Error: Expected " << cols << " columns, but got " << col << " in row " << row << "\n";
            return 1;
        }

        ++row;
    }

    if (row != rows) {
        std::cerr << "Error: Expected " << rows << " rows, but got " << row << "\n";
        return 1;
    }

    std::cout << "Loaded CSV successfully:\n";
    // for (const auto& r : data) {
    //     for (double val : r) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << "\n";
    // }

    //std::cout << data[0] << "\n";

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         std::cout << data[i][j] << "\n";
    //     }
    // }

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    std::cout << "OpenMP (" << num_threads << " threads)\n";


    int npoints = rows;
    double* distmat = new double[(npoints*(npoints-1))/2];
    Timer t;
    t.Start();
    #pragma omp parallel for //schedule(static,64)
    for (uint64_t i=0; i<npoints; i++) {
        for (uint64_t j=i+1; j<npoints; j++) {
            uint64_t k = (i * (2 * npoints - i - 1)) / 2 + (j - i - 1);
            distmat[k] = distance(data[i], data[j]);
        }
    }
    t.Stop();

    printf(">>Completed Distance Matrix: %lf\n\n", t.Seconds());



    return 0;
}
