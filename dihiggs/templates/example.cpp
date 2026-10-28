// ---------------------------------------------------------------------------
// example.cpp -- use the C++ version of the dihiggs model, and check it.
//
//     g++ -std=c++17 -O2 -o example example.cpp
//     ./example [heftnet_weights.txt] [reference.txt]
//
// 1. Prints the Standard Model spectrum at a few masses.
// 2. Compares the model with reference values computed by the original
//    PyTorch model, and reports PASS or FAIL.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "heftnet.hpp"

int main(int argc, char** argv) {
    std::string weights   = argc > 1 ? argv[1] : "heftnet_weights.txt";
    std::string reference = argc > 2 ? argv[2] : "reference.txt";

    dihiggs::HEFTNet model(weights);

    // --- 1. the SM spectrum ------------------------------------------------
    const dihiggs::Theta SM{1, 1, 0, 0, 0};   // klambda, ct, ctt, cggh, cgghh

    std::printf("Standard Model, sigma per 15 GeV bin\n");
    std::printf("  m_hh (GeV)   sigma (pb)\n");
    double total = 0;
    for (int k = 0; k < 80; ++k) {                // the 80 bin centres
        double mhh = 262.5 + 15.0 * k;
        double sigma = model(mhh, SM);
        total += sigma;
        if (k % 10 == 0) std::printf("  %9.1f   %12.5e\n", mhh, sigma);
    }
    std::printf("  total over 255-1455 GeV: %.5e pb\n\n", total);

    // --- 2. check against the PyTorch reference values ---------------------
    // each line: mhh klambda ct ctt cggh cgghh sigma   ('#' lines are comments)
    std::ifstream ref(reference);
    if (!ref) { std::printf("cannot open %s\n", reference.c_str()); return 1; }

    std::vector<double> model_values, ref_values;
    std::string line;
    while (std::getline(ref, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        double mhh, sigma_ref;
        dihiggs::Theta t{};
        in >> mhh >> t.klambda >> t.ct >> t.ctt >> t.cggh >> t.cgghh >> sigma_ref;
        model_values.push_back(model(mhh, t));
        ref_values.push_back(sigma_ref);
    }

    double scale = 0;
    for (double s : ref_values) scale = std::max(scale, std::fabs(s));
    double worst = 0;
    for (std::size_t n = 0; n < ref_values.size(); ++n) {
        double error = std::fabs(model_values[n] - ref_values[n])
                     / (std::fabs(ref_values[n]) + 1e-12 * scale);
        worst = std::max(worst, error);
    }
    bool ok = worst < 1e-8;

    std::printf("checked %zu points against %s\n", ref_values.size(), reference.c_str());
    std::printf("largest relative difference: %.2e\n", worst);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
