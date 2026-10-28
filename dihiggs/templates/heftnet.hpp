// ---------------------------------------------------------------------------
// heftnet.hpp -- the dihiggs HEFT di-Higgs model in standard C++ (C++17).
//
// Header-only: #include "heftnet.hpp" and nothing else is needed. Only the
// C++ standard library is used, so the header also works in ROOT macros.
//
// What the model computes
// -----------------------
// For a di-Higgs invariant mass m_hh (in GeV) and the five HEFT Wilson
// coefficients theta = (klambda, ct, ctt, cggh, cgghh), the model returns the
// gluon-fusion di-Higgs cross section, in pb, in a 15 GeV wide bin centred at
// m_hh (pp collisions at 13 TeV, POWHEG NLO). Divide by 15 to get
// dsigma/dm_hh in pb/GeV.
//
// The cross section is the dot product of two 23-component vectors,
//
//     sigma(m_hh, theta) = sum_i  c_i(theta) * a_i(m_hh),
//
//   c_i(theta): 23 KNOWN monomials in the Wilson coefficients (monomials()),
//   a_i(m_hh):  23 functions of the mass learned by a small neural network,
//
//     a_i = P_i(x) * exp(Q_i(x)),     x = (m_hh - 255 GeV) / 1500 GeV,
//
// where P is a fully connected network (hidden activation, tanh output) and
// Q is a single linear layer. See docs/dihiggs_model.pdf in
// https://github.com/hbprosper/dihiggs for the physics.
//
// Usage
// -----
//     dihiggs::HEFTNet model("heftnet_weights.txt");
//     dihiggs::Theta   sm{1, 1, 0, 0, 0};      // klambda, ct, ctt, cggh, cgghh
//     double sigma = model(400.0, sm);          // pb per 15 GeV bin at 400 GeV
// ---------------------------------------------------------------------------
#ifndef DIHIGGS_HEFTNET_HPP
#define DIHIGGS_HEFTNET_HPP

#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dihiggs {

constexpr int N_COEFFS = 23;                          // monomials / learned functions
inline const std::string FORMAT = "dihiggs-heftnet-text-v1";

using Vector = std::vector<double>;
using Coeffs = std::array<double, N_COEFFS>;

// The five HEFT Wilson coefficients. The Standard Model is {1, 1, 0, 0, 0}.
struct Theta {
    double klambda;   // Higgs self-coupling modifier (c_hhh)
    double ct;        // top Yukawa coupling modifier
    double ctt;       // t tbar h h contact coupling
    double cggh;      // effective g g h coupling
    double cgghh;     // effective g g h h coupling
};

// ---------------------------------------------------------------------------
// A fully connected layer:  y = W x + b
// W has n_out rows and n_in columns, stored row by row in a flat vector.
// ---------------------------------------------------------------------------
struct Layer {
    int n_in = 0;
    int n_out = 0;
    Vector W;   // n_out * n_in values; W[j * n_in + k] is row j, column k
    Vector b;   // n_out values

    Vector apply(const Vector& x) const {
        Vector y(b);                                   // start from the bias
        for (int j = 0; j < n_out; ++j)
            for (int k = 0; k < n_in; ++k)
                y[j] += W[j * n_in + k] * x[k];
        return y;
    }
};

// ---------------------------------------------------------------------------
// Reading the weights file
//
// The file is a sequence of whitespace-separated tokens; everything after a
// '#' on a line is a comment. It contains
//
//     key value                        header entries, e.g. "activation relu"
//     tensor NAME NDIM D1 ... DNDIM    followed by D1*...*DNDIM numbers,
//                                      in row-major (C) order
// ---------------------------------------------------------------------------
struct Tensor {
    std::vector<int> shape;
    Vector values;
};

struct WeightsFile {
    std::map<std::string, std::string> header;
    std::map<std::string, Tensor> tensors;

    explicit WeightsFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("cannot open " + filename);

        // collect all tokens, dropping comments
        std::stringstream tokens;
        std::string line;
        while (std::getline(file, line))
            tokens << line.substr(0, line.find('#')) << '\n';

        std::string key;
        while (tokens >> key) {
            if (key == "tensor") {
                std::string name;
                int ndim = 0;
                tokens >> name >> ndim;
                Tensor t;
                std::size_t size = 1;
                for (int i = 0; i < ndim; ++i) {
                    int d = 0;
                    tokens >> d;
                    t.shape.push_back(d);
                    size *= static_cast<std::size_t>(d);
                }
                t.values.resize(size);
                for (auto& v : t.values) tokens >> v;
                if (!tokens) throw std::runtime_error(filename + ": bad tensor " + name);
                tensors[name] = t;
            } else {
                tokens >> header[key];
            }
        }
    }

    const std::string& get(const std::string& key) const {
        auto it = header.find(key);
        if (it == header.end()) throw std::runtime_error("missing header entry: " + key);
        return it->second;
    }

    // Build a layer from the tensors NAME.weight [n_out, n_in] and NAME.bias [n_out].
    Layer layer(const std::string& name) const {
        auto w = tensors.find(name + ".weight");
        auto b = tensors.find(name + ".bias");
        if (w == tensors.end() || b == tensors.end())
            throw std::runtime_error("missing tensor: " + name);
        Layer L;
        L.n_out = w->second.shape.at(0);
        L.n_in  = w->second.shape.at(1);
        L.W = w->second.values;
        L.b = b->second.values;
        return L;
    }
};

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------
class HEFTNet {
public:
    explicit HEFTNet(const std::string& filename) {
        WeightsFile f(filename);

        if (f.get("format") != FORMAT)
            throw std::runtime_error(filename + ": expected format " + FORMAT);
        if (std::stoi(f.get("n_coeffs")) != N_COEFFS)
            throw std::runtime_error(filename + ": expected 23 coefficients");

        activation_ = f.get("activation");
        if (activation_ != "relu" && activation_ != "tanh" && activation_ != "sin")
            throw std::runtime_error(filename + ": unknown activation " + activation_);

        // the mass mapping used in training: x = (m_hh - mhh_min) / mhh_scale
        mhh_min_   = std::stod(f.get("mhh_min_GeV"));
        mhh_max_   = std::stod(f.get("mhh_max_GeV"));
        mhh_scale_ = std::stod(f.get("mhh_scale_GeV"));

        int n_layers = std::stoi(f.get("p_layers"));
        for (int i = 0; i < n_layers; ++i)
            P_.push_back(f.layer("P." + std::to_string(i)));
        Q_ = f.layer("Q");
    }

    // Cross section in pb per 15 GeV bin centred at mhh (GeV).
    double operator()(double mhh, const Theta& theta) const {
        Coeffs a = coeffs(mhh);
        Coeffs c = monomials(theta);
        double sigma = 0;
        for (int i = 0; i < N_COEFFS; ++i) sigma += c[i] * a[i];
        return sigma;
    }

    // The 23 learned functions a_i(m_hh) = P_i(x) * exp(Q_i(x)).
    Coeffs coeffs(double mhh) const {
        Vector x{to_x(mhh)};

        // P: hidden layers followed by the activation, the last one by tanh
        Vector h = x;
        for (std::size_t i = 0; i < P_.size(); ++i) {
            h = P_[i].apply(h);
            bool last = (i + 1 == P_.size());
            for (auto& v : h) v = last ? std::tanh(v) : activate(v);
        }

        // Q: one linear layer
        Vector q = Q_.apply(x);

        Coeffs a;
        for (int i = 0; i < N_COEFFS; ++i) a[i] = h[i] * std::exp(q[i]);
        return a;
    }

    // The 23 monomials c_i(theta), in the order of Eq. (4.1) of
    // arXiv:2304.01968 and Table 2 of docs/dihiggs_model.pdf. Terms 1-15
    // suffice at leading order; terms 16-23 appear at next-to-leading order.
    static Coeffs monomials(const Theta& t) {
        const double kl = t.klambda, ct = t.ct, ctt = t.ctt,
                     cggh = t.cggh, cgghh = t.cgghh;
        return Coeffs{
            ct*ct*ct*ct,             //  1
            ctt*ctt,                 //  2
            ct*ct * kl*kl,           //  3
            cggh*cggh * kl*kl,       //  4
            cgghh*cgghh,             //  5
            ctt * ct*ct,             //  6
            kl * ct*ct*ct,           //  7
            ct * kl * ctt,           //  8
            cggh * kl * ctt,         //  9
            ctt * cgghh,             // 10
            cggh * kl * ct*ct,       // 11
            cgghh * ct*ct,           // 12
            kl*kl * cggh * ct,       // 13
            cgghh * ct * kl,         // 14
            cggh * cgghh * kl,       // 15
            ct*ct*ct * cggh,         // 16  (NLO)
            ct * ctt * cggh,         // 17  (NLO)
            ct * cggh*cggh * kl,     // 18  (NLO)
            ct * cggh * cgghh,       // 19  (NLO)
            ct*ct * cggh*cggh,       // 20  (NLO)
            ctt * cggh*cggh,         // 21  (NLO)
            cggh*cggh*cggh * kl,     // 22  (NLO)
            cggh*cggh * cgghh,       // 23  (NLO)
        };
    }

    // Map m_hh in GeV to the dimensionless network input x.
    double to_x(double mhh) const { return (mhh - mhh_min_) / mhh_scale_; }

    // True if mhh lies in the mass range used in training (no extrapolation).
    bool in_range(double mhh) const { return mhh >= mhh_min_ && mhh <= mhh_max_; }

    double mhh_min() const { return mhh_min_; }
    double mhh_max() const { return mhh_max_; }

private:
    double activate(double z) const {
        if (activation_ == "relu") return z > 0 ? z : 0;
        if (activation_ == "tanh") return std::tanh(z);
        return std::sin(z);                            // "sin"
    }

    std::vector<Layer> P_;
    Layer Q_;
    std::string activation_;
    double mhh_min_ = 0, mhh_max_ = 0, mhh_scale_ = 1;
};

}  // namespace dihiggs

#endif  // DIHIGGS_HEFTNET_HPP
