#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>

#include <sstream>

#include <vector>

#include <cmath>

#include <algorithm>

#include <cstring>

#include <cstdlib>

#include <cstdio>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"

double sqr(double x) { return x * x; };

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}


class Polygon {
public:

    double area() {
        if (vertices.size() < 3) return 0;
        // TODO Lab 2
        // Compute the area of the polygon
        double aire = 0.0;
        int n = (int)vertices.size();
        for (int i = 0; i < n; i++) {
            const Vector& A = vertices[i];
            const Vector& B = vertices[(i + 1) % n];
            aire += A[0] * B[1] - B[0] * A[1];
        }
        return std::abs(aire) * 0.5;
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // TODO Lab 2
        // Compute the centroid of the polygon

        double somme_cross = 0.0;
        double cx_tmp = 0.0;
        double cy_tmp = 0.0;
        int n = (int)vertices.size();
        for (int k = 0; k < n; k++) {
            const Vector& p = vertices[k];
            const Vector& q = vertices[(k + 1) % n];
            double det_pq = p[0] * q[1] - q[0] * p[1];
            somme_cross += det_pq;
            cx_tmp += (p[0] + q[0]) * det_pq;
            cy_tmp += (p[1] + q[1]) * det_pq;
        }
        if (std::abs(somme_cross) < 1e-14) return Vector(0, 0);
        return Vector(cx_tmp / (3.0 * somme_cross),
                    cy_tmp / (3.0 * somme_cross));
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;

        // TODO Lab 2
        // Compute the integral of ||x-Pi||^2 over the polygon

        double somme = 0.0;
        double int_x = 0.0;
        double int_y = 0.0;
        double int_x2_y2 = 0.0;
        int n = (int)vertices.size();
        for (int k = 0; k < n; k++) {
            const Vector& A = vertices[k];
            const Vector& B = vertices[(k + 1) % n];
            double cross_ab = A[0] * B[1] - B[0] * A[1];
            somme += cross_ab;
            int_x += (A[0] + B[0]) * cross_ab;
            int_y += (A[1] + B[1]) * cross_ab;
            double terme_x = A[0] * A[0] + A[0] * B[0] + B[0] * B[0];
            double terme_y = A[1] * A[1] + A[1] * B[1] + B[1] * B[1];
            int_x2_y2 += cross_ab * (terme_x + terme_y);
        }
        double signe = 1.0;
        if (somme < 0) signe = -1.0;
        double aire = signe * somme / 2.0;
        int_x = signe * int_x / 6.0;
        int_y = signe * int_y / 6.0;
        int_x2_y2 = signe * int_x2_y2 / 12.0;
        return int_x2_y2
            - 2.0 * Pi[0] * int_x
            - 2.0 * Pi[1] * int_y
            + Pi.norm2() * aire;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}

void save_frame_fluide(const std::vector<Polygon>& fluid_cells,
                       std::string filename,
                       int frameid = 0) {

    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> dedans_fluide(W * H, 0);
    std::vector<unsigned char> bord(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)fluid_cells.size(); i++) {

        const auto& V = fluid_cells[i].vertices;
        int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;

        for (int j = 0; j < n; j++) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {

                double px = x + 0.5;
                double py = y + 0.5;

                int signe_avant = 0;
                bool dedans = true;
                bool sur_bord = false;

                for (int j = 0; j < n; j++) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];

                    double dx = bx - ax;
                    double dy = by - ay;

                    double qx = px - ax;
                    double qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int signe = (det > 1e-12) - (det < -1e-12);

                    if (signe != 0) {
                        if (signe_avant != 0 && signe != signe_avant) {
                            dedans = false;
                            break;
                        }
                        signe_avant = signe;
                    }

                    double len2 = dx * dx + dy * dy;
                    double proj = qx * dx + qy * dy;

                    if (proj >= 0.0 && proj <= len2 && det * det <= edge_width2 * len2) {
                        sur_bord = true;
                    }
                }

                if (dedans) {
                    int id = (H - 1 - y) * W + x;
                    dedans_fluide[id] = 1;
                    if (sur_bord) bord[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; i++) {
        if (dedans_fluide[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }

        if (bord[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };




        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi

void compute() {

    cells.clear();
    laguerre_cells.clear();

    cells.resize(points.size());
    laguerre_cells.resize(points.size());

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)points.size(); i++) {

        Polygon cellule;

        cellule.vertices.push_back(Vector(0, 0));
        cellule.vertices.push_back(Vector(1, 0));
        cellule.vertices.push_back(Vector(1, 1));
        cellule.vertices.push_back(Vector(0, 1));

        for (int j = 0; j < (int)points.size(); j++) {

            if (i == j) continue;

            double poids_i = weights.empty() ? 0.0 : weights[i];
            double poids_j = weights.empty() ? 0.0 : weights[j];

            cellule = clip_by_bisector(cellule, points[i], points[j], poids_i, poids_j);

            if (cellule.vertices.empty()) break;
        }

        laguerre_cells[i] = cellule;

        if (!weights.empty() && (int)weights.size() == (int)points.size() + 1) {

            double w_air = weights[points.size()];
            double rayon2 = weights[i] - w_air;

            if (rayon2 <= 0.0) {
                cellule.vertices.clear();
            }
            else {
                double rayon = std::sqrt(rayon2);
                int nb_bords = 40;

                for (int k = 0; k < nb_bords; k++) {

                    double a0 = 2.0 * M_PI * k / nb_bords;
                    double a1 = 2.0 * M_PI * (k + 1) / nb_bords;

                    Vector u(points[i][0] + rayon * std::cos(a0),
                             points[i][1] + rayon * std::sin(a0));

                    Vector v(points[i][0] + rayon * std::cos(a1),
                             points[i][1] + rayon * std::sin(a1));

                    cellule = clip_by_edge(cellule, u, v);

                    if (cellule.vertices.empty()) break;
                }
            }
        }

        cells[i] = cellule;
    }
}


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Polygon result;

        int n = (int)V.vertices.size();

        if (n == 0) return result;

        Vector cote = v - u;

        auto val = [&](const Vector& X) {

            Vector ux = X - u;

            return cote[0] * ux[1] - cote[1] * ux[0];

        };

        auto dedans = [&](const Vector& X) {

            return val(X) >= -1e-12;

        };

        for (int k = 0; k < n; k++) {

            Vector A = V.vertices[k];

            Vector B = V.vertices[(k + 1) % n];

            double va = val(A);

            double vb = val(B);

            bool A_in = dedans(A);

            bool B_in = dedans(B);

            if (A_in && B_in) {

                result.vertices.push_back(B);

            }

            else if (A_in && !B_in) {

                double t = va / (va - vb);

                result.vertices.push_back(A + t * (B - A));

            }

            else if (!A_in && B_in) {

                double t = va / (va - vb);

                result.vertices.push_back(A + t * (B - A));

                result.vertices.push_back(B);

            }

        }

        return result;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {

        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Polygon result;
        if (V.vertices.size() == 0) return result;
        Vector dir = Pi - P0;
        double limite = Pi.norm2() - P0.norm2() + w0 - wi;
        auto inside = [&](const Vector& X) {
            double val = 2.0 * dot(X, dir) - limite;
            return val <= 1e-12;
        };
        auto valeur = [&](const Vector& X) {
            return 2.0 * dot(X, dir) - limite;
        };
        int n = (int)V.vertices.size();
        for (int i = 0; i < n; i++) {
            Vector A = V.vertices[i];
            Vector B = V.vertices[(i + 1) % n];
            bool A_in = inside(A);
            bool B_in = inside(B);
            double vA = valeur(A);
            double vB = valeur(B);
            if (A_in && B_in) {
                result.vertices.push_back(B);
            }
            else if (A_in && !B_in) {
                double t = vA / (vA - vB);
                Vector inter = A + t * (B - A);
                result.vertices.push_back(inter);
            }
            else if (!A_in && B_in) {
                double t = vA / (vA - vB);
                Vector inter = A + t * (B - A);
                result.vertices.push_back(inter);
                result.vertices.push_back(B);
            }
        }
        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    // Lab 1 : the polygons representing each individual cell
    std::vector<Polygon> cells;
    std::vector<Polygon> laguerre_cells;
};


// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;
    std::vector<double> lambdas;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);

    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();

    lbfgsfloatval_t fx = 0.0;

    int nb_fluid = (int)ot->vor.points.size();
    bool avec_air = (n == nb_fluid + 1);

    double aire_fluide = 0.0;
    double cible_fluide = 0.0;

    for (int i = 0; i < nb_fluid; i++) {

        double cible_i;
        if (ot->lambdas.empty()) cible_i = 1.0 / nb_fluid;
        else cible_i = ot->lambdas[i];

        double aire_i = ot->vor.cells[i].area();
        double integ_i = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);

        g[i] = aire_i - cible_i;

        fx += x[i] * (aire_i - cible_i) - integ_i;

        aire_fluide += aire_i;
        cible_fluide += cible_i;
    }

    if (avec_air) {
        g[nb_fluid] = cible_fluide - aire_fluide;
        fx += x[nb_fluid] * g[nb_fluid];
    }

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(

    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,

    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,

    int n, int k, int ls) {

    return 0;

}


// Lab 2
void OptimalTransport::optimize() {

    lbfgsfloatval_t fx;
    std::vector<double> poids(vor.weights);

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);

    param.max_iterations = 80;
    param.epsilon = 1e-5;

    int ret = lbfgs((int)poids.size(), &poids[0], &fx, evaluate, NULL, (void*)this, &param);

    vor.weights = poids;
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 1000) : N_particles(N_particles) {
    }

    void init_particules() {

        double R = 0.23;
        Vector centre_depart(0.5, 0.75);

        fluid_volume = M_PI * R * R;

        particles.clear();
        velocities.clear();

        particles.resize(N_particles);
        velocities.resize(N_particles);

        srand(7);

        int nb = 0;

        while (nb < N_particles) {

            double x = centre_depart[0] + R * (2.0 * rand() / RAND_MAX - 1.0);
            double y = centre_depart[1] + R * (2.0 * rand() / RAND_MAX - 1.0);

            Vector p(x, y);

            if ((p - centre_depart).norm2() <= R * R) {

                particles[nb] = p;

                double vx = 0.15 * (2.0 * rand() / RAND_MAX - 1.0);
                double vy = -0.35;

                velocities[nb] = Vector(vx, vy);

                nb++;
            }
        }

        ot.vor.points = particles;

        double r2_debut = (fluid_volume / N_particles) / M_PI;

        ot.vor.weights.clear();

        for (int i = 0; i < N_particles; i++) {
            ot.vor.weights.push_back(0.0);
        }

        ot.vor.weights.push_back(-r2_debut);
    }

    void time_step(double dt) {

        double eps2 = 0.004 * 0.004;
        Vector gravite(0, -13.5);
        double masse = 200.0;

        ot.vor.points = particles;

        if ((int)ot.vor.weights.size() != N_particles + 1) {

            double r2_debut = (fluid_volume / N_particles) / M_PI;

            ot.vor.weights.assign(N_particles + 1, 0.0);
            ot.vor.weights[N_particles] = -r2_debut;
        }

        ot.lambdas.assign(N_particles + 1, 0.0);

        for (int i = 0; i < N_particles; i++) {
            ot.lambdas[i] = fluid_volume / N_particles;
        }

        ot.lambdas[N_particles] = 1.0 - fluid_volume;

        ot.optimize();

        for (int i = 0; i < N_particles; i++) {

            Vector force = masse * gravite;

            if (ot.vor.cells[i].vertices.size() >= 3) {
                Vector c_i = ot.vor.cells[i].centroid();
                force = force + (1.0 / eps2) * (c_i - particles[i]);
            }

            velocities[i] = velocities[i] + dt * (force / masse);

            particles[i] = particles[i] + dt * velocities[i];

            for (int d = 0; d < 2; d++) {

                if (particles[i][d] < 0.0) {
                    particles[i][d] = -particles[i][d];
                    velocities[i][d] *= -0.55;
                }

                if (particles[i][d] > 1.0) {
                    particles[i][d] = 2.0 - particles[i][d];
                    velocities[i][d] *= -0.55;
                }
            }
        }
    }

    void run_simulation() {

        init_particules();

        double dt = 0.0024;

        for (int k = 0; k < 360; k++) {

            time_step(dt);

            if (k % 2 == 0) {
                save_frame_fluide(ot.vor.cells, "water_sim_", k / 2);
                std::cout << "frame " << k / 2 << std::endl;
            }
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;
    std::vector<Vector> velocities;
    double fluid_volume;
};

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");

    }

    fprintf(f, "</svg>\n");
    fclose(f);
}







int main() {


    Fluid fluide(100);

    fluide.run_simulation();

    return 0;

}




// beginning of Fluid template : May 26th, 2026