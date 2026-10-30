#pragma once
#include "../airy_plate/kirsch.hpp"
#include "../airy_plate/shape_change.hpp"
#include "../complementary_field.hpp"
#include "../spline_field.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace plate4k {
constexpr int W = 3840, H = 2160, nx = W / 16, ny = H / 16, N = nx * ny,
              frames = 601;
constexpr double cx = 1920, cy = 1080, pixel = .01, radius = 112;
using Clock = std::chrono::steady_clock;
inline double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
inline long long ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}
inline FILE *file(const std::string &p, const char *m = "w") {
  auto *f = std::fopen(p.c_str(), m);
  if (!f)
    throw std::runtime_error("open " + p);
  return f;
}
inline double vm(double x, double y, double s) {
  return 2. / 3 * std::sqrt(x * x - x * y + y * y + 3 * s * s);
}
struct Geometry {
  std::vector<double> ref = std::vector<double>(2 * N),
                      weights = std::vector<double>(2 * N, 0);
  std::vector<unsigned char> support = std::vector<unsigned char>(N, 0),
                             all = std::vector<unsigned char>(N, 1),
                             near = std::vector<unsigned char>(N, 0);
  Geometry() {
    int cols = 0;
    for (int j = 0; j < nx; ++j) {
      double x = 8 + 16 * j;
      if (x >= 56 && x <= 3784)
        ++cols;
    }
    for (int i = 0; i < N; ++i) {
      double x = 8 + 16 * (i % nx), y = 8 + 16 * (i / nx),
             r = std::hypot(x - cx, y - cy);
      ref[2 * i] = x;
      ref[2 * i + 1] = y;
      support[i] = x >= 56 && x <= 3784 && y >= 120 && y <= 2040 && r >= 144;
      near[i] = support[i] && r <= 288;
      if (x >= 56 && x <= 3784) {
        if (y == 120 || y == 136)
          weights[2 * i + 1] = -1. / (2 * cols);
        if (y == 2024 || y == 2040)
          weights[2 * i + 1] = 1. / (2 * cols);
      }
    }
  }
  double extension(const std::vector<double> &u) const {
    double d = 0;
    for (int j = 0; j < 2 * N; ++j)
      if (weights[j])
        d += weights[j] * u[j];
    return d;
  }
};
// Image-independent exact analytical solution: generation/scoring only.
struct Solution {
  double pin = 0, hpin = 0, ratio = 0, command = 0;
  Solution(const Geometry &g) {
    std::vector<double> v(2 * N), h(2 * N);
    for (int i = 0; i < N; ++i)
      if (g.weights[2 * i + 1]) {
        auto a = raw(g.ref[2 * i], g.ref[2 * i + 1], false),
             b = raw(g.ref[2 * i], g.ref[2 * i + 1], true);
        v[2 * i + 1] = a.y;
        h[2 * i + 1] = b.y;
        if (g.weights[2 * i + 1] < 0) {
          pin -= g.weights[2 * i + 1] * a.y;
          hpin -= g.weights[2 * i + 1] * b.y;
        }
      }
    command = g.extension(v);
    ratio = g.extension(h) / command;
  }
  static airy::Vec raw(double X, double Y, bool horizontal) {
    double x = (X - cx) * pixel, y = (Y - cy) * pixel;
    auto a =
        horizontal ? airy::displacement(y, x, 1) : airy::displacement(x, y, 1);
    return horizontal ? airy::Vec{a.y / pixel, a.x / pixel}
                      : airy::Vec{a.x / pixel, a.y / pixel};
  }
  airy::Vec unit(double X, double Y, bool horizontal) const {
    auto v = raw(X, Y, false);
    v.y -= pin;
    if (!horizontal)
      return v;
    auto h = raw(X, Y, true);
    return {h.x - ratio * v.x, h.y - hpin - ratio * v.y};
  }
  airy::Vec extended(double X, double Y, bool horizontal) const {
    double x = X - cx, y = Y - cy, r = std::hypot(x, y);
    if (r >= radius)
      return unit(X, Y, horizontal);
    double shift = horizontal ? hpin : pin;
    if (r < 1e-10)
      return {0, -shift};
    auto u = unit(cx + x * radius / r, cy + y * radius / r, horizontal);
    return {r / radius * u.x, r / radius * (u.y + shift) - shift};
  }
  airy::Strain strain(double X, double Y, double p, double h) const {
    double x = (X - cx) * pixel, y = (Y - cy) * pixel;
    auto v = airy::stress(x, y, p - ratio * h), s = airy::stress(y, x, h);
    double xx = v.xx + s.yy, yy = v.yy + s.xx, xy = v.xy + s.xy;
    double ex =
               ((1 - airy::poisson) * xx - airy::poisson * yy) / (2 * airy::mu),
           ey =
               ((1 - airy::poisson) * yy - airy::poisson * xx) / (2 * airy::mu),
           es = xy / (2 * airy::mu);
    return {ex, ey, es, vm(ex, ey, es)};
  }
};
struct Row {
  int count = 0;
  std::array<int, 25> ids{};
  std::array<double, 25> dx{}, dy{};
};
struct StrainOperator {
  const Geometry &g;
  const int radius;
  std::vector<Row> rows = std::vector<Row>(N);
  StrainOperator(const Geometry &geom, int radius_ = 2) : g(geom), radius(radius_) {
    if (radius != 1 && radius != 2)
      throw std::invalid_argument("strain radius must be 1 or 2");
    build(g.support);
  }
  void build(const std::vector<unsigned char> &mask) {
    for (int i = 0; i < N; ++i) {
      Row r;
      if (!mask[i]) {
        rows[i] = r;
        continue;
      }
      double sx = 0, sy = 0, xx = 0, yy = 0, xy = 0;
      for (int b = -radius; b <= radius; ++b)
        for (int a = -radius; a <= radius; ++a) {
          int x = i % nx + a, y = i / nx + b;
          if (x < 0 || x >= nx || y < 0 || y >= ny || !mask[y * nx + x])
            continue;
          int j = r.count++;
          r.ids[j] = y * nx + x;
          r.dx[j] = 16 * a;
          r.dy[j] = 16 * b;
          sx += 16 * a;
          sy += 16 * b;
          xx += 256 * a * a;
          yy += 256 * b * b;
          xy += 256 * a * b;
        }
      if (r.count < (radius == 1 ? 6 : 9)) {
        r.count = 0;
        rows[i] = r;
        continue;
      }
      xx -= sx * sx / r.count;
      yy -= sy * sy / r.count;
      xy -= sx * sy / r.count;
      double det = xx * yy - xy * xy;
      if (det < 1e-6) {
        r.count = 0;
        rows[i] = r;
        continue;
      }
      for (int j = 0; j < r.count; ++j) {
        double x = r.dx[j] - sx / r.count, y = r.dy[j] - sy / r.count;
        r.dx[j] = (yy * x - xy * y) / det;
        r.dy[j] = (xx * y - xy * x) / det;
      }
      rows[i] = r;
    }
  }
  std::vector<std::array<double, 4>>
  apply(const std::vector<double> &pos) const {
    std::vector<std::array<double, 4>> out(N, {NAN, NAN, NAN, NAN});
    for (int i = 0; i < N; ++i) {
      auto &r = rows[i];
      if (!r.count)
        continue;
      double ex = 0, ey = 0, uyx = 0, uxy = 0;
      for (int j = 0; j < r.count; ++j) {
        int id = r.ids[j];
        double ux = pos[2 * id] - g.ref[2 * id],
               uy = pos[2 * id + 1] - g.ref[2 * id + 1];
        ex += r.dx[j] * ux;
        uxy += r.dy[j] * ux;
        uyx += r.dx[j] * uy;
        ey += r.dy[j] * uy;
      }
      double s = .5 * (uxy + uyx);
      out[i] = {ex, ey, s, vm(ex, ey, s)};
    }
    return out;
  }
};
inline bool read(FILE *f, std::vector<double> &p,
                 std::vector<unsigned char> &v) {
  return std::fread(p.data(), 8, p.size(), f) == p.size() &&
         std::fread(v.data(), 1, v.size(), f) == v.size();
}
inline std::vector<double> commands(const std::string &p) {
  FILE *f = file(p, "r");
  char line[256];
  std::vector<double> d;
  if (!std::fgets(line, 256, f))
    throw std::runtime_error("empty command");
  while (std::fgets(line, 256, f)) {
    int k;
    double t, v;
    if (std::sscanf(line, "%d,%lf,%lf", &k, &t, &v) != 3 || k != int(d.size()))
      throw std::runtime_error("command row");
    d.push_back(v);
  }
  std::fclose(f);
  return d;
}
} // namespace plate4k
