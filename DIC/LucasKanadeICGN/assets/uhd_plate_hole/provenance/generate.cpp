#include "common.hpp"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace plate4k;
static double sample(const std::vector<float> &v, double x, double y) {
  x = std::clamp(x, 0., double(W - 1));
  y = std::clamp(y, 0., double(H - 1));
  int a = int(x), b = int(y), c = std::min(a + 1, W - 1),
      d = std::min(b + 1, H - 1);
  double f = x - a, g = y - b;
  return (1 - g) * ((1 - f) * v[b * W + a] + f * v[b * W + c]) +
         g * ((1 - f) * v[d * W + a] + f * v[d * W + c]);
}
static double cubic(double a, double b, double c, double d, double t) {
  return b + .5 * t *
                 (c - a +
                  t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}
static double texture(const cv::Mat &m, double x, double y) {
  int ix = std::floor(x), iy = std::floor(y);
  double r[4];
  for (int j = -1; j <= 2; ++j) {
    double p[4];
    int yy = std::clamp(iy + j, 0, H - 1);
    const auto *row = m.ptr<unsigned char>(yy);
    for (int i = -1; i <= 2; ++i)
      p[i + 1] = row[std::clamp(ix + i, 0, W - 1)];
    r[j + 1] = cubic(p[0], p[1], p[2], p[3], x - ix);
  }
  return cubic(r[0], r[1], r[2], r[3], y - iy);
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  try {
    std::string work = argv[1], out = argv[2];
    std::filesystem::create_directories(work);
    std::filesystem::create_directories(out);
    Geometry g;
    Solution sol(g);
    cv::Mat base(H, W, CV_8UC1, cv::Scalar(220));
    cv::RNG rng(20261020);
    for (int i = 0; i < int(12000. * W * H / (400 * 1040)); ++i)
      cv::circle(base, {rng.uniform(0, W), rng.uniform(0, H)},
                 rng.uniform(1, 4), cv::Scalar(rng.uniform(25, 85)), -1);
    cv::GaussianBlur(base, base, {0, 0}, .7, .7, cv::BORDER_REFLECT_101);
    std::vector<float> a(W * H), b(W * H), c(W * H), d(W * H);
#pragma omp parallel for
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        auto u = sol.extended(x, y, false), v = sol.extended(x, y, true);
        int i = y * W + x;
        a[i] = u.x;
        b[i] = u.y;
        c[i] = v.x;
        d[i] = v.y;
      }
    FILE *raw = file(work + "/raw.y", "wb"), *cmd = file(work + "/command.csv"),
         *schedule = file(out + "/schedule.csv");
    std::fprintf(cmd, "frame,time_s,displacement_px\n");
    std::fprintf(schedule, "frame,primary_MPa,secondary_MPa\n");
    std::vector<unsigned char> pixels(W * H);
    double lastp = -1, lasth = -1, maxerr = 0;
    auto start = Clock::now();
    for (int k = 0; k < frames; ++k) {
      double p = airy::load_at(k), h = shape::load_at(k);
      if (p != lastp || h != lasth) {
#pragma omp parallel for
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            double X = x, Y = y;
            for (int it = 0; it < 6; ++it) {
              double X1 = x - p * sample(a, X, Y) - h * sample(c, X, Y),
                     Y1 = y - p * sample(b, X, Y) - h * sample(d, X, Y);
              X = X1;
              Y = Y1;
            }
            pixels[y * W + x] =
                std::hypot(X - cx, Y - cy) < radius
                    ? 12
                    : static_cast<unsigned char>(std::clamp(
                          std::lround(texture(base, X, Y)), 0L, 255L));
          }
        if (k == 95 || k == 225)
          for (int i = 0; i < N; ++i)
            if (g.support[i]) {
              double x = g.ref[2 * i], y = g.ref[2 * i + 1], X = x, Y = y;
              for (int it = 0; it < 6; ++it) {
                double X1 = x - p * sample(a, X, Y) - h * sample(c, X, Y),
                       Y1 = y - p * sample(b, X, Y) - h * sample(d, X, Y);
                X = X1;
                Y = Y1;
              }
              auto u = sol.unit(X, Y, false), v = sol.unit(X, Y, true);
              maxerr = std::max(maxerr, std::hypot(X + p * u.x + h * v.x - x,
                                                   Y + p * u.y + h * v.y - y));
            }
      }
      if (std::fwrite(pixels.data(), 1, pixels.size(), raw) != pixels.size())
        throw std::runtime_error("raw write");
      std::fprintf(cmd, "%d,%.9f,%.12f\n", k, k / 30., p * sol.command);
      std::fprintf(schedule, "%d,%.9f,%.9f\n", k, p, h);
      lastp = p;
      lasth = h;
      if (k % 30 == 0) {
        std::printf("frame=%d elapsed_s=%.1f\n", k, ms(start) / 1000);
        std::fflush(stdout);
      }
      if (k == 0 || k == 95 || k == 225 || k == 365 || k == 545)
        cv::imwrite(out + "/raw_" + std::to_string(k) + ".png",
                    cv::Mat(H, W, CV_8UC1, pixels.data()));
    }
    std::fclose(raw);
    std::fclose(cmd);
    std::fclose(schedule);
    std::filesystem::copy_file(
        work + "/command.csv", out + "/command.csv",
        std::filesystem::copy_options::overwrite_existing);
    std::printf("command_max_px=%.12f ratio=%.12f inverse_error_px=%.12g "
                "elapsed_s=%.3f\n",
                250 * sol.command, sol.ratio, maxerr, ms(start) / 1000);
    if (maxerr > .001)
      throw std::runtime_error("inverse mapping error exceeds protocol");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "generate: %s\n", e.what());
    return 1;
  }
}
