#include "/home/jeff/CNRS/Code/DIC_CODEC/DIC_FROM_H264/experiments/decoded_control_filter/airy_plate_4k/common.hpp"
#include <array>
int main() {
  plate4k::Geometry g;
  plate4k::Solution s(g);
  std::fprintf(stderr, "pin=%.17g hpin=%.17g ratio=%.17g command=%.17g\n", s.pin,s.hpin,s.ratio,s.command);
  std::puts("x,y,primary,secondary,ux,uy,exx,eyy,exy,vm");
  const std::array<std::array<double,2>,8> points{{{2032,1080},{1808,1080},{1920,968},{1920,1192},{2040,1080},{2007.5,1167.5},{56,120},{3784,2040}}};
  for (auto p : points) for (double h : {0.,80.}) {
    auto v=s.unit(p[0],p[1],false), q=s.unit(p[0],p[1],true);
    auto e=s.strain(p[0],p[1],250,h);
    std::printf("%.17g,%.17g,250,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",p[0],p[1],h,250*v.x+h*q.x,250*v.y+h*q.y,e.xx,e.yy,e.xy,e.vm);
  }
}
