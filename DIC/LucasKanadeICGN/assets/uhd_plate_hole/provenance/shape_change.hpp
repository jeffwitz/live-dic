// Elastic nonproportional control: two independent Kirsch load modes.
#pragma once
#include "kirsch.hpp"
namespace shape {
inline double load_at(int k) {
    if(k<=165)return 0;
    if(k<=225)return 80.*(k-165)/60;
    if(k<=275)return 80;
    if(k<=365)return 80.*(365-k)/90;
    return 0;
}
inline double holdout_load_at(int k) {
    if(k<=177)return 0;
    if(k<=231)return 65.*(k-177)/54;
    if(k<=289)return 65;
    if(k<=347)return 65.*(347-k)/58;
    return 0;
}
struct Mode {
    airy::Model vertical;
    double ratio=0,pin=0;
    Mode() {
        double low=0,high=0;
        for(int j=0;j<19;++j) {
            const double x=(56.+16*j-airy::cx)*airy::pixel_mm;
            low+=airy::displacement((80-airy::cy)*airy::pixel_mm,x,1).x;
            high+=airy::displacement((960-airy::cy)*airy::pixel_mm,x,1).x;
        }
        ratio=(high-low)/(19*airy::pixel_mm*vertical.command_per_mpa);
        // vertical.pixels already removes its own lower-section translation.
        pin=low/(19*airy::pixel_mm);
    }
    airy::Vec pixels(double X,double Y,double load)const {
        const auto h=airy::displacement((Y-airy::cy)*airy::pixel_mm,(X-airy::cx)*airy::pixel_mm,load);
        const auto v=vertical.pixels(X,Y,load);
        return {h.y/airy::pixel_mm-ratio*v.x,h.x/airy::pixel_mm-ratio*v.y-load*pin};
    }
    airy::Vec extended_pixels(double X,double Y,double load)const {
        const double x=X-airy::cx,y=Y-airy::cy,r=std::hypot(x,y);
        if(r>=airy::radius_px)return pixels(X,Y,load);
        if(r<1e-12)return {0,-load*pin};
        const auto edge=pixels(airy::cx+x*airy::radius_px/r,airy::cy+y*airy::radius_px/r,load);
        return {r/airy::radius_px*edge.x,r/airy::radius_px*(edge.y+load*pin)-load*pin};
    }
    airy::Stress stress(double x,double y,double load)const {
        const auto h=airy::stress(y,x,load),v=airy::stress(x,y,load);
        return {h.yy-ratio*v.xx,h.xx-ratio*v.yy,h.xy-ratio*v.xy,h.zz-ratio*v.zz};
    }
    airy::Strain strain(double x,double y,double primary,double secondary)const {
        const auto v=airy::stress(x,y,primary),h=stress(x,y,secondary);
        const double xx=((1-airy::poisson)*(v.xx+h.xx)-airy::poisson*(v.yy+h.yy))/(2*airy::mu);
        const double yy=((1-airy::poisson)*(v.yy+h.yy)-airy::poisson*(v.xx+h.xx))/(2*airy::mu);
        const double xy=(v.xy+h.xy)/(2*airy::mu);
        return {xx,yy,xy,2./3*std::sqrt(xx*xx-xx*yy+yy*yy+3*xy*xy)};
    }
};
} // namespace shape
