// Kirsch displacement from Kolosov-Muskhelishvili, plane strain.
// TU Wien Papaleo appendix A: A.19, A.32, A.37. MPa and mm internally.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>

namespace airy {
constexpr int width=400,height=1040,frames=601;
constexpr double cx=192,cy=544,pixel_mm=.02,radius_px=56;
constexpr double young=210000,poisson=.3,mu=young/(2*(1+poisson));
constexpr double kappa=3-4*poisson,a=radius_px*pixel_mm;
struct Vec {double x,y;};
struct Strain {double xx,yy,xy,vm;};
struct Stress {double xx,yy,xy,zz;};
using C=std::complex<double>;
inline C psi(C z,double load){return load/4.*(z+2*a*a/z);}
inline C dpsi(C z,double load){return load/4.*(1.-2*a*a/(z*z));}
inline C chi(C z,double load){return -load/4.*(z*z+2*a*a*std::log(z/a)+a*a*a*a/(z*z));}
inline C dchi(C z,double load){return -load/2.*(z+a*a/z-a*a*a*a/(z*z*z));}
inline double potential(double x,double y,double load) {
    const C z(y,-x);return std::real(std::conj(z)*psi(z,load)+chi(z,load));
}
inline Vec displacement(double x,double y,double load) {
    const C z(y,-x);
    const C u=(kappa*psi(z,load)-z*std::conj(dpsi(z,load))-std::conj(dchi(z,load)))/(2*mu);
    return {-std::imag(u),std::real(u)};
}
// Independent polar expression, including homogeneous far-field motion.
inline Vec polar_displacement(double x,double y,double load) {
    const double r=std::hypot(x,y),c=y/r,s=-x/r,q=a*a/r;
    const double ur=load/(8*mu)*((kappa-1)*r+2*q+
        (2*r+2*(kappa+1)*q-2*q*a*a/(r*r))*(c*c-s*s));
    const double ut=-load/(8*mu)*(2*r+2*(kappa-1)*q+2*q*a*a/(r*r))*(2*c*s);
    return {-(ur*s+ut*c),ur*c-ut*s};
}
inline Stress stress(double x,double y,double load) {
    const double r=std::hypot(x,y),c=y/r,s=-x/r,q=a*a/(r*r),c2=c*c-s*s,s2=2*c*s;
    const double rr=load/2*((1-q)+(1-4*q+3*q*q)*c2);
    const double tt=load/2*((1+q)-(1+3*q*q)*c2);
    const double rt=-load/2*(1+2*q-3*q*q)*s2;
    const double xx=rr*s*s+tt*c*c+2*rt*c*s,yy=rr*c*c+tt*s*s-2*rt*c*s;
    return {xx,yy,-((rr-tt)*c*s+rt*c2),poisson*(xx+yy)};
}
inline Strain strain(double x,double y,double load) {
    const auto s=stress(x,y,load);
    const double xx=((1-poisson)*s.xx-poisson*s.yy)/(2*mu);
    const double yy=((1-poisson)*s.yy-poisson*s.xx)/(2*mu),xy=s.xy/(2*mu);
    return {xx,yy,xy,2./3*std::sqrt(xx*xx-xx*yy+yy*yy+3*xy*xy)};
}
inline double load_at(int k) {
    if(k<=95)return 250.*k/95;
    if(k<=275)return 250;
    if(k<=365)return 250.*(365-k)/90;
    if(k<=455)return 0;
    if(k<=545)return 250.*(k-455)/90;
    return 250;
}
inline const char* phase(int k) {
    return k<=95?"load":k<=275?"hold_high":k<=365?"unload":
        k<=455?"hold_zero":k<=545?"reload":"hold_final";
}
struct Model {
    double pin_y=0,command_per_mpa=0;
    Model() {
        double low=0,high=0;
        for(int j=0;j<19;++j) {
            const double x=(56.+16*j-cx)*pixel_mm;
            low+=displacement(x,(80-cy)*pixel_mm,1).y;
            high+=displacement(x,(960-cy)*pixel_mm,1).y;
        }
        pin_y=low/19;command_per_mpa=(high-low)/(19*pixel_mm);
    }
    Vec pixels(double X,double Y,double load)const {
        const auto u=displacement((X-cx)*pixel_mm,(Y-cy)*pixel_mm,load);
        return {u.x/pixel_mm,(u.y-load*pin_y)/pixel_mm};
    }
    Vec extended_pixels(double X,double Y,double load)const {
        const double x=X-cx,y=Y-cy,r=std::hypot(x,y);
        if(r>=radius_px)return pixels(X,Y,load);
        if(r<1e-12)return {0,-load*pin_y/pixel_mm};
        const auto edge=displacement(x*pixel_mm*radius_px/r,y*pixel_mm*radius_px/r,load);
        return {r/radius_px*edge.x/pixel_mm,(r/radius_px*edge.y-load*pin_y)/pixel_mm};
    }
};
} // namespace airy
