/** Scoring only: the archived infinite plane-strain plate, MPa/mm, 0.01 mm/px.
 * Port of python/tests/plate_hole_reference.py. No truth enters registration.
 */
const A = 1.12, NU = .3, MU = 210000 / (2 * (1 + NU)), KAPPA = 3 - 4 * NU;
const add = (a,b) => [a[0]+b[0],a[1]+b[1]];
const scale = (a,s) => [a[0]*s,a[1]*s];
const mul = (a,b) => [a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const conj = a => [a[0],-a[1]];
const inverse = a => scale(conj(a),1/(a[0]*a[0]+a[1]*a[1]));
function raw(x,y,horizontal=false) {
  let xx=(x-1920)*.01, yy=(y-1080)*.01;
  if(horizontal) [xx,yy]=[yy,xx];
  const z=[yy,-xx], iz=inverse(z), iz2=mul(iz,iz), iz3=mul(iz2,iz);
  const psi=scale(add(z,scale(iz,2*A*A)),.25);
  const derivative=scale(add([1,0],scale(iz2,-2*A*A)),.25);
  const chi=scale(add(add(z,scale(iz,A*A)),scale(iz3,-(A**4))),-.5);
  const u=scale(add(add(scale(psi,KAPPA),scale(mul(z,conj(derivative)),-1)),scale(conj(chi),-1)),1/(2*MU));
  return horizontal ? [u[0]/.01,-u[1]/.01] : [-u[1]/.01,u[0]/.01];
}
let calibration;
function calibrate() {
  if(calibration) return calibration;
  const mean=(ys,horizontal)=>{let sum=0,n=0;for(const y of ys)for(let x=56;x<3785;x+=16){sum+=raw(x,y,horizontal)[1];n++;}return sum/n;};
  const pin=mean([120,136],false),hpin=mean([120,136],true);
  const command=mean([2024,2040],false)-pin;
  return calibration={pin,hpin,ratio:(mean([2024,2040],true)-hpin)/command};
}
function stress(x,y,load) {
  const r=Math.hypot(x,y),c=y/r,s=-x/r,q=A*A/(r*r),c2=c*c-s*s,s2=2*c*s;
  const rr=load/2*((1-q)+(1-4*q+3*q*q)*c2);
  const tt=load/2*((1+q)-(1+3*q*q)*c2),rt=-load/2*(1+2*q-3*q*q)*s2;
  return [rr*s*s+tt*c*c+2*rt*c*s,rr*c*c+tt*s*s-2*rt*c*s,-((rr-tt)*c*s+rt*c2)];
}
export function plateTruth(x,y,primary,secondary=0) {
  if(Math.hypot(x-1920,y-1080)<112) return null;
  const {pin,hpin,ratio}=calibrate(), v=raw(x,y), h=raw(x,y,true);
  v[1]-=pin;h[1]-=hpin;
  const displacement=v.map((u,i)=>primary*u+secondary*(h[i]-ratio*u));
  const xx=(x-1920)*.01,yy=(y-1080)*.01;
  const a=stress(xx,yy,primary-ratio*secondary),b=stress(yy,xx,secondary);
  const sx=a[0]+b[1],sy=a[1]+b[0],t=a[2]+b[2];
  const exx=((1-NU)*sx-NU*sy)/(2*MU),eyy=((1-NU)*sy-NU*sx)/(2*MU),exy=t/(2*MU);
  return {displacement,strain:[exx,eyy,exy],vm:2/3*Math.sqrt(Math.max(0,exx*exx-exx*eyy+eyy*eyy+3*exy*exy))};
}
export function scorePlate(field,load,selected) {
  let count=0,population=0,squared=0,vmSquared=0;
  const size=field.subsetSize,step=field.subsetStride;
  for(let i=0;i<field.status.length;i++) {
    const x=i%field.grid[0]*step+(size-1)/2,y=Math.floor(i/field.grid[0])*step+(size-1)/2;
    if(!selected[i] || x<32 || x>=3808 || y<32 || y>=2128 || Math.hypot(x-1920,y-1080)<288)continue;
    population++;
    if(field.status[i]<.5)continue;
    const truth=plateTruth(x,y,load.primary,load.secondary),p=field.parameters.subarray(i*6,i*6+6);
    squared+=(p[0]-truth.displacement[0])**2+(p[3]-truth.displacement[1])**2;
    const vm=2/3*Math.sqrt(Math.max(0,p[1]*p[1]-p[1]*p[5]+p[5]*p[5]+3*(.5*(p[2]+p[4]))**2));
    vmSquared+=(vm-truth.vm)**2;count++;
  }
  return {population:'selected reference centres, r >= 288 px and 32 px image margin; ezz = 0',
    count,requested:population,coverage:population?count/population:null,
    displacementRmsePx:count?Math.sqrt(squared/count):null,vmRmse:count?Math.sqrt(vmSquared/count):null};
}
