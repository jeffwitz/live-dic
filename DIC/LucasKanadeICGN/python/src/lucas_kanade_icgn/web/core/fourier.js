/** CPU bootstrap for sparse image pairs; the GPU solver remains unchanged.
 * A <=1024² window is box-averaged by two, Hann tapered, then zero padded.
 * The cached reference spectrum and reusable work planes bound memory.
 */
class FFT2D {
  constructor(size) {
    this.size=size;this.reverse=new Uint32Array(size);
    this.cos=new Float64Array(size/2);this.sin=new Float64Array(size/2);
    const bits=Math.log2(size);
    for(let i=0;i<size;i++){let x=i,r=0;for(let b=0;b<bits;b++){r=r*2+(x&1);x>>=1;}this.reverse[i]=r;}
    for(let i=0;i<size/2;i++){this.cos[i]=Math.cos(2*Math.PI*i/size);this.sin[i]=Math.sin(2*Math.PI*i/size);}
  }
  // Iterative radix-2 butterflies; every line uses the cached twiddle table.
  line(real,imag,start,stride,inverse) {
    const n=this.size;
    for(let i=0;i<n;i++){const j=this.reverse[i];if(j>i){const a=start+i*stride,b=start+j*stride;
      [real[a],real[b]]=[real[b],real[a]];[imag[a],imag[b]]=[imag[b],imag[a]];}}
    for(let span=2;span<=n;span*=2){const half=span/2,step=n/span;
      for(let base=0;base<n;base+=span)for(let k=0;k<half;k++){
        const a=start+(base+k)*stride,b=a+half*stride,c=this.cos[k*step],s=(inverse?1:-1)*this.sin[k*step];
        const re=c*real[b]-s*imag[b],im=s*real[b]+c*imag[b];
        real[b]=real[a]-re;imag[b]=imag[a]-im;real[a]+=re;imag[a]+=im;
      }}
    if(inverse)for(let i=0;i<n;i++){const k=start+i*stride;real[k]/=n;imag[k]/=n;}
  }
  transform(real,imag,inverse=false) {
    const n=this.size;
    for(let y=0;y<n;y++)this.line(real,imag,y*n,1,inverse);
    this.transpose(real,imag);
    for(let y=0;y<n;y++)this.line(real,imag,y*n,1,inverse);
    this.transpose(real,imag);
  }
  transpose(real,imag) {
    // Cache-sized blocks avoid strided FFT butterflies over full image columns.
    const n=this.size,block=32;
    for(let y0=0;y0<n;y0+=block)for(let x0=y0;x0<n;x0+=block)
      for(let y=y0;y<Math.min(y0+block,n);y++)for(let x=Math.max(x0,y+1);x<Math.min(x0+block,n);x++){
        const a=y*n+x,b=x*n+y,re=real[a],im=imag[a];real[a]=real[b];imag[a]=imag[b];real[b]=re;imag[b]=im;
      }
  }
}

export class FourierInitializer {
  constructor(reference,width,height,mask=null) {
    const size=2**Math.min(10,Math.floor(Math.log2(Math.min(width,height))));
    if(size<32)throw new Error('Fourier initialization needs an image of at least 32 pixels per side');
    this.width=width;this.height=height;this.size=size;
    const positions=[[.5,.5],...[0,.5,1].flatMap(y=>[0,.5,1].map(x=>[x,y]))];let best=null;
    for(const [fx,fy] of positions){
      const x=Math.floor((width-size)*fx),y=Math.floor((height-size)*fy);let sum=0,squares=0,count=0;
      for(let j=0;j<size;j++)for(let i=0;i<size;i++){const k=(y+j)*width+x+i;if(mask&&!mask[k])continue;const v=reference[k];sum+=v;squares+=v*v;count++;}
      const coverage=count/(size*size),std=count?Math.sqrt(Math.max(0,squares/count-(sum/count)**2)):0;
      if(coverage>=.25&&std>=1){const candidate={score:coverage*std,x,y,mean:sum/count};if(!best||candidate.score>best.score)best=candidate;
        if(fx===.5&&fy===.5&&coverage>=.5)break;}
    }
    if(!best)throw new Error('Fourier initialization has no sufficiently textured ROI window');
    this._window={x:best.x,y:best.y,size,sampleStep:2};this.fft=new FFT2D(size);
    this.taper=Float64Array.from({length:size/2},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(size/2-1)));
    this.referenceSamples=new Float64Array(size*size/4);this.samples=new Float64Array(size*size/4);
    this.allowedSamples=new Uint8Array(size*size/4);
    for(let j=0;j<size/2;j++)for(let i=0;i<size/2;i++){let allowed=true;for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)if(mask&&!mask[(best.y+2*j+dy)*width+best.x+2*i+dx])allowed=false;this.allowedSamples[j*size/2+i]=+allowed;}
    this.referenceReal=new Float64Array(size*size);this.referenceImag=new Float64Array(size*size);
    this.real=new Float64Array(size*size);this.imag=new Float64Array(size*size);
    this.fill(reference,this.referenceReal,best.mean,mask,width,best.x,best.y,this.referenceSamples);
    this.fft.transform(this.referenceReal,this.referenceImag);
  }
  get window(){return {...this._window};}
  fill(pixels,output,mean,mask,width,x,y,samples) {
    const n=this.size,m=n/2;output.fill(0);
    for(let j=0;j<m;j++)for(let i=0;i<m;i++){
      let sum=0;
      for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){const k=(y+2*j+dy)*width+x+2*i+dx;if(!mask||mask[k])sum+=pixels[k]-mean;}
      samples[j*m+i]=sum*.25;output[j*n+i]=samples[j*m+i]*this.taper[j]*this.taper[i];
    }
  }
  // The target is not masked: the ROI lives in reference coordinates.
  estimatePatch(pixels) {
    const started=performance.now(),n=this.size;
    if(pixels.length!==n*n)throw new Error('Invalid Fourier target window');
    let sum=0,squares=0;for(const v of pixels){sum+=v;squares+=v*v;}const mean=sum/pixels.length;
    if(Math.sqrt(Math.max(0,squares/pixels.length-mean*mean))<1)throw new Error('Fourier initialization rejected a target window without texture');
    this.fill(pixels,this.real,mean,null,n,0,0,this.samples);this.imag.fill(0);this.fft.transform(this.real,this.imag);
    for(let i=0;i<this.real.length;i++){
      const re=this.real[i]*this.referenceReal[i]+this.imag[i]*this.referenceImag[i];
      const im=this.imag[i]*this.referenceReal[i]-this.real[i]*this.referenceImag[i],norm=Math.max(Math.hypot(re,im),1e-12);
      this.real[i]=re/norm;this.imag[i]=im/norm;
    }
    this.fft.transform(this.real,this.imag,true);
    let peak=-Infinity,index=0;for(let i=0;i<this.real.length;i++)if(this.real[i]>peak){peak=this.real[i];index=i;}
    const px=index%n,py=Math.floor(index/n);let side=0,sideSquares=0;const count=n*n-121;
    for(const v of this.real){side+=v;sideSquares+=v*v;}
    for(let dy=-5;dy<=5;dy++)for(let dx=-5;dx<=5;dx++){
      const v=this.real[((py+dy+n)%n)*n+(px+dx+n)%n];side-=v;sideSquares-=v*v;
    }
    const score=(peak-side/count)/Math.max(Math.sqrt(Math.max(0,sideSquares/count-(side/count)**2)),1e-12);
    const dx=2*(px<n/2?px:px-n),dy=2*(py<n/2?py:py-n),overlap=Math.max(0,n-Math.abs(dx))*Math.max(0,n-Math.abs(dy))/(n*n);
    // Reject unrelated images that happen to produce a spectral peak.
    // This check uses untapered overlap samples, not the phase-correlation peak.
    const m=n/2,sx=dx/2,sy=dy/2;let countSamples=0,sa=0,sb=0,saa=0,sbb=0,sab=0;
    for(let y=Math.max(0,-sy);y<Math.min(m,m-sy);y++)for(let x=Math.max(0,-sx);x<Math.min(m,m-sx);x++){
      const k=y*m+x;if(!this.allowedSamples[k])continue;const a=this.referenceSamples[k],b=this.samples[(y+sy)*m+x+sx];
      countSamples++;sa+=a;sb+=b;saa+=a*a;sbb+=b*b;sab+=a*b;
    }
    const norm=Math.sqrt(Math.max(0,saa-sa*sa/countSamples)*Math.max(0,sbb-sb*sb/countSamples));
    const correlation=norm>0?(sab-sa*sb/countSamples)/norm:0;
    if(score<8||overlap<.25||correlation<.2)throw new Error(`Fourier initialization is ambiguous (peak score ${score.toFixed(2)}, overlap ${overlap.toFixed(3)}); no local field was calculated`);
    return {dx,dy,peakScore:score,overlap,correlation,window:this.window,method:'linear-phase-correlation',wallMs:performance.now()-started};
  }
  estimate(pixels,width=this.width,height=this.height) {
    if(width!==this.width||height!==this.height||pixels.length!==width*height)throw new Error('Fourier target dimensions must match the prepared reference');
    const {x,y,size}=this.window,patch=new Uint8Array(size*size);
    for(let j=0;j<size;j++)patch.set(pixels.subarray((y+j)*width+x,(y+j)*width+x+size),j*size);
    return this.estimatePatch(patch);
  }
  memoryBytes(){return [this.referenceReal,this.referenceImag,this.real,this.imag,this.referenceSamples,this.samples,this.allowedSamples,this.taper,this.fft.reverse,this.fft.cos,this.fft.sin].reduce((sum,array)=>sum+array.byteLength,0);}
}
