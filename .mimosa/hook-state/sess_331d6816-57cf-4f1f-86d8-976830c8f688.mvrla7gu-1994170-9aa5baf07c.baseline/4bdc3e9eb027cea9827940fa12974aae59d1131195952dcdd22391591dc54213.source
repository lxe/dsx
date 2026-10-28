#pragma once
struct RejectDraw {
    float temperature; int active, qrow, draft;
    double accept_uniform, recover_uniform;
};
__device__ __forceinline__ double reject_weight(float l,float mx,float threshold,double inv_t) {
    return l >= threshold ? exp((double)__fsub_rn(l,mx)*inv_t) : 0.;
}
// Exact speculative rejection for a sampled proposal q: accept with min(1,p(d)/q(d)),
// else recover from positive(p-q). qrow<0 samples p directly (lookup/bonus rows).
extern "C" __global__ void __launch_bounds__(1024) reject_logits_b(
    const float* logits, const float* proposals, int n, const RejectDraw* draws,
    const int* greedy, int* next) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const RejectDraw draw=draws[row];
    if(tid==0)next[row]=greedy[row];
    if(!draw.active)return;
    const float* p=logits+(size_t)row*n;
    const float pmx=p[greedy[row]];
    if(!isfinite(pmx))return;
    const double inv_t=1./(double)draw.temperature;
    const float pth=__fsub_rn(pmx,__fmul_rn(30.f,draw.temperature));
    bool probabilistic=draw.qrow>=0;
    const float* q=probabilistic?proposals+(size_t)draw.qrow*n:nullptr;
    __shared__ float q_wave[16],qmx_s;
    __shared__ double pw[16],qw[16],ptotal,qtotal;
    __shared__ int accepted,residual;
    float qmx=-INFINITY;
    if(probabilistic){
        for(int i=tid;i<n;i+=1024)qmx=fmaxf(qmx,q[i]);
        for(int d=32;d;d>>=1)qmx=fmaxf(qmx,__shfl_xor(qmx,d,64));
        if(lane==0)q_wave[wave]=qmx;
        __syncthreads();
        if(wave==0){
            float m=lane<16?q_wave[lane]:-INFINITY;
            for(int d=32;d;d>>=1)m=fmaxf(m,__shfl_xor(m,d,64));
            if(lane==0)qmx_s=m;
        }
        __syncthreads();
        qmx=qmx_s;
        if(!isfinite(qmx))probabilistic=false;
    }
    const float qth=__fsub_rn(qmx,__fmul_rn(30.f,draw.temperature));
    double ps=0.,qs=0.;
    for(int i=tid;i<n;i+=1024){
        ps+=reject_weight(p[i],pmx,pth,inv_t);
        if(probabilistic)qs+=reject_weight(q[i],qmx,qth,inv_t);
    }
    for(int d=32;d;d>>=1){ps+=__shfl_down(ps,d,64);qs+=__shfl_down(qs,d,64);}
    if(lane==0){pw[wave]=ps;qw[wave]=qs;}
    __syncthreads();
    if(wave==0){
        double a=lane<16?pw[lane]:0.,b=lane<16?qw[lane]:0.;
        for(int d=32;d;d>>=1){a+=__shfl_down(a,d,64);b+=__shfl_down(b,d,64);}
        if(lane==0){ptotal=a;qtotal=b;}
    }
    __syncthreads();
    if(tid==0){
        accepted=0;residual=0;
        if(probabilistic&&draw.draft>=0&&draw.draft<n){
            const double pd=reject_weight(p[draw.draft],pmx,pth,inv_t)/ptotal;
            const double qd=reject_weight(q[draw.draft],qmx,qth,inv_t)/qtotal;
            if(qd>0.){
                residual=1;
                if(pd>=qd||draw.accept_uniform<pd/qd){accepted=1;next[row]=draw.draft;}
            }
        }
    }
    __syncthreads();
    if(accepted)return;
    double mass=0.;
    for(int i=tid;i<n;i+=1024){
        double w=reject_weight(p[i],pmx,pth,inv_t)/ptotal;
        if(residual)w=fmax(w-reject_weight(q[i],qmx,qth,inv_t)/qtotal,0.);
        mass+=w;
    }
    double prefix=mass;
    for(int d=1;d<64;d*=2){double a=__shfl_up(prefix,d,64);if(lane>=d)prefix+=a;}
    __shared__ double totals[16],bases[16],total;
    __shared__ int winner,lasts[16],last_positive;
    const unsigned long long positive=__ballot(mass>0.);
    if(lane==63){totals[wave]=prefix;lasts[wave]=positive?wave*64+63-__clzll(positive):-1;}
    if(tid==0)winner=1024;
    __syncthreads();
    if(tid==0){last_positive=-1;for(int w=0;w<16;w++)last_positive=max(last_positive,lasts[w]);}
    if(wave==0){
        double v=lane<16?totals[lane]:0.;
        for(int d=1;d<16;d*=2){double a=__shfl_up(v,d,64);if(lane>=d)v+=a;}
        if(lane<16)bases[lane]=v-totals[lane];
        if(lane==15)total=v;
    }
    __syncthreads();
    double r=draw.recover_uniform*total-bases[wave]-(prefix-mass);
    if(mass>0.&&r>=0.&&r<mass)atomicMin(&winner,tid);
    __syncthreads();
    const bool endpoint=winner==1024&&tid==last_positive;
    if(tid!=winner&&!endpoint)return;
    if(endpoint)r=nextafter(mass,0.);
    int last=greedy[row];
    for(int i=tid;i<n;i+=1024){
        double w=reject_weight(p[i],pmx,pth,inv_t)/ptotal;
        if(residual)w=fmax(w-reject_weight(q[i],qmx,qth,inv_t)/qtotal,0.);
        if(w<=0.)continue;
        last=i;r-=w;if(r<0.){next[row]=i;return;}
    }
    next[row]=last;
}
