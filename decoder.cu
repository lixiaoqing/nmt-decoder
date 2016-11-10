#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <queue>
#include <string>
#include <sstream>
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

// get embeddings for words, grid shape (2-dim): B x E/n, block shape (1-dim): n
__global__ void lookup_kernel(float *d_lookup, const float *d_Wemb, int *d_word_indices, int E, int B, int V)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j >= E)
        return;
    if (d_word_indices[i] < 0)
        d_lookup[IDX2C(i,j,B)] = 0;
    else
        d_lookup[IDX2C(i,j,B)] = d_Wemb[IDX2C(d_word_indices[i],j,V)];
}

__global__ void tanh(float *v1, float *v2, int len)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= len ) 
       return;
   v1[idx] = tanhf(v1[idx]+v2[idx]);
}

// grid shape (3-dim): T x B x 2H/n, block shape (1-dim): n
__global__ void tanh(float *t3, float *m1, float *m2, float *bias, int T, int B, int HH)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x + blockIdx.z*blockDim.x;
    if (k >= HH)
        return;
    t3[IDX2C(i+j*gridDim.x,k,T*B)] = tanhf(m1[IDX2C(i,k,T)]+m2[IDX2C(j,k,B)]+bias[k]);
}

// grid shape (2-dim): B x E/n, block shape (1-dim): n
__global__ void tanh(float *m, float *m1, float *m2, float *m3, float *b1, float *b2, float *b3, int B, int E)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j >= E)
        return;
    m[IDX2C(i,j,B)] = tanhf(m1[IDX2C(i,j,B)]+m2[IDX2C(i,j,B)]+m3[IDX2C(i,j,B)]+b1[j]+b2[j]+b3[j]);
}

__global__ void exp_c(float *v1, float *b, int len)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= len ) 
       return;
   v1[idx] = expf(v1[idx]+b[0]);
}

// grid shape (2-dim): B x V/n, block shape (1-dim): n
__global__ void exp_v(float *m, float *b, int B, int V)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j >= V ) 
        return;
    m[IDX2C(i,j,B)] = expf(m[IDX2C(i,j,B)]+b[j]);
}

// ctx_ = (ctx[:,None,:] * alpha[:,:,None]).sum(0)
// grid shape (3-dim): T x B x 2H/n, block shape (1-dim): n
__global__ void pointwise_prod(float *t3, float *m1, float *m2, int T, int B, int HH)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x + blockIdx.z*blockDim.x;
    if (k >= HH)
        return;
    t3[IDX2C(i+j*gridDim.x,k,T*B)] = m1[IDX2C(i,k,T)] * m2[IDX2C(i,j,T)];
}

__global__ void divide(float *v1, float *v2, int T)
{
   int i = blockIdx.x;
   int j = threadIdx.x;
   v1[IDX2C(i,j,T)] = v1[IDX2C(i,j,T)]/v2[j];
}

// grid shape (2-dim): B x V/n, block shape (1-dim): n
__global__ void divide_log(float *m, float *v, int B, int V)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j >= V ) 
        return;
    m[IDX2C(i,j,B)] = logf(m[IDX2C(i,j,B)]/v[i]);
}

__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

__global__ void elementwise_op(float *h_t,float *ax_t, float *ah_t, float *bias, float *h_tm1, int H)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= H ) 
       return;
   float r, u, hbar;
   r = sigmoidf(ax_t[idx] + ah_t[idx] + bias[idx]);
   u = sigmoidf(ax_t[H+idx] + ah_t[H+idx] + bias[H+idx]);
   hbar = tanhf(ax_t[2*H+idx] + ah_t[2*H+idx]*r + bias[2*H+idx]);
   h_t[idx] = u * h_tm1[idx] + (1-u) * hbar;
}

// grid shape: B x H/n, block shape: n
__global__ void elementwise_op_dec(float *s_t, float *ay_t, float *as_t, float *ac_t, float *bias, float *s_tm1, int B, int H)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j >= H)
        return;

    float r, u, sbar;
    r = sigmoidf(ay_t[IDX2C(i,j,B)] + as_t[IDX2C(i,j,B)] + ac_t[IDX2C(i,j,B)] + bias[j]);
    u = sigmoidf(ay_t[IDX2C(i,j+H,B)] + as_t[IDX2C(i,j+H,B)] + ac_t[IDX2C(i,j+H,B)] + bias[j+H]);
    sbar = tanhf(ay_t[IDX2C(i,j+2*H,B)] + as_t[IDX2C(i,j+2*H,B)]*r + ac_t[IDX2C(i,j+2*H,B)] + bias[j+2*H]);
    s_t[IDX2C(i,j,B)] = u * s_tm1[IDX2C(i,j,B)] + (1-u) * sbar;
}

void show_matrix(float *d_m, int r, int c)
{
    vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            cout<<m[IDX2C(i,j,r)]<<' ';
        }
        cout<<endl;
    }
}

int main()
{
    cudaSetDevice(2);
    ifstream flist("param-list");
    if (!flist.is_open())
    {
        cerr<<"cannot open param-list!\n";
        return 0;
    }
    string s;
    vector<string> param_names;
    vector<int> offset;
    offset.push_back(0);
    while(getline(flist,s))
    {
        stringstream ss;
        ss << s;
        string pn;
        int x,y;
        ss>>pn>>x>>y;
        param_names.push_back(pn);
        offset.push_back(offset.back()+x*y);
    }

    ifstream fmodel("model.bin",ios::binary);
    if (!fmodel.is_open())
    {
        cerr<<"cannot open model.bin!\n";
        return 0;
    }
    int total_size = offset.back();
    vector<float> all_params;
    all_params.resize(total_size);
    fmodel.read((char*)&all_params[0],sizeof(float)*total_size);

    float *d_all_params;
    cudaMalloc((void**)&d_all_params, total_size*sizeof(float));
    cudaMemcpy(d_all_params, &all_params[0], total_size*sizeof(float), cudaMemcpyHostToDevice);
    
    map<string,float*> d_params;
    for (int i=0; i<param_names.size(); i++)
    {
        d_params[param_names[i]] = d_all_params + offset[i];
    }

    ifstream fsrc_w2i("1m.ch.w2i");
    if (!fsrc_w2i.is_open())
    {
        cerr<<"cannot open 1m.ch.w2i!\n";
        return 0;
    }
    map<string,int> src_w2i;
    while(getline(fsrc_w2i,s))
    {
        stringstream ss;
        ss << s;
        string w;
        int idx;
        ss >> w >> idx;
        src_w2i[w] = idx;
    }
    ifstream ftgt_i2w("1m.en.i2w");
    if (!ftgt_i2w.is_open())
    {
        cerr<<"cannot open 1m.en.i2w!\n";
        return 0;
    }
    map<int,string> tgt_i2w;
    while(getline(ftgt_i2w,s))
    {
        stringstream ss;
        ss << s;
        int idx;
        string w;
        ss >> idx >> w;
        tgt_i2w[idx] = w;
    }

    ifstream ftest("03.seg");
    if (!ftest.is_open())
    {
        cerr<<"cannot open 03.seg!\n";
        return 0;
    }

    int E = 500;
    int H = 1024;
    int B = 1;
    int V = 30000;
    int K = 10;
    int Tmax = 200;

    vector<float> ones(V,1);
    float *d_ones;
    cudaMalloc((void**)&d_ones, V*sizeof(float));
    cudaMemcpy(d_ones, &ones[0], V*sizeof(float), cudaMemcpyHostToDevice);

    int *d_father_idx;
    cudaMalloc((void**)&d_father_idx, K*sizeof(int));

    // allocate memory for x_t, h_{0:T}, affine_x = x_t x [U_z,U_r,U_h], affine_h = h_{t-1} x [W_z,W_r,W_h] on device
    // forward: d_x_t, d_ax_t, d_ah_t; backward: d_xr_t, d_axr_t, d_ahr_t; both: d_h_all = [h_0,hr_{T+1},h1,hr_T,...,h_{T+1},hr_0]
    float *d_x_t, *d_ax_t, *d_ah_t;
    float *d_xr_t, *d_axr_t, *d_ahr_t;
    float *d_h_all;

    cudaMalloc((void**)&d_x_t, E*sizeof(float));
    cudaMalloc((void**)&d_ax_t, 3*H*sizeof(float));
    cudaMalloc((void**)&d_ah_t, 3*H*sizeof(float));

    cudaMalloc((void**)&d_xr_t, E*sizeof(float));
    cudaMalloc((void**)&d_axr_t, 3*H*sizeof(float));
    cudaMalloc((void**)&d_ahr_t, 3*H*sizeof(float));

    cudaMalloc((void**)&d_h_all, 2*H*(Tmax+2)*sizeof(float));

    int *d_word_indices;
    cudaMalloc((void**)&d_word_indices, sizeof(int));
    int *d_word_indices_r;
    cudaMalloc((void**)&d_word_indices_r, sizeof(int));

    float *d_ctx_mean, *d_init_state;
    cudaMalloc((void**)&d_ctx_mean, 2*H*sizeof(float));
    cudaMalloc((void**)&d_init_state, H*sizeof(float));

    float *d_h_all_trans;
    cudaMalloc((void**)&d_h_all_trans, Tmax*2*H*sizeof(float));

    float *d_pctx_;
    cudaMalloc((void**)&d_pctx_, Tmax*2*H*sizeof(float));

    int *d_tgt_word_indices;
    cudaMalloc((void**)&d_tgt_word_indices, K*sizeof(int));

    float *d_s_tm1;
    cudaMalloc((void**)&d_s_tm1, K*H*sizeof(float));

    float *d_y_tm1, *d_pstate_, *d_pctx__, *d_att, *d_att_sum, *d_ctx, *d_ctx_sum;

    cudaMalloc((void**)&d_y_tm1, K*E*sizeof(float));
    cudaMalloc((void**)&d_pstate_, K*2*H*sizeof(float));
    cudaMalloc((void**)&d_pctx__, Tmax*K*2*H*sizeof(float));
    cudaMalloc((void**)&d_att, Tmax*K*sizeof(float));
    cudaMalloc((void**)&d_att_sum, K*sizeof(float));
    cudaMalloc((void**)&d_ctx, Tmax*K*2*H*sizeof(float));
    cudaMalloc((void**)&d_ctx_sum, K*2*H*sizeof(float));

    // allocate memory for affine_y = y_{t-1} x [U_z,U_r,U_h]
    // affine_s = s_{t-1} x [W_z,W_r,W_h] on device
    // affine_c = ctx_ x [Wc_z,Wc_r,Wc_h] on device
    float *d_ay_t, *d_as_t, *d_ac_t;
    float *d_s_t;
    float *d_logit_lstm, *d_logit_prev, *d_logit_ctx;
    float *d_logit, *d_logit_softmax, *d_logit_softmax_sum;

    cudaMalloc((void**)&d_ay_t, K*3*H*sizeof(float));
    cudaMalloc((void**)&d_as_t, K*3*H*sizeof(float));
    cudaMalloc((void**)&d_ac_t, K*3*H*sizeof(float));
    cudaMalloc((void**)&d_s_t, K*H*sizeof(float));
    cudaMalloc((void**)&d_logit_lstm, K*E*sizeof(float));
    cudaMalloc((void**)&d_logit_prev, K*E*sizeof(float));
    cudaMalloc((void**)&d_logit_ctx, K*E*sizeof(float));
    cudaMalloc((void**)&d_logit, K*E*sizeof(float));
    cudaMalloc((void**)&d_logit_softmax, K*V*sizeof(float));
    cudaMalloc((void**)&d_logit_softmax_sum, K*sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha=1.0;  
    float beta=0.0;  

    while(getline(ftest,s))
    {
        stringstream ss;
        ss << s;
        string w;
        vector<int> word_indices;
        while (ss>>w)
        {
            if (src_w2i.find(w) != src_w2i.end() && src_w2i[w] < 30000)
                word_indices.push_back(src_w2i[w]);
            else
                word_indices.push_back(1);
        }
        word_indices.push_back(0);
        int T = word_indices.size();

        // init h_0, hr_0 with all zeros
        cudaMemset(d_h_all,0,2*H*(T+2)*sizeof(float));

        /**********************************************/
        /************** encoder part ******************/
        /**********************************************/

        B = 1;
        for (int i=0;i<T;i++)
        {
            // fill x_t; B x E => [B x E/n] x n
            dim3 block_shape(128,1,1);
            dim3 grid_shape(B,(E + block_shape.x - 1)/block_shape.x,1);
            cudaMemcpy(d_word_indices, &word_indices[i], sizeof(int), cudaMemcpyHostToDevice);  // TODO could be done outside
            lookup_kernel<<<grid_shape,block_shape>>>(d_x_t,d_params["Wemb"],d_word_indices,E,B,V);

            // backward
            cudaMemcpy(d_word_indices_r, &word_indices[T-1-i], sizeof(int), cudaMemcpyHostToDevice);
            lookup_kernel<<<grid_shape,block_shape>>>(d_xr_t,d_params["Wemb"],d_word_indices_r,E,B,V);
            // get ax_t
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,E,&alpha,d_x_t,B,d_params["encoder_W"],E,&beta,d_ax_t,B);
            // backward
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,E,&alpha,d_xr_t,B,d_params["encoder_r_W"],E,&beta,d_axr_t,B);
            //get h_{t-1}
            float *d_h_tm1 = d_h_all + 2*i*H;
            //backward
            float *d_hr_Tp1mt = d_h_all + (2*T+3-2*i)*H;
            // get ah_t
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,H,&alpha,d_h_tm1,B,d_params["encoder_U"],H,&beta,d_ah_t,B);
            // backward
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,H,&alpha,d_hr_Tp1mt,B,d_params["encoder_r_U"],H,&beta,d_ahr_t,B);
            // point wise operation
            dim3 block_shape1(256,1,1);
            dim3 grid_shape1((H + block_shape1.x - 1)/block_shape1.x,1,1);
            elementwise_op<<<grid_shape1,block_shape1>>>(d_h_all+2*(i+1)*H,d_ax_t,d_ah_t,d_params["encoder_b"],d_h_tm1,H);
            //backward
            elementwise_op<<<grid_shape1,block_shape1>>>(d_h_all+(2*T+1-2*i)*H,d_axr_t,d_ahr_t,d_params["encoder_r_b"],d_hr_Tp1mt,H);
        }

        //cublasSgemv(handle,CUBLAS_OP_N,2*H,T,&alpha,d_h_all+2*H,2*H,d_ones,1,&beta,d_ctx_mean,1);
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,2*H,1,T,&alpha,d_h_all+2*H,2*H,d_ones,T,&beta,d_ctx_mean,2*H);  // skip h0
        float alpha1 = 1.0/T;
        cublasSscal(handle, 2*H, &alpha1, d_ctx_mean, 1); 

        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,H,2*H,&alpha,d_ctx_mean,1,d_params["ff_state_W"],2*H,&beta,d_init_state,1);

        dim3 block_shape(256,1,1);
        dim3 grid_shape((H + block_shape.x - 1)/block_shape.x,1,1);
        tanh<<<grid_shape,block_shape>>>(d_init_state,d_params["ff_state_b"],H);

        /**********************************************/
        /************** decoder part ******************/
        /**********************************************/
        vector<vector<int> > final_samples;
        vector<float> final_scores;
        
        int dead_k = 0;

        vector<vector<int> > hyp_samples(1,vector<int>());
        vector<float> hyp_scores(1,0.0);

        // transpose ctx (2H x T) to T x 2H
        cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, T, 2*H, &alpha, d_h_all+2*H, 2*H, &beta, d_h_all+2*H, 2*H, d_h_all_trans, T );
        // prod ctx with decoder_Wc_att to get pctx_ (T x 2H)
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,T,2*H,2*H,&alpha,d_h_all_trans,T,d_params["decoder_Wc_att"],2*H,&beta,d_pctx_,T);

        vector<int> tgt_word_indices;
        tgt_word_indices.push_back(-1);

        // prev state on target side
        cudaMemcpy(d_s_tm1, d_init_state, H*sizeof(float), cudaMemcpyDeviceToDevice);

        vector<float> logit_softmax,s_t;
        logit_softmax.resize(K*V);
        s_t.resize(K*H);

        for (int j=0; j < Tmax; j++)
        {
            // fill y_{t-1}; B x E => [B x E/n] x n
            dim3 block_shape(128,1,1);
            dim3 grid_shape(B,(E + block_shape.x - 1)/block_shape.x,1);
            cudaMemcpy(d_tgt_word_indices, &tgt_word_indices[0], B*sizeof(int), cudaMemcpyHostToDevice);
            lookup_kernel<<<grid_shape,block_shape>>>(d_y_tm1,d_params["Wemb_dec"],d_tgt_word_indices,E,B,V);

            // product prev_state with decoder_Wd_att to get pstate_
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,2*H,H,&alpha,d_s_tm1,B,d_params["decoder_Wd_att"],H,&beta,d_pstate_,B);

            // product prev_emb with decoder_Wi_att and add the result to pstate_
            float beta1 = 1;
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,2*H,E,&alpha,d_y_tm1,B,d_params["decoder_Wi_att"],E,&beta1,d_pstate_,B);

            // pctx__ = tanh(pctx_[:,None,:] + pstate_[None,:,:] + b_att[None,None,:])
            dim3 block_shape1(128,1,1);
            dim3 grid_shape1(T,B,(2*H + block_shape1.x - 1)/block_shape1.x);
            tanh<<<grid_shape1,block_shape1>>>(d_pctx__,d_pctx_,d_pstate_,d_params["decoder_b_att"],T,B,2*H);

            // product pctx__ with decoder_U_att (2H x 1) to get alpha (T x B)
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,T*B,1,2*H,&alpha,d_pctx__,T*B,d_params["decoder_U_att"],2*H,&beta,d_att,T*B);
            
            // normalize alpha: alpha = exp(alpha)/exp(alpha).sum(0)
            dim3 block_shape2(256,1,1);
            dim3 grid_shape2((T*B + block_shape2.x - 1)/block_shape2.x,1,1);
            exp_c<<<grid_shape2,block_shape2>>>(d_att,d_params["decoder_c_tt"],T*B);

            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,B,T,&alpha,d_ones,1,d_att,T,&beta,d_att_sum,1);

            dim3 block_shape3(B,1,1);
            dim3 grid_shape3(T,1,1);
            divide<<<grid_shape3,block_shape3>>>(d_att,d_att_sum,T);

            // ctx_ = (ctx[:,None,:] * alpha[:,:,None]).sum(0)
            dim3 block_shape4(128,1,1);
            dim3 grid_shape4(T,B,(2*H + block_shape4.x - 1)/block_shape4.x);
            pointwise_prod<<<grid_shape4,block_shape4>>>(d_ctx,d_h_all_trans,d_att,T,B,2*H);
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,B*2*H,T,&alpha,d_ones,1,d_ctx,T,&beta,d_ctx_sum,1);

            // get ay_t
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,E,&alpha,d_y_tm1,B,d_params["decoder_W"],E,&beta,d_ay_t,B);
            // get as_t
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,H,&alpha,d_s_tm1,B,d_params["decoder_U"],H,&beta,d_as_t,B);
            // get ac_t
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,3*H,2*H,&alpha,d_ctx_sum,B,d_params["decoder_Wc"],2*H,&beta,d_ac_t,B);
            // point wise operation
            dim3 block_shape5(256,1,1);
            dim3 grid_shape5(B,(H + block_shape5.x - 1)/block_shape5.x,1);
            elementwise_op_dec<<<grid_shape5,block_shape5>>>(d_s_t,d_ay_t,d_as_t,d_ac_t,d_params["decoder_b"],d_s_tm1,B,H);

            // logit = tanh(W x s_t + bs + U x y_tm1 + by + V x ctx + bv)
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,E,H,&alpha,d_s_t,B,d_params["ff_logit_lstm_W"],H,&beta,d_logit_lstm,B);
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,E,E,&alpha,d_y_tm1,B,d_params["ff_logit_prev_W"],E,&beta,d_logit_prev,B);
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,E,2*H,&alpha,d_ctx_sum,B,d_params["ff_logit_ctx_W"],2*H,&beta,d_logit_ctx,B);
            tanh<<<grid_shape,block_shape>>>(d_logit,d_logit_lstm,d_logit_prev,d_logit_ctx,
                    d_params["ff_logit_lstm_b"],d_params["ff_logit_prev_b"],d_params["ff_logit_ctx_b"],B,E);

            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,V,E,&alpha,d_logit,B,d_params["ff_logit_W"],E,&beta,d_logit_softmax,B);

            dim3 block_shape6(256,1,1);
            dim3 grid_shape6(B,(V + block_shape6.x - 1)/block_shape6.x,1);
            exp_v<<<grid_shape6,block_shape6>>>(d_logit_softmax, d_params["ff_logit_b"], B, V);
            // sum
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,B,1,V,&alpha,d_logit_softmax,B,d_ones,V,&beta,d_logit_softmax_sum,B);
            // divide
            divide_log<<<grid_shape6,block_shape6>>>(d_logit_softmax,d_logit_softmax_sum,B,V);

            int live_k = K - dead_k;
            cudaMemcpy(&logit_softmax[0], d_logit_softmax, B*V*sizeof(float), cudaMemcpyDeviceToHost);
            priority_queue<pair<float,pair<int,int> >,vector<pair<float,pair<int,int> > >,greater<pair<float,pair<int,int> > > > q;
            for (int i=0;i<B;i++)
            {
                for (int j=0;j<V;j++)
                {
                    float score = logit_softmax[IDX2C(i,j,B)] + hyp_scores[i];
                    if (q.size() < live_k )
                        q.push(make_pair(score, make_pair(i,j)));
                    else
                    {
                        if (q.top().first < score)
                        {
                            q.pop();
                            q.push(make_pair(score, make_pair(i,j)));
                        }
                    }
                }
            }

            vector<vector<int> > new_hyp_samples;
            vector<float> new_hyp_scores;
            vector<int> father_idx;
            tgt_word_indices.clear();
            for (int k = 0; k < live_k; k++) 
            {
                float score = q.top().first;
                int i = q.top().second.first;
                int j = q.top().second.second;
                vector<int> sample(hyp_samples[i]);
                sample.push_back(j);
                if (j == 0)
                {
                    dead_k += 1;
                    final_samples.push_back(sample);
                    final_scores.push_back(score);
                }
                else
                {
                    new_hyp_samples.push_back(sample);
                    new_hyp_scores.push_back(score);
                    tgt_word_indices.push_back(j);
                    father_idx.push_back(i);
                }
                q.pop();
            }
            cudaMemcpy(d_father_idx, &father_idx[0], (K-dead_k)*sizeof(int), cudaMemcpyHostToDevice);
            dim3 block_shape7(256,1,1);
            dim3 grid_shape7(K-dead_k,(H + block_shape7.x - 1)/block_shape7.x,1);
            lookup_kernel<<<grid_shape7,block_shape7>>>(d_s_tm1,d_s_t,d_father_idx,H,K-dead_k,B);

            hyp_samples.swap(new_hyp_samples);
            hyp_scores.swap(new_hyp_scores);
            B = K-dead_k;
            if (B<=0)
                break;
        }
        if (K - dead_k > 0)
        {
            for (int k=0;k<K-dead_k;k++)
            {
                final_samples.push_back(hyp_samples[k]);
                final_scores.push_back(hyp_scores[k]);
            }
        }
        float best_score = -9999;
        int best_k = 0;
        for (int k=0;k<final_samples.size();k++)
        {
            float score = final_scores[k]/final_samples[k].size();
            if (score > best_score)
            {
                best_score = score;
                best_k = k;
            }
        }
        for (int i=0;i<final_samples[best_k].size() - 1;i++)
        {
            cout<<tgt_i2w[final_samples[best_k][i]]<<' ';
        }
        cout<<endl;
    }
}
