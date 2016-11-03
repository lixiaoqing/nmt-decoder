#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

// get embeddings for words, grid shape (2-dim): B x n, block shape (1-dim): E/n
__global__ void lookup_kernel(float *d_lookup, const float *d_Wemb, int *d_word_indices, int E, int B, int V)
{
    int i = blockIdx.x;
    int j = threadIdx.x + blockIdx.y*blockDim.x;
    if (j < E)
    {
        d_lookup[IDX2C(i,j,B)] = d_Wemb[IDX2C(d_word_indices[i],j,V)];
    }
}

__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

__global__ void elementwise_op(int H, float *ax_t, float *ah_t, float *bias, float *h_tm1, float *h_t)
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

void show_matrix(float *d_m, int r, int c, string flag)
{
    vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
    if (flag == "row") //print rows
    {
        for (int i=0;i<r;i++)
        {
            for (int j=0;j<c;j++)
            {
                cout<<m[IDX2C(i,j,r)]<<' ';
            }
            cout<<endl;
        }
    }
    else              //print cols
    {
        for (int j=0;j<c;j++)
        {
            for (int i=0;i<r;i++)
            {
                cout<<m[IDX2C(i,j,r)]<<' ';
            }
            cout<<endl;
        }
    }
}

int main()
{
    ifstream flist("param-list");
    if (!flist.is_open())
    {
        cerr<<"cannot open param-list!\n";
        return 0;
    }
    string s;
    vector<string> param_names;
    map<string,pair<int,int> > param2shape;
    vector<int> offset;
    offset.push_back(0);
    while(getline(flist,s))
    {
        stringstream ss;
        ss << s;
        string pn;
        int x,y;
        ss>>pn>>x>>y;
        //cout<<pn<<' '<<x<<' '<<y<<endl;
        param_names.push_back(pn);
        param2shape[pn] = make_pair(x,y);
        offset.push_back(offset.back()+x*y);
        //cout<<offset.back()<<endl;
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

    /*
    int E = 500;
    int B = 2;
    int V = 30000;

    vector<int> word_indices;
    word_indices.push_back(1);
    word_indices.push_back(2);
    int *d_word_indices;
    cudaMalloc((void**)&d_word_indices, B*sizeof(int));
    cudaMemcpy(d_word_indices, &word_indices[0], B*sizeof(int), cudaMemcpyHostToDevice);

    int block_shape = 128;
    dim3 grid_shape(B,(E + block_shape - 1)/block_shape,1);
    float *d_lookup;
    cudaMalloc((void**)&d_lookup, B*E*sizeof(float));

    // B x E => [B x n] x E/n
    lookup_kernel<<<grid_shape,block_shape>>>(d_lookup,d_params["Wemb"],d_word_indices,E,B,V);
    vector<float> lookup;
    lookup.resize(B*E);
    cudaMemcpy(&lookup[0], d_lookup, B*E*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0;i<B;i++)
    {
        for (int j=0;j<E;j++)
        {
            cout<<lookup[IDX2C(i,j,B)]<<endl;
        }
    }
    */

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
        int E = 500;
        int H = 1024;
        int B = 1;
        int V = 30000;


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

        cudaMalloc((void**)&d_h_all, 2*H*(T+2)*sizeof(float));

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha=1.0;  
        float beta=0.0;  

        // init h_0, hr_0 with all zeros
        cudaMemset(d_h_all,0,2*H*(T+2)*sizeof(float));
        //show_matrix(d_h_all,2*H,T+2,"col");

        int *d_word_indices;
        cudaMalloc((void**)&d_word_indices, B*sizeof(int));
        int *d_word_indices_r;
        cudaMalloc((void**)&d_word_indices_r, B*sizeof(int));

        for (int i=0;i<T;i++)
        {
            // fill x_t; B x E => [B x n] x E/n
            dim3 block_shape(128,1,1);
            dim3 grid_shape(B,(E + block_shape.x - 1)/block_shape.x,1);
            cudaMemcpy(d_word_indices, &word_indices[i], B*sizeof(int), cudaMemcpyHostToDevice);
            lookup_kernel<<<grid_shape,block_shape>>>(d_x_t,d_params["Wemb"],d_word_indices,E,B,V);
            // backward
            cudaMemcpy(d_word_indices_r, &word_indices[T-1-i], B*sizeof(int), cudaMemcpyHostToDevice);
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
            elementwise_op<<<grid_shape1,block_shape1>>>(H,d_ax_t,d_ah_t,d_params["encoder_b"],d_h_tm1,d_h_all+2*(i+1)*H);
            //backward
            elementwise_op<<<grid_shape1,block_shape1>>>(H,d_axr_t,d_ahr_t,d_params["encoder_r_b"],d_hr_Tp1mt,d_h_all+(2*T+1-2*i)*H);
        }
        show_matrix(d_h_all,2*H,T+1,"col");
        break;
    }
}
