#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <sstream>
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

class Decoder
{
    public:
        Decoder();
        string translate(string input_sen);

    private:
        void load_params(map<string,float*> &d_params);
        void load_dict(map<string,int> &src_w2i,map<int,string> &tgt_i2w);
        vector<int> w2id(string input_sen);
        string id2w(vector<int> output_wids);
        void encode(vector<int> input_wids);
        vector<int> decode();
        void init_decoder_state();
        void get_next_prob(vector<int> tgt_wids);
        void generate_new_samples(vector<vector<int> > &hyp_samples,vector<float> &hyp_scores,
                vector<vector<int> > &final_samples, vector<float> &final_scores,
                vector<float> &logit_softmax,int &dead_k,vector<int> &tgt_wids);

        map<string,float*> d_params;
        map<string,int> src_w2i;
        map<int,string> tgt_i2w;

        int E, H, B, V, K, Tmax, T;
        float *d_ones;
        //encoder
        int *d_wid, *d_wid_r;
        float *d_x_t, *d_ax_t, *d_ah_t;
        float *d_xr_t, *d_axr_t, *d_ahr_t;
        float *d_h_all;

        //decoder
        int *d_tgt_wids;
        int *d_father_idx;
        float *d_ctx_mean, *d_init_state;
        float *d_h_all_trans;
        float *d_pctx_;
        float *d_s_tm1;
        float *d_y_tm1, *d_pstate_, *d_pctx__, *d_att, *d_att_sum, *d_ctx, *d_ctx_sum;
        float *d_ay_t, *d_as_t, *d_ac_t;
        float *d_s_t;
        float *d_logit_lstm, *d_logit_prev, *d_logit_ctx;
        float *d_logit, *d_logit_softmax, *d_logit_softmax_sum;

        cublasHandle_t handle;
        float alpha;  
        float beta;  
};
