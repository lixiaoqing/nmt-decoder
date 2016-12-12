import numpy as np

pp = np.load('model.npz')
d = {}
for k,v in pp.items():
    d[k] = v.transpose()

fout = open('model.bin','wb')
li = ['Wemb', 'Wemb_dec', 'encoder_W', 'encoder_Wx', 'encoder_U', 'encoder_Ux', 'encoder_b', 'encoder_bx', 'encoder_r_W', 'encoder_r_Wx', 'encoder_r_U', 'encoder_r_Ux', 'encoder_r_b', 'encoder_r_bx', 'decoder_W', 'decoder_Wx', 'decoder_U', 'decoder_Ux', 'decoder_b', 'decoder_bx', 'decoder_Wc', 'decoder_Wcx', 'decoder_Wi_att', 'decoder_Wc_att', 'decoder_Wd_att', 'decoder_U_att', 'decoder_b_att', 'decoder_c_tt', 'ff_state_W', 'ff_state_b', 'ff_logit_W', 'ff_logit_b', 'ff_logit_lstm_W', 'ff_logit_lstm_b', 'ff_logit_prev_W', 'ff_logit_prev_b', 'ff_logit_ctx_W', 'ff_logit_ctx_b']
for e in li:
    d[e].tofile(fout)
    print e,d[e].shape[0]*(d[e].shape[1] if len(d[e].shape) == 2 else 1)
fout.close()

