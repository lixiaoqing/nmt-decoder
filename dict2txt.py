import cPickle as pkl

src_dict = pkl.load(open('1m.ch.pkl','rb'))
fs = open('1m.ch.w2i','w')
for k,v in src_dict.items():
    if v >= 30000:
        break
    print >>fs,k,v

tgt_dict = pkl.load(open('1m.en.pkl','rb'))
ft = open('1m.en.i2w','w')
for k,v in tgt_dict.items():
    if k == 'eos':
        print >>ft,0,'eos'
        continue
    if v >= 30000:
        break
    print >>ft,v,k

