from collections import defaultdict
import cPickle as pkl

src_dict = pkl.load(open('1m.ch.pkl','rb'))
tgt_dict = pkl.load(open('1m.en.pkl','rb'))

d = defaultdict(list)
for s in open('lex.c2e'):
    s = s.split()
    e,c,p = s[0],s[1],float(s[2])
    if e != 'NULL':
        d[c].append((e,p))

for k,v in src_dict.items():
    if v >= 30000:
        break
    #if v < 2:
        #continue
    print v,
    #print k,
    ep_list = sorted(d.get(k,[]),key=lambda x:x[1],reverse=True)[:50]
    for e,p in ep_list:
        if tgt_dict.get(e,99999) < 30000:
            print tgt_dict[e],
            #print e,
    print

