import numpy as np
import progressbar
import sys
import operator
import pickle


def softmax(w):
    exp_mat = np.exp(w)
    exp_sum = np.sum(exp_mat, axis=1)
    exp_sum = np.tile(exp_sum, (w.shape[1], 1)).transpose()
    return exp_mat / exp_sum

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    #x1 = [b[0] for b in boxes]
    #x2 = [b[1] for b in boxes]
    #s = [b[-1] for b in boxes]
    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

#IoU=0.1

cat_index_dict={
0:("Background",0),
1:("BaseballPitch",7),
2:("BasketballDunk",9),
3:("Billiards",12),
4:("CleanAndJerk",21),
5:("CliffDiving",22),
6:("CricketBowling",23),
7:("CricketShot",24),
8:("Diving",26),
9:("FrisbeeCatch",31),
10:("GolfSwing",33),
11:("HammerThrow",36),
12:("HighJump",40),
13:("JavelinThrow",45),
14:("LongJump",51),
15:("PoleVault",68),
16:("Shotput",79),
17:("SoccerPenalty",85),
18:("TennisSwing",92),
19:("ThrowDiscus",93),
20:("VolleyballSpiking",97)
}

pkl_file = sys.argv[1]
IoU=float(sys.argv[2])
movie_fps=pickle.load(open("./movie_fps.pkl"))
nms_delta=0.2
feq_refine_early=True
act_refine_freq = np.load('val_training_samples_16-512_window_freq.npy')
test_movie_interval_label_samples={}
output_result_file=open("./after_postprocessing/"+pkl_file.split(".pkl")[0]+"_IoU_"+str(IoU)+".txt","w")


results_dict=pickle.load(open("./test_results/"+pkl_file))
bar = progressbar.ProgressBar(maxval=len(results_dict)).start()
for m_id, movie in enumerate(results_dict):
    bar.update(m_id+1)  
    feats = results_dict[movie][2]
    starts=results_dict[movie][0]
    ends= results_dict[movie][1]

    if len(starts) > 0:
        starts = np.array(starts).astype(dtype='float32')
        ends = np.array(ends).astype(dtype='float32')
        feats = np.array(feats).astype(dtype='float32')

        category_responces=feats[:, :]
        for i in range(len(starts)):
            cur_movie_start = starts[i]
            cur_movie_end = ends[i]
            if feq_refine_early:
                clip_length_index=[16,32,64,128,256,512].index(min([16,32,64,128,256,512],key=lambda x:abs(x-int(ends[i]-starts[i]))))
                category_responces[i,:] = act_refine_freq[1:, clip_length_index]*category_responces[i,:]
        category_preds=np.argmax(category_responces,axis=1)+1
        assert category_preds.shape[0]==len(starts)

        overlap_nms = IoU-nms_delta
        if overlap_nms < 0 : overlap_nms=0.0
        for cat_index in range(1,21):
            cat_starts=[]
            cat_ends=[]
            cat_scores=[]
            for k in range(len(category_preds)):
                pred=category_preds[k]
                if cat_index==pred:
                    cat_starts.append(starts[k])
                    cat_ends.append(ends[k])
                    if not feq_refine_early:
                        clip_length_index=[16,32,64,128,256,512].index(min([16,32,64,128,256,512],key=lambda x:abs(x-int(ends[i]*-starts[i]))))
                        cat_scores.append(category_responces[k,cat_index-1]*act_refine_freq[cat_index, clip_length_index])
                    else:
                        cat_scores.append(category_responces[k,cat_index-1])
            picks=nms_temporal(cat_starts,cat_ends,cat_scores , overlap_nms)
            for pick in picks:
                output_result_file.write(movie+" "+str(cat_starts[pick]/movie_fps[movie])+" "+str(cat_ends[pick]/movie_fps[movie])+" "+str(cat_index_dict[cat_index][1])+" "+str(cat_scores[pick])+"\n")

