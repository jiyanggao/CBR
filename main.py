import tensorflow as tf
import numpy as np
from six.moves import xrange
import time
import pickle
import operator
import os 

import cbr_model
import dataset

ctx_num = 2
unit_size = 16.0
unit_feature_size = 4096
lr = 0.001
lambda_reg = 1.0
batch_size = 128
test_steps = 500
action_class_num = 20
cas_step = 3

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


def load_c3d_weights(weights_path):
    data = pickle.load(open(weights_path))
    Ws, Bs = data['W'], data['B']
    return Ws, Bs


def get_pooling_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end):
    swin_step = unit_size
    all_feat = np.zeros([0,unit_feature_size], dtype=np.float32)
    current_pos = start
    while current_pos<end:
        swin_start = current_pos
        swin_end = swin_start+swin_step
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))
            all_feat = np.vstack((all_feat, feat))
        current_pos+=swin_step
    pool_feat = np.mean(all_feat, axis=0)
    return pool_feat

def get_left_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    count = 0
    current_pos = start
    context_ext = False
    while  count<ctx_num:
        swin_start = current_pos-swin_step
        swin_end = current_pos
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))    
            all_feat = np.vstack((all_feat, feat))
            context_ext = True
        current_pos-=swin_step
        count+=1
    if context_ext:
        pool_feat = np.mean(all_feat, axis=0)
    else:
        pool_feat = np.zeros([unit_feature_size], dtype=np.float32)
    return np.reshape(pool_feat, [unit_feature_size])

def get_right_context_feature(flow_feat_dir, appr_feat_dir, movie_name, start, end):
    swin_step = unit_size
    all_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
    count = 0
    current_pos = end
    context_ext = False
    while  count<ctx_num:
        swin_start = current_pos
        swin_end = current_pos+swin_step
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))
            all_feat = np.vstack((all_feat, feat))
            context_ext = True
        current_pos+=swin_step
        count+=1
    if context_ext:
        pool_feat = np.mean(all_feat, axis=0)
    else:
        pool_feat = np.zeros([unit_feature_size], dtype=np.float32)
    return np.reshape(pool_feat, [unit_feature_size])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def do_eval_slidingclips(sess, vs_eval_op, model, test_set, iter_step):
    
    result_dict = {}
    reg_result_dict = {}
    prob_weights = np.array([0.8,0.1,0.1])
    for k,test_sample in enumerate(test_set.test_samples):
        if k%1000==0:
            print str(k)+"/"+str(len(test_set.test_samples))
        movie_name = test_sample[0]
        if not movie_name in result_dict: 
            result_dict[movie_name] = []
            result_dict[movie_name].append([]) #start
            result_dict[movie_name].append([]) #end
            result_dict[movie_name].append([]) #feats
            reg_result_dict[movie_name]=[]
            reg_result_dict[movie_name].append([]) #start
            reg_result_dict[movie_name].append([]) #end
            reg_result_dict[movie_name].append([]) #feats
        init_clip_start = test_sample[1]
        init_clip_end = test_sample[2]
        clip_start = init_clip_start
        clip_end = init_clip_end
        final_action_prob = np.zeros([action_class_num])
        for i in range(cas_step):
            featmap = get_pooling_feature(test_set.flow_feat_dir, test_set.appr_feat_dir, movie_name,clip_start, clip_end)
            left_feat = get_left_context_feature(test_set.flow_feat_dir, test_set.appr_feat_dir, movie_name, clip_start, clip_end)
            right_feat = get_right_context_feature(test_set.flow_feat_dir, test_set.appr_feat_dir, movie_name, clip_start, clip_end)
            feat = np.hstack((left_feat, featmap, right_feat))
            feat = np.reshape(feat, [1, unit_feature_size*3])
        
            feed_dict = {
                model.visual_featmap_ph_test: feat
                }
        
            outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
            action_score = outputs[1:action_class_num+1]
            action_prob = softmax(action_score)
            # In BMVC paper, we used prob multiplication to calculate final prob, but later experiments showed that weighted average gives more stable results.
            final_action_prob = final_action_prob+prob_weights[i]*action_prob
            action_cat = np.argmax(action_prob)+1
            round_reg_end = clip_end+round(outputs[(action_class_num+1)*2+action_cat])*unit_size
            round_reg_start = clip_start+round(outputs[action_class_num+1+action_cat])*unit_size
            reg_end = clip_end+outputs[(action_class_num+1)*2+action_cat]*unit_size
            reg_start = clip_start+outputs[action_class_num+1+action_cat]*unit_size
            clip_start = round_reg_start
            clip_end = round_reg_end
        result_dict[movie_name][0].append(clip_start)
        result_dict[movie_name][1].append(clip_end)
        result_dict[movie_name][2].append(outputs[:action_class_num+1])
        reg_result_dict[movie_name][0].append(reg_start)
        reg_result_dict[movie_name][1].append(reg_end)
        reg_result_dict[movie_name][2].append(action_prob)
    pickle.dump(reg_result_dict, open("./eval/test_results/twostream_CBR_4_"+str(iter_step)+".pkl","w"))


def run_training():
    max_steps = 4000
    train_clip_path = "./val_training_samples.txt"
    background_path = "./background_samples.txt"
    train_flow_featmap_dir = "../val_fc6_16_overlap0.5_denseflow/"
    train_appr_featmap_dir = "../val_fc6_16_overlap0.5_resnet/"
    test_flow_featmap_dir = "../test_fc6_16_overlap0.5_denseflow/" 
    test_appr_featmap_dir = "../test_fc6_16_overlap0.5_resnet/"  
    test_clip_path = "./test_proposals_from_TURN.txt"

    model = cbr_model.CBR_Model(batch_size, ctx_num, unit_size, unit_feature_size, action_class_num, lr, lambda_reg, train_clip_path, background_path, test_clip_path, train_flow_featmap_dir, train_appr_featmap_dir, test_flow_featmap_dir, test_appr_featmap_dir)

    with tf.Graph().as_default():
		
        loss, vs_train_op, vs_eval_op, loss_reg = model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) # 30% memory of TITAN is enough
        sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train()

            _, loss_value, loss_reg_value = sess.run([vs_train_op, loss, loss_reg], feed_dict=feed_dict)
            duration = time.time()-start_time
            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f, loss_reg=%.2f, (%.3f sec)' % (step, loss_value, loss_reg_value, duration))
            

            if step>=1999 and (step+1) % test_steps==0:
                print "Start to test:-----------------\n"
                do_eval_slidingclips(sess, vs_eval_op, model, model.test_set, step+1)
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



