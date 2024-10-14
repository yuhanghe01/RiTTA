import pickle
import numpy as np
import os
import yaml
import librosa
import RelationEvaluator


def summarize_relation_result(data_dict):
    maincate_name_list = ['Count', 'Compositionality', 'Temporal_Order', 'Spatial_Distance']
    subcate_name_list = ['count', 'before', 'after', 'together', 'closefirst', 'farfirst', 'equaldist', 'and', 'or', 'not', 'if_then_else']

    data_dict['relation_eval_result'] = dict()

    APre, ARel, APar = summarize_relation_result_4onerelation(data_dict, None)
    AMSR = (APre*ARel*APar)

    data_dict['relation_eval_result']['Overall'] = {'APre': APre, 'ARel': ARel, 'APar': APar, 'AMSR': AMSR}

    for main_cate in maincate_name_list:
        APre, ARel, APar = summarize_relation_result_4onerelation(data_dict, main_cate)
        AMSR = (APre*ARel*APar)
        data_dict['relation_eval_result'][main_cate] = {'APre': APre, 'ARel': ARel, 'APar': APar, 'AMSR': AMSR}

    for sub_cate in subcate_name_list:
        APre, ARel, APar = summarize_relation_result_4onerelation(data_dict, sub_cate)
        AMSR = (APre*ARel*APar)
        data_dict['relation_eval_result'][sub_cate] = {'APre': APre, 'ARel': ARel, 'APar': APar, 'AMSR': AMSR}

    return data_dict
    

def summarize_relation_result_4onerelation(data_dict, relation_name):
    presence_score_list = []
    relation_score_list = []
    parsimony_score_list = []
    main_catename_list = ['Count', 'Compositionality', 'Temporal_Order', 'Spatial_Distance']
    sub_name_dict = dict()
    sub_name_dict['count'] = 'Count'
    #['before', 'after', 'together']
    sub_name_dict['before'] = 'Temporal_Order'
    sub_name_dict['after'] = 'Temporal_Order'
    sub_name_dict['together'] = 'Temporal_Order'
    #['closefirst', 'farfirst', 'equaldist']
    sub_name_dict['closefirst'] = 'Spatial_Distance'
    sub_name_dict['farfirst'] = 'Spatial_Distance'
    sub_name_dict['equaldist'] = 'Spatial_Distance'
    #['and', 'or', 'not', 'if_then_else']
    sub_name_dict['and'] = 'Compositionality'
    sub_name_dict['or'] = 'Compositionality'
    sub_name_dict['not'] = 'Compositionality'
    sub_name_dict['if_then_else'] = 'Compositionality'

    if relation_name is None:
        across_all = True
    else:
        across_all = False

    if relation_name in main_catename_list:
        across_main = True
    else:
        across_main = False

    if relation_name in sub_name_dict.keys():
        across_sub = True
    else:
        across_sub = False
    
    for main_cate in data_dict.keys():
        if main_cate in ['author', 'time']:
            continue
        if across_main:
            if main_cate != relation_name:
                continue
        for sub_cate in data_dict[main_cate].keys():
            if across_sub:
                if relation_name != sub_cate:
                    continue
            for relation_rst_tmp in data_dict[main_cate][sub_cate]['relation_result']:
                presence_score_list.append(relation_rst_tmp['presence_score'])
                relation_score_list.append(relation_rst_tmp['relation_score'])
                parsimony_score_list.append(relation_rst_tmp['parsimony_score'])
            # presence_score_list.append(data_dict[main_cate][sub_cate]['relation_result']['presence_score'])
            # relation_score_list.append(data_dict[main_cate][sub_cate]['relation_result']['relation_score'])
            # parsimony_score_list.append(data_dict[main_cate][sub_cate]['relation_result']['parsimony_score'])

    presence_score = np.mean(np.array(presence_score_list, np.float32))
    relation_score = np.mean(np.array(relation_score_list, np.float32))
    parsimony_score = np.mean(np.array(parsimony_score_list, np.float32))

    return presence_score, relation_score, parsimony_score

def get_mean_average_score(data_dict, score_list):
    maincate_name_list = ['Count', 'Compositionality', 'Temporal_Order', 'Spatial_Distance']
    subcate_name_list = ['count', 'before', 'after', 'together', 'closefirst', 'farfirst', 'equaldist', 'and', 'or', 'not', 'if_then_else']
    catename_list = ['Overall']
    catename_list.extend(maincate_name_list)
    catename_list.extend(subcate_name_list)

    data_dict['relation_result'] = dict()

    for catename in catename_list:
        assert len(score_list) > 0
        APre_list, ARel_list, APar_list, AMSR_list = list(), list(), list(), list()
        for score in score_list:
            APre_list.append( data_dict[score]['relation_eval_result'][catename]['APre'] )
            ARel_list.append( data_dict[score]['relation_eval_result'][catename]['ARel'] )
            APar_list.append( data_dict[score]['relation_eval_result'][catename]['APar'] )
            AMSR_list.append( data_dict[score]['relation_eval_result'][catename]['AMSR'] )

        mAPre = np.mean(np.array(APre_list, np.float32))
        mARel = np.mean(np.array(ARel_list, np.float32))
        mAPar = np.mean(np.array(APar_list, np.float32))
        mAMSR = np.mean(np.array(AMSR_list, np.float32))

        data_dict['relation_result'][catename] = dict()
        data_dict['relation_result'][catename] = {'mAPre': mAPre, 'mARel': mARel, 'mAPar': mAPar, 'mAMSR': mAMSR}

    return data_dict


def relation_eval(config,
                  data_dict_filename, 
                  pred_audio_dir, 
                  confidence_score=0.5,
                  save_dir=None,
                  eval4method='audioldm',):
    with open(data_dict_filename, 'rb') as f:
        data_dict = pickle.load(f)

    dettagging_extractor = RelationEvaluator.DetTagInfoExtractor()
    relation_evaluator = RelationEvaluator.RelationEvaluator(parsimony_weight = config['parsimony_weight'], 
                                                             dist_loudness_reduce_reatio = config['dist_loudness_reduce_reatio'], 
                                                             equaldist_loudness_tolerance_ratio = config['equaldist_loudness_tolerance_ratio'],)
    output_dict = dict()
    for main_cate in data_dict.keys():
        if main_cate in ['author', 'time']:
            continue
        output_dict[main_cate] = {}
        for sub_cate in data_dict[main_cate].keys():
            output_dict[main_cate][sub_cate] = {}
            output_dict[main_cate][sub_cate]['relation_result'] = list()
            output_dict[main_cate][sub_cate]['data_info'] = list()
            for data_instance in data_dict[main_cate][sub_cate]:
                audio_label_list = data_instance['audio_label_list']
                ref_audio_name = data_instance['reference_audio']
                sub_relation = data_instance['sub_relation']
                if sub_relation in ['closefirst', 'farfirst','equaldist']:
                    if len(audio_label_list) == 1:
                        audio_label_list = audio_label_list*2
                pred_audio_basename = ref_audio_name.replace('.wav', '_{}.wav'.format(eval4method))
                pred_audio = librosa.load(os.path.join(pred_audio_dir, pred_audio_basename), sr=16000)[0]
                # assert os.path.exists(os.path.join(pred_audio_dir, pred_audio_basename))
                pred_audio_det_basename = pred_audio_basename.replace('.wav', '_panns_det.npy')
                assert os.path.exists(os.path.join(pred_audio_dir, pred_audio_det_basename))
                pred_all_audioevents = dettagging_extractor.get_all_det_audioevents(det_filename=os.path.join(pred_audio_dir, pred_audio_det_basename),
                                                                                    confidence_threshold=confidence_score,
                                                                                    min_event_lensec=config['min_event_lensec'],)
                presence_score, relation_score, parsimony_score = relation_evaluator.get_MSR_score(gt_label_list=audio_label_list,
                                                                                                        pred_audioevent_list=pred_all_audioevents,
                                                                                                        pred_audio=pred_audio,
                                                                                                        sub_relation=sub_relation,)

                # print('main_cate: {}, sub_cate: {}, presence_score: {:.4f}, relation_score: {:.4f}, parsimony_score: {:.4f}'.format(main_cate, 
                #                                                                                                                     sub_cate, 
                #                                                                                                                     presence_score, 
                #                                                                                                                     relation_score, 
                #                                                                                                                     parsimony_score))
                
                output_dict[main_cate][sub_cate]['relation_result'].append( {'presence_score': presence_score,
                                                                     'relation_score': relation_score,
                                                                     'parsimony_score': parsimony_score} )
                output_dict[main_cate][sub_cate]['data_info'].append(data_instance)
                # print('main_cate: {}, sub_cate: {}, presence_score: {:.4f}, relation_score: {:.4f}, parsimony_score: {:.4f}'.format(main_cate, sub_cate, presence_score, relation_score, parsimony_score))
    # # summarize the result
    # presence_score, relation_score, parsimony_score = summarize_relation_result(output_dict)
    # print('Overall: presence_score: {:.4f}, relation_score: {:.4f}, parsimony_score: {:.4f}'.format(presence_score, relation_score, parsimony_score))
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, 'relation_result_{}.pkl'.format(eval4method)), 'wb') as f:
    #     pickle.dump(output_dict, f)
    return output_dict

def get_dataset():
    pred_audio_dirlist = list()
    eval4method_list = list()

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/audioldm_data')
    # eval4method_list.append('audioldm')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/audiogen_data')
    # eval4method_list.append('audioldm')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/makeanaudio_data')
    # eval4method_list.append('makeanaudio')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/tango_data')
    # eval4method_list.append('tango')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/tango2_data')
    # eval4method_list.append('tango2')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/audioldm_LFull_data')
    # eval4method_list.append('audioldm')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/tango2_finetune')
    # eval4method_list.append('tango2')

    pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/tango-finetuned')
    eval4method_list.append('tango')

    # pred_audio_dirlist.append('/mnt/nas/yuhang/audioldm/audioldm2_LFull_data')
    # eval4method_list.append('audioldm')

    return pred_audio_dirlist, eval4method_list
                    
def main():
    config_filename = 'eval_config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dict_filename = '/mnt/nas/yuhang/audioldm/gen_data_fixed/data_dict.pkl'
    pred_audio_dir_list, eval4method_list = get_dataset()
    save_dir = '/mnt/nas/yuhang/audioldm/benchmark_eval_0.1'
    os.makedirs(save_dir, exist_ok=True)

    # confidence_score_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    confscore_list = config['confscore']
    confscore_min = confscore_list[0]
    confscore_stepsize = confscore_list[1]
    confscore_max = confscore_list[2]


    score_list = [round(i, 2) for i in np.arange(confscore_min, 
                                                 confscore_max + confscore_stepsize, 
                                                 confscore_stepsize)]

    for pred_audio_dir, eval4method in zip(pred_audio_dir_list, eval4method_list):
        output_dict = dict()
        for confidence_score in score_list:
            eval_dict = relation_eval(config, 
                                      data_dict_filename, 
                                      pred_audio_dir, 
                                      confidence_score=confidence_score,
                                      save_dir=save_dir,
                                      eval4method=eval4method,)

            eval_dict = summarize_relation_result(eval_dict)
            output_dict[confidence_score] = eval_dict

        output_dict = get_mean_average_score(output_dict, target_score_list)
        with open(os.path.join(save_dir, os.path.basename(pred_audio_dir)+'_relation_rst.pkl'), 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()