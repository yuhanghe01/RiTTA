import numpy as np
import torch
import scipy.linalg as linalg
import itertools

class DetTagInfoExtractor(object):
    def __init__(self, dist_loudness_thred = 0.2, equaldist_loudness_range = 0.3 ) -> None:
        self.dist_loudness_thred = dist_loudness_thred
        self.equaldist_loudness_range = equaldist_loudness_range

    def get_all_det_audioevents(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1,):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    event_list.append([event_id + 1, start_time, end_time])
                    assert event_id < det_score.shape[1]
                    for i in range(start_time, min(end_time+1, det_score.shape[0])): #remove the detected event from the detection score
                        det_score[i, event_id] = 0

        return event_list

    def get_det_result_with_timestep(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1,):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append([event_id + 1, start_time, end_time])
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.append(parallel_events)

        #get the final event list, get all posible combinations
        det_label_list = list()

        all_combinations = list(itertools.product(*event_list))
        for combination in all_combinations:
            det_label_list.append(list(combination))

        return det_label_list

    def get_dettagging_result(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1,):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        det_label_list = []
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append(event_id + 1)
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.extend(parallel_events)

        return event_list
    
    def get_det_result(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1,):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        det_label_list = []
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append(event_id + 1)
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.append(parallel_events)

        #get the final event list, get all posible combinations
        det_label_list = list()

        all_combinations = list(itertools.product(*event_list))
        for combination in all_combinations:
            det_label_list.append(list(combination))

        return det_label_list

    def get_tagging_result(self, tagging_filename, confidence_threshold = 0.5,):
        """
        det_filename: npy file containing the tagging score
        output: a list of ordered detected audio events labels
        """
        tagging_score = np.load(tagging_filename).squeeze() #[25]
        tagging_result = np.where(tagging_score >= confidence_threshold)[0]
        if len(tagging_result) == 0:
            return []
        tagging_result = tagging_result + 1

        return tagging_result.tolist()

class RelationEvaluator(object):
    def __init__(self, parsimony_weight = 0.1, dist_loudness_reduce_reatio = 0.2, equaldist_loudness_tolerance_ratio = 0.2) -> None:
        self.parsimony_weight = parsimony_weight
        self.dist_loudness_reduce_reatio = dist_loudness_reduce_reatio
        self.equaldist_loudness_tolerance_ratio = equaldist_loudness_tolerance_ratio

    def get_all_after_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that are after the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        after_event_list = list()
        target_event_end_time = target_event[2]
        for ref_event in ref_event_list:
            if ref_event[1] > target_event_end_time:
                after_event_list.append(ref_event)
        
        return after_event_list

    def get_all_together_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that are after the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        together_event_list = list()
        target_event_start_time = target_event[1]
        target_event_end_time = target_event[2]
        for ref_event in ref_event_list:
            ref_event_start_time= ref_event[1]
            ref_event_end_time = ref_event[2]
            min_start_time = min(ref_event_start_time, target_event_start_time)
            max_end_time = max(ref_event_end_time, target_event_end_time)
            if max_end_time - min_start_time + 1 < ref_event_end_time - ref_event_start_time + 1 + target_event_end_time - target_event_start_time + 1:
                together_event_list.append(ref_event)
        
        return together_event_list
    
    def get_all_before_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that are after the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        before_event_list = list()
        target_event_start_time = target_event[1]
        for ref_event in ref_event_list:
            if ref_event[2] < target_event_start_time:
                before_event_list.append(ref_event)
        
        return before_event_list

    def check_all_include(self, ref_label_list, label2check_list):
        '''if all labels in ref_label_list are in pred_label_list, return 1, else return 0'''
        exist_ids = list()
        for label2check in label2check_list:
            for ref_id, ref_label in enumerate(ref_label_list):
                if label2check == ref_label and ref_id not in exist_ids:
                    exist_ids.append(ref_id)

        return  len(exist_ids) == len(label2check_list)

    def check_not_include(self, ref_label_list, label2check_list):
        for label2check in label2check_list:
            if label2check in ref_label_list:
                return False
            
        return True
    
    def get_MSR_score(self, gt_label_list, pred_audioevent_list, pred_audio, sub_relation = 'count'):
        if len(pred_audioevent_list) == 0:
            return 0., 0., 0.
        pred_label_list = [audioevent[0] for audioevent in pred_audioevent_list]
        presence_score = self.get_presence_score(pred_label_list, gt_label_list)
        if presence_score == 0.:
            relation_score = 0.
            parsimony_score = 0.
            return presence_score, relation_score, parsimony_score

        parsimony_score = self.parsimony_score(gt_label_list, pred_label_list)
        if sub_relation == 'count':
            relation_score = self.get_presence_score(gt_label_list, pred_label_list)
        if sub_relation == 'before':
            relation_score = self.get_before_relation_score(gt_label_list, pred_audioevent_list)
        if sub_relation == 'after':
            relation_score = self.get_before_relation_score([gt_label_list[1],gt_label_list[0]], pred_audioevent_list)
        if sub_relation == 'together':
            relation_score = self.get_together_relation_score(gt_label_list, pred_audioevent_list)

        if sub_relation == 'and':
            relation_score = 1. if self.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list) else 0.
        if sub_relation == 'or':
            any_include = self.check_any_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            all_include = self.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            if (not all_include) and any_include:
                relation_score = 1.
            else:
                relation_score = 0.
        if sub_relation == 'not':
            relation_score = 1. if self.check_not_include(ref_label_list=pred_label_list, label2check_list=gt_label_list) else 0.
        if sub_relation == 'if_then_else':
            if self.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list[:2]) and not self.check_all_include(ref_label_list=pred_label_list, label2check_list=[gt_label_list[2]]):
                relation_score = 1.
            elif self.check_not_include(ref_label_list=pred_label_list, label2check_list=gt_label_list[:2]) and self.check_all_include(ref_label_list=pred_label_list, label2check_list=[gt_label_list[2]]):
                relation_score = 1.
            else:
                relation_score = 0.
        if sub_relation in ['closefirst', 'farfirst','equaldist']:
            relation_score = self.get_spatialdist_relation_score(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        
        return presence_score, relation_score, parsimony_score

    def get_spatialdist_relation_score(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = 'closefirst'):
        # breakpoint()
        for pred_event in pred_audioevent_list:
            if pred_event[0] == gt_label_list[0]:
                first_audioevent = pred_event
                first_audio = pred_audio[first_audioevent[1]*8000:first_audioevent[2]*8000+1]
                first_audio_loudness = np.linalg.norm(first_audio)
                all_after_events = self.get_all_after_audioevents(pred_audioevent_list, first_audioevent)
                for after_event in all_after_events:
                    if after_event[0] == gt_label_list[0]:
                        after_audio = pred_audio[after_event[1]*8000:after_event[2]*8000+1]
                        second_audio_loudness = np.linalg.norm(after_audio)
                        if sub_relation == 'closefirst':
                            # breakpoint()
                            if first_audio_loudness - second_audio_loudness > self.dist_loudness_reduce_reatio*first_audio_loudness:
                                return 1.
                            else:
                                return 0.
                        if sub_relation == 'farfirst':
                            if second_audio_loudness - first_audio_loudness > self.dist_loudness_reduce_reatio*first_audio_loudness:
                                return 1.
                            else:
                                return 0.
                        if sub_relation == 'equaldist':
                            max_loudness = max(first_audio_loudness, second_audio_loudness)
                            if abs(first_audio_loudness-second_audio_loudness) < self.equaldist_loudness_tolerance_ratio*max_loudness:
                                return 1.
                            else:
                                return 0.
        return 0.
                            
    def get_together_relation_score(self, gt_label_list, pred_audioevent_list):
        first_label = gt_label_list[0]
        second_label = gt_label_list[1]

        for pred_audioevent in pred_audioevent_list:
            if pred_audioevent[0] == first_label:
                first_audioevent = pred_audioevent
                after_events = self.get_all_together_audioevents(pred_audioevent_list, first_audioevent)
                if self.check_any_include([audioevent[0] for audioevent in after_events], [second_label]):
                    return 1.

        return 0.
    
    def get_before_relation_score(self, gt_label_list, pred_audioevent_list):
        first_label = gt_label_list[0]
        second_label = gt_label_list[1]

        for pred_audioevent in pred_audioevent_list:
            if pred_audioevent[0] == first_label:
                first_audioevent = pred_audioevent
                after_events = self.get_all_after_audioevents(pred_audioevent_list, first_audioevent)
                if self.check_any_include([audioevent[0] for audioevent in after_events], [second_label]):
                    return 1.
        
        return 0.

    def check_any_include(self, ref_label_list, label2check_list):
        for label2check in label2check_list:
            if label2check in ref_label_list:
                return True
            
        return False
    
    def get_presence_score(self, ref_label_list, pred_label_list):
        '''if all labels in ref_label_list are in pred_label_list, return 1, else return 0'''
        if self.check_all_include(ref_label_list, pred_label_list):
            return 1.
        else:
            return 0.

    

    def target_audio_presence(self, gt_label_list, pred_label_list):
        '''if all labels in gt_label_list are in pred_label_list, return 1, else return 0'''
        gt_label_set = set(gt_label_list)
        pred_label_set = set(pred_label_list)

        intersection = gt_label_set.intersection(pred_label_set)

        return  float(intersection == gt_label_set)
    
    def is_relation_correct(self, gt_label_list, pred_label_list, relation = 'count'):
        """Check if the input relation is correctly predicted in the predicted label list.
        """
        if relation in ['count', 'together', 'and']:
            max_num = max(max(gt_label_list), max(pred_label_list))
            gt_label_num = np.zeros(max_num+1, np.int32)
            pred_label_num = np.zeros(max_num+1, np.int32)
            for gt_label in gt_label_list:
                gt_label_num[gt_label] += 1
            for pred_label in pred_label_list:
                pred_label_num[pred_label] += 1

            return float(np.all(gt_label_num <= pred_label_num))
        
        if relation in ['before', 'after']: 
            # before or after, the label order in gt should be the same as in pred
            # it returns true if at least one sub-label can be found in the pred_label_list that match the label
            # in the gt_label_list
            gtlabel_index_in_pred = -1*np.ones(len(gt_label_list), np.int32)
            for i, gt_label in enumerate(gt_label_list):
                for j, pred_label in enumerate(pred_label_list):
                    if gt_label == pred_label and j >= gtlabel_index_in_pred[max(i-1,0)]:
                        gtlabel_index_in_pred[i] = j
                        break
                if gtlabel_index_in_pred[i] == -1:
                    return 0.
                
            return float(np.all(gtlabel_index_in_pred>=0))
        
        if relation in ['or']:
            # at least one label in gt_label_list should be in pred_label_list
            gt_label_set = set(gt_label_list)
            pred_label_set = set(pred_label_list)
            intersection = gt_label_set.intersection(pred_label_set)
            return float(len(intersection) > 0)
        
        if relation in ['not']:
            # no label in gt_label_list should be in pred_label_list
            gt_label_set = set(gt_label_list)
            pred_label_set = set(pred_label_list)
            intersection = gt_label_set.intersection(pred_label_set)
            return float(len(intersection) == 0)
        
        if relation in ['if_then_else']:
            # gt_label_list[0] and gt_label_list[1] should be in pred_label_list at the same time, or gt_label_list[2] should be in pred_label_list,
            # but not the same time.
            # if the first label in gt_label_list is in pred_label_list, the second label should be in pred_label_list
            if gt_label_list[0] in pred_label_list and gt_label_list[1] in pred_label_list and gt_label_list[2] not in pred_label_list:
                return 1.0
            elif gt_label_list[0] not in pred_label_list and gt_label_list[1] not in pred_label_list and gt_label_list[2] in pred_label_list:
                return 1.0
            else:
                return 0.
            
        # if relation in ['closefirst', 'farfirst','equaldist']:
        #     NotImplementedError("This method should be implemented by the subclass")
    def is_relation_correct_spatialdist(self, gt_label_list, 
                                        pred_label_list, 
                                        pred_audios_list,
                                        relation = 'closefirst',):
        #first check if the relation is correct
        appear_times = 0
        for label in pred_audios_list:
            if label == gt_label_list[0]:
                appear_times += 1
        if appear_times < len(gt_label_list):
            return 0.
        
        for pred_label_id1 in range(len(pred_label_list)-1):
            if pred_label_id1 != gt_label_list[0]:
                continue
            for pred_label_id2 in range(pred_label_id1+1, len(pred_label_list)):
                if pred_label_id2 != gt_label_list[1]:
                    continue
                pred_audio1 = pred_audios_list[pred_label_id1]
                pred_audio2 = pred_audios_list[pred_label_id2]
                pred_audio1_loudness = np.linalg.norm(pred_audio1)
                pred_audio2_loudness = np.linalg.norm(pred_audio2)

                if relation == 'closefirst':
                    if pred_audio1_loudness - pred_audio2_loudness > self.dist_loudness_thred*pred_audio1_loudness:
                        return 1.
                    else:
                        return 0.
                if relation == 'farfirst':
                    if pred_audio2_loudness - pred_audio1_loudness > self.dist_loudness_thred*pred_audio2_loudness:
                        return 1.
                    else:
                        return 0.
                if relation == 'equaldist':
                    loudness_mean = (pred_audio1_loudness, pred_audio2_loudness)/2.0
                    if abs(pred_audio1_loudness - pred_audio2_loudness) < self.equaldist_loudness_range*loudness_mean:
                        return 1.
                    else:
                        return 0.

    def parsimony_score(self, gt_label_list, pred_label_list):
        '''The parsimony score is the number of labels in the predicted label list that are not in the ground truth label list.
        '''
        redundant_label_num = abs(len(pred_label_list) - len(gt_label_list))

        return np.exp(-1.0*redundant_label_num*self.parsimony_weight)

    def relation_score(self, gt_label_list, pred_label_list, relation = 'count'):
        '''The relation score is the product of the target_audio_presence and is_relation_correct.
        '''
        presence_score = self.target_audio_presence(gt_label_list, pred_label_list)
        relation_correct_score = self.is_relation_correct(gt_label_list, pred_label_list, relation)
        parsimony_score = self.parsimony_score(gt_label_list, pred_label_list)
        print('presence_score: ', presence_score)
        print('relation_correct_score: ', relation_correct_score)
        print('parsimony_score: ', parsimony_score)

        relation_score = presence_score * relation_correct_score * parsimony_score

        return relation_score