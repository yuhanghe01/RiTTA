import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import random
import yaml
from datetime import datetime
import pickle

class TextAudioCreator(object):
    def __init__(self, config):
        self.config = config
        random.seed(self.config['DATA_CREATION_CONFIG']['random_seed'])
        np.random.seed(self.config['DATA_CREATION_CONFIG']['random_seed'])
        self.seed_audio_path = self.config['DATA_CREATION_CONFIG']['seed_audio_dir']
        self.audio_category_list = self.config['AUDIO_CATEGORY']
        self.audio_len = self.config['DATA_CREATION_CONFIG']['audio_length']
        assert len(self.audio_category_list) > 0
        self.get_audioname_corpus()
        self.get_textprompt_template()
        self.get_name2label_dict()
        self.check_config()

    def check_config(self):
        assert len(self.config['DATA_CREATION_CONFIG']['relations2create']) > 0
        if self.config['DATA_CREATION_CONFIG']['ensure_nonoverlapping']:
            for relation in self.config['DATA_CREATION_CONFIG']['relations2create']:
                max_audio_len = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(relation)]['max_audio_len']
                if max_audio_len + max_audio_len - 1 >= self.config['DATA_CREATION_CONFIG']['audio_length']:
                    raise ValueError('The maximum audio length is too small to ensure non-overlapping for relation {}'.format(relation))
        
        #check relation2create in relation corpus
        assert len(self.config['RELATION_CORPUS']) > 0
        for relation in self.config['DATA_CREATION_CONFIG']['relations2create']:
            assert relation in self.config['RELATION_CORPUS']

    def get_name2label_dict(self):
        name2label_dict = dict()

        with open(self.config['DATA_CREATION_CONFIG']['name2label_filename'], 'rb') as f:
            for line in f:
                name, label = line.decode().strip().rstrip('\t').split(' ')
                name2label_dict[name] = int(label)
        
        self.name2label_dict = name2label_dict

    def get_textprompt_template(self):
        self.textprompt_templates = dict()
        self.textprompt_templates['Temporal_Order'] = dict()
        self.textprompt_templates['Temporal_Order']['before_or_after'] = ['generate {} audio followed by {} audio',
                                                                          'create an audio sequence starting with {} audio and then followed by {} audio',
                                                                          'make an audio recording play {} audio initially, {} audio afterwards',
                                                                          'generate {} audio preceded by {} audio',
                                                                          'produce an audio recording with {} audio in the begining, {} audio coming next']
        self.textprompt_templates['Temporal_Order']['together'] = ['create {} audio and {} audio at the same time',
                                                                   'produce {} audio and {} audio simultaneously',
                                                                   'generate audio for {} audio and {} audio concurrently',
                                                                   'create audio tracks for {} audio and {} audio in parallel',
                                                                   'produce {} audio and {} audio together']
        self.textprompt_templates['Count'] = ['generate {} audios including {}',
                                             'create {} audio clips, including {}',
                                             'produce {} audio samples, they are {}',
                                             'generate {} audio, among them are {}',
                                             'create {} audio, involving {}']
        self.textprompt_templates['Spatial_Continuality'] = dict()
        self.textprompt_templates['Spatial_Continuality']['close'] = ['generate {} audio that is moving closer',
                                                            'produce {} audio that moves closer to the listener',
                                                            'generate {}, an audio that advances toward the listener'
                                                            'produce {} audio that simulates coming closere',
                                                            'generate {} audio that gives the impression of approaching']
        self.textprompt_templates['Spatial_Continuality']['far'] = ['generate {} audio that is moving away',
                                                            'create {} audio that is that is fading into the distance',
                                                            'generate {} audio that gives the impression of receding',
                                                            'produce {} audio that that drifts away over time',
                                                            'produce {} audio that simulates moving farther into the distance']
        self.textprompt_templates['Spatial_Distance'] = dict()
        self.textprompt_templates['Spatial_Distance']['closefirst'] = ['generate {} audio 1 meter away, followed by the same {} audio that is 5 meters away',
                                                         'produce {} audio located at 2 meters away preceded by the same {} audio located 3 meters away',
                                                         'generate {} audio 1 meter away at begining, then the same {} audio at a distance of 7 meters',
                                                         'generate {} audio 3 meters away initially, then generate the same {} audio 7 meters away',
                                                         'create {} audio positioned 1 meter away, followed by the same {} audio at 5 meters']
        self.textprompt_templates['Spatial_Distance']['equaldist'] = ['generate {} audio 1 meter away, followed by the same {} audio that is also 1 meter away',
                                                         'create {} audio 2 meters away at begining, preceded by the same {} audio that is also 2 meters away',
                                                         'generate {} audio 7 meters away initially, then generate {} audio that is 7 meters away too',
                                                         'create {} audio at a distance of 1 meter preceded by another identical {} audio also 1 meter away',
                                                         'produce {} audio located 7 meter away, followed by the same {} audio at the same 7 meter distance']
        self.textprompt_templates['Compositionality'] = dict()
        self.textprompt_templates['Compositionality']['and'] = ['generate {} audio and {} audio',
                                                                'produce {} audio clip and {} audio clip together',
                                                                'create an audio for {} audio sample and {} audio sample',
                                                                'create an audio containing both {} audio and {} audio',
                                                                'generate an audio tracking for {} audio and {} audio']
        self.textprompt_templates['Compositionality']['or'] = ['generate {} audio or {} audio',
                                                               'create an audio containing either {} audio or {} audio',
                                                               'produce {} audio or {} audio sample',
                                                               'either to generate {} audio or generate {} audio',
                                                               'create {} audio or {} audio, not both']
        self.textprompt_templates['Compositionality']['not'] = ['do not generate {} audio',
                                                               'not create {} audio, but any other audio is fine',
                                                               'please do not generate {} audio sample',
                                                               'skip generating {} audio',
                                                               'avoid generating {} audio']
        self.textprompt_templates['Compositionality']['if_then_else'] = ['generate {} audio if generated {} audio else just generate {} audio',
                                                                 'if generated {} audio, then continue to generate {} audio else to generate {} audio',
                                                                 'just generate {} audio if can not generate {} audio else further genetate {} audio',
                                                                 'produce {} audio if {} audio has been generated, otherwise simply generate {} audio',
                                                                 'generate {} audio if {} audio is present, otherwise simply create {} audio']
    
    # def get_reference_audio(self, audio1, audio2, relation = 'before'):
    def get_reference_audio(self, audio_data_list, relation = 'count'):
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
        if relation == 'count':
            #if Count, each audio has independent start time,
            #DOTO: need to consider the overlap of the audio
            for audio in audio_data_list:
                audio_sec = audio.shape[0] // audio_sr
                audio_start_sec = random.randint(0, max(0, audio_len2gen - audio_sec))
                composite_audio[audio_start_sec*audio_sr: (audio_start_sec + audio_sec)*audio_sr] += audio
            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        
        # temporal order
        if relation in ['before', 'after', 'together']:
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr

            if relation == 'before':
                audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
                audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            elif relation == 'after':
                audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
                audio1_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio1_sec + 1))
            elif relation == 'together':
                audio_start_sec = random.randint(0, max(0, audio_len2gen - max(audio1_sec, audio2_sec)))
                audio1_start_sec = audio_start_sec
                audio2_start_sec = audio_start_sec

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[1]
            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        
        if relation in ['close', 'far']:
            pass

        if relation == 'not':
            #just return an empty audio
            composite_audio = composite_audio.astype(np.float16) * 32767
            return composite_audio

        if relation in ['and', 'or', 'if_then_else']:
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            if relation == 'and':
                audio1_start_sec = random.randint(0, max(0,audio_len2gen - audio1_sec))
                audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[1]
                composite_audio = composite_audio / np.max(np.abs(composite_audio))
                composite_audio = composite_audio.astype(np.float16) * 32767

                return composite_audio
            
            if relation == 'or':
                audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
                audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
                
                composite_audio2 = np.zeros([composite_audio.shape[0]], np.float32)
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
                composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[1]
                composite_audio = composite_audio / np.max(np.abs(composite_audio))
                composite_audio = composite_audio.astype(np.float16) * 32767

                composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
                composite_audio2 = composite_audio2.astype(np.float16) * 32767

                return np.stack([composite_audio, composite_audio2], axis=0)

            if relation == 'if_then_else':
                audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
                audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
                audio3_sec = audio_data_list[2].shape[0] // audio_sr
                audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
                composite_audio2 = np.zeros([composite_audio.shape[0]], np.float32)
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[1]
                composite_audio = composite_audio / np.max(np.abs(composite_audio))
                composite_audio = composite_audio.astype(np.float16) * 32767

                # audio1 = audio_data_list[0]
                # audio1 = audio1 / np.max(np.abs(audio1))
                # audio1 = audio1.astype(np.float16) * 32767
                # wavfile.write('audio1.wav', audio_sr, audio1.astype(np.int16))

                # audio2 = audio_data_list[1]
                # audio2 = audio2 / np.max(np.abs(audio2))
                # audio2 = audio2.astype(np.float16) * 32767
                # wavfile.write('audio2.wav', audio_sr, audio2.astype(np.int16))

                # audio3 = audio_data_list[2]
                # audio3 = audio3 / np.max(np.abs(audio3))
                # audio3 = audio3.astype(np.float16) * 32767
                # wavfile.write('audio3.wav', audio_sr, audio3.astype(np.int16))

                # wavfile.write('composite_audio.wav', audio_sr, composite_audio.astype(np.int16))


                composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] = audio_data_list[2]
                composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
                composite_audio2 = composite_audio2.astype(np.float16) * 32767

                return np.stack([composite_audio, composite_audio2], axis=0)
            
        if relation in ['closefirst', 'farfirst', 'equaldist']:
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio1_sec
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['loudness_reduction_ratio']
            reduced_audio = audio_data_list[0] * loudness_reduction_ratio
            if relation == 'closefirst':
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = reduced_audio
            elif relation == 'farfirst':
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[0]
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = reduced_audio
            elif relation == 'equaldist':
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[0]
            
            # audio_ori = audio_data_list[0]/np.max(np.abs(audio_data_list[0]))
            # audio_ori = audio_ori.astype(np.float16) * 32767
            # wavfile.write('audio_ori.wav', audio_sr, audio_ori.astype(np.int16))
            # audio_reduce = audio_data_list[0] * loudness_reduction_ratio
            # audio_reduce = audio_reduce / np.max(np.abs(audio_reduce))
            # audio_reduce /= 10.0
            # audio_reduce = audio_reduce.astype(np.float16) * 32767
            # # audio_reduce = audio_reduce / np.max(np.abs(audio_reduce))
            # wavfile.write('audio_reduce.wav', audio_sr, audio_reduce.astype(np.int16))

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            # wavfile.write('composite_audio_total.wav', audio_sr, composite_audio.astype(np.int16))

            return composite_audio

    def get_audioname_corpus(self):
        self.audio_category_dict = dict()
        for cate_name in self.audio_category_list:
            self.audio_category_dict[cate_name] = []
            for audio_name in self.config[cate_name]:
                self.audio_category_dict[cate_name].append(audio_name)

    def get_text_prompt(self, audio_name_list, relation):
        """
        preprocess audio name to change '_' to ' '
        """
        audio_name_list = [audio_name.replace('_', ' ') for audio_name in audio_name_list]
        if relation == 'count':
            template = random.choice(self.textprompt_templates['Count'])
            audio_name_str = ''
            for idx, audio_name in enumerate(audio_name_list):
                if idx == len(audio_name_list) - 1:
                    audio_name_str += audio_name
                elif idx == len(audio_name_list) - 2:
                    audio_name_str += audio_name + ' and '
                else:
                    audio_name_str += audio_name + ', '
            return template.format(len(audio_name_list), audio_name_str)
        if relation in ['before','after','together']:
            if relation in ['before', 'after']:
                template = random.choice(self.textprompt_templates['Temporal_Order']['before_or_after'])
                if relation == 'before':
                    return template.format(audio_name_list[0], audio_name_list[1])
                else:
                    return template.format(audio_name_list[1], audio_name_list[0])
            if relation == 'together':
                template = random.choice(self.textprompt_templates['Temporal_Order']['together'])
                return template.format(audio_name_list[0], audio_name_list[1])
        if relation in ['close', 'far']:
            if relation == 'close':
                template = random.choice(self.textprompt_templates['Spatial_Continuality']['close'])
                return template.format(audio_name_list[0])
            if relation == 'far':
                template = random.choice(self.textprompt_templates['Spatial_Continuality']['far'])
                return template.format(audio_name_list[0])
        if relation in ['and', 'or', 'not', 'if_then_else']:
            if relation == 'and':
                template = random.choice(self.textprompt_templates['Compositionality'][relation])
                return template.format(audio_name_list[0], audio_name_list[1])
            if relation == 'or':
                template = random.choice(self.textprompt_templates['Compositionality'][relation])
                return template.format(audio_name_list[0], audio_name_list[1])
            if relation == 'not':
                template = random.choice(self.textprompt_templates['Compositionality'][relation])
                return template.format(audio_name_list[0])
            if relation == 'if_then_else':
                template = random.choice(self.textprompt_templates['Compositionality'][relation])
                return template.format(audio_name_list[0], 
                                       audio_name_list[1], 
                                       audio_name_list[2])
        if relation in ['closefirst', 'farfirst', 'equaldist']:
            if relation == 'closefirst':
                template = random.choice(self.textprompt_templates['Spatial_Distance']['closefirst'])
                try:
                    return template.format(audio_name_list[0], audio_name_list[0])
                except:
                    breakpoint()
            elif relation == 'farfirst':
                template = random.choice(self.textprompt_templates['Spatial_Distance']['closefirst'])
                return template.format(audio_name_list[0], audio_name_list[0])
            elif relation == 'equaldist':
                template = random.choice(self.textprompt_templates['Spatial_Distance']['equaldist'])
                return template.format(audio_name_list[0], audio_name_list[0])

        raise ValueError('Unknown relation')

    def get_N_audios(self,
                     inter_source_category = True, 
                     inter_audio_category = True, 
                     max_audio_len = 5,
                     min_audio_len = 2,
                     audio_num2get = 3):
        """
        inter_source_category: whether the two audios are from different source categories, the source means how the source generated.
        inter_audio_category: whether the two audios are from different audio categories, the audio category means the audio content,
                              the audio content must be in the same source category
        max audio length: 5 seconds
        """
        assert min_audio_len <= max_audio_len
        assert min_audio_len in [1,2,3,4,5]
        assert max_audio_len in [1,2,3,4,5]
        category_list = list(self.audio_category_dict.keys())

        if inter_source_category:
            audio_source_list = random.sample(category_list, audio_num2get)
        else:
            audio_source_tmp = random.choice(category_list)
            audio_source_list = [audio_source_tmp] * audio_num2get

        audio_cate_list = [1]*len(audio_source_list)
        for source_id, source_tmp in enumerate(audio_source_list):
            if audio_cate_list[source_id] != 1:
                continue
            source_tmp_num = 0
            for source_tmp_new in audio_source_list:
                if source_tmp_new == source_tmp:
                    source_tmp_num += 1
            if inter_audio_category:
                sampled_audio_cates = random.sample(self.audio_category_dict[source_tmp], source_tmp_num)
            else:
                sampled_audio_cates = [random.choice(self.audio_category_dict[source_tmp])] * source_tmp_num
            
            assigned_id = 0
            for source_id_new, source_tmp_new in enumerate(audio_source_list):
                if source_tmp_new == source_tmp:
                    audio_cate_list[source_id_new] = sampled_audio_cates[assigned_id]
                    assigned_id += 1

        assert 1 not in audio_cate_list
        audio_filename_list = list()

        for audio_cate, audio_name in zip(audio_source_list, audio_cate_list):
            audio_basefilename_list = list()
            for filename in os.listdir(os.path.join(self.seed_audio_path, audio_cate, audio_name)):
                audio_len_tmp = int(filename.split('_')[-1].split('.')[0])
                if audio_len_tmp <= max_audio_len and audio_len_tmp >= min_audio_len:
                    audio_basefilename_list.append(filename)
            assert len(audio_basefilename_list) > 0

            audio_filename_list.append(random.choice(audio_basefilename_list))

        audio_data_list, audio_name_list, audio_label_list = list(), list(), list()

        '''read audios'''
        for audio_sourcename, audio_catename, audio_filename in zip(audio_source_list, audio_cate_list, audio_filename_list):
            audio, sr = librosa.load(os.path.join(self.seed_audio_path, audio_sourcename, audio_catename, audio_filename),  
                                    sr=self.config['DATA_CREATION_CONFIG']['audio_sr'], mono=True)
            audio_data_list.append(audio)
            audio_name_list.append(audio_catename)
            audio_label_list.append(self.name2label_dict[audio_catename])

        assert len(audio_data_list) == len(audio_name_list) == audio_num2get == len(audio_label_list)

        return audio_data_list, audio_name_list, audio_label_list


    def get_two_audios(self, inter_source_category = True, inter_audio_category = True, 
                       max_audio_len = 5):
        """
        inter_source_category: whether the two audios are from different source categories, the source means how the source generated.
        inter_audio_category: whether the two audios are from different audio categories, the audio category means the audio content,
                              the audio content must be in the same source category
        max audio length: 5 seconds
        """
        category_list = list(self.audio_category_dict.keys())
        if inter_source_category:
            audio_cate1, audio_cate2 = random.sample(category_list, 2)
        else:
            audio_cate1 = random.choice(category_list)
            audio_cate2 = audio_cate1

        if inter_audio_category:
            audio_name1, audio_name2 = random.sample(self.audio_category_dict[audio_cate1], 2)
        else:    
            audio_name1 = random.choice(self.audio_category_dict[audio_cate1])
            audio_name2 = audio_name1

        audio_filename1_list = list()
        for filename in os.listdir(os.path.join(self.seed_audio_path, audio_cate1, audio_name1)):
            audio_len_tmp = int(filename.split('_')[-1].split('.')[0])
            if audio_len_tmp <= max_audio_len:
                audio_filename1_list.append(filename)
        assert len(audio_filename1_list) > 0

        audio_filename2_list = list()
        for filename in os.listdir(os.path.join(self.seed_audio_path, audio_cate2, audio_name2)):
            audio_len_tmp = int(filename.split('_')[-1].split('.')[0])
            if audio_len_tmp <= max_audio_len:
                audio_filename2_list.append(filename)
        assert len(audio_filename2_list) > 0

        audio_filename1 = random.choice(audio_filename1_list)
        audio_filename2 = random.choice(audio_filename2_list)

        '''read two audios'''
        audio1, sr = librosa.load(os.path.join(self.seed_audio_path, audio_cate1, audio_name1, audio_filename1),  
                                  sr=self.target_sr, mono=True)
        audio2, sr = librosa.load(os.path.join(self.seed_audio_path, audio_cate2, audio_name2, audio_filename2),  
                                  sr=self.target_sr, mono=True)

        return audio_name1, audio_name2, audio1, audio2
    
    def get_textprompt_and_audio(self):
        output_dir = self.config['DATA_CREATION_CONFIG']['save_dir']
        os.makedirs(output_dir, exist_ok=True) if not os.path.exists(output_dir) else None
        relations2create = self.config['DATA_CREATION_CONFIG']['relations2create']
        assert len(relations2create) > 0
        data_dict = dict()
        data_dict['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_dict['author'] = 'Yuhang He'
        for task_name in relations2create:
            print('generating for main category: {}'.format(task_name))
            data_dict[task_name] = dict()
            num2gen = self.config['DATA_CREATION_CONFIG']['each_relation_num2gen']
            min_audio_num = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['min_count']
            max_audio_num = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['max_count']
            inter_source = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['inter_source']
            inter_audio = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['inter_audio']
            max_audio_len = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['max_audio_len']
            min_audio_len = self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['min_audio_len']
            for sub_relation in self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['relations']:
                print('generating for sub category: {}'.format(sub_relation))
                data_dict[task_name][sub_relation] = list()
                text_prompt_list = list()
                for num_id in range(num2gen):
                    # get audios, audio names, audio labels
                    # random.randint(min_audio_num, max_audio_num), both included
                    # sub_relation = random.choice(self.config['DATA_CREATION_CONFIG']['{}_Config'.format(task_name)]['relations'])
                    if sub_relation == 'count':
                        audio_num2get=random.randint(min_audio_num, max_audio_num)
                    elif sub_relation == 'if_then_else':
                        audio_num2get = 3
                    elif sub_relation in ['not', 'farfirst', 'closefirst', 'equaldist']:
                        audio_num2get = 1
                    else:
                        audio_num2get = 2

                    audio_data_list, audio_name_list, audio_label_list = self.get_N_audios(inter_source_category=inter_source,
                                                                                        inter_audio_category=inter_audio,
                                                                                        max_audio_len=max_audio_len,
                                                                                        min_audio_len=min_audio_len,
                                                                                        audio_num2get=audio_num2get)
                    
                    reference_audio = self.get_reference_audio(audio_data_list, relation=sub_relation)

                    textprompt_try_times = 1000
                    temp_try_times = 0
                    while temp_try_times < textprompt_try_times:
                        text_prompt = self.get_text_prompt(audio_name_list, relation=sub_relation)
                        if text_prompt not in text_prompt_list:
                            break
                        temp_try_times += 1

                    reference_audio_savename = os.path.join(output_dir, '{}_refaudio_{}_{}.wav'.format(task_name, 
                                                                                                    num_id,
                                                                                                    text_prompt.replace(' ', '_'.strip())))
                    two_ref_audios = False
                    if len(reference_audio.shape) == 1:
                        wavfile.write(reference_audio_savename, 
                                    self.config['DATA_CREATION_CONFIG']['audio_sr'], 
                                    reference_audio.astype(np.int16))
                    else:
                        two_ref_audios = True
                        two_ref_basenames = list()
                        for audio_id in range(reference_audio.shape[0]):
                            ref_basename_tmp = os.path.basename(reference_audio_savename.replace('.wav', '_{}.wav'.format(audio_id)))
                            two_ref_basenames.append(ref_basename_tmp)
                            wavfile.write(reference_audio_savename.replace('.wav', '_{}.wav'.format(audio_id)), 
                                        self.config['DATA_CREATION_CONFIG']['audio_sr'], 
                                        reference_audio[audio_id,:].astype(np.int16))

                    one_data_dict = dict()
                    one_data_dict['text_prompt'] = text_prompt
                    if two_ref_audios:
                        one_data_dict['reference_audio'] = two_ref_basenames
                    else:
                        one_data_dict['reference_audio'] = [os.path.basename(reference_audio_savename)]
                    one_data_dict['audio_name_list'] = audio_name_list
                    one_data_dict['audio_label_list'] = audio_label_list
                    one_data_dict['sub_relation'] = sub_relation

                    data_dict[task_name][sub_relation].append(one_data_dict)

                    text_prompt_list.append(text_prompt)

                with open(os.path.join(output_dir, '{}_{}_text_prompt.txt'.format(task_name, sub_relation)), 'w') as f:
                    for text_prompt in text_prompt_list:
                        f.writelines(text_prompt + '\n')

        # with open(text_prompt_filename, 'w') as f:
        with open(os.path.join(output_dir, 'data_dict.pkl'), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Done!')

if __name__ == '__main__':
    yaml_config_filename = 'tta_datagen_config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        config = yaml.safe_load(f)
    text_audio_creator = TextAudioCreator(config)

    text_audio_creator.get_textprompt_and_audio()