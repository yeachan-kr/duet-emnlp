from util import save_json, load_json, save_pkl, load_pkl, makedir, parse_args
from torch.utils.data import Dataset
import pandas as pd
import pdb
from pprint import pprint


class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        '''
        num_examples_to_run < 0: run all
        '''
        self.args = args
        self.narrations = self.get_descriptions()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        self.durations = load_json(args.duration_path)  # uid --> float
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run):
        if quids_to_exclude is not None:
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('uid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = '.\n'.join([f'{int(i*self.args.caption_every)}: {cap}' for i, cap in enumerate(narr[::self.args.caption_every])])
        return narr

    def get_anno(self):
        anno = load_json(self.args.anno_path)  # uid --> {question, option 0, option 1, option 2, option 3, option 4, truth (optional)}
        return anno

    def build(self):
        subset = None
        if self.args.subvideo_path is not None:
            subset = load_json(self.args.subvideo_path)
        frame_list = {}

        if subset is not None:
            for ss in subset:
                fl = ss['sorted_values']
                fl = sorted(fl)

                uid = ss['name']
                frame_list[uid] = fl

        data = []
        for uid, item in self.anno.items():
            # if type(self.narrations['data'][uid]) == type({}):
            if 'data' in self.narrations.keys():
                if uid not in self.narrations['data']:
                    continue
                narration = self.narrations['data'][uid]['narration']
                summ = self.narrations['data'][uid]['response'].strip()

                duration = self.narrations['data'][uid]['duration']
                if 'captions' in self.narrations['data'][uid].keys() and type(self.narrations['data'][uid]['pred']) == type([]):
                    pred = sorted(self.narrations['data'][uid]['pred'])
                    frames = [self.narrations['data'][uid]['frames'][x] for x in pred]
                    captions = '\n'.join([f'{frames[i]}: ' + self.narrations['data'][uid]['captions'][i] for i in range(len(frames))])
                elif 'captions' in self.narrations['data'][uid].keys() and self.narrations is not None:
                    captions = self.narrations['data'][uid]['captions']
                else:
                    captions = None

            else:
                if uid not in self.narrations:
                    continue
                # tmp_narration = [f'{x}: ' + self.narrations[uid][x].replace('#C ', '').replace('#c ', '').replace('#O ', '').replace('#o ', '').strip() for x in frame_list[uid]] if len(frame_list) != 0 else self.narrations[uid]
                tmp_narration = [self.narrations[uid][x].strip() for x in frame_list[uid]] if len(frame_list) != 0 else self.narrations[uid]
                narration = self.format_narration(tmp_narration) #+ '\n\n' + top_k_captions
            
                duration = len(tmp_narration)
                summ = None
                captions = None
            
            # narration = self.format_narration(self.narrations[uid])
            question = item['question']
            choices = [item['option 0'], item['option 1'], item['option 2'], item['option 3'], item['option 4']] 
            truth = item['truth'] if 'truth' in item else -1
            # duration = int(self.durations[uid])
            frames = frame_list[uid] if len(frame_list) != 0 else None

            data.append({
                'uid': uid,
                'narration': narration,
                'question': question,
                'optionA': choices[0],
                'optionB': choices[1],
                'optionC': choices[2],
                'optionD': choices[3],
                'optionE': choices[4],
                'truth': truth,
                'duration': duration,
                'frames': frames,
                'summary': summ,
                'captions': captions,
            })
        return data


class NextDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('quid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, dict):
            narr = list(narr.items())
            narr.sort(key=lambda x: int(x[0]))
            narr = [el[1] for el in narr]
        narr = '.\n'.join([f'{int(i*self.args.caption_every)}: {cap}' for i, cap in enumerate(narr[::self.args.caption_every])])
        return narr

    def get_anno(self):
        return pd.read_csv(self.args.anno_path)  # video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
         
    def build(self):
        subset = None
        if self.args.subvideo_path is not None:
            subset = load_json(self.args.subvideo_path)
        frame_list = {}
        
        if subset is not None:
            for ss in subset:
                fl = ss['sorted_values']
                fl = sorted(fl)

                uid = ss['name']
                frame_list[uid] = fl

        data = []
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            uid = str(row['video'])

            qid, q_type = row['qid'], row['type']
            quid = f'{uid}_{qid}'

            if 'data' in self.narrations.keys():
                if quid not in self.narrations['data']:
                    continue
                narration = self.narrations['data'][quid]['narration']
                summ = self.narrations['data'][quid]['response'].replace('[SUMMARY]\n', '').strip()

                duration = self.narrations['data'][quid]['duration']
                if 'captions' in self.narrations['data'][quid].keys() and type(self.narrations['data'][quid]['pred']) == type([]):
                    pred = sorted(self.narrations['data'][quid]['pred'])
                    frames = [self.narrations['data'][quid]['frames'][x] for x in pred]
                    captions = '\n'.join([f'{frames[i]}: ' + self.narrations['data'][quid]['captions'][i] for i in range(len(frames))])
                elif 'captions' in self.narrations['data'][quid].keys() and self.narrations is not None:
                    captions = self.narrations['data'][quid]['captions']
                else:
                    captions = None

            else:
                if uid not in self.narrations:
                    continue
                # tmp_narration = [f'{x}: ' + self.narrations[uid][x].replace('#C ', '').replace('#c ', '').replace('#O ', '').replace('#o ', '').strip() for x in frame_list[uid]] if len(frame_list) != 0 else self.narrations[uid]
                tmp_narration = [self.narrations[uid][x].strip() for x in frame_list[uid]] if len(frame_list) != 0 else self.narrations[uid]
                narration = self.format_narration(tmp_narration) #+ '\n\n' + top_k_captions
            
                duration = len(tmp_narration)
                summ = None
                captions = None

            # duration = int(self.durations[uid])
            frames = frame_list[uid] if len(frame_list) != 0 else None

            question, truth = row['question'], row['answer']
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            frames = frame_list[uid] if len(frame_list) != 0 else None

            data.append({
                'quid': quid,
                'uid': uid,
                'qid': qid,
                'q_type': q_type,
                'narration': narration,
                'question': question,
                'optionA': choices[0],
                'optionB': choices[1],
                'optionC': choices[2],
                'optionD': choices[3],
                'optionE': choices[4],
                'truth': truth,
                'duration': duration,
                'frames': frames,
                'summary': summ,
                'captions': captions,
            })
        return data


class VideoMMEDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('q_uid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = '.\n'.join([f'{int(i*self.args.caption_every)}: {cap}' for i, cap in enumerate(narr[::self.args.caption_every])])
        return narr

    def get_anno(self):
        anno = load_json(self.args.anno_path) 
        return anno

    def build(self):
        data = []
        for quid, item in self.anno.items():
            if item['videoID'] not in self.narrations:
                continue
            narration = self.format_narration(self.narrations[item['videoID']])
            question = item['question']
            choices = [item['option 0'], item['option 1'], item['option 2'], item['option 3']] 
            truth = item['truth']
            data.append({
                'q_uid': quid,
                'narration': narration,
                'question': question,
                'optionA': choices[0],
                'optionB': choices[1],
                'optionC': choices[2],
                'optionD': choices[3],
                'truth': truth,
            })
        return data
    
    # def build(self):
    #     data = []
    #     for video_data in self.anno:
    #         narr_id = video_data['url'].split('?v=')[-1]
    #         if narr_id not in self.narrations:
    #             continue
    #         narration = self.format_narration(self.narrations[narr_id])
    #         for question_info in video_data['questions']:
    #             choices = question_info['choices']
    #             info = question_info
    #             info.update({
    #                 'question_id': question_info['question_id'],
    #                 'video_id': video_data['video_id'],
    #                 'narration': narration,
    #                 'question': question_info['question'],
    #                 'optionA': choices[0],
    #                 'optionB': choices[1],
    #                 'optionC': choices[2],
    #                 'optionD': choices[3],
    #                 'answer': question_info['answer'],
    #             })
    #             data.append(info)
    #     return data
    

def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1):
    if args.dataset == 'egoschema':
        return EgoSchemaDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset in {'nextqa', 'nextgqa', 'intentqa'}:
        return NextDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset == 'videomme':
        return VideoMMEDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    print(len(dataset))
    # for data in dataset:
    #     pprint(data)
