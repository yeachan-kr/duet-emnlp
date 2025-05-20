from string import Template
import random
import re

def first_char_as_answer(res):
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    if len(res) == 0:
        return -1
    if res[0] in mapping:
        return mapping[res[0]]
    return -1

def first_char_as_answer_raw(res):
    candidates = ['A', 'B', 'C', 'D']
    if len(res) == 0:
        return random.choice(candidates)
    if res[0] in candidates:
        return res[0]
    return random.choice(candidates)

def identity(res):
    return res

def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            pred_letter = res[anchor_index+len(anchor)]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred
    return f

def get_intervals_as_list(text):
    text = text.split('.')[0]
    text = text.strip()
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index+1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        try:
            start, end = int(start_text), int(end_text)
        except:
            start, end = 0, 1
        intervals.append([start, end])
    return intervals

def update_pred_response(text):
    prediction_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        # print("`item` is", item)
    response = text
    # print("response",response)

    prediction_match = re.search(r"prediction: ([A-E])", response, re.IGNORECASE)
    explanation = re.search(r"explanation: (\d+)", response, re.IGNORECASE)
    if prediction_match:
        # Update 'pred' with the numerical value of the prediction
        pred = prediction_map[prediction_match.group(1).upper()]
    else:
        pred = 0
    return pred


def get_frame(text):
    response = text
    # print("response",response)

    frames_match = re.search(r"FRAMES:\s*([\d,\s]+)", response)

    if frames_match:
        try:
            frames_list = [int(frame.strip()) for frame in frames_match.group(1).split(",")]
        except:
            frames_list = []
    else:
        frames_list = []
    return frames_list



def update_pred_response2(text):
    response = text

    prediction_match = re.search(r"ANSWER:\s*(\d+)", response, re.IGNORECASE)

    if prediction_match:
        pred = int(prediction_match.group(1))
    else:
        pred = 0  # 기본값 (없을 때)

    return pred

def update_pred_response3(text):
    response = text

    prediction_match = re.search(r"FINAL_ANSWER:\s*(\d+)", response, re.IGNORECASE)

    if prediction_match:
        pred = int(prediction_match.group(1))
    else:
        pred = 0  # 기본값 (없을 때)

    return pred


class PromptTemplate(object):
    def __init__(self, head, template, post_process_fn):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        # match variable names: duration, narration, question, optionA, optionB, optionC, optionD, optionE, num_words
        prompt_filled = []
        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled


class PromptFactory(object):
    def __init__(self):
        self.prompt_templates = self.build()
    
    def build(self):
        prompt_templates = {}

        # egoschema QA (raw captions as input)
        prompt_templates['qa_standard'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n\n In your response, the first character should be your answer to this multiple choice question."),
            ],
            post_process_fn = first_char_as_answer
        )
  
        # egoschema QA (GT as input)
        prompt_templates['qa_standard_gt'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n\n In your response, the first character should be your answer to this multiple choice question."),
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema QA (raw captions as input) few-shot
        prompt_templates['qa_standard_fewshot'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. You are also given a question and five potential choices. Your task is to answer with a correct choice based on the video descriptions. \nHere are a few examples. \n${examplars}\n\n Now answer this question.\nDescriptions: ${narration}.\n Question: ${question}\n A: ${optionA}.\n B: ${optionB}.\n C: ${optionC}.\n D: ${optionD}.\n E: ${optionE}.\n Answer: "),
            ],
            post_process_fn = first_char_as_answer
        )
        
        # egoschema QA (summary as input)
        prompt_templates['qa_sum'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n\nIn your response, the first character should be your answer to this multiple choice question."),
            ],
            post_process_fn = first_char_as_answer
        )

        prompt_templates['qa_sum_hcqa'] = PromptTemplate(
            head = "You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION], and [REASON]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.",
            template = [
                Template('''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents a caption of a 1s video clip, each caption is separated by a semicolon, with a total of $duration lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, you need to get it from the [CAPTION].
Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give right number of [OPTION] as [ANSWER]. Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even given a low confidence.
[CAPTION]
$narration
[SUMMARY]
$summary
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE
                         
You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION, SUMMARY, and CAPTION.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()

                ),
            ],
            post_process_fn = update_pred_response2
        )

        prompt_templates['qa_sum_nocap'] = PromptTemplate(
    head = "You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [SUMMARY], [QUESTION], and [REASON]. The [SUMMARY] is an overall description of a 180-second first-person perspective video, generated based on textual descriptions (captions) of individual 1-second clips.",
    template = [
        Template('''
[SUMMARY]: An overall description of a 180-second first-person perspective video, generated based on detailed textual captions describing each 1-second clip.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, assume the missing details from its source captions.

Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give the right number of [OPTION] as [ANSWER]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even if your confidence is low.

[SUMMARY]
$summary
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and SUMMARY.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = update_pred_response2
        )

        prompt_templates['frame_selection'] = PromptTemplate(
    head = "You are a visual question answering expert. You can select the most relevant frames to infer the correct answer from the five options of [OPTION], based on the [CAPTION], [QUESTION], and [REASON]. The [CAPTION] consists of textual descriptions of the video as seen from a first-person perspective, with each description labeled by its corresponding frame number.",
    template = [
        Template('''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behavior. Each line represents a caption of a 1s video clip, each caption is separated by a semicolon, with a total of $duration lines describing 180 seconds of video. Each caption begins with its corresponding frame number.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [CAPTION], reason step by step to determine which frames contain the necessary information to infer the correct answer. If direct information is missing, select frames that provide the closest possible context.

Now, you should first make a [REASON] based on [QUESTION] and [CAPTION], then select the most relevant frames as [FRAMES]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in selecting the correct frames, on a scale from 1 to 5. You MUST select at least one frame, even if confidence is low.

[CAPTION]
$narration
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the FRAMES as a comma-separated list of frame numbers. DO NOT provide answers in any other format.  
If unsure, you must still pick at least one frame number and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and CAPTION.]  
FRAMES: [Comma-separated list of frame numbers.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = get_frame
        )

        prompt_templates['qa_nosum'] = PromptTemplate(
    head = "You are a visual question answering expert. You can infer the correct answer from five options of [OPTION] based on the [CAPTION], [DETAILED_CAPTION], [QUESTION], and [REASON]. The [CAPTION] consists of textual descriptions of the video as seen from a first-person perspective, with each description labeled by its corresponding frame number. The [DETAILED_CAPTION] contains detailed descriptions of the most relevant frames providing additional context related to the question.",
    template = [
        Template('''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behavior. Each line represents a caption of a 1s video clip, each caption is separated by a semicolon, with a total of $duration lines describing 180 seconds of video. Each caption begins with its corresponding frame number.
[DETAILED_CAPTION]: Detailed descriptions of the most relevant frames providing additional context related to the question.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [DETAILED_CAPTION], reason step by step to determine the correct answer. If necessary, refer back to the [CAPTION] for additional context.

Now, you should first make a [REASON] based on [QUESTION] and [DETAILED_CAPTION], then give the right number of [OPTION] as [ANSWER]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even if confidence is low.

[CAPTION]
$narration
[DETAILED_CAPTION]
$captions
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and DETAILED_CAPTION.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = update_pred_response2
        )

        # egoschema sum (standard)
        prompt_templates['sum_standard'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n Please give me a $num_words words summary.')
            ],
            post_process_fn = identity
        )

        prompt_templates['sum_hcqa'] = PromptTemplate(
            head = "You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.",
            template = [
                Template(
'''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents a caption of a ${clip_length}s video clip, each caption is separated by a semicolon, with a total of $duration lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
[CAPTION]
#C C picks a bottle of juice; #C C picks a bottle of juice from the fridge with her right hand; #C C moves a bottle of water in the refrigerator with her right hand; #C C pushes the bottle in the fridge; #C C picks a bottle of water from the fridge with her right hand
#C C opens a refrigerator with her left hand.; #C C places the tin in the fridge; #O The man X opens the fridge with his right hand.; #O The man X puts the container in the sink with his right hand; #C C looks at man X
#C C places the can in her right hand into the fridge.; #C C drops the carton of milk on the fridge with her left hand; #C C moves a container in the fridge with her left hand. ; #C C picks a bottle of juice; #C C puts the container in the fridge 
#C C picks a container from the refrigerator with her left hand. ; #C C places the plastic container on the fridge with her right hand.; #C C puts the pack in her left hand into the fridge.; #C C puts the container in the fridge with her left hand.; #C C drops the jar of fruit juice in the fridge with her right hand
#C C picks a container from the fridge; #C C holds a tin; #C C picks a plate of food from the refrigerator with her right hand; #C C picks a plastic container from the refrigerator with her right hand.; #C C picks a plastic container from the fridge with her right hand.
#C C looks around; #C C stares a; #C C looks around; #C C looks around; #C C looks around
#C C looks at the fridge ; #C C closes the fridge with his left hand; #C C looks inside the fridge; #C C looks around the fridge; #C C looks inside the fridge
#C C closes the fridge; #C C walks towards the kitchen; #C C walks to the kitchen sink; #C C opens the fridge; #C C closes the cabinet
#C C looks around the room; #C C puts the bottle in the kitchen cabinet with her right hand; #C C looks around; #C C picks the shopping from the shelves; #C C turns towards the shelf.
#C C closes the door with her right hand.; #C C closes the closet door; #C C looks around ; #C C closes the cabinet; #C C opens the door of the kitchen cabinet with her right hand
#O The man X opens the refrigerator door with his right hand; #O The man X closes the refrigerator with his right hand.; #O The man X closes the fridge door with his right hand; #O The man X opens a refrigerator with his right hand.; #C C interacts with the man X.
#O The man X closes the fridge with his right hand.; #O The man X holds the fridge door with his right hand.; #O The man X opens the fridge with his right hand.; #O The man X picks a plate from the kitchen counter with his right hand; #O the man X closes the fridge with his right hand.
#C C puts the meat into the cabinet; #C C opens a cabinet with his right hand; #C C opens a cabinet door with her right hand.; #C C opens the cupboard with her right hand.; #C C picks a plate from the kitchen cabinet with his right hand
#C C picks a bag from the floor with her left hand; #C C picks a paper bag from the floor; #C C holds the shopping bag; #C C moves the bag on the floor with her right hand; #C C picks up the shopping bag
#C C picks a bag from the floor ; #C C picks the paper bag; #C C puts the piece of cucumber on the floor; #C C takes the blue bag from the floor; #C C picks up the bag of chips from the floor with her right hand
#C C looks around the house; #C C walks to the kitchen; #C C walks around; #C C picks a paper bag; #C C walks around
#O the man A picks a phone from the kitchen table with his left hand.; #C C interacts with the man X.; #O the man A cleans a cooker with the rag in his right hand.; #O person X operates the cooker; #O The man X holds a phone in his left hand.
#C C looks around ; #C C moves towards the kitchen sink.; #C C picks the plastic bowl from the kitchen counter with her right hand; #C C looks around; #C C looks around
#C C walks around; #C C walks to a kitchen; #C C walks around; #C C walks to the kitchen; #C C walks towards the kitchen
#C C looks around; #C C looks around; #C C interacts with the woman; #C C looks around; #C C looks around 
#C C picks the paper from the table with her right hand.; #C C picks a paper from the table with her right hand.; #C C picks a paper from the dining table with her left hand; #C C picks a paper on the table with her left hand.; #C C picks a paper from the table
#C C moves a paper on the table with her right hand.; #C C picks up a paper on the table with her left hand; #C C moves a wire with her right hand.; #C C picks up a book from a table with her right hand; #C C picks a paper from the table with her right hand.
#C C opens a purse on the table with her right hand.; #C C picks up a cable from the table with her left hand; #C C picks a hair band from a table with her right hand; #C C picks the earphones from the table with her left hand; #C C picks a pen
#C C picks a paper from the table with her right hand.; #C C drops the paper bag on the table with her right hand; #C C puts a hand towel on the table; #C C picks the paper; #C C drops the envelope on the table with her right hand.
#O The man X opens a kitchen cabinet with his left hand.; #O The man X drops a plastic bag on the kitchen countertop with his right hand; #O The man X picks a bottle from the countertop with his right hand.; #C C walks around; #O The man X picks the bottle of spice from the counter with his right hand.
#O The man M cleans the plate with the towel in his right hand; #O man X cleans the plate with the cloth; #O man X wipes his hands with the hand towel; #O The man M holds the cloth with both hands.; #C C interacts with the man X.
#O The man B picks a towel from the table with his left hand; #O man X talks to C ; #C C looks around; #O The man A cleans his hands with the handkerchief. ; #C C converses with the man X.
#C C picks a paper from a table with her right hand.; #C C picks up a paper from the table with her right hand.; #C C picks up a paper from a table with her right hand.; #C C takes a paper from the table with her right hand.; #C C picks up a paper on the dining table with her right hand
#C C opens a bag with both hands.; #C C opens the bag; #C C picks the nylon bag from the table with both hands; #C C lifts a leash with her right hand.; #C C moves the dog leash
#C C passes the rope from her left hand to her right hand; #C C converses with the man A.; #C C holds the rope with both hands.; #C C walks to the table; #C C walks around the house
#C C walks to the bedroom; #C C walks in the house; #C C moves to the bed; #C C walks around the house; #C C walks in the room
#C C holds the nylon in his right hand; #C C looks around; #C C looks around; #C C walks to the table.; #C C walks to the table
#C C holds the bag of grapes in both hands.; #C C holds the bag of carrots with both hands.; #C C holds the bag with both hands.; #C C holds the nylon bag with both hands.; #C C holds the bag of grapes with her right hand
#C C walks around the; #C C looks around; #C C looks around; #C C looks around; #C C talks to the
#C C walks towards a kitchen.; #C C walks towards a living room; #C C walks into the living room; #C C walks in the house; #C C walks around
#C C walks into the room.; #C C walks into the sitting room from the corridor; #C C walks around the house.; #C C walks into the sitting room.; #C C walks into the bedroom from the room.
#C C walks around; #C C looks around ; #C C walks around; #C C walks into the room; #C C walks around
#C C walks into the room.; #C C walks to the bedroom from the kitchen; #C C walks into a room.; #C C walks into the bedroom.; #C C walks into the living room.
#C C walks around the house ; #C C walks towards a dining area.; #C C walks towards the dining table.; #C C walks to the table in the living room; #C C walks to the living room.
#C C walks to the kitchen from the dining area; #C C walks around ; #C C looks around; #C C walks towards the table.; #C C walks towards a table.
#C C walks around the house.; #C C walks around the house; #C C walks into a living room; #C C walks towards a sitting room; #C C walks in the room 
#C C closes the cabinet; #C C talks to a man X ; #C C walks towards the door; #C C closes the door; #C C walks towards the door of the storage room with the man X
#C C walks around the room; #C C walks around; #C C opens a door with his right hand; #C C walks into the sitting room.; #C C walks around the room
#C C walks towards the sitting room from the kitchen; #C C walks around; #C C walks into the apartment.; #C C walks towards the door; #C C walks to the kitchen from the dining room
#C C walks towards the table; #C C walks towards the dining table; #C C walks towards the dining area; #C C walks into the kitchen.; #C C walks to the kitchen.
[SUMMARY]
C picks a bottle from the fridge with her right hand, then C places the tin in the fridge. C picks a container from the fridge, then closes the fridge with her right hand. C puts the bottle in the kitchen cabinet with her right hand. The man X closes the fridge door with his right hand, then C picks a paper bag from the floor and puts the piece of cucumber on the floor. Then C picks the paper from the table with her right hand, then walks to the kitchen and interacts with the man X. C picks a hair band from a table with her right hand and moves to the bed. C opens a door with his right hand and walks around the house. C walks to the table in the living room, then opens a door with his right hand. Finally, C walks to the kitchen from the dining room.

Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
$narration
[SUMMARY]
'''.strip())
            ],
            post_process_fn = identity
        )

        prompt_templates['qa_sum_nocap_intendqa'] = PromptTemplate(
    head = "You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [SUMMARY], [QUESTION], and [REASON]. The [SUMMARY] is an overall description of a 180-second first-person perspective video, generated based on textual descriptions (captions) of individual 1-second clips.",
    template = [
        Template('''
[SUMMARY]: An overall description of a 180-second first-person perspective video, generated based on detailed textual captions describing each 1-second clip.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, assume the missing details from its source captions.

Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give the right number of [OPTION] as [ANSWER]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even if your confidence is low.

[SUMMARY]
$narration
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and SUMMARY.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = update_pred_response2
        )
        prompt_templates['next_sum'] = PromptTemplate(
            head = "You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.",
            template = [
                Template(
'''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents a caption of a ${clip_length}s video clip, each caption is separated by a semicolon, with a total of $duration lines describing the video. At the beginning of each caption.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
[CAPTION]
A woman wearing a hat and a purple shirt is standing in front of a large rock. She is smiling and appears to be enjoying her time outdoors.
A woman wearing a hat and sunglasses is smiling and looking at a large rock.
A woman wearing sunglasses is biting into a rock.
A woman with a hat and sunglasses is sticking her tongue out.
A woman wearing a hat and sunglasses is leaning against a rock, with her mouth open.
A woman wearing a hat and sunglasses is standing in front of a large rock formation. She is wearing a backpack and has a handbag with her.
A woman is standing in front of a large rock, wearing a backpack.
A woman is standing in front of a large rock, smiling and posing for a picture. She is wearing a backpack and sunglasses. 
[SUMMARY]
A woman wearing a hat, sunglasses, and a purple shirt is exploring a large rock formation outdoors. She appears cheerful and is enjoying the experience, smiling and interacting playfully with the rock. At one point, she pretends to bite into the rock and sticks her tongue out, adding a humorous element to her behavior. She leans against the rock and poses for pictures, carrying a backpack and a handbag. Throughout the video, she continues to smile and engage with the surroundings, clearly enjoying her time in nature.

Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
$narration
[SUMMARY]
'''.strip())
            ],
            post_process_fn = identity
        )

        prompt_templates['qa_next_global'] = PromptTemplate(
    head = "You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [SUMMARY], [QUESTION], and [REASON]. The [SUMMARY] is an overall description of a video, generated based on textual descriptions (captions) of individual 1-second clips.",
    template = [
        Template('''
[SUMMARY]: An overall description of a video, generated based on detailed textual captions describing each 1-second clip.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, assume the missing details from its source captions.

Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give the right number of [OPTION] as [ANSWER]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even if your confidence is low.

[SUMMARY]
$summary
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and SUMMARY.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = update_pred_response2
        )

        prompt_templates['frame_selection_next'] = PromptTemplate(
    head = "You are a visual question answering expert. You can select the most relevant frames to infer the correct answer from the five options of [OPTION], based on the [CAPTION], [QUESTION], and [REASON]. The [CAPTION] consists of textual descriptions of the video as seen from a first-person perspective, with each description labeled by its corresponding frame number.",
    template = [
        Template('''
[CAPTION]: Textual descriptions of videos, about natural human activities and behavior. Each line represents a caption of a 1s video clip, each caption is separated by a semicolon, with a total of $duration lines describing the video. Each caption begins with its corresponding frame number.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [CAPTION], reason step by step to determine which frames contain the necessary information to infer the correct answer. If direct information is missing, select frames that provide the closest possible context.

Now, you should first make a [REASON] based on [QUESTION] and [CAPTION], then select the most relevant frames as [FRAMES]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in selecting the correct frames, on a scale from 1 to 5. You MUST select at least one frame, even if confidence is low.

[CAPTION]
$narration
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the FRAMES as a comma-separated list of frame numbers. DO NOT provide answers in any other format.  
If unsure, you must still pick at least one frame number and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide step-by-step reasoning based on QUESTION and CAPTION.]  
FRAMES: [Comma-separated list of frame numbers.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = get_frame
        )
        
        prompt_templates['qa_next_local'] = PromptTemplate(
    head = "You are a visual question answering expert. You can infer the correct answer from five options of [OPTION] based on the [CAPTION], [DETAILED_CAPTION], [QUESTION], and [REASON]. The [CAPTION] consists of textual descriptions of the video as seen from a first-person perspective, with each description labeled by its corresponding frame number. The [DETAILED_CAPTION] contains detailed descriptions of the most relevant frames providing additional context related to the question.",
    template = [
        Template('''
[CAPTION]: Textual descriptions of videos, about natural human activities and behavior. Each line represents a caption of a 1s video clip, each caption is separated by a semicolon, with a total of $duration lines describing the video. Each caption begins with its corresponding frame number.
[DETAILED_CAPTION]: Detailed descriptions of the most relevant frames providing additional context related to the question.
[QUESTION]: A question about the video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [DETAILED_CAPTION], very short reason step by step to determine the correct answer. If necessary, refer back to the [CAPTION] for additional context.

Now, you should first make a [REASON] based on [QUESTION] and [DETAILED_CAPTION], then give the right number of [OPTION] as [ANSWER]. Additionally, you need to provide [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You MUST answer the question, even if confidence is low.

[CAPTION]
$narration
[DETAILED_CAPTION]
$captions
[QUESTION]
$question
[OPTION]
OPTION 0: $optionA
OPTION 1: $optionB
OPTION 2: $optionC
OPTION 3: $optionD
OPTION 4: $optionE

You MUST provide the ANSWER as a single digit (0-4). DO NOT provide answers in any other format.  
If unsure, you must still pick a numeric ANSWER (0-4) and lower your CONFIDENCE accordingly.

Your final response must strictly follow this format:

REASON: [Provide very short step-by-step reasoning based on QUESTION and DETAILED_CAPTION.]  
ANSWER: [Only one number from 0 to 4.]  
CONFIDENCE: [Confidence level (integer) from 1 (lowest) to 5 (highest).]
'''.strip()
                ),
            ],
            post_process_fn = update_pred_response2
        )

        # egoschema sum (q)
        prompt_templates['sum_q'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions: $narration.\n Please give me a $num_words words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: $question'),
            ],
            post_process_fn = identity
        )

        # egoschema sum (qa)
        prompt_templates['sum_qa'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template('You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n Please give me a $num_words words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: $question\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n Do not answer this question directly. Instead, use the question and choices to guide your summary.')
            ],
            post_process_fn = identity
        )

        # egoschema QA zero-shot-CoT
        prompt_templates['qa_zs-cot'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n Before answering this question, let's think step by step."),
                Template("Please provide a single-letter answer (A, B, C, D, E) to the multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. Your response should only contain one letter.")
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema QA plan-and-solve
        prompt_templates['qa_plansolve'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. Here are the descriptions: $narration.\n You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen from the choices.\n Here is the question: $question.\n Here are the choices.\n A: $optionA\n B: $optionB\n C: $optionC\n D: $optionD\n E: $optionE\n To answer this question, let's first prepare relevant information and decompose it into 3 sub-questions. Then, let's answer the sub-questions one by one. Finally, let's answer the multiple choice question."),
                Template("Please provide a single-letter answer (A, B, C, D, E) to the multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. Your response should only contain one letter.")
            ],
            post_process_fn = first_char_as_answer
        )

        # next-qa QA, intentQA QA
        prompt_templates['qa_next'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. If you are not sure, answer with the most likely answer. You are given some language descriptions of a first person view video. The video is ${fps} FPS and the descriptions are the captions every 2 frames. Each caption starts with the frame number.\nHere are the descriptions:\n$narration\n Here is the question: $question?\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n\nIn your response, the first character should be your answer to this multiple choice question."),
            ],
            post_process_fn = first_char_as_answer
        )

        # next-gqa GQA
        prompt_templates['gqa'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("I will provide video descriptions and one question about the video. The video is ${fps} FPS and the descriptions are the captions every ${caption_every} frames. Each caption starts with the frame number.\n To answer this question, what is the minimun frame interval to check?\n Follow this format: [frame_start_index, frame_end_index]. Do not provide any explanation.\n Here are the descriptions:\n$narration\n Here is the question: $question?\n Please follow the output format as follows:\n #Example1: [5, 19]\n #Example2: [30, 60]\n #Example3: [1, 10] and [50, 60]"),
            ],
            post_process_fn = get_intervals_as_list
        )

        # egoschema QA llama-2
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        anchor = 'The most correct answer is ('
        prompt_templates['qa_standard_llama'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly." + E_SYS + 'Here are the descriptions:\n$narration\n Here is the question: $question.\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n' + E_INST + anchor),
            ],
            post_process_fn = first_char_after_anchor(anchor)
        )

        # egoschema QA llama-3
        anchor = 'The most correct answer is ('
        prompt_templates['qa_standard_llama-3'] = PromptTemplate(
            head = "",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language descriptions of a first person view video. The video is $duration seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly." + "\n\n" + 'Here are the descriptions:\n$narration\n Here is the question: $question.\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n' + anchor),
            ],
            post_process_fn = first_char_after_anchor(anchor)
        )

        # videoMME QA GPT
        prompt_templates['qa_videomme'] = PromptTemplate(
            head = "You are a helpful expert in video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, or D). You must not provide any other response or explanation. If you are not sure, answer with the most likely answer. I will provide video descriptions and one question about the video. The video is ${fps} FPS and the descriptions are the captions every ${caption_every} frames. Each caption starts with the frame number. \nHere are the descriptions:\n$narration\n Here is the question: $question?\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n\nIn your response, the first character should be your answer to this multiple choice question."),
            ],
            post_process_fn = first_char_as_answer
        )
        return prompt_templates
    

    def get(self, prompt_type):
        return self.prompt_templates[prompt_type]
