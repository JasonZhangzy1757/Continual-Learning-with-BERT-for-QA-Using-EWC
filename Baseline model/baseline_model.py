import json, collections, os, random, glob, math, string, re, torch
import numpy as np
from tqdm import trange, tqdm_notebook as tqdm 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import WEIGHTS_NAME, BertConfig, BertForQuestionAnswering, BertTokenizer, XLMConfig, XLMForQuestionAnswering, XLMTokenizer, XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer, DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer, AdamW, BasicTokenizer
from transformers.tokenization_bert import whitespace_tokenize
from transformers import get_linear_schedule_with_warmup 

class SquadExample(object):
    """
    A single training/test example for the SQuAD dataset.
    For those examples without an answer, the start and end position are -1.
    """
    
    def __init__(self, 
                 qas_id, 
                 question_text, 
                 doc_tokens, 
                 orig_answer_text=None, 
                 start_position=None, 
                 end_position=None, 
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

    
import json
from transformers.tokenization_bert import whitespace_tokenize

def read_squad_examples(input_file,
                       is_training, 
                       version_2_with_negative):
    """
    Read a SQuAD JSON file into a list of `SquadExample`s.
    """
    
    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
        
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    
    examples = []
    
    for entry in input_data:
        # print(entry)
        # {
        #   "title": "Economy_of_Greece",
        #   "paragraph": [
        #     {
        #       "qas": [
        #         {
        #           "question": "What type of country is Greece?",
        #           "id": "5731e7900fdd8d15006c662d"
        #           "answers": [
        #             {
        #               "text": "developed",
        #               "answer_start": 12
        #             }
        #           ],
        #           "is_impossible": False
        #         },
        #         {
        #            "question": ...
        #             ...
        #         }
        #       ],
        #       "context": "Greece is a developed country with an economy based on the service (82.8%) and industrial sectors (13.3%). The agricultural sector contributed 3.9% of national economic output in 2015. Important Greek industries include tourism and shipping. With 18 million international tourists in 2013, Greece was the 7th most visited country in the European Union and 16th in the world. The Greek Merchant Navy is the largest in the world, with Greek-owned vessels accounting for 15% of global deadweight tonnage as of 2013. The increased demand for international maritime transportation between Greece and Asia has resulted in unprecedented investment in the shipping industry."
        #     }
        #   ]
        # }
        # {
        #   "title": ...
        #   ...
        # }
        for paragraph in entry["paragraphs"]:
            
            # Extract words in `context` and put them into `doc_tokens`, also record their positions in `char_to_word_offset`   
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            
            for c in paragraph_text:
                
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                # print(doc_tokens)
                # ['Beyoncé', 'Giselle', 'Knowles-C']
                char_to_word_offset.append(len(doc_tokens) - 1)
                # print(char_to_word_offset)
                # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                #  B  e  y  o  n  c  é     G  i  s  e  l  l  e     K  n  o  w  l  e  s  -  C 
        
            for qa in paragraph["qas"]:
                
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                
                # The reason using `is_training` is because train and dev sets are different 
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError("For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        
                        # Only add answers where the text can be exactly recovered from the document. If this CAN'T happen
                        # it's likely due to weird Unicode stuff so we will just skip the example. Note that this means for
                        # training mode, every example is NOT guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue
                    else:
                        # No answer
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                        
                # Create a `SquadExample` instance
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                # repr(example)
                # [qas_id: 56be85543aeaaa14008c9063, 
                #  question_text: When did Beyonce start becoming popular?, 
                #  doc_tokens: [Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".],
                #  start_position: 39,
                #  end_position: 42,
                #  qas_id: ...,
                #  question_text: ...,
                #  doc_tokens: ...,
                #  ...]
                
                # Return a list of `SquadExample`
                examples.append(example)    
                
    return examples

# examples = read_squad_examples(input_file="datasets/squad/train-v2.0.json", 
#                                is_training=True,
#                                version_2_with_negative=True)


class InputFeatures(object):
    """
    A single set of features of data.
    """
    
    def __init__(self,
                 unique_id, 
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        

def _improve_answer_span(doc_tokens, 
                         input_start, 
                         input_end, 
                         tokenizer,
                         orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    
    # print(doc_tokens)
    # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']

    # print(input_start)
    # 66
    
    # print(input_end) 
    # 69
    
    # print(tokenizer)
    # <transformers.tokenization_bert.BertTokenizer object at 0x000001C1CE9D4F98>
    
    # print(orig_answer_text)
    # in the late 1990s
    
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    # print(tok_answer_text)
    # in the late 1990s

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            # print(text_span)
            # in the late 1990s
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the "max context" doc span for the token.
    """
    
    best_score = None
    best_span_index = None
    
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(examples, 
                                 tokenizer, 
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]", 
                                 sep_token="[SEP]", 
                                 pad_token=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0, 
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of `InputBatch`s.
    """
    
    unique_id = 1000000000
    
    features = []
    for (example_index, example) in enumerate(examples):
        # print(example_index)
        # 0
        
        # print(example)
        # qas_id: 56be85543aeaaa14008c9063, 
        # question_text: When did Beyonce start becoming popular?,
        # doc_tokens: [Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".], 
        # start_position: 39, 
        # end_position: 42
        
        query_tokens = tokenizer.tokenize(example.question_text)
        # print(query_tokens)
        # ['when', 'did', 'beyonce', 'start', 'becoming', 'popular', '?']
    
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
            
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        
        # `token`s are separated by whitespace; `sub_token`s are separated in a `token` by symbol
        for (i, token) in enumerate(example.doc_tokens):
            # print(i)
            # 0
            
            # print(token)
            # Beyoncé
            
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            # print(sub_tokens)
            # ['beyonce']
            # ['gi', '##selle']
            # ['knowles', '-', 'carter']
            # ['(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/']
            # ...
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        # print(tok_to_orig_index)
        # [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 51, 52, 53, 54, 54, 55, 56, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 64, 64, 64, 65, 66, 67, 68, 69, 69, 70, 71, 72, 73, 74, 75, 76, 76, 76, 77, 78, 78, 79, 80, 81, 82, 82, 82, 82, 83, 84, 85, 86, 87, 88, 89, 90, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 108, 108]
        
        # print(orig_to_tok_index)
        # [0, 1, 3, 6, 16, 23, 25, 26, 28, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 80, 83, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 102, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123, 124, 125, 126, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 153, 155, 156, 158, 159, 161]
        
        # print(all_doc_tokens)
        # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']
        
        tok_start_position = None
        tok_end_position = None
        
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # print(tok_start_position)
            # 66
            
            # print(tok_end_position)
            # 69
            (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, 
                                                                          tok_start_position, 
                                                                          tok_end_position, 
                                                                          tokenizer, 
                                                                          example.orig_answer_text)
            # print(tok_start_position)
            # 66
            
            # print(tok_end_position)
            # 69
            
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        
        # We can have documents that are longer than the maximum sequence length. To deal with this we do a 
        # sliding window approach, where we take chunks of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        
        while start_offset < len(all_doc_tokens):
            # print(len(all_doc_tokens))
            # 426
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            # Take an example with stride
            # 
            # print(doc_spans)
            # [DocSpan(start=0, length=373)]
            # 
            # In this case, `start` will move a `doc_strike`, 128, so the new `start` is 128  
            # And the new `length` is 426 - 128 = 298
            # 
            # [DocSpan(start=0, length=373), DocSpan(start=128, length=298)]
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            # `p_mask`: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keeps the classification token (set to 0) (not sure why...)
            p_mask = []

            # `[CLS]` token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
                
            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # [SEP] token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph built based on `doc_span`
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # [SEP] token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # [CLS] token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                # Index of classification token
                cls_index = len(tokens) - 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            # print(input_ids)
            # [101, 2043, 2106, 20773, 2707, 3352, 2759, 1029, 102, 20773, 21025, 19358, 22815, 1011, 5708, 1006, 1013, 12170, 23432, 29715, 3501, 29678, 12325, 29685, 1013, 10506, 1011, 10930, 2078, 1011, 2360, 1007, 1006, 2141, 2244, 1018, 1010, 3261, 1007, 2003, 2019, 2137, 3220, 1010, 6009, 1010, 2501, 3135, 1998, 3883, 1012, 2141, 1998, 2992, 1999, 5395, 1010, 3146, 1010, 2016, 2864, 1999, 2536, 4823, 1998, 5613, 6479, 2004, 1037, 2775, 1010, 1998, 3123, 2000, 4476, 1999, 1996, 2397, 4134, 2004, 2599, 3220, 1997, 1054, 1004, 1038, 2611, 1011, 2177, 10461, 1005, 1055, 2775, 1012, 3266, 2011, 2014, 2269, 1010, 25436, 22815, 1010, 1996, 2177, 2150, 2028, 1997, 1996, 2088, 1005, 1055, 2190, 1011, 4855, 2611, 2967, 1997, 2035, 2051, 1012, 2037, 14221, 2387, 1996, 2713, 1997, 20773, 1005, 1055, 2834, 2201, 1010, 20754, 1999, 2293, 1006, 2494, 1007, 1010, 2029, 2511, 2014, 2004, 1037, 3948, 3063, 4969, 1010, 3687, 2274, 8922, 2982, 1998, 2956, 1996, 4908, 2980, 2531, 2193, 1011, 2028, 3895, 1000, 4689, 1999, 2293, 1000, 1998, 1000, 3336, 2879, 1000, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # print(input_mask)
            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # print(segment_ids)
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Only `sequence_b_segment_id` is set to 1
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            
            # Get `start_position` and `end_position`
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation we throw it out, 
                # since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
           
            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index
        
            # Display some examples
            if example_index < 20:              
                print("*** Example ***")
                # *** Example ***
                print("unique_id: %s" % (unique_id))
                # unique_id: 1000000000
                print("example_index: %s" % (example_index))
                # example_index: 0
                print("doc_span_index: %s" % (doc_span_index))
                # doc_span_index: 0
                print("tokens: %s" % " ".join(tokens))
                # tokens: [CLS] when did beyonce start becoming popular ? [SEP] beyonce gi ##selle knowles - carter ( / bi ##ː ##ˈ ##j ##ɒ ##nse ##ɪ / bee - yo ##n - say ) ( born september 4 , 1981 ) is an american singer , songwriter , record producer and actress . born and raised in houston , texas , she performed in various singing and dancing competitions as a child , and rose to fame in the late 1990s as lead singer of r & b girl - group destiny ' s child . managed by her father , mathew knowles , the group became one of the world ' s best - selling girl groups of all time . their hiatus saw the release of beyonce ' s debut album , dangerously in love ( 2003 ) , which established her as a solo artist worldwide , earned five grammy awards and featured the billboard hot 100 number - one singles " crazy in love " and " baby boy " . [SEP]
                print("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                # token_to_orig_map: 9:0 10:1 11:1 12:2 13:2 14:2 15:3 16:3 17:3 18:3 19:3 20:3 21:3 22:3 23:3 24:3 25:4 26:4 27:4 28:4 29:4 30:4 31:4 32:5 33:5 34:6 35:7 36:7 37:8 38:8 39:9 40:10 41:11 42:12 43:12 44:13 45:13 46:14 47:15 48:16 49:17 50:17 51:18 52:19 53:20 54:21 55:22 56:22 57:23 58:23 59:24 60:25 61:26 62:27 63:28 64:29 65:30 66:31 67:32 68:33 69:34 70:34 71:35 72:36 73:37 74:38 75:39 76:40 77:41 78:42 79:43 80:44 81:45 82:46 83:47 84:47 85:47 86:48 87:48 88:48 89:49 90:49 91:49 92:50 93:50 94:51 95:52 96:53 97:54 98:54 99:55 100:56 101:56 102:57 103:58 104:59 105:60 106:61 107:62 108:63 109:63 110:63 111:64 112:64 113:64 114:65 115:66 116:67 117:68 118:69 119:69 120:70 121:71 122:72 123:73 124:74 125:75 126:76 127:76 128:76 129:77 130:78 131:78 132:79 133:80 134:81 135:82 136:82 137:82 138:82 139:83 140:84 141:85 142:86 143:87 144:88 145:89 146:90 147:90 148:91 149:92 150:93 151:94 152:95 153:96 154:97 155:98 156:99 157:100 158:101 159:101 160:101 161:102 162:103 163:103 164:104 165:105 166:105 167:106 168:107 169:107 170:108 171:108 172:108
                print("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                # token_is_max_context: 9:True 10:True 11:True 12:True 13:True 14:True 15:True 16:True 17:True 18:True 19:True 20:True 21:True 22:True 23:True 24:True 25:True 26:True 27:True 28:True 29:True 30:True 31:True 32:True 33:True 34:True 35:True 36:True 37:True 38:True 39:True 40:True 41:True 42:True 43:True 44:True 45:True 46:True 47:True 48:True 49:True 50:True 51:True 52:True 53:True 54:True 55:True 56:True 57:True 58:True 59:True 60:True 61:True 62:True 63:True 64:True 65:True 66:True 67:True 68:True 69:True 70:True 71:True 72:True 73:True 74:True 75:True 76:True 77:True 78:True 79:True 80:True 81:True 82:True 83:True 84:True 85:True 86:True 87:True 88:True 89:True 90:True 91:True 92:True 93:True 94:True 95:True 96:True 97:True 98:True 99:True 100:True 101:True 102:True 103:True 104:True 105:True 106:True 107:True 108:True 109:True 110:True 111:True 112:True 113:True 114:True 115:True 116:True 117:True 118:True 119:True 120:True 121:True 122:True 123:True 124:True 125:True 126:True 127:True 128:True 129:True 130:True 131:True 132:True 133:True 134:True 135:True 136:True 137:True 138:True 139:True 140:True 141:True 142:True 143:True 144:True 145:True 146:True 147:True 148:True 149:True 150:True 151:True 152:True 153:True 154:True 155:True 156:True 157:True 158:True 159:True 160:True 161:True 162:True 163:True 164:True 165:True 166:True 167:True 168:True 169:True 170:True 171:True 172:True
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                # input_ids: 101 2043 2106 20773 2707 3352 2759 1029 102 20773 21025 19358 22815 1011 5708 1006 1013 12170 23432 29715 3501 29678 12325 29685 1013 10506 1011 10930 2078 1011 2360 1007 1006 2141 2244 1018 1010 3261 1007 2003 2019 2137 3220 1010 6009 1010 2501 3135 1998 3883 1012 2141 1998 2992 1999 5395 1010 3146 1010 2016 2864 1999 2536 4823 1998 5613 6479 2004 1037 2775 1010 1998 3123 2000 4476 1999 1996 2397 4134 2004 2599 3220 1997 1054 1004 1038 2611 1011 2177 10461 1005 1055 2775 1012 3266 2011 2014 2269 1010 25436 22815 1010 1996 2177 2150 2028 1997 1996 2088 1005 1055 2190 1011 4855 2611 2967 1997 2035 2051 1012 2037 14221 2387 1996 2713 1997 20773 1005 1055 2834 2201 1010 20754 1999 2293 1006 2494 1007 1010 2029 2511 2014 2004 1037 3948 3063 4969 1010 3687 2274 8922 2982 1998 2956 1996 4908 2980 2531 2193 1011 2028 3895 1000 4689 1999 2293 1000 1998 1000 3336 2879 1000 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                # input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # segment_ids: 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                if is_training and span_is_impossible:
                    print("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % (start_position))
                    # start_position: 75
                    print("end_position: %d" % (end_position))
                    # end_position: 78
                    print("answer: %s" % (answer_text))
                    # answer: in the late 1990s

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible)
            )
            
            unique_id += 1

    return features

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# features = convert_examples_to_features(examples, 
#                                         tokenizer, 
#                                         max_seq_length=384,
#                                         doc_stride=128,
#                                         max_query_length=64,
#                                         is_training=True,
#                                         cls_token_at_end=False,
#                                         cls_token="[CLS]", 
#                                         sep_token="[SEP]", 
#                                         pad_token=0,
#                                         sequence_a_segment_id=0, 
#                                         sequence_b_segment_id=1,
#                                         cls_token_segment_id=0, 
#                                         pad_token_segment_id=0,
#                                         mask_padding_with_zero=True)


def load_and_cache_examples(args,
                            tokenizer, 
                            evaluate=False, 
                            output_examples=False):
    
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args["local_rank"] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  
    
    # Load data features from cache or dataset file
    input_file = args["predict_file"] if evaluate else args["train_file"]
    cached_features_file = os.path.join(
        os.path.dirname(input_file), 
        'cached_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args["model_name_or_path"].split('/'))).pop(),
            str(args["max_seq_length"])
        )
    )
    
    if os.path.exists(cached_features_file) and not args["overwrite_cache"] and not output_examples:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", input_file)
        
        # Call `read_squad_examples()`
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args["version_2_with_negative"])

        # Call `convert_examples_to_features()`
        features = convert_examples_to_features(examples, 
                                                tokenizer, 
                                                max_seq_length=args["max_seq_length"],
                                                doc_stride=args["doc_stride"],
                                                max_query_length=args["max_query_length"],
                                                is_training=True,
                                                cls_token_at_end=False,
                                                cls_token="[CLS]", 
                                                sep_token="[SEP]", 
                                                pad_token=0,
                                                sequence_a_segment_id=0, 
                                                sequence_b_segment_id=1,
                                                cls_token_segment_id=0, 
                                                pad_token_segment_id=0,
                                                mask_padding_with_zero=True)
        
        if args["local_rank"] in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args["local_rank"] == 0 and not evaluate:
        torch.distributed.barrier()  

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids,
                                all_example_index, 
                                all_cls_index, 
                                all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids,
                                all_start_positions, 
                                all_end_positions,
                                all_cls_index, 
                                all_p_mask)

    if output_examples:
        return dataset, examples, features

    return dataset

# load_and_cache_examples(args,
#                         input_file="datasets/squad/train-v2.0.json",
#                         model_name_or_path="bert-base-uncased",
#                         max_seq_length=384,
#                         tokenizer=BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True))


def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])
        
def train(args,        
          train_dataset,
          model,
          tokenizer):
    """
    Train the model.
    """
    
    args["train_batch_size"] = args["per_gpu_train_batch_size"] * max(1, args["n_gpu"])
    train_sampler = RandomSampler(train_dataset) if args["local_rank"] == -1 else DistributedSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args["train_batch_size"])
    
    if args["max_steps"] > 0:
        t_total = args["max_steps"]
        args["num_train_epochs"] = args["max_steps"] // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args["learning_rate"], 
                      eps=args["adam_epsilon"])
    #scheduler = WarmupLinearSchedule(optimizer, 
    #                                 warmup_steps=args["warmup_steps"], 
    #                                 t_total=t_total)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                     num_warmup_steps=args["warmup_steps"],num_training_steps=t_total)
    
    # Multiple GPU training
    if args["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args["local_rank"] != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args["local_rank"]],
                                                          output_device=args["local_rank"],
                                                          find_unused_parameters=True)
    
    # Training
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args["num_train_epochs"])
    print("  Instantaneous batch size per GPU = %d", args["per_gpu_train_batch_size"])
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args["train_batch_size"] * args["gradient_accumulation_steps"] * (torch.distributed.get_world_size() if args["local_rank"] != -1 else 1))
    print("  Gradient Accumulation steps = %d", args["gradient_accumulation_steps"])
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["local_rank"] not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)  
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, 
                              desc="Iteration", 
                              disable=args["local_rank"] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args["device"]) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'start_positions': batch[3],
                      'end_positions':   batch[4]}
            if args["model_type"] != 'distilbert':
                inputs['token_type_ids'] = None if args["model_type"] == 'xlm' else batch[2]
            if args["model_type"] in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask': batch[6]})
            outputs = model(**inputs)
            # Model outputs are always tuple in transformers (see doc)
            loss = outputs[0]  

            if args["n_gpu"] > 1:
                # `mean()` to average on multi-gpu parallel (not distributed) training
                loss = loss.mean() 
            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            tr_loss += loss.item()
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                optimizer.step()
                # Update learning rate schedule
                scheduler.step()
                model.zero_grad()
                global_step += 1

                #if args["local_rank"] in [-1, 0] and args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    #if local_rank == -1 and evaluate_during_training:  
                    #    results = evaluate(args, model, tokenizer)
                    #    for key, value in results.items():
                    #        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    #tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    #tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args["logging_steps"], global_step)
                    #logging_loss = tr_loss

                if args["local_rank"] in [-1, 0] and args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args["output_dir"], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    print("Saving model checkpoint to %s", output_dir)

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                epoch_iterator.close()
                break
                
        if args["max_steps"] > 0 and global_step > args["max_steps"]:
            train_iterator.close()
            break

    #if args["local_rank"] in [-1, 0]:
    #    tb_writer.close()

    return global_step, tr_loss / global_step



if __name__ == "__main__":
    
    MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
    }

    args = {
        "local_rank": -1,
        "model_type": "bert",
        "config_name": "", 
        "model_name_or_path": "bert-base-uncased", 
        "tokenizer_name": "",
        "max_seq_length": 384,
        "overwrite_cache": False,
        "do_lower_case": True,
        "do_train": True,
        "output_dir": "models/bert-finetuned", 
        "version_2_with_negative": True,
        "doc_stride": 128,
        "max_query_length": 64,
        "train_file": "datasets/squad/train-v2.0.json",
        "predict_file": "datasets/squad/dev-v2.0.json",  
        "per_gpu_train_batch_size": 4,
        "max_steps": -1,
        "num_train_epochs": 2,
        "learning_rate": 3e-5,
        "adam_epsilon": 1e-8,
        "warmup_steps": 100,
        "no_cuda": False,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "model_type": "bert",
        "save_steps": 1000,
        "seed": 42,
        "do_eval": True,
        "eval_all_checkpoints": True,
        # "eval_batch_size": 8,
        "per_gpu_eval_batch_size": 4,
        "n_best_size": 20,
        "max_answer_length": 300,
        "null_score_diff_threshold": 0.0
    }
    
    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        args["n_gpu"] = torch.cuda.device_count()
    else:  
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args["local_rank"])
        device = torch.device("cuda", args["local_rank"])
        torch.distributed.init_process_group(backend='nccl')
        args["n_gpu"] = 1
        
    args["device"] = device
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    config = config_class.from_pretrained(args["config_name"] if args["config_name"] else args["model_name_or_path"])
    tokenizer = tokenizer_class.from_pretrained(args["tokenizer_name"] if args["tokenizer_name"] else args["model_name_or_path"], do_lower_case=args["do_lower_case"])
    model = model_class.from_pretrained(args["model_name_or_path"], from_tf=bool('.ckpt' in args["model_name_or_path"]), config=config)
    
    # Make sure only the first process in distributed training will download model & vocab
    if args["local_rank"] == 0:
        torch.distributed.barrier()  
    
    model.to(args["device"])
    
    print("Training/evaluation parameters %s", args)
    
    # Training
    if args["do_train"]:
        train_dataset = load_and_cache_examples(args, 
                                                tokenizer, 
                                                evaluate=False, 
                                                output_examples=False)
        global_step, tr_loss = train(args, 
                                     train_dataset, 
                                     model, 
                                     tokenizer)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    # Save the trained model and the tokenizer
    if args["do_train"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args["output_dir"]) and args["local_rank"] in [-1, 0]:
            os.makedirs(args["output_dir"])
    
        print("Saving model checkpoint to %s", args["output_dir"])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(args["output_dir"])
        tokenizer.save_pretrained(args["output_dir"])
    
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args["output_dir"], 'training_args.bin'))
    
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args["output_dir"])
        tokenizer = tokenizer_class.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
        model.to(device)
        
        
        
    results = {}
    if args["do_eval"] and args["local_rank"] in [-1, 0]:
        checkpoints = [args["output_dir"]]
        if args["eval_all_checkpoints"]:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args["output_dir"] + '/**/' + WEIGHTS_NAME, recursive=True)))
    
        print("Evaluate the following checkpoints: %s", checkpoints)
    
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
    
            # Evaluate
            # result = evaluate(args, model, tokenizer, prefix=global_step)
    
            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)
    
    print("Results: {}".format(results))
    
    with open("Result.csv", "w") as f:
        f.write(str(results))