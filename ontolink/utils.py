import re
from termcolor import colored
from typing import List

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def insert_if_anybig(lalist,elem):
  """Insert elem into the lalist if elem is bigger than 
    any item of the list.
    Returns -1 if there's was no insertion
  """
  k =len(lalist)
  for i,val in enumerate(lalist):
    if elem > val:
      # lalist[i] = elem
      return i,lalist[:i] + [elem] +lalist[i:k-1] 

  return -1,lalist


def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]


def mark_word(string,start,end):
    mark1 ='<mark>'
    mark2 = '</mark>' 

    return string[:start] + mark1+ string[start:end+1] +mark2+ string[end+1:]
    #the <>children


# Helper function to check
# if the two intervals overlaps
def is_overlaping(a, b):
  if b[0] > a[0] and b[0] < a[1]:
    return True
  else:
    return False


def merge_intervals(arr):
  if len(arr) < 1:
    return []
  #sort the intervals by its first value
  arr.sort(key=lambda x: x[0])

  merged_list = []
  merged_list.append(arr[0])
  for i in range(1, len(arr)):
    pop_element = merged_list.pop()
    if is_overlaping(pop_element, arr[i]):
      new_element = pop_element[0], max(pop_element[1], arr[i][1])
      merged_list.append(new_element)
    else:
      merged_list.append(pop_element)
      merged_list.append(arr[i])
  return merged_list


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]
#From https://github.com/facebookresearch/BLINK/ with some modifications
def _print_colorful_text(input_sentence, samples):
    """
    pred_triples:
        Assumes no overlapping triples
    """


    sort_idxs = sorted(range(len(samples)),
                   key=lambda idx: samples[idx]['start_pos'])
    # init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0: int(samples[sort_idxs[0]]["start_pos"])]
        for i,idx in enumerate(sort_idxs):
            sample = samples[idx]
            msg += colored(
                input_sentence[int(sample["start_pos"]): int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[i % len(HIGHLIGHTS)],
            )
            if i < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]): int(samples[sort_idxs[i + 1]]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]):]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def is_overlaping(a:List[int], b:List[int]):
  """Check if the two interval `a` and `b` are overlapping

  :param a: [description]
  :type a: List[int]
  :param b: [description]
  :type b: List[int]
  :return: [description]
  :rtype: [type]
  """
  return overlaps(a,b) > 0

#https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python
def overlaps(a, b):
    """
    Return the amount of overlap, between a and b.
    If >0, the amount overlap
    If 0,  their limits touch.
    If <0, distance"""
    
    return min(a[1], b[1]) - max(a[0], b[0])


DEBUG = False


def log(*args):
    if DEBUG:
        print(args)
      
