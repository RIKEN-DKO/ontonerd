import re


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
