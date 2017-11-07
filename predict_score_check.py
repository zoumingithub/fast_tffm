import sys
import re

idx = 0
model = {}
file = open("small_model", 'r')
for line in file:
  line = line.strip()
  weights = re.findall(r'\[([-.\w\s]*)\]', line)
  for weight in weights:
    model[idx] = [float(elem) for elem in weight.split(' ')]
    idx += 1


def comput_fm_score():
  fea_ids = [0,1,3,5,6]
  pscore = 0
  factor_sum = [0 for i in range(0,9)]
  for idx in fea_ids:
    bias = model[idx][0]
    fsum_2 = 0
    for i in range(1, 9):
      fsum_2 += model[idx][i] * model[idx][i]
      factor_sum[i] += model[idx][i]
    pscore += bias
    pscore -= fsum_2 * 0.5
  for i in range(1, 9):
    pscore += factor_sum[i] * factor_sum[i] * 0.5
  return pscore


print comput_fm_score()
      
  
  
