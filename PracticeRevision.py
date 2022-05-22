# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:34:45 2021

@author: User
"""
#Dictionary ex

phone = input('phone :')
digit_mapping = {
    "1":"One",
    "2":"Two",
    "3":"Three",
    "4":"Four"
    }
output = ''
for ch in phone:
    output += digit_mapping.get(ch, "!") +' '
print(output)

# for i in ['XXXXX','XX','XXXXX','XX','XX']:
#     print(i)
# for i in [5,2,5,2,2]:
#     print(i*'X')
# for i in "python":
#     print(i)
  
# for i in range(4):
#     for j in range(3):
#         print(f'({i}, {j})')

# for i in range(2):
#      for j in range(1,3):
#         if (i%j == 0):
#             print('XXXXX')
#         else:
#             print('XX')
    