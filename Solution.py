# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:28:41 2020

@author: User
"""
#%%
# Given an integer, return the integer with reversed digits.
# Note: The integer could be either positive or negative.

#Reverse number
def solution(num):
    string=str(num)
    # num=split(num)
    if string[0]=='-':
        return int('-'+string[:0:-1])
    else:
        return int(string[::-1])

#import numpy as np        
print(solution(2345))
print(solution(-3451))
#print(np.reversed(2345))

#%%    
# For a given sentence, return the average word length. 
# Note: Remember to remove punctuation first.

sentence1 = "Hi all, my name is Tom...I am originally from Australia."
sentence2 = "I   need to work very hard to learn more about algorithms in Python!"

def sol(sentence):
    for i in "'?!,.;":
        sentence=sentence.replace(i,'')
    words=sentence.split()
    return round(sum(len(word) for word in words)/len(words),2)
print(sol(sentence1))

print(sol(sentence2))    

#%%
# Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.
# You must not use any built-in BigInteger library or convert the inputs to integer directly.

#Notes:
#Both num1 and num2 contains only digits 0-9.
#Both num1 and num2 does not contain any leading zero.

num1 = '3640'
num2 = '1836'

def solu(num1,num2):
    s=eval(num1)+ eval(num2)
    #print(type(s))
    return str(s)
print(solu(num1,num2))
ord('0')
#%%

#Approach2 
#Given a string of length one, the ord() function returns an integer representing the Unicode code point of the character 
#when the argument is a unicode object, or the value of the byte when the argument is an 8-bit string.
num1 = '364'
num2 = '1836'

def solution(num1, num2):
    n1, n2 = 0, 0
    m1, m2 = 10**(len(num1)-1), 10**(len(num2)-1)

    for i in num1:
        n1 += (ord(i) - ord("0")) * m1 
        m1 = m1//10        

    for i in num2:
        n2 += (ord(i) - ord("0")) * m2
        m2 = m2//10

    return str(n1 + n2)
print(solution(num1, num2))

#%%

# Given a string, find the first non-repeating character in it and return its index. 
# If it doesn't exist, return -1. # Note: all the input strings are already lowercase.

#Approach 1

def solut(s):
    freq={}
    for i in s:
        if i not in freq:
            freq[i]=1
        else:    
            freq[i]+=1
    for i in range(len(s)):
        if freq[s[i]]==1:
            return i
    return -1
print(solut('alphalpha'))
print(solut('barbados'))
print(solut('crunchy'))
#%%
#Approach 2
import collections

def solution(s):
    # build hash map : character and how often it appears
    count = collections.Counter(s) # <-- gives back a dictionary with words occurrence count 
                                         #Counter({'l': 1, 'e': 3, 't': 1, 'c': 1, 'o': 1, 'd': 1})
    # find the index
    for idx, ch in enumerate(s):
        if count[ch] == 1:
            return idx     
    return -1

print(solution('alphabet'))
print(solution('barbados'))
print(solution('crunchy'))

#%%%

# Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.
# The string will only contain lowercase characters a-z.

s = 'rakar'
def solution(s):
    # for i in range(len(s)):
    #     t=s[i:]+s[i+1:]
    #     if t == t[::-1]:  
    #         True
    #     else:
    #         False
    return s==s[::-1]

print(solution(s))    
#%%
#To check palindrom
s='rasksar'
def solution(s):
    for i in range(len(s)):
        if s==s[::-1]:
            return True
        else:
            return False
print(solution(s))
#%%

# Given an array of integers, determine whether the array is monotonic or not.
A = [6, 5, 4, 4,0] 
B = [1,1,1,3,3,4,3,2,4,2]
C = [0,1,1,2,3,7]

def solution(n):
    return (all(n[i]>=n[i+1] for i in range(len(n)-1)) or all(n[i]<=n[i+1] for i in range(len(n)-1)))

print(solution(A))
print(solution(B))
print(solution(C))    

#%%
#Given an array nums, write a function to move all zeroes to the end of it while maintaining the relative order of 
#the non-zero elements.

array1 = [0,1,0,3,12]
array2 = [1,7,0,0,8,0,10,12,0,4]

def solution(n):
    for i in n:
        if 0 in n:
            n.remove(0)
            n.append(0)
    return (n)
    
    
print (solution(array1))
print (solution(array2))            
#%%
# Given an array containing None values fill in the None values with most recent 
# non None value in the array 
array1 = [None,1,None,2,3,None,None,5,None]

def solution(array):
    valid=0
    for i in range(len(array)):
        if array[i]!=None:
            valid=i
        else:
            array[i]=valid
    return array    
        
print(solution(array1))

#%%
#Approach2
array1 = [None,1,None,2,3,None,None,5,None]
def solution(array):
    ar=[]
    valid=0
    for i in array:
        if i is not None:
            ar.append(i)
            valid=i
        else:
            ar.append(valid)
    return ar    
        
print(solution(array1))

#%%

#Given two sentences, return an array that has the words that appear in one sentence and not
#the other and an array with the words in common. 

sentence1 = 'We are really pleased to meet you in our city'
sentence2 = 'The city was hit by a really heavy storm'

def solution(sentence1, sentence2):
    words1=set(sentence1.split())
    words2=set(sentence2.split())
    
    return sorted(words1 ^ words2),sorted(words1 & words2)

print(solution(sentence1, sentence2))    

#%%
# Given k numbers which are less than n, return the set of prime number among them
# Note: The task is to write a program to print all Prime numbers in an Interval.
# Definition: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. 

n = 65
def solution(n):
    prime=[]
    for n1 in range (n):
        if n1>1: #all prime >1
         for i in range(2,n1):       
             if (n1%i) == 0:
                 break
         else:
              prime.append(n1)
                 
    return(prime)
print(solution(n))                 

#%%        
from array import *
ar=array('i',[1,2,3,4])
ar.insert(1,100)
type(range1)
ar.extend((4,5,3,5))
ar.fromlist(lst)

ar.tolist()
type(ar)
del lst[2]
lst
ar
lst.extend(str1)
del lst
str1='learning is fun!'
str1.isalnum()
lst=[1,2,'a','sam',2]
lst.insert(9,25)
tup=(1,2,3,4,3,'py')
range1=range(1,12,4)
dic={1:'first','second':2,3:'three','four':4}
dic['five']=5
dic.update(second = 2)
dic.get(3)
dic.items()
sets={'example',24,87.5,'data',24,'data'}
dic
list(dic)
print("string={},list = {},array={},tuple={},dictionary={},set={},range={}".
      format(len(str1),len(lst),len(ar),len(tup),len(dic),len(sets),len(range1)))
for i in range1: print(i)
range1.reverse()
lst.append([2,3,'b'])
ar.append(5)
del ar[3:5]
dic.pop(3)
sets.add(24)
lst.remove(2)
lst
sets.discard(24)
sets
n1='Gitaa'
n2='Pvt'
n3='Ltd'
n='{} {}. {}.'.format(n1,n2,n3)
n
sets.clear()
sets
def number(n):
    return[i for i in range(1,n+1)]
print(number(10))
#%%
len('python')
# 26.Length of string
def str_len(s1):
    count=0
    for ch in s1:
        count+=1
    return count
print(str_len('Python'))
#%%
#27.character freq
def char_freq(s3):
    dic={}
    for n1 in s3:
        keys=dic.keys()
        if n1 in keys:
            dic[n1] += 1
        else:
            dic[n1]=1
    return dic
print(char_freq('google.com'))
    
#%%

#28Replace second occ of first char with $
def change_char(s11):
    char=s11[0]
    s11=s11.replace(char,'$')
    #print(s11)
    s11 = char + s11[1:]
    return s11 
print(change_char('restart'))
#%%
l=[]
def char_freq(s2):
    for i in enumerate(s2):
        if i==i+1:
            l.append(i)
    return l
print(str_len('Python is pyari'))
#%%
#29. Remove nth char
def rem_char(s1,n):
    first=s1[:n]
    last=s1[n+1:]
    s1=first+last
    return s1
print(rem_char('python', 0))
print(rem_char('python', 3))
print(rem_char('python', 5))
#%%
#30: unique words in sorted order
#items=('red','black','green','blue','white','red')
items=input('please enter csv of colors')
def uniq_word(items):
    words = [word for word in items.split(',')]
    return (' , '.join(sorted(list(set(words)))))
print(uniq_word(items))
#%%
# 31. 3 chars of string
def subString(s):
    return s[:3] if len(s)>3 else (s)
print(subString('python'))
print(subString('hon'))
#%%

def ceasar_encript(realText,step):
    outText = []
    cryptText = []
    uppercase=['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    lowercase=['a','b','c','d','e','f','g','h','i','j','k','l','m'
               'n','o','p','q','r','s','t','u','v,''w','x','y','z']
    for eachLetter in realText:
        if  eachLetter in uppercase:
            index = uppercase.index(eachLetter)
            crypting=(index + step) % 26
            cryptText.append(crypting)
            newLetter = uppercase[crypting]
            outText.append(newLetter)
        elif eachLetter in lowercase:
            index = lowercase.index(eachLetter)
            crypting=(index + step) % 26
            cryptText.append(crypting)
            newLetter = lowercase[crypting]
            outText.append(newLetter)
    return outText
code1 = ceasar_encript('abc', 4)
code2 = ceasar_encript('ABC', 4)

print()
print(code1)
print(code2)
#%%
#floating number with no decimal
x=float(input('Enter flot to convert int'))
y=float(input('Enter flot to convert int'))
print('Formatted number:'+'{:.0f}'.format(x))
print('formatted number:'+'{:.0f}'.format(y))
#print()

#%%33 Result as 2.0?

a=1.0
b='1'
c='1.1'
print(a + float(b))
print(float(b)+float(c))       
# print(a + int(c))
print(a + int(float(c)))
print(int(a) + int(float(c)))
print(2.0 + b)
#%%35 print index of ch
s='Python'
x=print(input('Enter the letter to find index: '))
for i,x in enumerate(s):
    print('The index of char' ,x, 'is at: ',i)

#%%
s=input('Enter the letter to find index: ')
for i,x in enumerate(s):
    print('The index of char' ,x, 'is at: ',i)
    
    
#%%.36.

def vowel(text):
    vowels='aeiouAEIOU'
    print(len([letter for letter in text if letter  in vowels]))
    print([letter for letter in text if letter  in vowels])
vowel('Welcome')

#%%37. sum items in list
def ListSum(l):
    sum1=0
    for i in l:
        sum1 += i
    return sum1 
print(ListSum([-3,4,6,8]))

#%%.40. Largest number
def Largest(l):
    
    max=0
    for i in range(len(l)):
        if max < l[i]:
            max,l[i]=l[i],max
    return max
l=[23,45,76,45,34,65,34]
print(Largest(l))


#%% Bubble sort
l=[23,45,76,45,34,65,34]
for i in range(len(l)-1):
    for j in range(len(l)-i-1):
        if l[j-1] > l[j]:
            l[j-1],l[j] = l[j],l[j-1]
print(l)
#%% Insertion sort
for i in l:
    j=l.index(i)
    while j  > 0:
        if l[j-1] > l[j]:
            l[j-1],l[j] = l[j],l[j-1]
        else:
            break
    j=j-i 
print(l)
        
#%%.41. Remove duplicates
a=[10,20,30,20,10,50,60,40,80,50,40]
dup=set()
uniq= []
for i in a :
    if i not in dup:
        uniq.append(i)
        dup.add(i)
print(dup,uniq)
#%%.42.commer in two lists
s1=(1,2,3,4,5)
s2=[5,6,7,8,9,10]

def commanItem(s1,s2):
    res=False
    for i in s1:
        for j in s2:
            if i == j:
                res= True
    return res        
print(commanItem(s1, s2))

#%%43. shuffle
color=['red','green','red','white','blue']
from random import shuffle
shuffle(color)
print(color)

#%% count range in list
def cnt_rng(li,min,max):
    ctr = 0
    for x in li:
        if min<x<max:
            ctr += 1
    return ctr
print(cnt_rng([2,3,4,5,6,48,56],3,56))

#%%.45. 5 conseutive

#l=[10,20,30,20,10,50,60,40,80,50,40]
l=[[5*i+j for j in range(1,6)] for i in range(5)]
print(l)

#%% Appending a list in place of last element
s1=[1,2,3,4,5]
s2=[6,7,8,9,10]
s1[-1:]=s2
#s1=s1[:-1]+s2
print(s1)

#%% 47 dict from 2 lists
l1=['a','b','c','d']
l2=[1,2,2,4]
from collections import defaultdict
temp=defaultdict(set)
for c,i in zip(l1,l2):
    temp[c].add(i)
print(temp)

#%%.48. iteratge over dict
d={'a': {1}, 'b': {2}, 'c': {2}, 'd': {4}}
for key,value in d.items():
        print(key,'->',value)

#%% 49. 