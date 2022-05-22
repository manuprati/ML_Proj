# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 15:26:07 2021

@author: User
"""

print(''' 
Please give any of following command
start: To start the car 
stop : To stop the car
quit: to quit the program
       ''')

command=""
started=False

while(True):
     command=input(" >").lower()

     if (command == 'start'):
         if not started:
             print("Car started")
             started=True
         else:
             print ('Car already started')
         
     elif(command == 'stop' ):
         if started:
             print('Car is stopped' )
             started=False
         else:
             print('Car is already stopped')
     elif(command == 'quit'):
         break
     
     else:
         print("I don't know what you typed")
     
        
    
    
    
#print('Your command is: '+command)
#  */