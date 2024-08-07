# daily multiplicaation practice for kids
import random


def daily_practice(stats):
    stats['total'] += 1
    # generate two random numbers
    num1 = random.randint(1, 12)
    num2 = random.randint(1, 12)
    

    # mode=random.randint(1, 3)
    mode=0
    if mode ==0:
        # multiplication
        product=num1*num2
        print(f'{num1} * {num2} = ?')
        answer=product
    if mode==1:
        # division
        product=num1*num2
        print(f'{product} / {num1} = ?')
        answer=num2
    if mode==2:
        # subtraction
        sum=num1+num2
        print(f'{sum} - {num1} = ?')
        answer=num2
    # get the input
    user_input = input()
    # check if the input is a number
    if not user_input.isdigit():
        # print('Please enter a number')
        stats['skipped'] += 1
        pass
    else:
        # check the answer
        if int(user_input) == answer:
            stats['correct'] += 1
            print(f'Correct! You have answered {stats["correct"]}/{stats["total"]} questions correctly, skipped {stats["skipped"]} questions.')
            
        else:
            print('Wrong!')

    
        
stats={'correct':0, 'total':0, 'skipped':0}
while True:
    
    daily_practice(stats)