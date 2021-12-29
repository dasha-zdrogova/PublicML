import os

TEST_PATH = './tests/'

'''for i in os.listdir(TEST_PATH):
    with open(TEST_PATH + i) as f:'''
file = TEST_PATH + '1.txt'
with open(file, 'w') as f:
    f.write('a' * 10**3 + 'b' * 5 + '\n')
    f.write('b' * 5 + 'c' * 10**3)

file = TEST_PATH + '2.txt'
with open(file, 'w') as f:
    f.write('a' * 10**4 + '\n')
    f.write('b' * 5)

file = TEST_PATH + '3.txt'
with open(file, 'w') as f:
    f.write('a' * 10**4 + 5 * 'b' + 'a' * 10**4 + '\n')
    f.write('b' * 5)

file = TEST_PATH + '4.txt'
with open(file, 'w') as f:
    f.write('ab' * 10**4 + '\n')
    f.write('ba' * 10**4)

file = TEST_PATH + '5.txt'
with open(file, 'w') as f:
    f.write('ab' + 'a' * 10**4 + '\n')
    f.write('a' * 10**4 + 'b')