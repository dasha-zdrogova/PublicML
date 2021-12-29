import main
import os

TEST_PATH = './tests/'

for i in os.listdir(TEST_PATH):
    with open(TEST_PATH + i) as f:
        s1 = f.readline()[:-1]
        s2 = f.readline()
        print(i)
        print(len(s1), len(s2))
        s1n, s2n = main.main(s1, s2)
        print(len(s1n), len(s2n))
        print(s1n.count('-'), s2n.count('-'))
        print(s1n, s2n, sep='\n')
