import random

for i in range(1, 3):
    with open(f'test{13 + i}.txt', 'w') as f:
        n = random.randint(1200, 1300)
        s = list(set([f'{random.randint(0, n)} {random.randint(0, n)}' for _ in range(n)]))
        f.write(f'{len(s)}\n')
        f.write('\n'.join(s))