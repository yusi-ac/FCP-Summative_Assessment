import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(op = [], ttl = None):
    plt.figure(figsize=(8, 4))
    plt.hist(op, edgecolor='black')
    plt.title(ttl)
    plt.xlabel('Opinion')


def test_defuant(num_people = 50, threshold = 0.2, beta = 0.2):
    print(f'total number of people in this test function is {num_people}')
    print(f'threshold set at {threshold}')
    print(f'beta set at {beta}')

    opinions = np.random.rand(num_people)

    plot_hist(opinions,'Initial Opinions')
    plt.show()

    for each in range(3):
        person_idx = np.random.randint(0, num_people)
        print(f'------iteration {each}------')
        print(f'people No. {person_idx} chosen as Xi')
        
        direction = np.random.choice([-1, 1])
        print(f'direction {direction} chosen')
        neighbor_idx = (person_idx + direction) % num_people
        print(f'people No. {neighbor_idx} chosen as Xj')
        
        diff = abs(opinions[person_idx] - opinions[neighbor_idx])
        
        if diff < threshold:
            print('diff < threshold, updating opinions')
            print(f'Xi({each}) = {opinions[person_idx]}, Xj({each}) = {opinions[neighbor_idx]}')
            opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
            opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
            print(f'Xi({each+1}) = {opinions[person_idx]}, Xj({each+1}) = {opinions[neighbor_idx]}')
        else:
            print('diff >= threshold, no changes made')

    plot_hist(opinions,'Final Opinions')
    plt.show()

def main(num_people = 50, threshold = 0.2, beta = 0.2):

    opinions = np.random.rand(num_people)

    plot_hist(opinions,'Initial Opinions')
    plt.show()

    opinions_history = [opinions.copy()]

    for each in range(num_people*100):
        person_idx = np.random.randint(0, num_people)
        
        direction = np.random.choice([-1, 1])
        neighbor_idx = (person_idx + direction) % num_people
        diff = abs(opinions[person_idx] - opinions[neighbor_idx])
        
        if diff < threshold:
            opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
            opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
        opinions_history.append(opinions.copy())

    plt.figure(figsize=(10, 6))
    for i, opinion in enumerate(opinions_history):
        plt.plot([i]*num_people, opinion, 'o', markersize=2, color='blue', alpha=0.5)

    plt.title('Opinions Evolution During Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Opinion')
    plt.show()

    plot_hist(opinions,'Final Opinions')
    plt.show()

if __name__ == "__main__":
    num_people = 50
    threshold = 0.2
    beta = 0.2
    if '-threshold' in sys.argv:
        threshold_idx = sys.argv.index('-threshold')
        threshold = float(sys.argv[threshold_idx + 1])
    if '-beta' in sys.argv:
        beta_idx = sys.argv.index('-beta')
        beta = float(sys.argv[beta_idx + 1])
    if '-num_people' in sys.argv:
        num_people_idx = sys.argv.index('-num_people')
        num_people = int(sys.argv[num_people_idx + 1])
    if '-test_defuant' in sys.argv:
        test_defuant(num_people, threshold, beta)
    else:
        if '-defuant' in sys.argv:
            main(num_people, threshold, beta)