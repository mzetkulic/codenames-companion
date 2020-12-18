from pathlib import Path
import cluefinder

def main():
    path = Path(__file__).resolve()
    print(path)
    print(type(path))
    print("Hello World")
    print(path.parents[2])
    model = cluefinder.get_model(path.parents[2] / 'data' / 'GoogleNews-vectors-negative300.bin')
    positives = ['Earth','Moon']
    negatives = ['planet']
    clues = model.most_similar(positive = positives, negative = negatives, restrict_vocab=50000,topn=20)
    print(clues)
    print(type(clues))

if __name__=='__main__':
    main()