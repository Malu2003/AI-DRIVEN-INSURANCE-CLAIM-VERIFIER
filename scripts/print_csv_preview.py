import csv
import sys

def preview(path, n=5):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            if i >= n-1:
                break

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/mimic_training_sample.csv'
    preview(path)
