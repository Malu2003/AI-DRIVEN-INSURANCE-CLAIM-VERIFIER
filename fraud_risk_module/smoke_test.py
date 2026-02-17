from feature_builder import load_dataset, build_X_y

df = load_dataset()
X, y = build_X_y(df)
print('Loaded df shape:', df.shape)
print('X shape:', X.shape)
print('y counts:', dict(y.value_counts()))
