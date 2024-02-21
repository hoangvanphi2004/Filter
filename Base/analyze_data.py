import pandas;
import matplotlib.pyplot as plt;

df = pandas.read_csv("ImageSizeDataFrame.csv");

df['ratio-img-size'] = df['img-width'] / df['img-height'];
print(df.describe())
plt.hist(df['img-height']);
plt.show(); 

