import Data
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    data = Data.load_smarket_dataset()

    # Display data correlation
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(data.corr().round(2), annot=True, ax=ax)
    plt.show()

    # Display average volume against year
    plt.scatter(
        x=data['Year'].unique(),
        y=data.groupby('Year').mean()['Volume'])
    plt.xlabel('Year')
    plt.ylabel('Volume')
    plt.show()


if __name__ == '__main__':
    main()
