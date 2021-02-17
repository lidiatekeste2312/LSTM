import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 

# reading the data
missing_values=["?"]
df = pd.read_csv("household_power_consumption.txt", delimiter=';', na_values=missing_values)
#dropping the date column
df.drop('Date', axis = 'columns', inplace = True )
# dropping the non numeric values
# for col in df.columns:
#     print (df[col].str.isnumeric().isnull())
        
df.dropna(inplace=True)

titles = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

feats = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink"
]

date_time_key = "Time"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    print (time_data[:5])
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feats)):
        key = feats[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

show_raw_visualization(df[:5])

targ = ["Global_active_power"]
feats_values = df[feats][:-1]
targ_values = df[targ][1:]

split_ratio = 0.7
split_index = int(len(feats_values)*split_ratio)
train_x = feats_values[:split_index]
train_y = targ_values[:split_index]
test_x = feats_values[split_index:]
test_y = targ_values[split_index:]

#Normalization
x_mean = train_x.mean(axis = 0)
x_std = train_x.std(axis = 0)
y_mean = train_y.mean(axis = 0)
y_std = train_y.std(axis = 0)
train_x =np.array((train_x - x_mean) / x_std)
test_x = np.array((test_x - x_mean) / x_std) 
train_y = np.array((train_y - y_mean) / y_std)
test_y = np.array((test_y - y_mean) / y_std)



#model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 100, activation = "relu"))
model.add(keras.layers.Dense(units = 1, activation = None))
model.compile(loss = 'mse', optimizer = "adam")
history = model.fit(train_x, train_y, epochs = 3, batch_size = 32, validation_split = 0.1)
train_pred = model.predict(train_x)
test_pred = model.predict(test_x)
fig,axs = plt.subplots(nrows = 2)

plt_indx = 1000
axs[0].plot(train_y[:plt_indx])
axs[0].plot(train_pred[:plt_indx])
axs[1].plot(test_y[:plt_indx])
axs[1].plot(test_pred[:plt_indx])
axs[1].legend("True", "Predicted")
axs[0].set_title("Training set")
axs[1].plot("Testing set")
plt.show()